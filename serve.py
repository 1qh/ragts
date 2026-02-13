import asyncio
import json
import operator
import time
from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager
from pathlib import Path

import mlx.core as mx
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from mlx import nn
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tokenizer_utils import TokenizerWrapper
from pydantic import BaseModel

MODELS_DIR = Path.home() / '.lmstudio' / 'models'
EMBED_PATH = str(MODELS_DIR / 'arthurcollet' / 'Qwen3-VL-Embedding-2B-mlx-nvfp4')
RERANK_PATH = str(MODELS_DIR / 'arthurcollet' / 'Qwen3-VL-Reranker-2B-mlx-nvfp4')
CHAT_PATH = str(MODELS_DIR / 'lmstudio-community' / 'Qwen3-VL-8B-Instruct-MLX-4bit')

YES_TOKEN_ID = 9693
NO_TOKEN_ID = 2152

RERANK_SYSTEM = (
  'Judge whether the Document meets the requirements based on the Query'
  ' and the Instruct provided. Note that the answer can only be "yes" or "no".'
)
RERANK_INSTRUCT = 'Given a search query, retrieve relevant candidates that answer the query.'


def last_token_pool(hidden: mx.array, attention_mask: mx.array) -> mx.array:
  seq_lengths = attention_mask.sum(axis=1).astype(mx.int32) - 1
  batch_indices = mx.arange(hidden.shape[0])
  return hidden[batch_indices, seq_lengths]


def l2_normalize(x: mx.array) -> mx.array:
  norms = mx.sqrt((x * x).sum(axis=-1, keepdims=True))
  return x / mx.maximum(norms, mx.array(1e-12))


def sse_chunk(
  chunk_id: str,
  created: int,
  model: str,
  delta: dict,
  finish: str | None = None,
) -> str:
  payload = {
    'id': chunk_id,
    'object': 'chat.completion.chunk',
    'created': created,
    'model': model,
    'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish}],
  }
  return f'data: {json.dumps(payload, ensure_ascii=False)}\n\n'


class EmbedRequest(BaseModel):
  input: str | list[str]
  model: str = 'qwen3-vl-embedding'


class RerankRequest(BaseModel):
  query: str
  documents: list[str]
  model: str = 'qwen3-vl-reranker'


class ChatMessage(BaseModel):
  role: str
  content: str


class ChatRequest(BaseModel):
  model: str = 'qwen3-vl-chat'
  messages: list[ChatMessage]
  max_tokens: int = 2048
  temperature: float = 0.7
  top_p: float = 0.9
  stream: bool = False


class ModelRegistry:
  def __init__(  # noqa: PLR0913
    self,
    *,
    embed_model: nn.Module,
    embed_tokenizer: TokenizerWrapper,
    rerank_model: nn.Module,
    rerank_tokenizer: TokenizerWrapper,
    chat_model: nn.Module,
    chat_tokenizer: TokenizerWrapper,
  ) -> None:
    self.embed_model = embed_model
    self.embed_tokenizer = embed_tokenizer
    self.rerank_model = rerank_model
    self.rerank_tokenizer = rerank_tokenizer
    self.chat_model = chat_model
    self.chat_tokenizer = chat_tokenizer
    self.embed_lock = asyncio.Lock()
    self.rerank_lock = asyncio.Lock()
    self.chat_lock = asyncio.Lock()

  @staticmethod
  def create() -> 'ModelRegistry':
    print('Loading embedding model...')
    embed_m, embed_t, *_ = load(EMBED_PATH)
    print('Loading reranker model...')
    rerank_m, rerank_t, *_ = load(RERANK_PATH)
    print('Loading chat model...')
    chat_m, chat_t, *_ = load(CHAT_PATH)
    print('All models loaded.')
    return ModelRegistry(
      embed_model=embed_m,
      embed_tokenizer=embed_t,
      rerank_model=rerank_m,
      rerank_tokenizer=rerank_t,
      chat_model=chat_m,
      chat_tokenizer=chat_t,
    )


registry: ModelRegistry


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None]:  # noqa: RUF029
  global registry  # noqa: PLW0603
  registry = ModelRegistry.create()
  yield


app = FastAPI(lifespan=lifespan)


@app.get('/health')
async def health() -> dict[str, str]:
  return {'status': 'ok'}


@app.get('/v1/models')
async def list_models() -> dict[str, list[dict[str, str]]]:
  return {
    'data': [
      {'id': 'qwen3-vl-embedding', 'object': 'model'},
      {'id': 'qwen3-vl-reranker', 'object': 'model'},
      {'id': 'qwen3-vl-chat', 'object': 'model'},
    ]
  }


@app.post('/v1/embeddings')
async def embeddings(request: EmbedRequest) -> dict:
  async with registry.embed_lock:
    texts = [request.input] if isinstance(request.input, str) else request.input
    lm = registry.embed_model.language_model
    tok = registry.embed_tokenizer

    encoded = tok._tokenizer(  # noqa: SLF001
      texts,
      padding=True,
      truncation=True,
      max_length=2048,
      return_tensors='np',
    )
    input_ids = mx.array(encoded['input_ids'])
    attention_mask = mx.array(encoded['attention_mask'])

    hidden = lm.model(input_ids)
    pooled = last_token_pool(hidden, attention_mask)
    normalized = l2_normalize(pooled)
    mx.eval(normalized)

    data = []
    total_tokens = 0
    for i in range(len(texts)):
      vec = normalized[i].tolist()
      data.append({'object': 'embedding', 'index': i, 'embedding': vec})
      total_tokens += int(attention_mask[i].sum().item())

    return {
      'object': 'list',
      'data': data,
      'model': request.model,
      'usage': {'prompt_tokens': total_tokens, 'total_tokens': total_tokens},
    }


@app.post('/v1/rerank')
async def rerank(request: RerankRequest) -> dict:
  async with registry.rerank_lock:
    lm = registry.rerank_model.language_model
    tok = registry.rerank_tokenizer

    results = []
    for idx, doc in enumerate(request.documents):
      user_content = f'<Instruct>: {RERANK_INSTRUCT}\n<Query>: {request.query}\n<Document>: {doc}'
      messages = [
        {'role': 'system', 'content': RERANK_SYSTEM},
        {'role': 'user', 'content': user_content},
      ]
      prompt = tok.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
      )

      token_ids = tok.encode(prompt)
      input_ids = mx.array([token_ids])

      logits = lm(input_ids)
      yes_logit = logits[0, -1, YES_TOKEN_ID]
      no_logit = logits[0, -1, NO_TOKEN_ID]
      score = float(mx.sigmoid(yes_logit - no_logit).item())
      mx.eval(logits)

      results.append({
        'index': idx,
        'relevance_score': score,
        'document': {'text': doc},
      })

    results.sort(
      key=operator.itemgetter('relevance_score'),
      reverse=True,
    )
    return {'results': results, 'model': request.model}


def _stream_chat_sync(
  request: ChatRequest,
) -> Generator[str]:
  msgs = [{'role': m.role, 'content': m.content} for m in request.messages]
  tok = registry.chat_tokenizer
  prompt = tok.apply_chat_template(
    msgs,
    add_generation_prompt=True,
    tokenize=False,
  )

  created = int(time.time())
  chunk_id = f'chatcmpl-{created}'

  yield sse_chunk(
    chunk_id,
    created,
    request.model,
    {'role': 'assistant', 'content': ''},
  )

  sampler = make_sampler(temp=request.temperature, top_p=request.top_p)
  for resp in stream_generate(
    registry.chat_model,
    tok,
    prompt=prompt,
    max_tokens=request.max_tokens,
    sampler=sampler,
  ):
    if resp.text:
      yield sse_chunk(
        chunk_id,
        created,
        request.model,
        {'content': resp.text},
      )

  yield sse_chunk(
    chunk_id,
    created,
    request.model,
    {},
    finish='stop',
  )
  yield 'data: [DONE]\n\n'


@app.post('/v1/chat/completions', response_model=None)
async def chat_completions(
  request: ChatRequest,
) -> dict | StreamingResponse:
  async with registry.chat_lock:
    if request.stream:
      return StreamingResponse(
        _stream_chat_sync(request),
        media_type='text/event-stream',
      )

    msgs = [{'role': m.role, 'content': m.content} for m in request.messages]
    tok = registry.chat_tokenizer
    prompt = tok.apply_chat_template(
      msgs,
      add_generation_prompt=True,
      tokenize=False,
    )

    full_text = ''
    prompt_tokens = 0
    completion_tokens = 0
    sampler = make_sampler(temp=request.temperature, top_p=request.top_p)
    for resp in stream_generate(
      registry.chat_model,
      tok,
      prompt=prompt,
      max_tokens=request.max_tokens,
      sampler=sampler,
    ):
      full_text += resp.text
      if resp.prompt_tokens:
        prompt_tokens = resp.prompt_tokens
      completion_tokens += 1

    if full_text.startswith('<think>'):
      think_end = full_text.find('</think>')
      if think_end != -1:
        full_text = full_text[think_end + len('</think>') :].strip()

    created = int(time.time())
    return {
      'id': f'chatcmpl-{created}',
      'object': 'chat.completion',
      'created': created,
      'model': request.model,
      'choices': [
        {
          'index': 0,
          'message': {'role': 'assistant', 'content': full_text},
          'finish_reason': 'stop',
        }
      ],
      'usage': {
        'prompt_tokens': prompt_tokens,
        'completion_tokens': completion_tokens,
        'total_tokens': prompt_tokens + completion_tokens,
      },
    }


if __name__ == '__main__':
  import uvicorn

  uvicorn.run(app, host='0.0.0.0', port=8000)  # noqa: S104
