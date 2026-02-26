# RULES

- only use `bun`, `yarn/npm/npx/pnpm` are forbidden
- `bun fix` must always pass
- `ruff format && ruff check --fix && ty check` must pass
- only use arrow functions
- all exports must be at end of file
- if a `.tsx` file only exports a single component, use `export default`
- `bun ts-unused-exports foo/bar/tsconfig.json` to detect and remove unused exports
- `bun why <package>` to check if a package is already installed, no need to install packages that are already dependencies of other packages

## Code Style

- consolidate into fewer files, co-locate small components
- `export default` for components, named exports for utilities/backend
- `catch (error)` is enforced by oxlint; name other error variables descriptively to avoid shadow

## Linting

| Linter | Ignore comment |
|--------|----------------|
| oxlint | `// oxlint-disable(-next-line) rule-name` |
| eslint | `// eslint-disable(-next-line) rule-name` |
| biomejs| `/** biome-ignore(-all) lint/category/rule: reason */` |

Run `bun fix` to auto-fix and verify all linters pass (zero errors, warnings allowed).

### Safe-to-ignore rules (only when cannot fix)

**oxlint:**

- `promise/prefer-await-to-then` - ky/fetch chaining

**eslint:**

- `no-await-in-loop`, `max-statements`, `complexity` - complex handlers
- `@typescript-eslint/no-unnecessary-condition` - type narrowing false positives
- `@typescript-eslint/promise-function-async` - functions returning thenable (not Promise)
- `@next/next/no-img-element` - external images without optimization
- `react-hooks/refs` - custom ref patterns

**biomejs:**

- `style/noProcessEnv` - env validation files
- `performance/noAwaitInLoops` - sequential async operations
- `nursery/noContinue`, `nursery/noForIn` - intentional control flow
- `performance/noImgElement` - external images
- `suspicious/noExplicitAny` - unavoidable generic boundaries

# PROHIBITIONS

- NEVER write comments at all (lint ignores are allowed)
- NEVER touch files inside `packages/ui` (shared frontend components, read-only)
- NEVER use `Array#reduce()`, use `for` loops instead
- NEVER use `forEach()`, use `for` loops instead
- NEVER use non-null assertion operator (`!`)
- NEVER use `any` type
- NEVER commit, only the user can commit
- NEVER use npm/yarn/npx/pnpm, only `bun`
- NEVER use ollama or lm-studio servers, we have our own custom `serve.py`
- NEVER hardcode data in code
- Max LLM size is 8b, max embedding/reranker is 2b

# RAG EXPERIMENT

## Goal

Maximize accuracy on 48 Vietnamese legal QA pairs (`exp/qna.json`).

## Architecture

### Inference Stack (`serve.py`)

3 MLX models served from a single Python FastAPI server on port 8000:

| Endpoint | Model | Purpose | Dim |
|----------|-------|---------|-----|
| `POST /v1/embeddings` | Qwen3-VL-Embedding-2B-mlx-nvfp4 | Embed text | 2048 |
| `POST /v1/rerank` | Qwen3-VL-Reranker-2B-mlx-nvfp4 | Rerank chunks | — |
| `POST /v1/chat/completions` | Qwen3-VL-8B-Instruct-MLX-4bit | Generate answers | — |
| `GET /health` | — | Health check | — |

Model paths: `~/models/`. Python 3.14.3 via `uv`, venv at `.venv/`.

### Pipeline

ingest (chunk → embed → store) → retrieve (vector+BM25 hybrid) → rerank → generate answer

### Tech Stack

- **DB**: `timescale/timescaledb-ha:pg18` via Docker (`packages/ragts/docker/docker-compose.yml`)
- **Search**: pgvectorscale DiskANN (vector) + pg_textsearch BM25 (fulltext), RRF fusion
- **Chunking**: Custom pure-TS chunker (`packages/ragts/src/chunk.ts`), hierarchical splitting with `unwrapHardBreaks` for PDF artifacts
- **Reranking**: Custom `RerankingModelV3` adapter in `apps/demo/scripts/utils.ts`
- **Generation**: AI SDK (`@ai-sdk/openai-compatible`, `generateText`)
- **BM25 text config**: `simple` (for Vietnamese, not `english`)

## serve.py Details

- `mlx_lm.load()` for all 3 models (all `qwen3_vl.Model` type)
- **Embedding**: Pipelined — processes texts individually (no padding), overlaps tokenization with GPU via `mx.async_eval`, uses `numpy` for fast array conversion. `mx.set_cache_limit(32GB)`. Each text: tokenize → `lm.language_model.model(input_ids)` → last-token-pool → L2 normalize → 2048-dim. ~3x faster than naive batched approach.
- **Reranking**: Forward pass → `sigmoid(logit_yes - logit_no)`. `yes_token_id=9693`, `no_token_id=2152`
- **Chat**: `make_sampler(temp=, top_p=)` from `mlx_lm.sample_utils` — `stream_generate` does NOT accept temp/top_p directly
- **Tokenizer**: `tok._tokenizer(texts, padding=True, truncation=True, max_length=2048, return_tensors='np')`
- **Strips** `<think>...</think>` tags from chat output
- **Locks**: Each model has its own `asyncio.Lock()` — MLX models are NOT thread-safe
- `ruff format && ruff check --fix && ty check` must pass

## Demo Scripts (`apps/demo/scripts/`)

| Script | Purpose | Key flags |
|--------|---------|-----------|
| `ingest.ts` | Chunk, embed, store docs | `--folder`, `--backup`, `--chunk-size`, `--batch-size` |
| `eval.ts` | Run QA evaluation | `--system`, `--limit`, `--rerank-top`, `--mode`, `--output`, `--no-rerank`, `--expand`, `--rrf-k`, `--temperature`, `--hyde`, `--multihop`, `--child-db`, `--agentic-rerank`, `--instruct`, `--vector-weight`, `--bm25-weight` |
| `selfcon.ts` | Self-consistency eval (majority vote) | `--system`, `--runs`, `--temperature`, `--limit`, `--rerank-top`, `--output`, `--rrf-k` |
| `score.ts` | Score eval results | `<eval-output.json>` |
| `answer.ts` | Single question answer | `--query`, `--system`, `--no-rerank` |
| `search.ts` | Search only | `--query`, `--limit`, `--mode` |
| `wipe.ts` | Drop all tables | `--yes` |
| `constants.ts` | Shared config (URLs, models, dimension) | — |
| `utils.ts` | Shared utilities (embed, rerank, parseArgs) | — |
| `verdict.ts` | Shared Vietnamese verdict extraction | — |

## Monitoring Rules

- ALWAYS use `--max-time` on curl (5s for health, 30s for embed, 60s for chat)
- NEVER chain long commands; run separately and check between
- For long operations: `nohup ... > exp/<name>.log 2>&1 &` then poll `tail -3 exp/<name>.log` every 30-60s
- Check `ps -p PID -o pid=` to verify process is alive before polling logs
- All `.log` files MUST go inside `exp/`, never in project root
