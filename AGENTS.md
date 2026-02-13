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

Beat 75.7% accuracy on 48 Vietnamese legal QA pairs (`exp/qna.json`). Previous best was 28/37 scoreable = 75.7% (v6 with Ollama, 768-dim, no reranking). A rerun showed non-determinism: 21/35 = 60.0%.

## Architecture

### Inference Stack (Custom FastAPI Server — `serve.py`)

3 MLX models served from a single Python FastAPI server on port 8000:

| Endpoint | Model | Purpose | Dim |
|----------|-------|---------|-----|
| `POST /v1/embeddings` | Qwen3-VL-Embedding-2B-mlx-nvfp4 | Embed text | 2048 |
| `POST /v1/rerank` | Qwen3-VL-Reranker-2B-mlx-nvfp4 | Rerank chunks | — |
| `POST /v1/chat/completions` | Qwen3-VL-8B-Instruct-MLX-4bit | Generate answers | — |
| `GET /health` | — | Health check | — |

Model paths: `~/.lmstudio/models/arthurcollet/` (embed, rerank) and `~/.lmstudio/models/lmstudio-community/` (chat).

### Pipeline

ingest (chunk → embed → store) → retrieve (vector+BM25 hybrid) → rerank → generate answer

### Tech Stack

- **DB**: `timescale/timescaledb-ha:pg18` via Docker (`packages/ragts/docker/docker-compose.yml`)
- **Search**: pgvectorscale DiskANN (vector) + pg_textsearch BM25 (fulltext), RRF fusion
- **Reranking**: Custom `RerankingModelV3` adapter in `apps/demo/scripts/utils.ts`
- **Generation**: AI SDK (`@ai-sdk/openai-compatible`, `generateText`) treating serve.py as OpenAI-compatible API
- **BM25 text config**: `simple` (for Vietnamese, not `english`)

## Critical Implementation Details

### serve.py

- Uses `mlx_lm.load()` for all 3 models (all are `qwen3_vl.Model` type)
- **Embedding**: `lm.language_model.model(input_ids)` → last-token-pool → L2 normalize → 2048-dim vectors
- **Reranking**: Full forward pass → `sigmoid(logit_yes - logit_no)` at last position. `yes_token_id=9693`, `no_token_id=2152`
- **Chat**: Must use `make_sampler(temp=, top_p=)` from `mlx_lm.sample_utils` — `stream_generate` does NOT accept temp/top_p directly
- **Tokenizer**: `tok._tokenizer(texts, padding=True, truncation=True, max_length=2048, return_tensors='np')` — `TokenizerWrapper` has no `batch_encode_plus()`
- **Strips** `<think>...</think>` tags from chat output
- **Locks**: Each model has its own `asyncio.Lock()` — MLX models are NOT thread-safe
- `ruff format && ruff check --fix && ty check` must pass. Uses `noqa: SLF001` for `tok._tokenizer`, `noqa: S104` for `0.0.0.0`
- Python 3.14.3 via `uv`, venv at `.venv/`

### AI SDK Reranking Adapter

`@ai-sdk/openai-compatible` does NOT have `.reranking()`. Custom `RerankingModelV3` implementation in `utils.ts`:
- `specificationVersion: 'v3'`, calls `POST /v1/rerank` directly
- In `rerank()` result, use `entry.originalIndex` and `entry.score` (not `entry.index` / `entry.relevanceScore`)

### Embedding Speed (Apple Silicon MLX)

- ~0.7-0.85s per long Vietnamese text (~500-700 tokens) regardless of batch size
- Batch size has minimal impact — model forward pass is the bottleneck, not padding
- `max_length=2048` in tokenizer (reduced from 8192, saves memory but minimal speed gain)
- With chunk_size=512: ~55 chunks/doc → ~40s/doc → ~24h total for 2189 docs
- With chunk_size=1024: ~17 chunks/doc → ~16s/doc → ~10h total
- With chunk_size=2048: ~13 chunks/doc → ~10s/doc → ~6h total
- Larger chunks = faster ingest AND potentially better retrieval for legal text (preserves context)

## Demo Scripts (`apps/demo/scripts/`)

| Script | Purpose | Key flags |
|--------|---------|-----------|
| `ingest.ts` | Chunk, embed, store docs | `--folder`, `--backup`, `--chunk-size`, `--batch-size` |
| `eval.ts` | Run QA evaluation | `--system`, `--limit`, `--rerank-top`, `--mode`, `--output`, `--no-rerank` |
| `score.ts` | Score eval results | `<eval-output.json>` |
| `answer.ts` | Single question answer | `--query`, `--system`, `--no-rerank` |
| `search.ts` | Search only | `--query`, `--limit`, `--mode` |
| `backup.ts` | Export backup | `--output` |
| `wipe.ts` | Drop all tables | — |

## Experiment Data (`exp/` — gitignored)

- `exp/data/` — 2189 Vietnamese law `.md` files (some are 1.6MB)
- `exp/qna.json` — 48 QnA pairs
- `exp/prompt.txt` — tuned Vietnamese legal system prompt
- `exp/backup_2048.jsonl` — partial backup (only 6 docs, chunk_size=2048)
- `exp/out_best.json` — previous best: 75.7% (Ollama era, 768-dim)
- `exp/out_rerun.json` — reproducibility test: 60.0%

## Monitoring Rules

- ALWAYS use `--max-time` on curl (5s for health, 30s for embed, 60s for chat)
- NEVER chain long commands; run separately and check between
- For long operations: `nohup ... > log 2>&1 &` then poll `tail -3 log` every 30-60s
- Check BOTH `serve.log` AND `ingest.log` for errors
- Check `ps -p PID -o pid=` to verify process is alive before polling logs
- Some docs are 1.6MB — they produce hundreds of chunks and take minutes each

## TODO: Unfinished Work

### 0. Environment Setup (new machine)

```bash
# Install bun (if not installed)
curl -fsSL https://bun.sh/install | bash

# Install uv (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone/pull repo and install Node dependencies
cd /path/to/ragts
bun install

# Setup Python venv (must use latest Python, currently 3.14)
uv python install 3.14
uv venv --python 3.14
source .venv/bin/activate
uv pip install mlx mlx-lm mlx-vlm fastapi uvicorn pydantic

# Verify Python setup
ruff format && ruff check --fix && ty check

# Start Docker (TimescaleDB)
docker compose -f packages/ragts/docker/docker-compose.yml up -d
# Wait ~5s, then verify:
docker exec docker-db-1 psql -U postgres -c "SELECT 1"

# Create test database (needed for `bun test`)
docker exec docker-db-1 psql -U postgres -c "CREATE DATABASE ragts_test;"

# Verify library
cd packages/ragts && bun test  # expect 38 pass
cd ../..
bun fix  # expect zero errors

# Download MLX models (if not present)
# These must exist at these exact paths:
#   ~/.lmstudio/models/arthurcollet/Qwen3-VL-Embedding-2B-mlx-nvfp4/
#   ~/.lmstudio/models/arthurcollet/Qwen3-VL-Reranker-2B-mlx-nvfp4/
#   ~/.lmstudio/models/lmstudio-community/Qwen3-VL-8B-Instruct-MLX-4bit/
# If missing, download via LM Studio or huggingface-cli:
#   huggingface-cli download arthurcollet/Qwen3-VL-Embedding-2B-mlx-nvfp4 --local-dir ~/.lmstudio/models/arthurcollet/Qwen3-VL-Embedding-2B-mlx-nvfp4
#   huggingface-cli download arthurcollet/Qwen3-VL-Reranker-2B-mlx-nvfp4 --local-dir ~/.lmstudio/models/arthurcollet/Qwen3-VL-Reranker-2B-mlx-nvfp4
#   huggingface-cli download lmstudio-community/Qwen3-VL-8B-Instruct-MLX-4bit --local-dir ~/.lmstudio/models/lmstudio-community/Qwen3-VL-8B-Instruct-MLX-4bit

# Start serve.py
nohup .venv/bin/python3 serve.py > serve.log 2>&1 &
# Wait ~10-30s for all 3 models to load (watch serve.log for "All models loaded.")
# Then verify:
curl -sf --max-time 10 http://localhost:8000/health
curl -sf --max-time 30 http://localhost:8000/v1/embeddings -H 'Content-Type: application/json' -d '{"input":["test"],"model":"embed"}' | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'dim={len(d[\"data\"][0][\"embedding\"])}')"
# Should print: dim=2048
```

### 1. Ingest all 2189 docs (BLOCKED on previous machine — RAM overflow)

DB has 6 docs ingested with chunk_size=2048. Backup has 6 lines. Need to wipe and start fresh:
```bash
# Wipe DB (clean slate)
docker compose -f packages/ragts/docker/docker-compose.yml down -v
docker compose -f packages/ragts/docker/docker-compose.yml up -d
# Wait for DB:
sleep 5 && docker exec docker-db-1 psql -U postgres -c "SELECT 1"
# Recreate test DB:
docker exec docker-db-1 psql -U postgres -c "CREATE DATABASE ragts_test;"

# Delete old backup
rm -f exp/backup_2048.jsonl

# Ensure serve.py is running (see step 0)

# Ingest (chunk-size=2048 for speed, or 1024 for balance)
nohup bun apps/demo/scripts/ingest.ts \
  --folder exp/data --backup exp/backup_2048.jsonl \
  --chunk-size 2048 --batch-size 4 > ingest.log 2>&1 &

# Monitor (poll every 30-60s, NEVER let it hang unnoticed):
tail -3 ingest.log
docker exec docker-db-1 psql -U postgres -d postgres -t -c "SELECT COUNT(*) FROM documents;"
ps -p <PID> -o pid=  # verify alive
tail -3 serve.log  # verify server not crashed
```

Expected timing: ~0.7-0.85s per chunk embedding. With chunk_size=2048 (~13 chunks/doc), ~10s/doc, ~6h total for 2189 docs. With 64GB RAM this should complete without OOM.

### 2. Run eval with reranking
```bash
bun apps/demo/scripts/eval.ts exp/qna.json \
  --system "$(cat exp/prompt.txt)" \
  --limit 20 --rerank-top 10 --mode hybrid \
  --output exp/out_mlx_v1.json
```

### 3. Score and iterate
```bash
bun apps/demo/scripts/score.ts exp/out_mlx_v1.json
```

Tunable parameters:
- `--chunk-size` (at ingest time): 512, 1024, 2048
- `--limit`: retrieve count before rerank (try 15, 20, 30)
- `--rerank-top`: keep top N after rerank (try 5, 8, 10, 15)
- `--mode`: hybrid (default), vector, bm25
- `--no-rerank`: disable reranking
- `exp/prompt.txt`: system prompt (Vietnamese legal expert instructions)
- temperature in serve.py ChatRequest (default 0.7, try 0.3 for less randomness)

### 4. Library optimizations still possible (lower priority)
- Consolidate init SQL into fewer roundtrips
- Streaming backup validation for large files
