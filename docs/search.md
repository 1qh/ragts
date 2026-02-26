# Search

## Query (High-Level)

`query()` chains retrieve, rerank, dedup, and context building:

```typescript
const { context, results } = await rag.query({
  query: 'What is TypeScript?',
  embed,
  rerank: { model: rerank, topN: 10 }
})
```

Pass `context` directly to your LLM.

## Retrieve (Low-Level)

`retrieve()` returns raw search results without reranking or context building:

```typescript
const results = await rag.retrieve({
  query: 'What is TypeScript?',
  embed,
  mode: 'hybrid',
  limit: 50
})
```

## Search Modes

| Mode | Engine |
|---|---|
| `hybrid` | RRF fusion of vector + BM25 (default) |
| `vector` | DiskANN approximate nearest neighbor |
| `bm25` | Full-text search via pg_textsearch |

```typescript
await rag.retrieve({ query, embed, mode: 'vector' })
await rag.retrieve({ query, embed, mode: 'bm25' })
await rag.retrieve({ query, embed, mode: 'hybrid' })
```

## RRF Tuning

Reciprocal Rank Fusion combines vector and BM25 results. Tune the weights:

```typescript
await rag.query({
  query,
  embed,
  vectorWeight: 1.5,
  bm25Weight: 0.8,
  rrfK: 60
})
```

- `vectorWeight` / `bm25Weight` — relative importance of each signal (default: 1)
- `rrfK` — RRF constant, higher = more uniform blending (default: 60)

## Reranking

Pass any AI SDK `RerankingModelV3`:

```typescript
const { results } = await rag.query({
  query,
  embed,
  rerank: { model: rerank, topN: 10 }
})
```

Or rerank manually:

```typescript
import { rerankChunks } from 'ragts'

const results = await rag.retrieve({ query, embed, limit: 50 })
const reranked = await rerankChunks(query, results, { model: rerank, topN: 10 })
```

## Dedup

`query()` removes chunks that are substrings of longer chunks (enabled by default):

```typescript
await rag.query({ query, embed, dedup: false })
```

Or manually:

```typescript
import { dedupSubstrings } from 'ragts'

const deduped = dedupSubstrings(results, { prefixLength: 100 })
```

## HyDE

Use a hypothetical document embedding for vector search while keeping the original query for BM25:

```typescript
const hypothesis = await generateText({ model: chat, prompt: `Answer: ${query}` })

await rag.query({
  query,
  embed,
  vectorQuery: hypothesis.text
})
```

## Similarity Threshold

Filter vector results below a minimum cosine similarity:

```typescript
await rag.query({ query, embed, threshold: 0.5 })
```
