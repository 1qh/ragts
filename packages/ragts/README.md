# ragts

TypeScript RAG with PostgreSQL, pgvectorscale (DiskANN), and BM25.

You bring the models. We handle chunk, embed, store, retrieve, rerank.

## Install

```bash
bun add ragts
bunx ragts gen-compose && docker compose up -d
```

## Quickstart

```typescript
import { createProvider, Rag } from 'ragts'
import { generateText } from 'ai'

const provider = createProvider({ baseURL: 'http://localhost:8000' })
const embed = provider.embedFn('my-embed-model')
const rerank = provider.rerankingModel('my-rerank-model')
const chat = provider.chatModel('my-chat-model')

const rag = new Rag({ connectionString: 'postgresql://postgres:postgres@localhost:5432/postgres' })

await rag.ingest([{ title: 'Intro', content: 'TypeScript is a typed superset of JavaScript.' }], { embed })

const { context } = await rag.query({
  query: 'What is TypeScript?',
  embed,
  rerank: { model: rerank, topN: 5 }
})

const { text } = await generateText({ model: chat, prompt: context, system: 'Answer from context.' })

await rag.close()
```

Works with any OpenAI-compatible server (vLLM, Ollama, MLX, etc.).

## Features

- **Hybrid search** — vector (DiskANN) + BM25 with RRF fusion
- **Graph-enhanced retrieval** — document relations, weighted edges, community detection, depth decay
- **Chunk dedup** — identical chunks stored once, linked via junction table
- **Hierarchical chunking** — markdown-aware splitting with hard break unwrapping
- **JSONL backup/restore** — export, import, validate
- **Drizzle ORM access** — custom queries on the underlying schema

## Guides

| Guide | Topics |
|---|---|
| [Ingestion](https://github.com/1qh/ragts/blob/main/docs/ingest.md) | Chunking, embedding, dedup, transforms, progress |
| [Search](https://github.com/1qh/ragts/blob/main/docs/search.md) | Modes, RRF tuning, reranking, HyDE, thresholds |
| [Graph retrieval](https://github.com/1qh/ragts/blob/main/docs/graph.md) | Relations, weights, decay, communities, global queries |
| [Backup & restore](https://github.com/1qh/ragts/blob/main/docs/backup.md) | Export, import, validate, JSONL format |
## API

```typescript
const rag = new Rag({ connectionString, dimension?, textConfig? })

await rag.ingest(docs, { embed, chunk?, batchSize?, relations? })
const { context, results } = await rag.query({ query, embed, rerank?, limit?, mode? })
const results = await rag.retrieve({ query, embed, mode?, limit? })

await rag.exportBackup(path)
await rag.importBackup(path)

await rag.detectCommunities()
await rag.buildCommunitySummaries({ embed, summarize })
const { answer } = await rag.globalQuery({ embed, query, generate })
const relations = await rag.fetchRelations(docIds)

await rag.close()
```

See [types.ts](https://github.com/1qh/ragts/blob/main/packages/ragts/src/types.ts) for all config options.

## Advanced: Graph-Enhanced Retrieval

Define document relations to expand search to related documents:

```typescript
await rag.ingest(docs, {
  embed,
  relations: {
    'doc-a': [{ title: 'doc-b', type: 'implements', weight: 0.9 }],
    'doc-b': ['doc-c']
  }
})

const { context } = await rag.query({
  query: 'my question',
  embed,
  graphHops: 2,
  graphWeight: 1.5,
  graphDecay: 0.7,
  communityBoost: 0.5
})
```

- `graphHops` — how many relation hops to traverse
- `graphWeight` — relative importance of graph results in RRF fusion
- `graphDecay` — per-hop decay factor (0.7 = 30% reduction per hop)
- `communityBoost` — expand from dominant community via vector similarity

Community detection runs automatically after ingest. For summaries and global queries:

```typescript
await rag.buildCommunitySummaries({ embed, summarize: async (docs) => '...' })
const { answer } = await rag.globalQuery({ embed, query: '...', generate: async (ctx, q) => '...' })
```

## Database

Uses `timescale/timescaledb-ha:pg18` with pgvectorscale + pg_textsearch. Schema is created automatically on first use.

```bash
bunx ragts gen-compose
docker compose up -d
```
