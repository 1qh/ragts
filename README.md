TypeScript RAG with PostgreSQL, pgvectorscale (DiskANN), and BM25.

You bring the embedding function and the LLM. We handles chunk, embed, store, and retrieve.

## Quickstart

```bash
bun add ragts
bunx ragts gen-compose && docker compose up -d
```

```typescript
import { Rag } from 'ragts'

const embed = async (texts: string[]) => {
  const res = await fetch('http://localhost:11434/api/embed', {
    method: 'POST',
    body: JSON.stringify({ model: 'nomic-embed-text', input: texts })
  })
  return (await res.json()).embeddings
}

const rag = new Rag({ connectionString: 'postgresql://postgres:postgres@localhost:5432/postgres' })

await rag.ingest([{ title: 'Intro', content: 'TypeScript is a typed superset of JavaScript.' }], { embed })

const results = await rag.retrieve({ query: 'What is TypeScript?', embed })

await rag.close()
```

## API

| Method | Description |
|--------|-------------|
| `new Rag({ connectionString, dimension? })` | Create instance (default dim: 768, lazy init) |
| `rag.ingest(docs, { embed })` | Chunk, embed, and store documents |
| `rag.retrieve({ query, embed, mode?, limit? })` | Search by vector, bm25, or hybrid (default) |
| `rag.exportBackup(path)` | Export to JSONL |
| `rag.importBackup(path)` | Import from JSONL |
| `rag.init()` | Explicit init (returns Drizzle db instance) |
| `rag.drop()` | Drop all tables |
| `rag.close()` | Close connection |

## Retrieve Modes

| Mode | Engine |
|------|--------|
| `vector` | DiskANN approximate nearest neighbor via pgvectorscale |
| `bm25` | Full-text search via pg_textsearch |
| `hybrid` | RRF fusion of vector + bm25 (default) |

## Drizzle Access

`rag.init()` returns the Drizzle db instance for custom queries.

```typescript
import { Rag, documents, count } from 'ragts'

const db = await rag.init()
const total = await db.select({ count: count() }).from(documents)
```

## Backup

```typescript
await rag.exportBackup('./backup.jsonl')
await rag.importBackup('./backup.jsonl')

import { validateBackup } from 'ragts'
validateBackup('./backup.jsonl')
```
