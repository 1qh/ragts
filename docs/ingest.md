# Ingestion

## Basic

```typescript
import { Rag } from 'ragts'

const rag = new Rag({ connectionString: 'postgresql://postgres:postgres@localhost:5432/postgres' })

await rag.ingest(
  [
    { title: 'doc-1', content: 'First document content...' },
    { title: 'doc-2', content: 'Second document content...' }
  ],
  { embed }
)
```

Documents with identical content hashes are skipped. Chunks with identical text are stored once and linked to all parent documents.

## Chunking

```typescript
await rag.ingest(docs, {
  embed,
  chunk: {
    chunkSize: 2048,
    overlap: 200,
    normalize: (text) => text.replaceAll('\r\n', '\n')
  }
})
```

The chunker is markdown-aware — it splits on headings, paragraphs, and list boundaries before falling back to character limits. Hard line breaks from OCR/PDF are unwrapped automatically.

## Transform Chunks

Prepend document metadata to each chunk before embedding:

```typescript
await rag.ingest(docs, {
  embed,
  transformChunk: (text, doc) => `Title: ${doc.title}\n\n${text}`
})
```

## Batch Size

Control how many chunks are embedded per API call:

```typescript
await rag.ingest(docs, { embed, batchSize: 32 })
```

## Progress

```typescript
await rag.ingest(docs, {
  embed,
  onProgress: (title, current, total) => {
    console.log(`[${current}/${total}] ${title}`)
  }
})
```

## Backup During Ingest

Write a JSONL backup while ingesting:

```typescript
await rag.ingest(docs, { embed, backupPath: './backup.jsonl' })
```

## Result

`ingest()` returns stats:

```typescript
const result = await rag.ingest(docs, { embed })

result.documentsInserted   // new documents added
result.duplicatesSkipped   // skipped (same content hash)
result.chunksInserted      // new chunks created
result.chunksReused        // existing chunks linked
result.relationsInserted   // graph edges created
result.communitiesDetected // communities found
result.unresolvedRelations // relation targets not found
```
