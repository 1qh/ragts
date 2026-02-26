# Backup & Restore

## Export

```typescript
const { documentsExported, outputPath } = await rag.exportBackup('./backup.jsonl')
```

Exports all documents, chunks, embeddings, and relations as JSONL (one document per line).

## Import

```typescript
const result = await rag.importBackup('./backup.jsonl')

result.documentsImported
result.chunksInserted
result.duplicatesSkipped
result.warnings
```

Embeddings are preserved from the backup. Duplicate documents (same content hash) are skipped.

## Validate

Check a backup file without importing:

```typescript
import { validateBackup } from 'ragts'

const result = validateBackup('./backup.jsonl')

result.valid
result.totalDocuments
result.totalChunks
result.dimensions
result.errors
result.duplicateHashes
```

## Format

Each line is a JSON object:

```json
{
  "title": "doc-1",
  "content": "Full document text...",
  "contentHash": "sha256...",
  "metadata": {},
  "communityId": 0,
  "chunks": [
    {
      "text": "Chunk text...",
      "embedding": [0.1, 0.2, ...],
      "startIndex": 0,
      "endIndex": 512,
      "tokenCount": 128
    }
  ],
  "relations": [
    { "title": "doc-2", "type": "implements", "weight": 0.9 }
  ]
}
```
