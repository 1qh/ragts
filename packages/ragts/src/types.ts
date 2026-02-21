interface BackupChunk {
  embedding: number[]
  endIndex: number
  startIndex: number
  text: string
  tokenCount: number
}

interface BackupDoc {
  chunks: BackupChunk[]
  content: string
  contentHash: string
  metadata: Record<string, unknown>
  title: string
}

interface ChunkConfig {
  chunkSize?: number
  normalize?: (text: string) => string
  overlap?: number
}

interface Doc {
  content: string
  metadata?: Record<string, unknown>
  title: string
}

type EmbedFn = (texts: string[]) => Promise<number[][]>

interface ExportResult {
  documentsExported: number
  outputPath: string
}

interface ImportResult {
  chunksInserted: number
  documentsImported: number
  duplicatesSkipped: number
  warnings: string[]
}

interface IngestConfig {
  backupPath?: string
  batchSize?: number
  chunk?: ChunkConfig
  embed: EmbedFn
  onProgress?: (doc: string, current: number, total: number) => void
}

interface IngestResult {
  chunksInserted: number
  chunksReused: number
  documentsInserted: number
  duplicatesSkipped: number
}

interface RagtsConfig {
  connectionString: string
  dimension?: number
  textConfig?: string
}

interface SearchConfig {
  embed: EmbedFn
  limit?: number
  mode?: SearchMode
  query: string
  rrfK?: number
  threshold?: number
}

type SearchMode = 'bm25' | 'hybrid' | 'vector'

interface SearchResult {
  documentId: number
  id: number
  mode: 'bm25' | 'vector'
  score: number
  text: string
  title: string
}

interface ValidationResult {
  dimensions: Set<number>
  duplicateHashes: string[]
  errors: string[]
  totalChunks: number
  totalDocuments: number
  valid: boolean
}

export type {
  BackupChunk,
  BackupDoc,
  ChunkConfig,
  Doc,
  EmbedFn,
  ExportResult,
  ImportResult,
  IngestConfig,
  IngestResult,
  RagtsConfig,
  SearchConfig,
  SearchMode,
  SearchResult,
  ValidationResult
}
