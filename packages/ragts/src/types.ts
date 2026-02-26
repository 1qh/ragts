import type { RerankingModelV3 } from '@ai-sdk/provider'

interface BackupChunk {
  embedding: number[]
  endIndex: number
  startIndex: number
  text: string
  tokenCount: number
}

interface BackupDoc {
  chunks: BackupChunk[]
  communityId?: number
  content: string
  contentHash: string
  metadata: Record<string, unknown>
  relations?: BackupRelation[]
  title: string
}

interface BackupRelation {
  title: string
  type?: string
  weight?: number
}

interface ChunkConfig {
  chunkSize?: number
  normalize?: (text: string) => string
  overlap?: number
}

interface CommunitySummaryConfig {
  chunk?: ChunkConfig
  embed: EmbedFn
  minCommunitySize?: number
  summarize: (docs: { content: string; title: string }[]) => Promise<string>
}

interface CommunitySummaryResult {
  communitiesProcessed: number
  summariesGenerated: number
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

interface GlobalQueryConfig {
  embed: EmbedFn
  generate: (context: string, query: string) => Promise<string>
  limit?: number
  maxCommunities?: number
  query: string
  rerank?: { model: RerankingModelV3; topN?: number }
}

interface GlobalQueryResult {
  answer: string
  partialAnswers: { answer: string; communityId: number; titles: string[] }[]
}

interface GraphRelation {
  sourceTitle: string
  targetTitle: string
  type?: string
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
  relations?: Record<string, RelationTarget[]>
  transformChunk?: (text: string, doc: Doc) => string
}

interface IngestResult {
  chunksInserted: number
  chunksReused: number
  communitiesDetected: number
  documentsInserted: number
  duplicatesSkipped: number
  relationsInserted: number
  unresolvedRelations: string[]
}

interface QueryConfig {
  bm25Weight?: number
  communityBoost?: number
  dedup?: boolean
  embed: EmbedFn
  graphChunkLimit?: number
  graphDecay?: number
  graphHops?: number
  graphWeight?: number
  limit?: number
  mode?: SearchMode
  query: string
  rerank?: { model: RerankingModelV3; topN?: number }
  rrfK?: number
  threshold?: number
  vectorQuery?: string
  vectorWeight?: number
}

interface QueryResult {
  context: string
  results: SearchResult[]
}

interface RagtsConfig {
  connectionString: string
  dimension?: number
  textConfig?: string
}

type RelationTarget = string | { title: string; type?: string; weight?: number }

interface SearchConfig {
  bm25Weight?: number
  communityBoost?: number
  embed: EmbedFn
  graphChunkLimit?: number
  graphDecay?: number
  graphHops?: number
  graphWeight?: number
  limit?: number
  mode?: SearchMode
  query: string
  rrfK?: number
  threshold?: number
  vectorQuery?: string
  vectorWeight?: number
}

type SearchMode = 'bm25' | 'hybrid' | 'vector'

interface SearchResult {
  communityId?: number
  documentId: number
  id: number
  mode: 'bm25' | 'community' | 'graph' | 'vector'
  relationType?: string
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
  BackupRelation,
  ChunkConfig,
  CommunitySummaryConfig,
  CommunitySummaryResult,
  Doc,
  EmbedFn,
  ExportResult,
  GlobalQueryConfig,
  GlobalQueryResult,
  GraphRelation,
  ImportResult,
  IngestConfig,
  IngestResult,
  QueryConfig,
  QueryResult,
  RagtsConfig,
  RelationTarget,
  SearchConfig,
  SearchMode,
  SearchResult,
  ValidationResult
}
