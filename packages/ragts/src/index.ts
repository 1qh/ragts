import type postgres from 'postgres'

import { sql } from 'drizzle-orm'

import type { DrizzleDb } from './db'
import type {
  Doc,
  EmbedFn,
  ExportResult,
  ImportResult,
  IngestConfig,
  IngestResult,
  RagtsConfig,
  SearchConfig,
  SearchResult
} from './types'

import { exportBackup, importBackup } from './backup'
import { createDb, dropSchema, initSchema } from './db'
import { ingest } from './ingest'
import { search } from './search'

const buildContext = (results: { text: string; title: string }[]): string => {
  let output = ''
  for (let i = 0; i < results.length; i += 1) {
    const row = results[i]
    if (row) output += `[${String(i + 1)}] ${row.title}\n${row.text}\n\n`
  }
  return output.trim()
}

class Rag {
  public db: DrizzleDb | undefined
  private _client: ReturnType<typeof postgres> | undefined
  private readonly connStr: string
  private readonly dim: number
  private readonly txtCfg: string

  public constructor(config: RagtsConfig) {
    this.connStr = config.connectionString
    this.dim = config.dimension ?? 768
    this.txtCfg = config.textConfig ?? 'english'
  }

  public close = async (): Promise<void> => {
    if (this._client) await this._client.end()
  }

  public drop = async (): Promise<void> => {
    if (this.db) {
      await this.db.execute(sql`DROP TABLE IF EXISTS chunks CASCADE`)
      await this.db.execute(sql`DROP TABLE IF EXISTS documents CASCADE`)
    } else await dropSchema(this.connStr)
    if (this._client) await this._client.end()
    this._client = undefined
    this.db = undefined
  }

  public exportBackup = async (outputPath: string): Promise<ExportResult> => {
    const db = await this.init()
    return exportBackup(db, outputPath)
  }

  public importBackup = async (filePath: string, config?: { embed?: EmbedFn }): Promise<ImportResult> => {
    const db = await this.init()
    return importBackup(db, filePath, config?.embed ? undefined : this.dim)
  }

  public ingest = async (docs: Doc[], config: IngestConfig): Promise<IngestResult> => {
    const db = await this.init()
    return ingest(db, docs, config)
  }

  public init = async (): Promise<DrizzleDb> => {
    if (this.db) return this.db
    const { client, db } = createDb(this.connStr)
    await initSchema(db, this.dim, this.txtCfg)
    this._client = client
    this.db = db
    return db
  }

  public retrieve = async (config: SearchConfig): Promise<SearchResult[]> => {
    const db = await this.init()
    return search(db, config)
  }
}

export { buildContext, Rag }
export { validateBackup } from './backup'
export { generateCompose } from './compose'
export type { DrizzleDb } from './db'
export { chunks, documents } from './schema'
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
} from './types'
export { count } from 'drizzle-orm'
