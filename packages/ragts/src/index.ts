import type { RerankingModelV3 } from '@ai-sdk/provider'
import type postgres from 'postgres'

import { createOpenAICompatible } from '@ai-sdk/openai-compatible'
import { embedMany, rerank } from 'ai'
import { sql } from 'drizzle-orm'
import { readdirSync, readFileSync } from 'node:fs'
import { basename, extname, join } from 'node:path'

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
  },
  createEmbedFn = (config: { baseURL: string; model: string }): EmbedFn => {
    const provider = createOpenAICompatible({ apiKey: 'none', baseURL: config.baseURL, name: 'openai-compatible' })
    return async (texts: string[]) => {
      const { embeddings } = await embedMany({ model: provider.embeddingModel(config.model), values: texts })
      return embeddings
    }
  },
  // eslint-disable-next-line max-statements
  dedupSubstrings = <T extends { text: string }>(items: T[], options?: { prefixLength?: number }): T[] => {
    const prefixLen = options?.prefixLength ?? 0,
      kept: T[] = [],
      prefixes = prefixLen > 0 ? new Set<string>() : undefined
    for (const item of items) {
      let isDupe = false
      for (const higher of kept)
        if (higher.text.length > item.text.length && higher.text.includes(item.text)) {
          isDupe = true
          break
        }
      if (!isDupe && prefixes) {
        const p = item.text.slice(0, prefixLen)
        if (p.length >= prefixLen && prefixes.has(p)) isDupe = true
        else prefixes.add(p)
      }
      if (!isDupe) kept.push(item)
    }
    return kept
  },
  getDbUrl = (flags: Map<string, boolean | string>): string => {
    const value = flags.get('--db')
    return typeof value === 'string' && value.length > 0 ? value : 'postgresql://postgres:postgres@localhost:5432/postgres'
  },
  getRequiredString = (flags: Map<string, boolean | string>, name: string): string => {
    const value = flags.get(name)
    if (typeof value !== 'string' || value.length === 0) throw new Error(`Missing required flag: ${name}`)
    return value
  },
  loadDocsFromFolder = (folderPath: string): Doc[] => {
    const names = readdirSync(folderPath).toSorted((a, b) => a.localeCompare(b)),
      docs: Doc[] = []
    for (const fileName of names)
      if (extname(fileName) === '.txt' || extname(fileName) === '.md') {
        const fullPath = join(folderPath, fileName),
          content = readFileSync(fullPath, 'utf8'),
          title = basename(fileName, extname(fileName))
        docs.push({ content, title })
      }
    return docs
  },
  loadJsonArray = <T>(filePath: string): T[] => {
    const raw = readFileSync(filePath, 'utf8'),
      parsed = JSON.parse(raw) as unknown
    if (!Array.isArray(parsed)) throw new Error(`Expected JSON array in ${filePath}`)
    return parsed as T[]
  },
  parseArgs = (values: string[]): Map<string, boolean | string> => {
    const flags = new Map<string, boolean | string>()
    for (let i = 0; i < values.length; i += 1) {
      const token = values[i]
      if (token?.startsWith('--')) {
        const nextValue = values[i + 1]
        if (!nextValue || nextValue.startsWith('--')) flags.set(token, true)
        else {
          flags.set(token, nextValue)
          i += 1
        }
      }
    }
    return flags
  },
  rerankChunks = async (
    query: string,
    chunks: SearchResult[],
    config: { model: RerankingModelV3; topN?: number }
  ): Promise<SearchResult[]> => {
    const documents: string[] = []
    for (const c of chunks) documents.push(c.text)
    const { ranking } = await rerank({ documents, model: config.model, query, topN: config.topN }),
      reranked: SearchResult[] = []
    for (const entry of ranking) {
      const original = chunks[entry.originalIndex]
      if (original) reranked.push({ ...original, score: entry.score })
    }
    return reranked
  }

class Rag {
  public db: DrizzleDb | undefined
  private _client: ReturnType<typeof postgres> | undefined
  private readonly connStr: string
  private readonly dim: number
  private readonly txtCfg: string

  public constructor(config: RagtsConfig) {
    this.connStr = config.connectionString
    this.dim = config.dimension ?? 2048
    this.txtCfg = config.textConfig ?? 'english'
  }

  public close = async (): Promise<void> => {
    if (this._client) await this._client.end()
  }

  public drop = async (): Promise<void> => {
    if (this.db) {
      await this.db.execute(sql`DROP TABLE IF EXISTS chunk_sources CASCADE`)
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

const runWithRag = async (config: RagtsConfig, fn: (rag: Rag) => Promise<void>): Promise<void> => {
  const rag = new Rag(config)
  try {
    await fn(rag)
  } finally {
    await rag.close()
  }
}

export {
  buildContext,
  createEmbedFn,
  dedupSubstrings,
  getDbUrl,
  getRequiredString,
  loadDocsFromFolder,
  loadJsonArray,
  parseArgs,
  Rag,
  rerankChunks,
  runWithRag
}
export { computeHash, validateBackup } from './backup'
export { chunkText } from './chunk'
export { generateCompose } from './compose'
export type { DrizzleDb } from './db'
export { chunks, chunkSources, documents } from './schema'
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
