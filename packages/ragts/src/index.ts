/* eslint-disable max-statements, no-await-in-loop */
/** biome-ignore-all lint/style/useConsistentMemberAccessibility: x */
/** biome-ignore-all lint/performance/noAwaitInLoops: sequential community queries */
import type { OpenAICompatibleProviderSettings } from '@ai-sdk/openai-compatible'
import type { LanguageModelV3, RerankingModelV3, RerankingModelV3CallOptions } from '@ai-sdk/provider'
import type postgres from 'postgres'

import { createOpenAICompatible } from '@ai-sdk/openai-compatible'
import { embedMany, rerank } from 'ai'
import { sql } from 'drizzle-orm'

import type { DrizzleDb } from './db'
import type {
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
  SearchConfig,
  SearchResult
} from './types'

import { exportBackup, importBackup } from './backup'
import { createDb, dropSchema, initSchema } from './db'
import { ingest } from './ingest'
import { fetchGraphRelations, search } from './search'

type ProviderConfig = Omit<OpenAICompatibleProviderSettings, 'name'> & { name?: string }

interface RerankConfig {
  apiKey?: string
  baseURL: string
  headers?: Record<string, string>
  model: string
}

const RE_V1_SUFFIX = /\/v1\/?$/u,
  buildContext = (results: { text: string; title: string }[]): string => {
    let output = ''
    for (let i = 0; i < results.length; i += 1) {
      const row = results[i]
      if (row) output += `[${String(i + 1)}] ${row.title}\n${row.text}\n\n`
    }
    return output.trim()
  },
  buildGraphContext = (results: { text: string; title: string }[], relations: GraphRelation[]): string => {
    let relSection = ''
    if (relations.length > 0) {
      relSection = '=== Document Relations ===\n'
      for (const rel of relations) {
        const typeLabel = rel.type ? ` [${rel.type}]` : ''
        relSection += `${rel.sourceTitle} → ${rel.targetTitle}${typeLabel}\n`
      }
      relSection += '\n'
    }

    let docSection = ''
    for (let i = 0; i < results.length; i += 1) {
      const row = results[i]
      if (row) docSection += `[${String(i + 1)}] ${row.title}\n${row.text}\n\n`
    }

    return `${relSection}${docSection}`.trim()
  },
  normalizeBaseURL = (url: string): string => url.replace(RE_V1_SUFFIX, ''),
  createEmbedFn = (config: {
    apiKey?: string
    baseURL: string
    headers?: Record<string, string>
    model: string
  }): EmbedFn => {
    const base = normalizeBaseURL(config.baseURL),
      provider = createOpenAICompatible({
        apiKey: config.apiKey ?? 'none',
        baseURL: `${base}/v1`,
        headers: config.headers,
        name: 'openai-compatible'
      })
    return async (texts: string[]) => {
      const { embeddings } = await embedMany({ model: provider.embeddingModel(config.model), values: texts })
      return embeddings
    }
  },
  createRerankingModel = (config: RerankConfig): RerankingModelV3 => {
    const base = normalizeBaseURL(config.baseURL),
      reqHeaders: Record<string, string> = { 'Content-Type': 'application/json' }
    if (config.apiKey) reqHeaders.Authorization = `Bearer ${config.apiKey}`
    if (config.headers) for (const [k, v] of Object.entries(config.headers)) reqHeaders[k] = v
    return {
      doRerank: async (options: RerankingModelV3CallOptions) => {
        const documents: string[] = []
        if (options.documents.type === 'text') for (const v of options.documents.values) documents.push(v)
        else for (const v of options.documents.values) documents.push(JSON.stringify(v))

        const body: Record<string, unknown> = { documents, model: config.model, query: options.query }
        if (options.topN !== undefined) body.top_n = options.topN

        const res = await fetch(`${base}/v1/rerank`, {
          body: JSON.stringify(body),
          headers: reqHeaders,
          method: 'POST'
        })
        if (!res.ok) throw new Error(`Rerank failed: ${String(res.status)} ${await res.text()}`)
        const data = (await res.json()) as {
            results: { document: { text: string }; index: number; relevance_score: number }[]
          },
          ranking: { index: number; relevanceScore: number }[] = []
        for (const r of data.results) ranking.push({ index: r.index, relevanceScore: r.relevance_score })

        return { ranking }
      },
      modelId: config.model,
      provider: 'custom',
      specificationVersion: 'v3'
    }
  },
  createProvider = (config: ProviderConfig) => {
    const base = normalizeBaseURL(config.baseURL),
      aiProvider = createOpenAICompatible({
        ...config,
        apiKey: config.apiKey ?? 'none',
        baseURL: `${base}/v1`,
        name: config.name ?? 'openai-compatible'
      })
    return {
      chatModel: (model: string): LanguageModelV3 => aiProvider.chatModel(model),
      embedFn: (model: string): EmbedFn => {
        const embModel = aiProvider.embeddingModel(model)
        return async (texts: string[]) => {
          const { embeddings } = await embedMany({ model: embModel, values: texts })
          return embeddings
        }
      },
      rerankingModel: (model: string): RerankingModelV3 =>
        createRerankingModel({ apiKey: config.apiKey, baseURL: base, headers: config.headers, model })
    }
  },
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
    this.txtCfg = config.textConfig ?? 'simple'
  }

  public buildCommunitySummaries = async (config: CommunitySummaryConfig): Promise<CommunitySummaryResult> => {
    const db = await this.init(),
      { buildCommunitySummaries: build } = await import('./community')
    return build(db, config)
  }

  public close = async (): Promise<void> => {
    if (this._client) await this._client.end()
  }

  public detectCommunities = async (): Promise<number> => {
    const db = await this.init(),
      { detectCommunities: detect } = await import('./community')
    return detect(db)
  }

  public drop = async (): Promise<void> => {
    if (this.db) {
      await this.db.execute(sql`DROP TABLE IF EXISTS document_relations CASCADE`)
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

  public fetchRelations = async (docIds: number[]): Promise<GraphRelation[]> => {
    const db = await this.init()
    return fetchGraphRelations(db, docIds)
  }

  public globalQuery = async (config: GlobalQueryConfig): Promise<GlobalQueryResult> => {
    const db = await this.init(),
      communityRows = await db.execute<{ community_id: number; member_titles: unknown }>(
        sql.raw(`
        SELECT
          (metadata->>'_ragts_community_id')::integer AS community_id,
          metadata->'_ragts_member_titles' AS member_titles
        FROM documents
        WHERE metadata->>'_ragts_type' = 'community_summary'
        ORDER BY community_id
      `)
      ),
      maxCommunities = config.maxCommunities ?? communityRows.length,
      communitiesToQuery = communityRows.slice(0, maxCommunities),
      partialAnswers: { answer: string; communityId: number; titles: string[] }[] = []

    for (const row of communitiesToQuery) {
      const summaryResults = await search(db, {
          embed: config.embed,
          limit: config.limit ?? 10,
          mode: 'vector',
          query: config.query
        }),
        communityChunks = summaryResults.filter(r => {
          const titles = Array.isArray(row.member_titles) ? (row.member_titles as string[]) : []
          return titles.includes(r.title) || r.title === `_ragts_community_${String(row.community_id)}`
        })

      if (communityChunks.length > 0) {
        let chunksToUse = communityChunks
        if (config.rerank) chunksToUse = await rerankChunks(config.query, communityChunks, config.rerank)

        const context = buildContext(chunksToUse),
          answer = await config.generate(context, config.query),
          titles: string[] = Array.isArray(row.member_titles) ? (row.member_titles as string[]) : []
        partialAnswers.push({ answer, communityId: row.community_id, titles })
      }
    }

    let finalAnswer = ''
    if (partialAnswers.length > 0) {
      let combinedContext = ''
      for (const pa of partialAnswers) combinedContext += `[Community ${String(pa.communityId)}]\n${pa.answer}\n\n`

      finalAnswer = await config.generate(combinedContext.trim(), config.query)
    }

    return { answer: finalAnswer, partialAnswers }
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

  public query = async (config: QueryConfig): Promise<QueryResult> => {
    const db = await this.init()
    let results = await search(db, {
      bm25Weight: config.bm25Weight,
      communityBoost: config.communityBoost,
      embed: config.embed,
      graphChunkLimit: config.graphChunkLimit,
      graphDecay: config.graphDecay,
      graphHops: config.graphHops,
      graphWeight: config.graphWeight,
      limit: config.limit ?? 50,
      mode: config.mode,
      query: config.query,
      rrfK: config.rrfK,
      threshold: config.threshold,
      vectorQuery: config.vectorQuery,
      vectorWeight: config.vectorWeight
    })
    if (config.rerank)
      results = await rerankChunks(config.query, results, {
        model: config.rerank.model,
        topN: config.rerank.topN ?? 10
      })

    if (config.dedup !== false) results = dedupSubstrings(results, { prefixLength: 100 })

    let context: string
    if (config.graphHops !== undefined && config.graphHops > 0) {
      const docIds: number[] = []
      for (const r of results) docIds.push(r.documentId)
      const uniqueDocIds = [...new Set(docIds)],
        relations = await fetchGraphRelations(db, uniqueDocIds)
      context = buildGraphContext(results, relations)
    } else context = buildContext(results)

    return { context, results }
  }

  public retrieve = async (config: SearchConfig): Promise<SearchResult[]> => {
    const db = await this.init()
    return search(db, config)
  }
}

export {
  buildContext,
  buildGraphContext,
  createEmbedFn,
  createProvider,
  createRerankingModel,
  dedupSubstrings,
  Rag,
  rerankChunks
}
export { computeHash, validateBackup } from './backup'
export { chunkText } from './chunk'
export { buildCommunitySummaries, detectCommunities } from './community'
export { generateCompose } from './compose'
export type { DrizzleDb } from './db'
export { chunks, chunkSources, documentRelations, documents } from './schema'
export { fetchGraphRelations } from './search'
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
} from './types'
export type { ProviderConfig, RerankConfig }
