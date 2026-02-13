import type { RerankingModelV3, RerankingModelV3CallOptions } from '@ai-sdk/provider'
import type { EmbedFn, SearchResult } from 'ragts'

import { createOpenAICompatible } from '@ai-sdk/openai-compatible'
import { embedMany, rerank } from 'ai'
import { readdirSync, readFileSync } from 'node:fs'
import { basename, extname, join } from 'node:path'
import { Rag } from 'ragts'

import { BASE_URL, DEFAULT_DB_URL, EMBEDDING_DIMENSION, EMBEDDING_MODEL, RERANK_MODEL } from './constants'

interface ServerRerankResult {
  document: { text: string }
  index: number
  relevance_score: number
}

const createRerankingModel = (modelId = RERANK_MODEL, baseURL = BASE_URL.replace('/v1', '')): RerankingModelV3 => ({
    doRerank: async (options: RerankingModelV3CallOptions) => {
      const documents: string[] = []
      if (options.documents.type === 'text') for (const v of options.documents.values) documents.push(v)
      else for (const v of options.documents.values) documents.push(JSON.stringify(v))

      const res = await fetch(`${baseURL}/v1/rerank`, {
        body: JSON.stringify({ documents, model: modelId, query: options.query }),
        headers: { 'Content-Type': 'application/json' },
        method: 'POST'
      })
      if (!res.ok) throw new Error(`Rerank failed: ${String(res.status)} ${await res.text()}`)
      const data = (await res.json()) as { results: ServerRerankResult[] }

      const ranking: { index: number; relevanceScore: number }[] = []
      for (const r of data.results) ranking.push({ index: r.index, relevanceScore: r.relevance_score })

      return { ranking }
    },
    modelId,
    provider: 'mlx',
    specificationVersion: 'v3'
  }),
  parseArgs = (values: string[]) => {
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
  getRequiredString = (flags: Map<string, boolean | string>, name: string) => {
    const value = flags.get(name)
    if (typeof value !== 'string' || value.length === 0) throw new Error(`Missing required flag: ${name}`)

    return value
  },
  getDbUrl = (flags: Map<string, boolean | string>) => {
    const value = flags.get('--db')
    return typeof value === 'string' && value.length > 0 ? value : DEFAULT_DB_URL
  },
  createEmbedFn = (model = EMBEDDING_MODEL, baseURL = BASE_URL): EmbedFn => {
    const provider = createOpenAICompatible({ apiKey: 'none', baseURL, name: 'mlx' })
    return async (texts: string[]) => {
      const { embeddings } = await embedMany({ model: provider.embeddingModel(model), values: texts })
      return embeddings
    }
  },
  rerankChunks = async (query: string, chunks: SearchResult[], topK?: number): Promise<SearchResult[]> => {
    const documents: string[] = []
    for (const c of chunks) documents.push(c.text)

    const { ranking } = await rerank({
      documents,
      model: createRerankingModel(),
      query,
      topN: topK
    })

    const reranked: SearchResult[] = []
    for (const entry of ranking) {
      const original = chunks[entry.originalIndex]
      if (original) reranked.push({ ...original, score: entry.score })
    }

    return reranked
  },
  runWithRag = async (flags: Map<string, boolean | string>, fn: (rag: Rag) => Promise<void>) => {
    const textCfg = flags.get('--text-config'),
      rag = new Rag({
        connectionString: getDbUrl(flags),
        dimension: EMBEDDING_DIMENSION,
        textConfig: typeof textCfg === 'string' ? textCfg : 'simple'
      })
    try {
      await fn(rag)
    } finally {
      await rag.close()
    }
  },
  loadDocsFromFolder = (folderPath: string) => {
    const names = readdirSync(folderPath).toSorted((a, b) => a.localeCompare(b)),
      docs: { content: string; title: string }[] = []

    for (const fileName of names)
      if (extname(fileName) === '.txt' || extname(fileName) === '.md') {
        const fullPath = join(folderPath, fileName),
          content = readFileSync(fullPath, 'utf8'),
          title = basename(fileName, extname(fileName))
        docs.push({ content, title })
      }

    return docs
  },
  loadQuestions = (filePath: string): string[] => {
    const raw = readFileSync(filePath, 'utf8'),
      parsed = JSON.parse(raw) as unknown
    if (!Array.isArray(parsed)) throw new Error(`Expected JSON array of strings in ${filePath}`)
    for (let i = 0; i < parsed.length; i += 1)
      if (typeof parsed[i] !== 'string') throw new Error(`Item at index ${String(i)} is not a string`)

    return parsed as string[]
  }

export {
  createEmbedFn,
  createRerankingModel,
  getDbUrl,
  getRequiredString,
  loadDocsFromFolder,
  loadQuestions,
  parseArgs,
  rerankChunks,
  runWithRag
}
