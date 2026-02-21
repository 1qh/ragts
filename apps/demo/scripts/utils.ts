import type { LanguageModelV3, RerankingModelV3, RerankingModelV3CallOptions } from '@ai-sdk/provider'
import type { SearchResult } from 'ragts'

import { generateText } from 'ai'

import { BASE_URL, RERANK_MODEL } from './constants'

interface ServerRerankResult {
  document: { text: string }
  index: number
  relevance_score: number
}

const EXPAND_PROMPT =
    'Decompose the following question into 2-3 focused search queries. Each query should target a different concept or aspect of the question.\nOutput format: one query per line, no numbering, no explanations.\nLanguage: same as the input question.',
  EXPAND_THRESHOLD = 200,
  RE_NUMBERING = /^\d+[.)]\s*/u,
  createRerankingModel = (modelId = RERANK_MODEL, baseURL = BASE_URL.replace('/v1', '')): RerankingModelV3 => ({
    doRerank: async (options: RerankingModelV3CallOptions) => {
      const documents: string[] = []
      if (options.documents.type === 'text') for (const v of options.documents.values) documents.push(v)
      else for (const v of options.documents.values) documents.push(JSON.stringify(v))

      const body: Record<string, unknown> = { documents, model: modelId, query: options.query }
      if (options.topN !== undefined) body.top_n = options.topN

      const res = await fetch(`${baseURL}/v1/rerank`, {
        body: JSON.stringify(body),
        headers: { 'Content-Type': 'application/json' },
        method: 'POST'
      })
      if (!res.ok) throw new Error(`Rerank failed: ${String(res.status)} ${await res.text()}`)
      const data = (await res.json()) as { results: ServerRerankResult[] },
        ranking: { index: number; relevanceScore: number }[] = []
      for (const r of data.results) ranking.push({ index: r.index, relevanceScore: r.relevance_score })

      return { ranking }
    },
    modelId,
    provider: 'mlx',
    specificationVersion: 'v3'
  }),
  expandQuery = async (question: string, model: LanguageModelV3): Promise<string[]> => {
    if (question.length <= EXPAND_THRESHOLD) return [question]
    /* eslint-disable @typescript-eslint/no-unsafe-assignment */
    const response = await generateText({
        abortSignal: AbortSignal.timeout(60_000),
        model,
        prompt: question,
        system: EXPAND_PROMPT
      }),
      /* eslint-enable @typescript-eslint/no-unsafe-assignment */
      lines: string[] = [question]
    for (const raw of response.text.trim().split('\n')) {
      const line = raw.replace(RE_NUMBERING, '').trim()
      if (line.length > 0) lines.push(line)
    }
    return lines
  }

const mergeRetrievals = (resultSets: SearchResult[][]): SearchResult[] => {
  const byId = new Map<number, SearchResult>()
  for (const set of resultSets)
    for (const c of set) {
      const existing = byId.get(c.id)
      if (!existing || c.score > existing.score) byId.set(c.id, c)
    }
  const merged: SearchResult[] = []
  for (const [, v] of byId) merged.push(v)
  merged.sort((a, b) => b.score - a.score)
  return merged
}

export { createRerankingModel, expandQuery, mergeRetrievals }
