/* eslint-disable max-statements */
import type { LanguageModelV3 } from '@ai-sdk/provider'
import type { SearchResult } from 'ragts'

import { generateText } from 'ai'
import { existsSync, readFileSync } from 'node:fs'

interface ChunkEntry {
  endIndex: number
  startIndex: number
  text: string
  textHash: string
  tokenCount: number
}

interface EmbeddedChunk {
  embedding: number[]
  text: string
  textHash: string
}

interface EvalEntry {
  generated: string
  gold?: string
  question: string
  retrievedDocs: { documentId: number; id: number; score: number; text: string; title: string }[]
}

interface Step1Output {
  config: { chunkSize: number; folder: string; overlap: number }
  docs: { chunkCount: number; chunks: ChunkEntry[]; title: string }[]
  stats: {
    duplicateChunks: number
    elapsedMs: number
    totalChunks: number
    totalDocs: number
    uniqueHashes: number
  }
}

interface Step2Line {
  chunks: EmbeddedChunk[]
  title: string
}

const AGENTIC_RERANK_PROMPT =
    'You are a relevance judge. For each numbered document, rate its relevance to the question on a scale of 0-10.\n0 = completely irrelevant, 10 = directly answers the question.\nOutput format: one line per document, format "N: score" where N is the document number.\nOutput ONLY the ratings, nothing else.',
  EXPAND_PROMPT =
    'Decompose the following question into 2-3 focused search queries. Each query should target a different concept or aspect of the question.\nOutput format: one query per line, no numbering, no explanations.\nLanguage: same as the input question.',
  EXPAND_THRESHOLD = 200,
  HYDE_PROMPT =
    'Given the following question, write a short answer paragraph as it would appear in an official legal document. Write ONLY the answer text in the same language as the question. No explanations, no preamble, no "Answer:" prefix.',
  MULTIHOP_PROMPT =
    'Based on the context provided, identify what additional legal provisions, articles, or concepts would be needed to fully answer the question. Output a single refined search query in the same language as the question. Output ONLY the query, nothing else.',
  RE_NUMBERING = /^\d+[.)]\s*/u,
  RE_SCORE_LINE = /^(?<idx>\d+)\s*:\s*(?<val>[\d.]+)/u,
  agenticRerank = async (
    question: string,
    items: SearchResult[],
    config: { model: LanguageModelV3; topN: number }
  ): Promise<SearchResult[]> => {
    let prompt = `Question: ${question}\n\n`
    for (let i = 0; i < items.length; i += 1) {
      const item = items[i]
      if (item) prompt += `[${String(i + 1)}] ${item.title}\n${item.text}\n\n`
    }

    const response = await generateText({
        abortSignal: AbortSignal.timeout(120_000),
        model: config.model,
        prompt,
        system: AGENTIC_RERANK_PROMPT
      }),
      scores = new Map<number, number>()
    for (const line of response.text.trim().split('\n')) {
      const match = RE_SCORE_LINE.exec(line.trim())
      if (match?.groups) {
        const idx = Number.parseInt(match.groups.idx ?? '0', 10) - 1,
          score = Number.parseFloat(match.groups.val ?? '0')
        if (!Number.isNaN(score) && idx >= 0 && idx < items.length) scores.set(idx, score)
      }
    }

    const scored: { index: number; score: number }[] = []
    for (let i = 0; i < items.length; i += 1) scored.push({ index: i, score: scores.get(i) ?? 0 })
    scored.sort((a, b) => b.score - a.score)

    const result: SearchResult[] = []
    for (let i = 0; i < Math.min(config.topN, scored.length); i += 1) {
      const entry = scored[i]
      if (entry) {
        const original = items[entry.index]
        if (original) result.push({ ...original, score: entry.score })
      }
    }
    return result
  },
  expandQuery = async (question: string, model: LanguageModelV3): Promise<string[]> => {
    if (question.length <= EXPAND_THRESHOLD) return [question]

    const response = await generateText({
        abortSignal: AbortSignal.timeout(60_000),
        model,
        prompt: question,
        system: EXPAND_PROMPT
      }),
      lines: string[] = [question]
    for (const raw of response.text.trim().split('\n')) {
      const line = raw.replace(RE_NUMBERING, '').trim()
      if (line.length > 0) lines.push(line)
    }
    return lines
  },
  formatRetrievedDocs = (results: SearchResult[]): EvalEntry['retrievedDocs'] => {
    const docs: EvalEntry['retrievedDocs'] = []
    for (const chunk of results)
      docs.push({
        documentId: chunk.documentId,
        id: chunk.id,
        score: Math.round(chunk.score * 1_000_000) / 1_000_000,
        text: chunk.text,
        title: chunk.title
      })
    return docs
  },
  getNumFlag = (parsed: Map<string, boolean | string>, name: string): number | undefined => {
    const v = parsed.get(name)
    return typeof v === 'string' ? Number(v) : undefined
  },
  getStringFlag = (parsed: Map<string, boolean | string>, name: string): string | undefined => {
    const v = parsed.get(name)
    return typeof v === 'string' ? v : undefined
  },
  generateHypothesis = async (question: string, model: LanguageModelV3): Promise<string> => {
    const response = await generateText({
      abortSignal: AbortSignal.timeout(60_000),
      model,
      prompt: question,
      system: HYDE_PROMPT
    })
    return response.text.trim()
  },
  generateMultihopQuery = async (question: string, context: string, model: LanguageModelV3): Promise<string> => {
    const response = await generateText({
      abortSignal: AbortSignal.timeout(240_000),
      model,
      prompt: `Question: ${question}\n\nContext:\n${context}`,
      system: MULTIHOP_PROMPT
    })
    return response.text.trim()
  },
  mergeRetrievals = (resultSets: SearchResult[][]): SearchResult[] => {
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
  },
  loadSystemPrompt = (parsed: Map<string, boolean | string>): string => {
    const raw = getStringFlag(parsed, '--system') ?? ''
    return raw && existsSync(raw) ? readFileSync(raw, 'utf8') : raw
  }

export type { ChunkEntry, EmbeddedChunk, EvalEntry, Step1Output, Step2Line }
export {
  agenticRerank,
  expandQuery,
  formatRetrievedDocs,
  generateHypothesis,
  generateMultihopQuery,
  getNumFlag,
  getStringFlag,
  loadSystemPrompt,
  mergeRetrievals
}
