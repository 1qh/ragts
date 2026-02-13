/* eslint-disable max-statements */
import { cosineDistance, desc, eq, gt, sql } from 'drizzle-orm'

import type { DrizzleDb } from './db'
import type { SearchConfig, SearchResult } from './types'

import { chunks, documents } from './schema'

const vectorSearch = async (params: {
    db: DrizzleDb
    limit: number
    queryEmbedding: number[]
    threshold?: number
  }): Promise<SearchResult[]> => {
    const { db, limit, queryEmbedding, threshold } = params,
      similarity = sql<number>`1 - (${cosineDistance(chunks.embedding, queryEmbedding)})`,
      query = db
        .select({
          documentId: chunks.documentId,
          id: chunks.id,
          score: similarity,
          text: chunks.text,
          title: documents.title
        })
        .from(chunks)
        .innerJoin(documents, eq(chunks.documentId, documents.id)),
      withThreshold = threshold === undefined ? query : query.where(gt(similarity, threshold)),
      rows = await withThreshold.orderBy(desc(similarity)).limit(limit),
      results: SearchResult[] = []
    for (const row of rows)
      results.push({
        documentId: row.documentId,
        id: row.id,
        mode: 'vector',
        score: row.score,
        text: row.text,
        title: row.title
      })

    return results
  },
  bm25Search = async (db: DrizzleDb, queryText: string, limit: number): Promise<SearchResult[]> => {
    const bm25q = sql`to_bm25query(${queryText}, 'chunks_text_idx')`,
      bm25Score = sql<number>`-(${chunks.text} <@> ${bm25q})`,
      rows = await db
        .select({
          documentId: chunks.documentId,
          id: chunks.id,
          score: bm25Score,
          text: chunks.text,
          title: documents.title
        })
        .from(chunks)
        .innerJoin(documents, eq(chunks.documentId, documents.id))
        .where(sql`${chunks.text} <@> ${bm25q} < 0`)
        .orderBy(sql`${chunks.text} <@> ${bm25q}`)
        .limit(limit),
      results: SearchResult[] = []
    for (const row of rows)
      results.push({
        documentId: row.documentId,
        id: row.id,
        mode: 'bm25',
        score: row.score,
        text: row.text,
        title: row.title
      })

    return results
  },
  hybridSearch = async (params: {
    db: DrizzleDb
    limit: number
    queryEmbedding: number[]
    queryText: string
    rrfK: number
    threshold?: number
  }): Promise<SearchResult[]> => {
    const { db, limit, queryEmbedding, queryText, rrfK, threshold } = params,
      fetchLimit = limit * 3,
      [vectorResults, bm25Results] = await Promise.all([
        vectorSearch({ db, limit: fetchLimit, queryEmbedding, threshold }),
        bm25Search(db, queryText, fetchLimit)
      ]),
      scoreMap = new Map<number, { bm25Rank: number; result: SearchResult; vectorRank: number }>()

    for (const [index, result] of vectorResults.entries())
      scoreMap.set(result.id, { bm25Rank: 0, result, vectorRank: index + 1 })

    for (const [index, result] of bm25Results.entries()) {
      const existing = scoreMap.get(result.id)
      if (existing) existing.bm25Rank = index + 1
      else scoreMap.set(result.id, { bm25Rank: index + 1, result, vectorRank: 0 })
    }

    const merged: { result: SearchResult; rrfScore: number }[] = []
    for (const [, entry] of scoreMap) {
      let rrfScore = 0
      if (entry.vectorRank > 0) rrfScore += 1 / (rrfK + entry.vectorRank)
      if (entry.bm25Rank > 0) rrfScore += 1 / (rrfK + entry.bm25Rank)
      merged.push({ result: { ...entry.result, score: rrfScore }, rrfScore })
    }

    merged.sort((a, b) => b.rrfScore - a.rrfScore)

    const results: SearchResult[] = []
    for (let i = 0; i < Math.min(limit, merged.length); i += 1) {
      const m = merged[i]
      if (m) results.push(m.result)
    }
    return results
  },
  search = async (db: DrizzleDb, config: SearchConfig): Promise<SearchResult[]> => {
    const mode = config.mode ?? 'hybrid',
      limit = config.limit ?? 10,
      rrfK = config.rrfK ?? 60

    if (mode === 'vector') {
      const [vectorEmb] = await config.embed([config.query])
      if (!vectorEmb) return []
      return vectorSearch({ db, limit, queryEmbedding: vectorEmb, threshold: config.threshold })
    }

    if (mode === 'bm25') return bm25Search(db, config.query, limit)

    const [hybridEmb] = await config.embed([config.query])
    if (!hybridEmb) return []
    return hybridSearch({
      db,
      limit,
      queryEmbedding: hybridEmb,
      queryText: config.query,
      rrfK,
      threshold: config.threshold
    })
  }

export { search }
