/* eslint-disable max-statements */
import { cosineDistance, desc, eq, gt, sql } from 'drizzle-orm'

import type { DrizzleDb } from './db'
import type { SearchConfig, SearchResult } from './types'

import { chunks, chunkSources, documents } from './schema'

const dedup = (results: SearchResult[], limit: number): SearchResult[] => {
    const seen = new Set<string>(),
      unique: SearchResult[] = []
    for (const r of results)
      if (!seen.has(r.text)) {
        seen.add(r.text)
        unique.push(r)
        if (unique.length >= limit) break
      }

    return unique
  },
  rawVectorSearch = async (params: {
    db: DrizzleDb
    limit: number
    queryEmbedding: number[]
    threshold?: number
  }): Promise<SearchResult[]> => {
    const { db, limit, queryEmbedding, threshold } = params,
      similarity = sql<number>`1 - (${cosineDistance(chunks.embedding, queryEmbedding)})`,
      newestDocId = sql<string>`MAX(${chunkSources.documentId})`,
      query = db
        .select({
          documentId: newestDocId,
          id: chunks.id,
          score: similarity,
          text: chunks.text,
          title: documents.title
        })
        .from(chunks)
        .innerJoin(chunkSources, eq(chunkSources.chunkId, chunks.id))
        .innerJoin(documents, eq(documents.id, chunkSources.documentId)),
      withThreshold = threshold === undefined ? query : query.where(gt(similarity, threshold)),
      rows = await withThreshold
        .groupBy(chunks.id, chunks.embedding, chunks.text, documents.title)
        .orderBy(desc(similarity))
        .limit(limit),
      results: SearchResult[] = []
    for (const row of rows)
      results.push({
        documentId: Number(row.documentId),
        id: row.id,
        mode: 'vector',
        score: row.score,
        text: row.text,
        title: row.title
      })

    return results
  },
  rawBm25Search = async (db: DrizzleDb, queryText: string, limit: number): Promise<SearchResult[]> => {
    const bm25q = sql`to_bm25query(${queryText}, 'chunks_text_idx')`,
      bm25Score = sql<number>`-(${chunks.text} <@> ${bm25q})`,
      newestDocId = sql<string>`MAX(${chunkSources.documentId})`,
      rows = await db
        .select({
          documentId: newestDocId,
          id: chunks.id,
          score: bm25Score,
          text: chunks.text,
          title: documents.title
        })
        .from(chunks)
        .innerJoin(chunkSources, eq(chunkSources.chunkId, chunks.id))
        .innerJoin(documents, eq(documents.id, chunkSources.documentId))
        .where(sql`${chunks.text} <@> ${bm25q} < 0`)
        .groupBy(chunks.id, chunks.text, documents.title)
        .orderBy(sql`${chunks.text} <@> ${bm25q}`)
        .limit(limit),
      results: SearchResult[] = []
    for (const row of rows)
      results.push({
        documentId: Number(row.documentId),
        id: row.id,
        mode: 'bm25',
        score: row.score,
        text: row.text,
        title: row.title
      })

    return results
  },
  rawHybridSearch = async (params: {
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
        rawVectorSearch({ db, limit: fetchLimit, queryEmbedding, threshold }),
        rawBm25Search(db, queryText, fetchLimit)
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
      rrfK = config.rrfK ?? 60,
      fetchLimit = limit * 3

    if (mode === 'vector') {
      const [vectorEmb] = await config.embed([config.query])
      if (!vectorEmb) return []
      const raw = await rawVectorSearch({ db, limit: fetchLimit, queryEmbedding: vectorEmb, threshold: config.threshold })
      return dedup(raw, limit)
    }

    if (mode === 'bm25') {
      const raw = await rawBm25Search(db, config.query, fetchLimit)
      return dedup(raw, limit)
    }

    const [hybridEmb] = await config.embed([config.query])
    if (!hybridEmb) return []
    const raw = await rawHybridSearch({
      db,
      limit: fetchLimit,
      queryEmbedding: hybridEmb,
      queryText: config.query,
      rrfK,
      threshold: config.threshold
    })
    return dedup(raw, limit)
  }

export { search }
