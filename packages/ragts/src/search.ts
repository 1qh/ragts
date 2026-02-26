/* eslint-disable complexity, max-statements */
/** biome-ignore-all lint/performance/noAwaitInLoops: x */
import { and, cosineDistance, desc, eq, gt, inArray, sql } from 'drizzle-orm'

import type { DrizzleDb } from './db'
import type { GraphRelation, SearchConfig, SearchResult } from './types'

import { chunks, chunkSources, documents } from './schema'

const DEFAULT_GRAPH_CHUNK_LIMIT = 200,
  dedup = (results: SearchResult[], limit: number): SearchResult[] => {
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
          communityId: documents.communityId,
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
        .groupBy(chunks.id, chunks.embedding, chunks.text, documents.title, documents.communityId)
        .orderBy(desc(similarity))
        .limit(limit),
      results: SearchResult[] = []
    for (const row of rows)
      results.push({
        communityId: row.communityId ?? undefined,
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
          communityId: documents.communityId,
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
        .groupBy(chunks.id, chunks.text, documents.title, documents.communityId)
        .orderBy(sql`${chunks.text} <@> ${bm25q}`)
        .limit(limit),
      results: SearchResult[] = []
    for (const row of rows)
      results.push({
        communityId: row.communityId ?? undefined,
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
    bm25Weight: number
    db: DrizzleDb
    limit: number
    queryEmbedding: number[]
    queryText: string
    rrfK: number
    threshold?: number
    vectorWeight: number
  }): Promise<SearchResult[]> => {
    const { bm25Weight, db, limit, queryEmbedding, queryText, rrfK, threshold, vectorWeight } = params,
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
      if (entry.vectorRank > 0) rrfScore += vectorWeight / (rrfK + entry.vectorRank)
      if (entry.bm25Rank > 0) rrfScore += bm25Weight / (rrfK + entry.bm25Rank)
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
  graphExpand = async (params: {
    chunkLimit: number
    db: DrizzleDb
    decay: number
    existingChunkIds: Set<number>
    graphHops: number
    initialDocIds: number[]
  }): Promise<SearchResult[]> => {
    const { chunkLimit, db, decay, existingChunkIds, graphHops, initialDocIds } = params
    if (initialDocIds.length === 0) return []

    const rows = await db.execute<{ doc_id: number; path_weight: number; rel_type: null | string }>(
        sql.raw(`
        WITH RECURSIVE related AS (
          SELECT target_id AS doc_id, 1 AS depth, ARRAY[target_id] AS visited,
                 COALESCE(weight, 1.0) * ${String(decay)} AS path_weight, rel_type
          FROM document_relations
          WHERE source_id = ANY(ARRAY[${initialDocIds.join(',')}])
          AND target_id != ALL(ARRAY[${initialDocIds.join(',')}])
          UNION
          SELECT source_id AS doc_id, 1 AS depth, ARRAY[source_id] AS visited,
                 COALESCE(weight, 1.0) * ${String(decay)} AS path_weight, rel_type
          FROM document_relations
          WHERE target_id = ANY(ARRAY[${initialDocIds.join(',')}])
          AND source_id != ALL(ARRAY[${initialDocIds.join(',')}])
          UNION ALL
          SELECT
            CASE WHEN dr.source_id = r.doc_id THEN dr.target_id ELSE dr.source_id END AS doc_id,
            r.depth + 1,
            r.visited || CASE WHEN dr.source_id = r.doc_id THEN dr.target_id ELSE dr.source_id END,
            r.path_weight * COALESCE(dr.weight, 1.0) * ${String(decay)},
            COALESCE(dr.rel_type, r.rel_type)
          FROM related r
          JOIN document_relations dr
            ON (dr.source_id = r.doc_id OR dr.target_id = r.doc_id)
          WHERE r.depth < ${String(graphHops)}
            AND CASE WHEN dr.source_id = r.doc_id THEN dr.target_id ELSE dr.source_id END != ALL(r.visited)
            AND CASE WHEN dr.source_id = r.doc_id THEN dr.target_id ELSE dr.source_id END != ALL(ARRAY[${initialDocIds.join(',')}])
        )
        SELECT DISTINCT ON (doc_id) doc_id, path_weight, rel_type
        FROM related
        ORDER BY doc_id, path_weight DESC
      `)
      ),
      docWeights = new Map<number, { relType: string | undefined; weight: number }>()

    for (const row of rows) {
      // eslint-disable-next-line @typescript-eslint/no-unnecessary-type-conversion
      const docId = Number(row.doc_id)
      docWeights.set(docId, {
        relType: row.rel_type ?? undefined,
        weight: row.path_weight
      })
    }

    const relatedDocIds: number[] = []
    // eslint-disable-next-line @typescript-eslint/no-unnecessary-type-conversion
    for (const row of rows) relatedDocIds.push(Number(row.doc_id))

    if (relatedDocIds.length === 0) return []

    const chunkRows = await db
        .select({
          communityId: documents.communityId,
          documentId: chunkSources.documentId,
          id: chunks.id,
          text: chunks.text,
          title: documents.title
        })
        .from(chunks)
        .innerJoin(chunkSources, eq(chunkSources.chunkId, chunks.id))
        .innerJoin(documents, eq(documents.id, chunkSources.documentId))
        .where(inArray(chunkSources.documentId, relatedDocIds))
        .limit(chunkLimit),
      results: SearchResult[] = []
    for (const row of chunkRows)
      if (!existingChunkIds.has(row.id)) {
        const docInfo = docWeights.get(row.documentId)
        results.push({
          communityId: row.communityId ?? undefined,
          documentId: row.documentId,
          id: row.id,
          mode: 'graph',
          relationType: docInfo?.relType,
          score: docInfo?.weight ?? 0,
          text: row.text,
          title: row.title
        })
      }

    return results
  },
  communityExpand = async (params: {
    db: DrizzleDb
    existingChunkIds: Set<number>
    limit: number
    queryEmbedding: number[]
    topCommunityId: number
  }): Promise<SearchResult[]> => {
    const { db, existingChunkIds, limit, queryEmbedding, topCommunityId } = params,
      similarity = sql<number>`1 - (${cosineDistance(chunks.embedding, queryEmbedding)})`,
      newestDocId = sql<string>`MAX(${chunkSources.documentId})`,
      rows = await db
        .select({
          communityId: documents.communityId,
          documentId: newestDocId,
          id: chunks.id,
          score: similarity,
          text: chunks.text,
          title: documents.title
        })
        .from(chunks)
        .innerJoin(chunkSources, eq(chunkSources.chunkId, chunks.id))
        .innerJoin(documents, eq(documents.id, chunkSources.documentId))
        .where(
          and(
            eq(documents.communityId, topCommunityId),
            sql`(${documents.metadata}->>'_ragts_type' IS NULL OR ${documents.metadata}->>'_ragts_type' != 'community_summary')`
          )
        )
        .groupBy(chunks.id, chunks.embedding, chunks.text, documents.title, documents.communityId)
        .orderBy(desc(similarity))
        .limit(limit),
      seen = new Set<number>(),
      results: SearchResult[] = []
    for (const row of rows)
      if (!(existingChunkIds.has(row.id) || seen.has(row.id))) {
        seen.add(row.id)
        results.push({
          communityId: row.communityId ?? undefined,

          documentId: Number(row.documentId),
          id: row.id,
          mode: 'community',
          score: row.score,
          text: row.text,
          title: row.title
        })
      }

    return results
  },
  fetchGraphRelations = async (db: DrizzleDb, docIds: number[]): Promise<GraphRelation[]> => {
    if (docIds.length === 0) return []

    const rows = await db.execute<{
        rel_type: null | string
        source_title: string
        target_title: string
      }>(
        sql.raw(`
        SELECT d1.title AS source_title, d2.title AS target_title, dr.rel_type
        FROM document_relations dr
        JOIN documents d1 ON d1.id = dr.source_id
        JOIN documents d2 ON d2.id = dr.target_id
        WHERE dr.source_id = ANY(ARRAY[${docIds.join(',')}])
          OR dr.target_id = ANY(ARRAY[${docIds.join(',')}])
      `)
      ),
      relations: GraphRelation[] = []
    for (const row of rows)
      relations.push({
        sourceTitle: row.source_title,
        targetTitle: row.target_title,
        type: row.rel_type ?? undefined
      })
    return relations
  },
  search = async (db: DrizzleDb, config: SearchConfig): Promise<SearchResult[]> => {
    const mode = config.mode ?? 'hybrid',
      limit = config.limit ?? 10,
      rrfK = config.rrfK ?? 60,
      fetchLimit = limit * 3,
      embedText = config.vectorQuery ?? config.query

    let queryEmbedding: number[] | undefined

    if (mode !== 'bm25') {
      const [emb] = await config.embed([embedText])
      queryEmbedding = emb
      if (!queryEmbedding) return []
    }

    let raw: SearchResult[]

    if (mode === 'vector') {
      if (!queryEmbedding) return []
      raw = await rawVectorSearch({ db, limit: fetchLimit, queryEmbedding, threshold: config.threshold })
    } else if (mode === 'bm25') raw = await rawBm25Search(db, config.query, fetchLimit)
    else {
      if (!queryEmbedding) return []
      raw = await rawHybridSearch({
        bm25Weight: config.bm25Weight ?? 1,
        db,
        limit: fetchLimit,
        queryEmbedding,
        queryText: config.query,
        rrfK,
        threshold: config.threshold,
        vectorWeight: config.vectorWeight ?? 1
      })
    }

    const results = dedup(raw, limit),
      allChunkIds = new Set<number>()
    for (const r of results) allChunkIds.add(r.id)

    let expanded = false

    if (config.graphHops !== undefined && config.graphHops > 0) {
      const docIdSet = new Set<number>()
      for (const r of results) docIdSet.add(r.documentId)

      const graphResults = await graphExpand({
        chunkLimit: config.graphChunkLimit ?? DEFAULT_GRAPH_CHUNK_LIMIT,
        db,
        decay: config.graphDecay ?? 1,
        existingChunkIds: allChunkIds,
        graphHops: config.graphHops,
        initialDocIds: [...docIdSet]
      })

      graphResults.sort((a, b) => b.score - a.score)
      const gw = config.graphWeight ?? 1
      for (let i = 0; i < graphResults.length; i += 1) {
        const gr = graphResults[i]
        if (gr) {
          gr.score = gw / (rrfK + i + 1)
          results.push(gr)
          allChunkIds.add(gr.id)
        }
      }
      expanded = true
    }

    if (config.communityBoost !== undefined && config.communityBoost > 0) {
      const commCounts = new Map<number, number>()
      for (const r of results)
        if (r.communityId !== undefined) commCounts.set(r.communityId, (commCounts.get(r.communityId) ?? 0) + 1)

      let topCommunityId = -1,
        maxCount = 0
      for (const [cid, cnt] of commCounts)
        if (cnt > maxCount) {
          maxCount = cnt
          topCommunityId = cid
        }

      if (topCommunityId >= 0) {
        if (!queryEmbedding) {
          const [emb] = await config.embed([embedText])
          queryEmbedding = emb
        }

        if (queryEmbedding) {
          const communityResults = await communityExpand({
            db,
            existingChunkIds: allChunkIds,
            limit: config.graphChunkLimit ?? DEFAULT_GRAPH_CHUNK_LIMIT,
            queryEmbedding,
            topCommunityId
          })

          for (let i = 0; i < communityResults.length; i += 1) {
            const cr = communityResults[i]
            if (cr) {
              cr.score = config.communityBoost / (rrfK + i + 1)
              results.push(cr)
            }
          }
          expanded = true
        }
      }
    }

    if (expanded) results.sort((a, b) => b.score - a.score)

    return results
  }

export { fetchGraphRelations, search }
