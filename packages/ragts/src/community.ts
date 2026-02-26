/* eslint-disable max-statements, no-await-in-loop */
/** biome-ignore-all lint/performance/noAwaitInLoops: x */
import { inArray, sql } from 'drizzle-orm'

import type { DrizzleDb } from './db'

import { documentRelations, documents } from './schema'

const detectCommunities = async (db: DrizzleDb): Promise<number> => {
    const docRows = await db.select({ id: documents.id }).from(documents)
    if (docRows.length === 0) return 0

    const allIds: number[] = []
    for (const r of docRows) allIds.push(r.id)

    const relRows = await db
        .select({ sourceId: documentRelations.sourceId, targetId: documentRelations.targetId })
        .from(documentRelations),
      parent = new Map<number, number>()
    for (const id of allIds) parent.set(id, id)

    const find = (x: number): number => {
        let root = x,
          p = parent.get(root)
        while (p !== undefined && p !== root) {
          root = p
          p = parent.get(root)
        }
        let curr = x
        while (curr !== root) {
          const next = parent.get(curr) ?? root
          parent.set(curr, root)
          curr = next
        }
        return root
      },
      union = (a: number, b: number) => {
        const ra = find(a),
          rb = find(b)
        if (ra !== rb) parent.set(ra, rb)
      }

    for (const rel of relRows) union(rel.sourceId, rel.targetId)

    const rootToId = new Map<number, number>()
    let nextId = 0
    for (const id of allIds) {
      const root = find(id)
      if (!rootToId.has(root)) {
        rootToId.set(root, nextId)
        nextId += 1
      }
    }

    const communityToIds = new Map<number, number[]>()
    for (const id of allIds) {
      const root = find(id),
        cId = rootToId.get(root)
      if (cId !== undefined) {
        const existing = communityToIds.get(cId)
        if (existing) existing.push(id)
        else communityToIds.set(cId, [id])
      }
    }

    for (const [cId, ids] of communityToIds) {
      const batchSize = 500
      for (let j = 0; j < ids.length; j += batchSize) {
        const batch = ids.slice(j, j + batchSize)
        await db.update(documents).set({ communityId: cId }).where(inArray(documents.id, batch))
      }
    }

    return rootToId.size
  },
  buildCommunitySummaries = async (
    db: DrizzleDb,
    config: {
      chunk?: { chunkSize?: number; overlap?: number }
      embed: (texts: string[]) => Promise<number[][]>
      minCommunitySize?: number
      summarize: (docs: { content: string; title: string }[]) => Promise<string>
    }
  ): Promise<{ communitiesProcessed: number; summariesGenerated: number }> => {
    /** biome-ignore lint/suspicious/noImportCycles: dynamic import breaks circular dep */
    const { ingest } = await import('./ingest')

    await db.execute(sql`DELETE FROM documents WHERE metadata->>'_ragts_type' = 'community_summary'`)

    const minSize = config.minCommunitySize ?? 2,
      communityRows = await db
        .select({ communityId: documents.communityId, content: documents.content, title: documents.title })
        .from(documents)
        .where(sql`community_id IS NOT NULL AND (metadata->>'_ragts_type') IS DISTINCT FROM 'community_summary'`)
        .orderBy(documents.communityId),
      communities = new Map<number, { content: string; title: string }[]>()
    for (const row of communityRows)
      if (row.communityId === null) {
        /* Skip */
      } else {
        const existing = communities.get(row.communityId)
        if (existing) existing.push({ content: row.content, title: row.title })
        else communities.set(row.communityId, [{ content: row.content, title: row.title }])
      }

    let communitiesProcessed = 0,
      summariesGenerated = 0

    for (const [cId, docs] of communities)
      if (docs.length < minSize) {
        /* Skip */
      } else {
        communitiesProcessed += 1
        const summaryText = await config.summarize(docs),
          titles: string[] = []
        for (const d of docs) titles.push(d.title)
        await ingest(
          db,
          [
            {
              content: summaryText,
              metadata: { _ragts_community_id: cId, _ragts_member_titles: titles, _ragts_type: 'community_summary' },
              title: `_ragts_community_${String(cId)}`
            }
          ],
          { chunk: config.chunk, embed: config.embed }
        )
        summariesGenerated += 1
      }

    return { communitiesProcessed, summariesGenerated }
  }

export { buildCommunitySummaries, detectCommunities }
