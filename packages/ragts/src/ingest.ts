/* eslint-disable complexity, max-statements, max-depth, no-await-in-loop */
/** biome-ignore-all lint/performance/noAwaitInLoops: x */
import { eq, inArray } from 'drizzle-orm'

import type { DrizzleDb } from './db'
import type { BackupChunk, BackupDoc, Doc, IngestConfig, IngestResult, RelationTarget } from './types'

import { appendBackupLine, computeHash } from './backup'
import { chunkText } from './chunk'
import { chunks, chunkSources, documentRelations, documents } from './schema'

const parseRelTarget = (target: RelationTarget): { title: string; type?: string; weight?: number } =>
  typeof target === 'string' ? { title: target } : target

interface DedupEntry {
  embedding?: number[]
  sources: { docId: number; endIndex: number; startIndex: number }[]
  text: string
  textHash: string
  tokenCount: number
}

const ingest = async (db: DrizzleDb, docs: Doc[], config: IngestConfig): Promise<IngestResult> => {
  let documentsInserted = 0,
    duplicatesSkipped = 0

  const dedupMap = new Map<string, DedupEntry>(),
    docIds: number[] = [],
    backupEntries: { backupChunks: BackupChunk[]; contentHash: string; doc: Doc }[] = [],
    titleToNewIds = new Map<string, number[]>()

  for (let i = 0; i < docs.length; i += 1) {
    const doc = docs[i]
    if (doc) {
      const contentHash = computeHash(`${doc.title}${doc.content}`),
        existing = await db
          .select({ id: documents.id })
          .from(documents)
          .where(eq(documents.contentHash, contentHash))
          .limit(1)
      if (existing.length > 0) duplicatesSkipped += 1
      else {
        const [inserted] = await db
          .insert(documents)
          .values({
            content: doc.content,
            contentHash,
            metadata: doc.metadata ?? {},
            title: doc.title
          })
          .returning({ id: documents.id })
        if (inserted) {
          docIds.push(inserted.id)
          const existingIds = titleToNewIds.get(doc.title)
          if (existingIds) existingIds.push(inserted.id)
          else titleToNewIds.set(doc.title, [inserted.id])
          documentsInserted += 1
          const textChunks = chunkText(doc.content, config.chunk)
          for (const c of textChunks) {
            const finalText = config.transformChunk ? config.transformChunk(c.text, doc) : c.text,
              hash = computeHash(finalText),
              entry = dedupMap.get(hash)
            if (entry) entry.sources.push({ docId: inserted.id, endIndex: c.endIndex, startIndex: c.startIndex })
            else
              dedupMap.set(hash, {
                sources: [{ docId: inserted.id, endIndex: c.endIndex, startIndex: c.startIndex }],
                text: finalText,
                textHash: hash,
                tokenCount: c.tokenCount
              })
          }
          backupEntries.push({ backupChunks: [], contentHash, doc })
        }
      }
      if (config.onProgress) config.onProgress(doc.title, i + 1, docs.length)
    }
  }

  const uniqueEntries: DedupEntry[] = []
  for (const [, entry] of dedupMap) uniqueEntries.push(entry)

  const allHashes: string[] = []
  for (const entry of uniqueEntries) allHashes.push(entry.textHash)

  const existingHashes = new Set<string>()
  if (allHashes.length > 0) {
    const lookupBatchSize = 500
    for (let j = 0; j < allHashes.length; j += lookupBatchSize) {
      const batch = allHashes.slice(j, j + lookupBatchSize),
        rows = await db.select({ textHash: chunks.textHash }).from(chunks).where(inArray(chunks.textHash, batch))
      for (const row of rows) existingHashes.add(row.textHash)
    }
  }

  const newTexts: string[] = [],
    newEntries: DedupEntry[] = []
  for (const entry of uniqueEntries)
    if (!existingHashes.has(entry.textHash)) {
      newTexts.push(entry.text)
      newEntries.push(entry)
    }

  const batchSize = config.batchSize ?? 64
  for (let j = 0; j < newTexts.length; j += batchSize) {
    const batch = newTexts.slice(j, j + batchSize),
      batchEmbeddings = await config.embed(batch)
    for (let k = 0; k < batchEmbeddings.length; k += 1) {
      const entry = newEntries[j + k]
      if (entry) entry.embedding = batchEmbeddings[k]
    }
  }

  const chunkRows: { embedding: number[]; text: string; textHash: string; tokenCount: number }[] = []
  for (const entry of newEntries)
    if (entry.embedding)
      chunkRows.push({
        embedding: entry.embedding,
        text: entry.text,
        textHash: entry.textHash,
        tokenCount: entry.tokenCount
      })

  if (chunkRows.length > 0) {
    const insertBatchSize = 500
    for (let j = 0; j < chunkRows.length; j += insertBatchSize) {
      const batch = chunkRows.slice(j, j + insertBatchSize)
      await db.insert(chunks).values(batch).onConflictDoNothing({ target: chunks.textHash })
    }
  }

  const hashToId = new Map<string, number>()
  if (allHashes.length > 0) {
    const lookupBatchSize = 500
    for (let j = 0; j < allHashes.length; j += lookupBatchSize) {
      const batch = allHashes.slice(j, j + lookupBatchSize),
        rows = await db
          .select({ id: chunks.id, textHash: chunks.textHash })
          .from(chunks)
          .where(inArray(chunks.textHash, batch))
      for (const row of rows) hashToId.set(row.textHash, row.id)
    }
  }

  const sourceRows: { chunkId: number; documentId: number; endIndex: number; startIndex: number }[] = []
  for (const entry of uniqueEntries) {
    const chunkId = hashToId.get(entry.textHash)
    if (chunkId !== undefined)
      for (const src of entry.sources)
        sourceRows.push({ chunkId, documentId: src.docId, endIndex: src.endIndex, startIndex: src.startIndex })
  }

  if (sourceRows.length > 0) {
    const insertBatchSize = 500
    for (let j = 0; j < sourceRows.length; j += insertBatchSize) {
      const batch = sourceRows.slice(j, j + insertBatchSize)
      await db.insert(chunkSources).values(batch)
    }
  }

  const chunksInserted = chunkRows.length,
    chunksReused = existingHashes.size

  if (config.backupPath) {
    const hashToEmbedding = new Map<string, number[]>()
    for (const entry of newEntries) if (entry.embedding) hashToEmbedding.set(entry.textHash, entry.embedding)

    if (existingHashes.size > 0) {
      const existArr = [...existingHashes],
        lbSize = 500
      for (let j = 0; j < existArr.length; j += lbSize) {
        const batch = existArr.slice(j, j + lbSize),
          rows = await db
            .select({ embedding: chunks.embedding, textHash: chunks.textHash })
            .from(chunks)
            .where(inArray(chunks.textHash, batch))
        for (const row of rows) {
          const emb = Array.isArray(row.embedding) ? row.embedding : (JSON.parse(row.embedding as string) as number[])
          hashToEmbedding.set(row.textHash, emb)
        }
      }
    }

    for (const be of backupEntries) {
      const textChunks = chunkText(be.doc.content, config.chunk)
      for (const c of textChunks) {
        const finalText = config.transformChunk ? config.transformChunk(c.text, be.doc) : c.text,
          hash = computeHash(finalText),
          emb = hashToEmbedding.get(hash)
        if (emb)
          be.backupChunks.push({
            embedding: emb,
            endIndex: c.endIndex,
            startIndex: c.startIndex,
            text: finalText,
            tokenCount: c.tokenCount
          })
      }
      const backupDoc: BackupDoc = {
        chunks: be.backupChunks,
        content: be.doc.content,
        contentHash: be.contentHash,
        metadata: be.doc.metadata ?? {},
        title: be.doc.title
      }
      appendBackupLine(config.backupPath, backupDoc)
    }
  }

  let relationsInserted = 0
  const unresolvedRelations: string[] = []

  if (config.relations) {
    const titleToAllIds = new Map<string, number[]>()
    for (const [title, ids] of titleToNewIds) titleToAllIds.set(title, [...ids])

    const allRelationTitles = new Set<string>()
    for (const [sourceTitle, targets] of Object.entries(config.relations)) {
      allRelationTitles.add(sourceTitle)
      for (const t of targets) {
        const parsed = parseRelTarget(t)
        allRelationTitles.add(parsed.title)
      }
    }

    const missingTitles: string[] = []
    for (const title of allRelationTitles) if (!titleToAllIds.has(title)) missingTitles.push(title)

    if (missingTitles.length > 0) {
      const lookupBatchSize = 500
      for (let j = 0; j < missingTitles.length; j += lookupBatchSize) {
        const batch = missingTitles.slice(j, j + lookupBatchSize),
          rows = await db
            .select({ id: documents.id, title: documents.title })
            .from(documents)
            .where(inArray(documents.title, batch))
        for (const row of rows) {
          const existingIds = titleToAllIds.get(row.title)
          if (existingIds) existingIds.push(row.id)
          else titleToAllIds.set(row.title, [row.id])
        }
      }
    }

    const relationRows: { relType?: string; sourceId: number; targetId: number; weight?: number }[] = []
    for (const [sourceTitle, targets] of Object.entries(config.relations)) {
      const sourceIds = titleToAllIds.get(sourceTitle)
      if (sourceIds)
        for (const raw of targets) {
          const target = parseRelTarget(raw)
          if (target.title === sourceTitle) {
            /* Skip self-reference */
          } else {
            const targetIds = titleToAllIds.get(target.title)
            if (targetIds)
              for (const sId of sourceIds)
                for (const tId of targetIds)
                  relationRows.push({ relType: target.type, sourceId: sId, targetId: tId, weight: target.weight })
            else unresolvedRelations.push(target.title)
          }
        }
    }

    if (relationRows.length > 0) {
      const insertBatchSize = 500
      for (let j = 0; j < relationRows.length; j += insertBatchSize) {
        const batch = relationRows.slice(j, j + insertBatchSize),
          result = await db
            .insert(documentRelations)
            .values(batch)
            .onConflictDoNothing()
            .returning({ id: documentRelations.id })
        relationsInserted += result.length
      }
    }
  }

  /** biome-ignore lint/suspicious/noImportCycles: dynamic import breaks circular dep */
  const { detectCommunities } = await import('./community'),
    communitiesDetected = config.relations ? await detectCommunities(db) : 0

  return {
    chunksInserted,
    chunksReused,
    communitiesDetected,
    documentsInserted,
    duplicatesSkipped,
    relationsInserted,
    unresolvedRelations: [...new Set(unresolvedRelations)]
  }
}

export { ingest }
