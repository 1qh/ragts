/* eslint-disable complexity, max-depth, max-statements, no-await-in-loop */
/** biome-ignore-all lint/performance/noAwaitInLoops: x */
import { CryptoHasher } from 'bun'
import { eq, inArray, or } from 'drizzle-orm'
import { appendFileSync, readFileSync, writeFileSync } from 'node:fs'

import type { DrizzleDb } from './db'
import type { BackupChunk, BackupDoc, BackupRelation, ExportResult, ImportResult, ValidationResult } from './types'

import { chunks, chunkSources, documentRelations, documents } from './schema'

const computeHash = (input: string): string => {
    const hasher = new CryptoHasher('sha256')
    hasher.update(input)
    return hasher.digest('hex')
  },
  appendBackupLine = (filePath: string, doc: BackupDoc) => {
    appendFileSync(filePath, `${JSON.stringify(doc)}\n`)
  },
  parseEmbedding = (raw: number[] | string): number[] => {
    if (Array.isArray(raw)) return raw
    return JSON.parse(raw) as number[]
  },
  normalizeRelations = (rels: BackupRelation[] | string[] | undefined): BackupRelation[] | undefined => {
    if (!rels || rels.length === 0) return
    const [first] = rels
    if (typeof first === 'string') {
      const result: BackupRelation[] = []
      for (const r of rels) result.push({ title: r as string })
      return result
    }
    return rels as BackupRelation[]
  },
  exportBackup = async (db: DrizzleDb, outputPath: string): Promise<ExportResult> => {
    const rows = await db
      .select({
        chunkEmbedding: chunks.embedding,
        chunkText: chunks.text,
        chunkTokenCount: chunks.tokenCount,
        communityId: documents.communityId,
        content: documents.content,
        contentHash: documents.contentHash,
        docId: documents.id,
        endIndex: chunkSources.endIndex,
        metadata: documents.metadata,
        startIndex: chunkSources.startIndex,
        title: documents.title
      })
      .from(documents)
      .leftJoin(chunkSources, eq(chunkSources.documentId, documents.id))
      .leftJoin(chunks, eq(chunks.id, chunkSources.chunkId))
      .orderBy(documents.id, chunks.id)

    writeFileSync(outputPath, '')

    const grouped = new Map<
      number,
      {
        chunks: BackupChunk[]
        communityId: number | undefined
        content: string
        contentHash: string
        metadata: Record<string, unknown>
        title: string
      }
    >()
    for (const row of rows) {
      let entry = grouped.get(row.docId)
      if (!entry) {
        entry = {
          chunks: [],
          communityId: row.communityId ?? undefined,
          content: row.content,
          contentHash: row.contentHash,
          metadata: row.metadata ?? {},
          title: row.title
        }
        grouped.set(row.docId, entry)
      }
      if (row.chunkText !== null && row.chunkEmbedding !== null)
        entry.chunks.push({
          embedding: parseEmbedding(row.chunkEmbedding),
          endIndex: row.endIndex ?? 0,
          startIndex: row.startIndex ?? 0,
          text: row.chunkText,
          tokenCount: row.chunkTokenCount ?? 0
        })
    }

    const docIds = [...grouped.keys()],
      docIdToTitle = new Map<number, string>()
    for (const [docId, entry] of grouped) docIdToTitle.set(docId, entry.title)

    const relationsMap = new Map<number, BackupRelation[]>()
    if (docIds.length > 0) {
      const batchSize = 500
      for (let j = 0; j < docIds.length; j += batchSize) {
        const batch = docIds.slice(j, j + batchSize),
          relRows = await db
            .select({
              relType: documentRelations.relType,
              sourceId: documentRelations.sourceId,
              targetId: documentRelations.targetId,
              weight: documentRelations.weight
            })
            .from(documentRelations)
            .where(or(inArray(documentRelations.sourceId, batch), inArray(documentRelations.targetId, batch)))
        for (const rel of relRows) {
          const targetTitle = docIdToTitle.get(rel.targetId)
          if (targetTitle) {
            const backupRel: BackupRelation = { title: targetTitle }
            if (rel.relType) backupRel.type = rel.relType
            if (rel.weight !== null && rel.weight !== 1) backupRel.weight = rel.weight
            const existing = relationsMap.get(rel.sourceId)
            if (existing) existing.push(backupRel)
            else relationsMap.set(rel.sourceId, [backupRel])
          }
        }
      }
    }

    for (const [docId, entry] of grouped) {
      const rels = relationsMap.get(docId),
        backupDoc: BackupDoc = { ...entry }
      if (rels && rels.length > 0) backupDoc.relations = rels
      appendBackupLine(outputPath, backupDoc)
    }

    return { documentsExported: grouped.size, outputPath }
  },
  validateBackupLines = (lines: string[]): ValidationResult => {
    const errors: string[] = [],
      dimensions = new Set<number>(),
      hashes = new Map<string, number>()
    let totalChunks = 0

    for (let i = 0; i < lines.length; i += 1) {
      const line = lines[i]
      if (line)
        try {
          const doc = JSON.parse(line) as BackupDoc

          if (!doc.title || typeof doc.title !== 'string') errors.push(`Line ${i + 1}: missing or invalid title`)
          if (!doc.content || typeof doc.content !== 'string') errors.push(`Line ${i + 1}: missing or invalid content`)
          if (!doc.contentHash || typeof doc.contentHash !== 'string')
            errors.push(`Line ${i + 1}: missing or invalid contentHash`)
          if (!Array.isArray(doc.chunks)) errors.push(`Line ${i + 1}: missing or invalid chunks array`)

          if (Array.isArray(doc.chunks))
            for (const chunk of doc.chunks) {
              if (Array.isArray(chunk.embedding)) dimensions.add(chunk.embedding.length)
              else errors.push(`Line ${i + 1}: chunk missing embedding array`)

              totalChunks += 1
            }

          const count = hashes.get(doc.contentHash) ?? 0
          hashes.set(doc.contentHash, count + 1)
        } catch {
          errors.push(`Line ${i + 1}: invalid JSON`)
        }
    }

    const duplicateHashes: string[] = []
    for (const [hash, count] of hashes) if (count > 1) duplicateHashes.push(hash)

    return {
      dimensions,
      duplicateHashes,
      errors,
      totalChunks,
      totalDocuments: lines.length,
      valid: errors.length === 0 && dimensions.size <= 1
    }
  },
  validateBackup = (filePath: string): ValidationResult => {
    const content = readFileSync(filePath, 'utf8'),
      lines = content.split('\n').filter(l => l.trim().length > 0)
    return validateBackupLines(lines)
  },
  importBackup = async (db: DrizzleDb, filePath: string, dimension?: number): Promise<ImportResult> => {
    const content = readFileSync(filePath, 'utf8'),
      lines = content.split('\n').filter(l => l.trim().length > 0),
      validation = validateBackupLines(lines)
    if (!validation.valid) {
      const errorParts: string[] = []
      if (validation.errors.length > 0) errorParts.push(validation.errors.join('; '))
      if (validation.dimensions.size > 1)
        errorParts.push(`inconsistent embedding dimensions: [${[...validation.dimensions].join(', ')}]`)
      const detail = errorParts.length > 0 ? errorParts.join('; ') : 'invalid backup file'
      throw new Error(`Backup validation failed: ${detail}`)
    }

    let documentsImported = 0,
      chunksInserted = 0,
      duplicatesSkipped = 0
    const warnings: string[] = []

    for (const line of lines) {
      const doc = JSON.parse(line) as BackupDoc
      let isDimensionMismatch = false

      if (dimension !== undefined && doc.chunks.length > 0) {
        const [firstChunk] = doc.chunks
        if (firstChunk && firstChunk.embedding.length !== dimension) {
          warnings.push(
            `Document "${doc.title}" has embedding dimension ${firstChunk.embedding.length}, expected ${dimension}`
          )
          isDimensionMismatch = true
        }
      }

      if (!isDimensionMismatch) {
        const existing = await db
          .select({ id: documents.id })
          .from(documents)
          .where(eq(documents.contentHash, doc.contentHash))
          .limit(1)

        if (existing.length > 0) {
          duplicatesSkipped += 1
          warnings.push(`Duplicate skipped: "${doc.title}" (hash: ${doc.contentHash.slice(0, 12)}...)`)
        } else {
          let insertedChunkCount = 0
          await db.transaction(async tx => {
            const [inserted] = await tx
              .insert(documents)
              .values({
                communityId: doc.communityId,
                content: doc.content,
                contentHash: doc.contentHash,
                metadata: doc.metadata,
                title: doc.title
              })
              .returning({ id: documents.id })

            if (inserted && doc.chunks.length > 0)
              for (const chunk of doc.chunks) {
                const textHash = computeHash(chunk.text)
                await tx
                  .insert(chunks)
                  .values({
                    embedding: chunk.embedding,
                    text: chunk.text,
                    textHash,
                    tokenCount: chunk.tokenCount
                  })
                  .onConflictDoNothing({ target: chunks.textHash })

                const [chunkRow] = await tx
                  .select({ id: chunks.id })
                  .from(chunks)
                  .where(eq(chunks.textHash, textHash))
                  .limit(1)

                if (chunkRow) {
                  await tx.insert(chunkSources).values({
                    chunkId: chunkRow.id,
                    documentId: inserted.id,
                    endIndex: chunk.endIndex,
                    startIndex: chunk.startIndex
                  })
                  insertedChunkCount += 1
                }
              }
          })
          chunksInserted += insertedChunkCount
          documentsImported += 1
        }
      }
    }

    const titleToImportedIds = new Map<string, number[]>()
    for (const line of lines) {
      const doc = JSON.parse(line) as BackupDoc,
        [inserted] = await db
          .select({ id: documents.id })
          .from(documents)
          .where(eq(documents.contentHash, doc.contentHash))
          .limit(1)
      if (inserted) {
        const existingIds = titleToImportedIds.get(doc.title)
        if (existingIds) existingIds.push(inserted.id)
        else titleToImportedIds.set(doc.title, [inserted.id])
      }
    }

    for (const line of lines) {
      const doc = JSON.parse(line) as BackupDoc,
        rels = normalizeRelations(doc.relations as BackupRelation[] | string[] | undefined)
      if (rels && rels.length > 0) {
        const sourceIds = titleToImportedIds.get(doc.title)
        if (sourceIds)
          for (const rel of rels)
            if (rel.title !== doc.title) {
              const targetIds = titleToImportedIds.get(rel.title)
              if (targetIds)
                for (const sId of sourceIds)
                  for (const tId of targetIds)
                    await db
                      .insert(documentRelations)
                      .values({
                        relType: rel.type,
                        sourceId: sId,
                        targetId: tId,
                        weight: rel.weight
                      })
                      .onConflictDoNothing()
            }
      }
    }

    return { chunksInserted, documentsImported, duplicatesSkipped, warnings }
  }

export { appendBackupLine, computeHash, exportBackup, importBackup, validateBackup }
