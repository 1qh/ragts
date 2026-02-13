/* eslint-disable max-statements, max-depth, no-await-in-loop */
/** biome-ignore-all lint/performance/noAwaitInLoops: x */
import { eq } from 'drizzle-orm'

import type { DrizzleDb } from './db'
import type { BackupChunk, BackupDoc, Doc, IngestConfig, IngestResult } from './types'

import { appendBackupLine, computeHash } from './backup'
import { chunkText } from './chunk'
import { chunks, documents } from './schema'

const ingest = async (db: DrizzleDb, docs: Doc[], config: IngestConfig): Promise<IngestResult> => {
  let documentsInserted = 0,
    chunksInserted = 0,
    duplicatesSkipped = 0
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
        const textChunks = await chunkText(doc.content, config.chunk),
          embedTexts: string[] = []
        for (const c of textChunks) embedTexts.push(c.text)
        const batchSize = config.batchSize ?? 64,
          allEmbeddings: number[][] = []
        for (let j = 0; j < embedTexts.length; j += batchSize) {
          const batch = embedTexts.slice(j, j + batchSize),
            batchEmbeddings = await config.embed(batch)
          for (const emb of batchEmbeddings) allEmbeddings.push(emb)
        }
        const backupChunks: BackupChunk[] = [],
          chunkRows: {
            documentId: number
            embedding: number[]
            endIndex: number
            startIndex: number
            text: string
            tokenCount: number
          }[] = []

        await db.transaction(async tx => {
          const [inserted] = await tx
            .insert(documents)
            .values({
              content: doc.content,
              contentHash,
              metadata: doc.metadata ?? {},
              title: doc.title
            })
            .returning({ id: documents.id })
          if (inserted) {
            for (let j = 0; j < textChunks.length; j += 1) {
              const chunk = textChunks[j],
                embedding = allEmbeddings[j]
              if (chunk && embedding) {
                chunkRows.push({
                  documentId: inserted.id,
                  embedding,
                  endIndex: chunk.endIndex,
                  startIndex: chunk.startIndex,
                  text: chunk.text,
                  tokenCount: chunk.tokenCount
                })
                backupChunks.push({
                  embedding,
                  endIndex: chunk.endIndex,
                  startIndex: chunk.startIndex,
                  text: chunk.text,
                  tokenCount: chunk.tokenCount
                })
              }
            }
            if (chunkRows.length > 0) await tx.insert(chunks).values(chunkRows)
          }
        })

        chunksInserted += chunkRows.length
        if (config.backupPath) {
          const backupDoc: BackupDoc = {
            chunks: backupChunks,
            content: doc.content,
            contentHash,
            metadata: doc.metadata ?? {},
            title: doc.title
          }
          appendBackupLine(config.backupPath, backupDoc)
        }
        documentsInserted += 1
      }
      if (config.onProgress) config.onProgress(doc.title, i + 1, docs.length)
    }
  }
  return { chunksInserted, documentsInserted, duplicatesSkipped }
}
export { ingest }
