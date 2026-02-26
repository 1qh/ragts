/* eslint-disable max-statements, @typescript-eslint/require-await */
/** biome-ignore-all lint/suspicious/useAwait: x */
/** biome-ignore-all lint/performance/noAwaitInLoops: x */

import type { Doc } from 'ragts'

import { createHash } from 'node:crypto'
import { createReadStream } from 'node:fs'
import { createInterface } from 'node:readline'
import { Rag } from 'ragts'
import { loadDocsFromFolder, parseArgs } from 'ragts/utils'

import type { Step2Line } from './utils'

import { EMBEDDING_DIMENSION } from './constants'
import { normalizeMarkdown } from './normalize'

const args = process.argv.slice(2),
  main = async () => {
    const parsed = parseArgs(args),
      folderFlag = parsed.get('--folder'),
      folderPath = typeof folderFlag === 'string' ? folderFlag : '',
      inputFlag = parsed.get('--input'),
      inputPath = typeof inputFlag === 'string' ? inputFlag : 'exp/step2-embedded.jsonl',
      dbFlag = parsed.get('--db'),
      dbUrl = typeof dbFlag === 'string' ? dbFlag : 'postgresql://postgres:postgres@localhost:5432/postgres',
      textCfgFlag = parsed.get('--text-config'),
      textCfg = typeof textCfgFlag === 'string' ? textCfgFlag : 'simple',
      chunkSizeFlag = parsed.get('--chunk-size'),
      chunkSize = typeof chunkSizeFlag === 'string' ? Number.parseInt(chunkSizeFlag, 10) : 2048,
      contextual = parsed.get('--contextual') === true

    if (!folderPath)
      throw new Error(
        'Usage: bun step3-insert.ts --folder <path> [--input <step2.jsonl>] [--db url] [--chunk-size N] [--contextual]'
      )

    process.stderr.write(`Loading docs from ${folderPath}...\n`)
    const docs = loadDocsFromFolder(folderPath)
    process.stderr.write(`Loaded ${String(docs.length)} documents\n`)

    process.stderr.write(`Loading embeddings from ${inputPath}...\n`)
    const hashToEmbedding = new Map<string, number[]>()
    let totalEmbedded = 0

    const rl = createInterface({ crlfDelay: Number.POSITIVE_INFINITY, input: createReadStream(inputPath, 'utf8') })
    for await (const line of rl)
      if (line.trim()) {
        const s2doc = JSON.parse(line) as Step2Line
        for (const c of s2doc.chunks) {
          hashToEmbedding.set(c.textHash, c.embedding)
          totalEmbedded += 1
        }
      }

    process.stderr.write(`Loaded ${String(totalEmbedded)} embeddings (${String(hashToEmbedding.size)} unique)\n\n`)

    const embed = async (texts: string[]): Promise<number[][]> => {
        const results: number[][] = []
        for (const text of texts) {
          const hash = createHash('sha256').update(text).digest('hex'),
            emb = hashToEmbedding.get(hash)
          if (emb) results.push(emb)
          else {
            process.stderr.write(`[WARN] No pre-computed embedding for hash ${hash.slice(0, 12)}...\n`)
            results.push(Array.from<number>({ length: EMBEDDING_DIMENSION }).fill(0))
          }
        }
        return results
      },
      rag = new Rag({ connectionString: dbUrl, dimension: EMBEDDING_DIMENSION, textConfig: textCfg }),
      startTime = performance.now(),
      contextualTransform = contextual ? (text: string, doc: Doc) => `[${doc.title}]\n${text}` : undefined,
      result = await rag.ingest(docs, {
        chunk: { chunkSize, normalize: normalizeMarkdown },
        embed,
        onProgress: (title, current, total) => {
          if (current % 200 === 0 || current === total) {
            const elapsed = ((performance.now() - startTime) / 1000).toFixed(1),
              rate = (current / ((performance.now() - startTime) / 1000)).toFixed(1)
            process.stderr.write(
              `[${String(current)}/${String(total)}] ${title.slice(0, 50)} | ${elapsed}s | ${rate} docs/s\n`
            )
          }
        },
        transformChunk: contextualTransform
      }),
      elapsedMs = Math.round(performance.now() - startTime)
    await rag.close()

    process.stderr.write('\n--- Step 3 Summary ---\n')
    process.stderr.write(`Documents inserted: ${String(result.documentsInserted)}\n`)
    process.stderr.write(`Documents skipped:  ${String(result.duplicatesSkipped)}\n`)
    process.stderr.write(`Chunks inserted:    ${String(result.chunksInserted)}\n`)
    process.stderr.write(`Chunks reused:      ${String(result.chunksReused)}\n`)
    process.stderr.write(`Elapsed:            ${String(elapsedMs)}ms\n`)

    process.stdout.write(`${JSON.stringify({ ...result, elapsedMs }, null, 2)}\n`)
  }

try {
  await main()
} catch (error) {
  console.error(error)
  process.exitCode = 1
}
