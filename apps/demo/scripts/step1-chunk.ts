/* eslint-disable complexity, max-statements */
import { createHash } from 'node:crypto'
import { writeFileSync } from 'node:fs'
import { chunkText } from 'ragts'
import { loadDocsFromFolder, parseArgs } from 'ragts/utils'

import type { ChunkEntry, Step1Output } from './utils'

import { normalizeMarkdown } from './normalize'

const args = process.argv.slice(2),
  main = () => {
    const parsed = parseArgs(args),
      folderFlag = parsed.get('--folder'),
      folderPath = typeof folderFlag === 'string' ? folderFlag : '',
      outputFlag = parsed.get('--output'),
      outputPath = typeof outputFlag === 'string' ? outputFlag : 'exp/step1-chunks.json',
      chunkSizeFlag = parsed.get('--chunk-size'),
      chunkSize = typeof chunkSizeFlag === 'string' ? Number.parseInt(chunkSizeFlag, 10) : 2048,
      overlapFlag = parsed.get('--overlap'),
      overlap = typeof overlapFlag === 'string' ? Number.parseInt(overlapFlag, 10) : 0,
      contextual = parsed.get('--contextual') === true

    if (!folderPath)
      throw new Error(
        'Usage: bun step1-chunk.ts --folder <path> [--output <path>] [--chunk-size N] [--overlap N] [--contextual]'
      )

    process.stderr.write(`Loading docs from ${folderPath}...\n`)
    const rawDocs = loadDocsFromFolder(folderPath)
    process.stderr.write(`Loaded ${String(rawDocs.length)} documents\n\n`)

    const startTime = performance.now(),
      allDocs: Step1Output['docs'] = [],
      globalHashes = new Map<string, number>()
    let totalChunks = 0

    for (let i = 0; i < rawDocs.length; i += 1) {
      const doc = rawDocs[i]
      if (!doc) break

      const textChunks = chunkText(doc.content, { chunkSize, normalize: normalizeMarkdown, overlap }),
        entries: ChunkEntry[] = []

      for (const c of textChunks) {
        const finalText = contextual ? `[${doc.title}]\n${c.text}` : c.text,
          textHash = createHash('sha256').update(finalText).digest('hex')
        globalHashes.set(textHash, (globalHashes.get(textHash) ?? 0) + 1)
        entries.push({
          endIndex: c.endIndex,
          startIndex: c.startIndex,
          text: finalText,
          textHash,
          tokenCount: c.tokenCount
        })
      }

      totalChunks += entries.length
      allDocs.push({ chunkCount: entries.length, chunks: entries, title: doc.title })

      if ((i + 1) % 100 === 0 || i === rawDocs.length - 1) {
        const elapsed = ((performance.now() - startTime) / 1000).toFixed(1)
        process.stderr.write(
          `[${String(i + 1)}/${String(rawDocs.length)}] ${doc.title.slice(0, 60)} | chunks so far: ${String(totalChunks)} | ${elapsed}s\n`
        )
      }
    }

    const elapsedMs = Math.round(performance.now() - startTime)
    let duplicateChunks = 0
    for (const [, count] of globalHashes) if (count > 1) duplicateChunks += count - 1

    const output: Step1Output = {
      config: { chunkSize, folder: folderPath, overlap },
      docs: allDocs,
      stats: {
        duplicateChunks,
        elapsedMs,
        totalChunks,
        totalDocs: rawDocs.length,
        uniqueHashes: globalHashes.size
      }
    }

    writeFileSync(outputPath, `${JSON.stringify(output, null, 2)}\n`)

    process.stderr.write('\n--- Step 1 Summary ---\n')
    process.stderr.write(`Documents:        ${String(output.stats.totalDocs)}\n`)
    process.stderr.write(`Total chunks:     ${String(output.stats.totalChunks)}\n`)
    process.stderr.write(`Unique hashes:    ${String(output.stats.uniqueHashes)}\n`)
    process.stderr.write(`Duplicate chunks: ${String(output.stats.duplicateChunks)}\n`)
    process.stderr.write(`Elapsed:          ${String(output.stats.elapsedMs)}ms\n`)
    process.stderr.write(`Output:           ${outputPath}\n`)

    const sizes: number[] = []
    for (const d of allDocs) for (const c of d.chunks) sizes.push(c.text.length)
    sizes.sort((a, b) => a - b)
    let sum = 0
    for (const s of sizes) sum += s
    const p50 = sizes[Math.floor(sizes.length * 0.5)] ?? 0,
      p90 = sizes[Math.floor(sizes.length * 0.9)] ?? 0,
      p99 = sizes[Math.floor(sizes.length * 0.99)] ?? 0,
      min = sizes[0] ?? 0,
      max = sizes.at(-1) ?? 0,
      avg = Math.round(sum / sizes.length)
    process.stderr.write(
      `Chunk sizes:      min=${String(min)} avg=${String(avg)} p50=${String(p50)} p90=${String(p90)} p99=${String(p99)} max=${String(max)}\n`
    )
  }

try {
  main()
} catch (error) {
  console.error(error)
  process.exitCode = 1
}
