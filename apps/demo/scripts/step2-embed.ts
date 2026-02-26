/** biome-ignore-all lint/performance/noAwaitInLoops: sequential batched embedding */
/* eslint-disable no-await-in-loop, max-statements */
import { appendFileSync, readFileSync, writeFileSync } from 'node:fs'
import { createEmbedFn } from 'ragts'
import { parseArgs } from 'ragts/utils'

import type { EmbeddedChunk, Step1Output, Step2Line } from './utils'

import { BASE_URL, EMBEDDING_MODEL } from './constants'

const args = process.argv.slice(2),
  main = async () => {
    const parsed = parseArgs(args),
      inputFlag = parsed.get('--input'),
      inputPath = typeof inputFlag === 'string' ? inputFlag : 'exp/step1-chunks.json',
      outputFlag = parsed.get('--output'),
      outputPath = typeof outputFlag === 'string' ? outputFlag : 'exp/step2-embedded.jsonl',
      batchSizeFlag = parsed.get('--batch-size'),
      batchSize = typeof batchSizeFlag === 'string' ? Number.parseInt(batchSizeFlag, 10) : 64

    process.stderr.write(`Loading chunks from ${inputPath}...\n`)
    const step1 = JSON.parse(readFileSync(inputPath, 'utf8')) as Step1Output
    process.stderr.write(`Loaded ${String(step1.stats.totalDocs)} docs, ${String(step1.stats.totalChunks)} chunks\n`)
    process.stderr.write(`Unique hashes: ${String(step1.stats.uniqueHashes)} (will embed only unique)\n\n`)

    const uniqueTexts: string[] = [],
      uniqueHashes: string[] = [],
      seen = new Set<string>()

    for (const doc of step1.docs)
      for (const chunk of doc.chunks)
        if (!seen.has(chunk.textHash)) {
          seen.add(chunk.textHash)
          uniqueTexts.push(chunk.text)
          uniqueHashes.push(chunk.textHash)
        }

    process.stderr.write(`Unique texts to embed: ${String(uniqueTexts.length)}\n`)
    process.stderr.write(`Batch size: ${String(batchSize)}\n\n`)

    const embedFn = createEmbedFn({ baseURL: BASE_URL, model: EMBEDDING_MODEL }),
      hashToEmbedding = new Map<string, number[]>(),
      totalBatches = Math.ceil(uniqueTexts.length / batchSize),
      startTime = performance.now()

    for (let b = 0; b < totalBatches; b += 1) {
      const batchStart = b * batchSize,
        batchEnd = Math.min(batchStart + batchSize, uniqueTexts.length),
        batchTexts = uniqueTexts.slice(batchStart, batchEnd),
        batchHashes = uniqueHashes.slice(batchStart, batchEnd),
        embedStart = performance.now(),
        embeddings = await embedFn(batchTexts),
        embedMs = Math.round(performance.now() - embedStart)

      for (let i = 0; i < batchHashes.length; i += 1) {
        const h = batchHashes[i],
          e = embeddings[i]
        if (h && e) hashToEmbedding.set(h, e)
      }

      const elapsed = ((performance.now() - startTime) / 1000).toFixed(1),
        done = batchEnd,
        rate = (done / ((performance.now() - startTime) / 1000)).toFixed(1)
      process.stderr.write(
        `[batch ${String(b + 1)}/${String(totalBatches)}] embedded ${String(batchStart)}-${String(batchEnd)} (${String(embedMs)}ms) | total: ${String(done)}/${String(uniqueTexts.length)} | ${elapsed}s | ${rate} chunks/s\n`
      )
    }

    const embedElapsed = Math.round(performance.now() - startTime)
    process.stderr.write(`\nEmbedding done in ${String(embedElapsed)}ms\n`)
    process.stderr.write(`Writing ${outputPath}...\n`)

    const writeStart = performance.now()
    writeFileSync(outputPath, '')

    for (const doc of step1.docs) {
      const embeddedChunks: EmbeddedChunk[] = []
      for (const chunk of doc.chunks) {
        const embedding = hashToEmbedding.get(chunk.textHash)
        if (embedding) embeddedChunks.push({ embedding, text: chunk.text, textHash: chunk.textHash })
      }
      const line: Step2Line = { chunks: embeddedChunks, title: doc.title }
      appendFileSync(outputPath, `${JSON.stringify(line)}\n`)
    }
    const writeMs = Math.round(performance.now() - writeStart),
      totalElapsed = Math.round(performance.now() - startTime)
    process.stderr.write('\n--- Step 2 Summary ---\n')
    process.stderr.write(`Unique chunks embedded: ${String(hashToEmbedding.size)}\n`)
    process.stderr.write(`Embedding time:        ${String(embedElapsed)}ms\n`)
    process.stderr.write(`Write time:            ${String(writeMs)}ms\n`)
    process.stderr.write(`Total time:            ${String(totalElapsed)}ms\n`)
    process.stderr.write(`Output:                ${outputPath}\n`)
  }

try {
  await main()
} catch (error) {
  console.error(error)
  process.exitCode = 1
}
