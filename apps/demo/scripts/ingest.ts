import { createEmbedFn, getRequiredString, loadDocsFromFolder, parseArgs, runWithRag } from './utils'

const args = process.argv.slice(2),
  main = async () => {
    const parsed = parseArgs(args),
      folderPath = getRequiredString(parsed, '--folder'),
      backupPath = getRequiredString(parsed, '--backup'),
      docs = loadDocsFromFolder(folderPath),
      embedFn = createEmbedFn(),
      chunkSizeFlag = parsed.get('--chunk-size'),
      chunkSize = typeof chunkSizeFlag === 'string' ? Number.parseInt(chunkSizeFlag, 10) : undefined,
      batchSizeFlag = parsed.get('--batch-size'),
      batchSize = typeof batchSizeFlag === 'string' ? Number.parseInt(batchSizeFlag, 10) : undefined
    await runWithRag(parsed, async rag => {
      const startTime = Date.now()
      let lastLog = Date.now()
      const result = await rag.ingest(docs, {
        backupPath,
        batchSize,
        chunk: chunkSize ? { chunkSize } : undefined,
        embed: embedFn,
        onProgress: (title, current, total) => {
          const now = Date.now()
          if (now - lastLog > 5000 || current === total) {
            const elapsed = ((now - startTime) / 1000).toFixed(0),
              rate = (current / ((now - startTime) / 1000)).toFixed(1)
            process.stderr.write(`[${String(current)}/${String(total)}] ${title} (${elapsed}s, ${rate} docs/s)\n`)
            lastLog = now
          }
        }
      })
      console.log(JSON.stringify({ docs: docs.length, result }, null, 2))
    })
  }
try {
  await main()
} catch (error) {
  console.error(error)
  process.exitCode = 1
}
