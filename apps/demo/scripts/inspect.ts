import { chunks, count, documents } from 'ragts'

import { parseArgs, runWithRag } from './utils'

const args = process.argv.slice(2),
  main = async () => {
    const parsed = parseArgs(args)
    await runWithRag(parsed, async rag => {
      const db = await rag.init(),
        extensionRows = await db.execute<{ extname: string; extversion: string }>(
          "SELECT extname, extversion FROM pg_extension WHERE extname IN ('pg_textsearch', 'vector', 'vectorscale') ORDER BY extname"
        ),
        documentCountRows = await db.select({ count: count(documents.id) }).from(documents),
        chunkCountRows = await db.select({ count: count(chunks.id) }).from(chunks),
        indexRows = await db.execute<{ indexdef: string; indexname: string; tablename: string }>(
          "SELECT tablename, indexname, indexdef FROM pg_indexes WHERE tablename IN ('ragts_chunks', 'ragts_documents') ORDER BY tablename, indexname"
        ),
        documentCount = documentCountRows[0]?.count ?? 0,
        chunkCount = chunkCountRows[0]?.count ?? 0
      console.log('Extensions')
      console.table(extensionRows)
      console.log('Counts')
      console.table([
        { metric: 'documents', value: documentCount },
        { metric: 'chunks', value: chunkCount }
      ])
      console.log('Indexes')
      console.table(indexRows)
    })
  }
try {
  await main()
} catch (error) {
  console.error(error)
  process.exitCode = 1
}
