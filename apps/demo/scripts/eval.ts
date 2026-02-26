/* eslint-disable no-await-in-loop,max-statements,complexity,no-continue */
/** biome-ignore-all lint/performance/noAwaitInLoops: x */
/** biome-ignore-all lint/nursery/noContinue: loop guard */
import type { SearchMode, SearchResult } from 'ragts'

import { generateText } from 'ai'
import { eq, inArray } from 'drizzle-orm'
import { writeFileSync } from 'node:fs'
import {
  buildContext,
  chunkSources,
  chunks as chunksTable,
  createProvider,
  dedupSubstrings,
  documents as documentsTable,
  Rag,
  rerankChunks
} from 'ragts'
import { getDbUrl, loadJsonArray, parseArgs, runWithRag } from 'ragts/utils'

import type { EvalEntry } from './utils'

import { BASE_URL, CHAT_MODEL, EMBEDDING_DIMENSION, EMBEDDING_MODEL, RERANK_MODEL } from './constants'
import {
  agenticRerank,
  expandQuery,
  formatRetrievedDocs,
  generateHypothesis,
  generateMultihopQuery,
  getNumFlag,
  getStringFlag,
  loadSystemPrompt,
  mergeRetrievals
} from './utils'

const args = process.argv.slice(2),
  main = async () => {
    const qnaFile = args.find(a => !a.startsWith('--'))
    if (!qnaFile)
      throw new Error(
        'Usage: bun eval.ts <qna.json> --system <prompt> [--limit N] [--mode hybrid|vector|bm25] [--output path] [--rerank-top N] [--hyde] [--multihop] [--child-db URL] [--agentic-rerank] [--instruct TEXT] [--vector-weight N] [--bm25-weight N]'
      )

    const parsed = parseArgs(args),
      systemPrompt = loadSystemPrompt(parsed),
      outputPath = getStringFlag(parsed, '--output') ?? 'out.json',
      limit = getNumFlag(parsed, '--limit') ?? 20,
      modeFlag = getStringFlag(parsed, '--mode'),
      mode: SearchMode = modeFlag === 'vector' || modeFlag === 'bm25' ? modeFlag : 'hybrid',
      chatModel = getStringFlag(parsed, '--model') ?? CHAT_MODEL,
      rerankTop = getNumFlag(parsed, '--rerank-top') ?? 10,
      noRerank = parsed.get('--no-rerank') === true,
      expand = parsed.get('--expand') === true,
      hyde = parsed.get('--hyde') === true,
      multihop = parsed.get('--multihop') === true,
      useAgenticRerank = parsed.get('--agentic-rerank') === true,
      childDbUrl = getStringFlag(parsed, '--child-db'),
      rrfK = getNumFlag(parsed, '--rrf-k'),
      temperature = getNumFlag(parsed, '--temperature'),
      instruct = getStringFlag(parsed, '--instruct'),
      vectorWeight = getNumFlag(parsed, '--vector-weight'),
      bm25Weight = getNumFlag(parsed, '--bm25-weight'),
      qna = loadJsonArray<{ answer?: string; question: string }>(qnaFile),
      provider = createProvider({ baseURL: BASE_URL }),
      embedFn = provider.embedFn(EMBEDDING_MODEL),
      chat = provider.chatModel(chatModel),
      reranker = provider.rerankingModel(RERANK_MODEL),
      textConfig = getStringFlag(parsed, '--text-config') ?? 'simple',
      results: EvalEntry[] = []

    let childRag: Rag | undefined
    if (childDbUrl) {
      childRag = new Rag({ connectionString: childDbUrl, dimension: EMBEDDING_DIMENSION, textConfig })
      await childRag.init()
    }

    try {
      await runWithRag(
        {
          connectionString: getDbUrl(parsed),
          dimension: EMBEDDING_DIMENSION,
          textConfig
        },
        async rag => {
          for (let i = 0; i < qna.length; i += 1) {
            const pair = qna[i]
            if (!pair) continue
            process.stderr.write(`[${String(i + 1)}/${String(qna.length)}] ${pair.question.slice(0, 80)}...\n`)

            let vectorQuery: string | undefined
            if (hyde) {
              process.stderr.write('  Generating HyDE hypothesis...\n')
              vectorQuery = await generateHypothesis(pair.question, chat)
              process.stderr.write(`  HyDE: ${vectorQuery.slice(0, 100)}...\n`)
            } else if (instruct) vectorQuery = `Instruct: ${instruct}\nQuery:${pair.question}`

            const retrieveFrom = childRag ?? rag,
              queries = expand ? await expandQuery(pair.question, chat) : [pair.question]
            if (queries.length > 1) process.stderr.write(`  Expanded into ${String(queries.length)} queries\n`)
            const resultSets = await Promise.all(
              queries.map(async q => {
                const vq = instruct && !hyde ? `Instruct: ${instruct}\nQuery:${q}` : vectorQuery
                return retrieveFrom.retrieve({
                  bm25Weight,
                  embed: embedFn,
                  limit,
                  mode,
                  query: q,
                  rrfK,
                  vectorQuery: vq,
                  vectorWeight
                })
              })
            )
            let chunks: SearchResult[] = mergeRetrievals(resultSets)

            if (childRag) {
              process.stderr.write(`  Parent-child: reranking ${String(chunks.length)} child chunks first...\n`)
              chunks = await rerankChunks(pair.question, chunks, { model: reranker, topN: rerankTop })
              process.stderr.write(`  Parent-child: ${String(chunks.length)} reranked child → mapping to parent...\n`)

              const docFreq = new Map<number, number>()
              for (const c of chunks) docFreq.set(c.documentId, (docFreq.get(c.documentId) ?? 0) + 1)
              const rankedDocs: { docId: number; freq: number }[] = []
              for (const [docId, freq] of docFreq) rankedDocs.push({ docId, freq })
              rankedDocs.sort((a, b) => b.freq - a.freq)
              const uniqueDocIds: number[] = []
              for (let d = 0; d < Math.min(10, rankedDocs.length); d += 1) {
                const entry = rankedDocs[d]
                if (entry) uniqueDocIds.push(entry.docId)
              }

              const parentDb = await rag.init(),
                parentRows = await parentDb
                  .select({
                    documentId: chunkSources.documentId,
                    id: chunksTable.id,
                    text: chunksTable.text,
                    title: documentsTable.title
                  })
                  .from(chunksTable)
                  .innerJoin(chunkSources, eq(chunkSources.chunkId, chunksTable.id))
                  .innerJoin(documentsTable, eq(documentsTable.id, chunkSources.documentId))
                  .where(inArray(chunkSources.documentId, uniqueDocIds))
                  .groupBy(chunksTable.id, chunksTable.text, chunkSources.documentId, documentsTable.title),
                parentChunks: SearchResult[] = [],
                seenIds = new Set<number>()
              for (const row of parentRows) {
                if (seenIds.has(row.id)) continue
                seenIds.add(row.id)
                parentChunks.push({
                  documentId: row.documentId,
                  id: row.id,
                  mode: 'vector',
                  score: 0,
                  text: row.text,
                  title: row.title
                })
              }
              process.stderr.write(
                `  Parent-child: ${String(uniqueDocIds.length)} docs → ${String(parentChunks.length)} parent chunks → reranking...\n`
              )
              const batchSize = 100,
                allReranked: SearchResult[] = []
              for (let b = 0; b < parentChunks.length; b += batchSize) {
                const batch = parentChunks.slice(b, b + batchSize),
                  batchResult = await rerankChunks(pair.question, batch, {
                    model: reranker,
                    topN: Math.min(rerankTop, batch.length)
                  })
                for (const r of batchResult) allReranked.push(r)
              }
              allReranked.sort((a, b) => b.score - a.score)
              chunks = allReranked.slice(0, rerankTop)
            }

            if (!(noRerank || useAgenticRerank || childRag)) {
              process.stderr.write(`  Reranking ${String(chunks.length)} chunks → top ${String(rerankTop)}...\n`)
              chunks = await rerankChunks(pair.question, chunks, { model: reranker, topN: rerankTop })
            }

            if (useAgenticRerank) {
              const prefiltered = noRerank
                ? chunks.slice(0, rerankTop)
                : await rerankChunks(pair.question, chunks, { model: reranker, topN: rerankTop })
              process.stderr.write(
                `  Agentic reranking ${String(prefiltered.length)} chunks → top ${String(rerankTop)}...\n`
              )
              chunks = await agenticRerank(pair.question, prefiltered, { model: chat, topN: rerankTop })
            }

            if (multihop) {
              process.stderr.write('  Multi-hop: generating refined query...\n')
              try {
                const firstPassContext = buildContext(chunks),
                  refinedQuery = await generateMultihopQuery(pair.question, firstPassContext, chat)
                process.stderr.write(`  Multi-hop query: ${refinedQuery.slice(0, 100)}...\n`)

                const secondPass = await rag.retrieve({ embed: embedFn, limit, mode, query: refinedQuery, rrfK }),
                  merged = mergeRetrievals([chunks.map(c => ({ ...c })), secondPass])

                if (noRerank) chunks = merged.slice(0, rerankTop)
                else {
                  process.stderr.write(
                    `  Multi-hop reranking ${String(merged.length)} chunks → top ${String(rerankTop)}...\n`
                  )
                  chunks = await rerankChunks(pair.question, merged, { model: reranker, topN: rerankTop })
                }
              } catch (hopError) {
                process.stderr.write(
                  `  [WARN] Multi-hop failed: ${hopError instanceof Error ? hopError.message : 'unknown'}, using first pass\n`
                )
              }
            }

            const before = chunks.length
            chunks = dedupSubstrings(chunks, { prefixLength: 100 })
            if (chunks.length < before)
              process.stderr.write(
                `  Substring dedup: ${String(before)} → ${String(chunks.length)} (removed ${String(before - chunks.length)})\n`
              )

            const retrievedDocs = formatRetrievedDocs(chunks),
              context = buildContext(chunks)
            let generated = '[ERROR: timeout]'
            try {
              const response = await generateText({
                abortSignal: AbortSignal.timeout(240_000),
                model: chat,
                prompt: `Question:\n${pair.question}\n\nContext:\n${context}`,
                system: systemPrompt,
                temperature
              })
              generated = response.text.trim()
            } catch (genError) {
              process.stderr.write(
                `  [WARN] Generation failed: ${genError instanceof Error ? genError.message : 'unknown'}\n`
              )
            }
            results.push({
              generated,
              gold: pair.answer,
              question: pair.question,
              retrievedDocs
            })
          }
        }
      )
    } finally {
      if (childRag) await childRag.close()
    }

    const output = JSON.stringify(results, null, 2)
    writeFileSync(outputPath, `${output}\n`)
    process.stdout.write(`Wrote ${String(results.length)} results to ${outputPath}\n`)
  }

try {
  await main()
} catch (error) {
  console.error(error)
  process.exitCode = 1
}
