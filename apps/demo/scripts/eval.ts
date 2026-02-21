/* eslint-disable no-await-in-loop,max-statements,max-depth */
/** biome-ignore-all lint/performance/noAwaitInLoops: x */
import type { SearchMode, SearchResult } from 'ragts'

import { createOpenAICompatible } from '@ai-sdk/openai-compatible'
import { generateText } from 'ai'
import { existsSync, readFileSync, writeFileSync } from 'node:fs'
import {
  buildContext,
  createEmbedFn,
  dedupSubstrings,
  getDbUrl,
  loadJsonArray,
  parseArgs,
  rerankChunks,
  runWithRag
} from 'ragts'

import { BASE_URL, CHAT_MODEL, EMBEDDING_DIMENSION, EMBEDDING_MODEL } from './constants'
import { createRerankingModel, expandQuery, mergeRetrievals } from './utils'

interface EvalEntry {
  generated: string
  gold?: string
  question: string
  retrievedDocs: { documentId: number; id: number; score: number; text: string; title: string }[]
}

const args = process.argv.slice(2),
  main = async () => {
    const qnaFile = args.find(a => !a.startsWith('--'))
    if (!qnaFile)
      throw new Error(
        'Usage: bun eval.ts <qna.json> --system <prompt> [--limit N] [--mode hybrid|vector|bm25] [--model name] [--output path] [--rerank-top N]'
      )

    const parsed = parseArgs(args),
      systemFlag = parsed.get('--system'),
      systemRaw = typeof systemFlag === 'string' ? systemFlag : '',
      systemPrompt = systemRaw && existsSync(systemRaw) ? readFileSync(systemRaw, 'utf8') : systemRaw,
      outputFlag = parsed.get('--output'),
      outputPath = typeof outputFlag === 'string' ? outputFlag : 'out.json',
      limitFlag = parsed.get('--limit'),
      limit = typeof limitFlag === 'string' ? Number.parseInt(limitFlag, 10) : 20,
      modeFlag = parsed.get('--mode'),
      mode: SearchMode = modeFlag === 'vector' || modeFlag === 'bm25' ? modeFlag : 'hybrid',
      modelFlag = parsed.get('--model'),
      chatModel = typeof modelFlag === 'string' ? modelFlag : CHAT_MODEL,
      rerankTopFlag = parsed.get('--rerank-top'),
      rerankTop = typeof rerankTopFlag === 'string' ? Number.parseInt(rerankTopFlag, 10) : 10,
      noRerank = parsed.get('--no-rerank') === true,
      expand = parsed.get('--expand') === true,
      rrfKFlag = parsed.get('--rrf-k'),
      rrfK = typeof rrfKFlag === 'string' ? Number.parseInt(rrfKFlag, 10) : undefined,
      tempFlag = parsed.get('--temperature'),
      temperature = typeof tempFlag === 'string' ? Number.parseFloat(tempFlag) : undefined,
      qna = loadJsonArray<{ answer?: string; question: string }>(qnaFile),
      embedFn = createEmbedFn({ baseURL: BASE_URL, model: EMBEDDING_MODEL }),
      provider = createOpenAICompatible({ apiKey: 'none', baseURL: BASE_URL, name: 'mlx' }),
      textCfg = parsed.get('--text-config'),
      results: EvalEntry[] = []

    await runWithRag(
      {
        connectionString: getDbUrl(parsed),
        dimension: EMBEDDING_DIMENSION,
        textConfig: typeof textCfg === 'string' ? textCfg : 'simple'
      },
      async rag => {
        for (let i = 0; i < qna.length; i += 1) {
          const pair = qna[i]
          if (pair) {
            process.stderr.write(`[${String(i + 1)}/${String(qna.length)}] ${pair.question.slice(0, 80)}...\n`)

            const queries = expand ? await expandQuery(pair.question, provider.chatModel(chatModel)) : [pair.question]
            if (queries.length > 1) process.stderr.write(`  Expanded into ${String(queries.length)} queries\n`)
            const resultSets = await Promise.all(
              queries.map(async q => rag.retrieve({ embed: embedFn, limit, mode, query: q, rrfK }))
            )
            let chunks: SearchResult[] = mergeRetrievals(resultSets)

            if (!noRerank) {
              process.stderr.write(`  Reranking ${String(chunks.length)} chunks → top ${String(rerankTop)}...\n`)
              chunks = await rerankChunks(pair.question, chunks, { model: createRerankingModel(), topN: rerankTop })
            }

            const before = chunks.length
            chunks = dedupSubstrings(chunks, { prefixLength: 100 })
            if (chunks.length < before)
              process.stderr.write(
                `  Substring dedup: ${String(before)} → ${String(chunks.length)} (removed ${String(before - chunks.length)})\n`
              )

            const retrievedDocs: EvalEntry['retrievedDocs'] = []
            for (const chunk of chunks)
              retrievedDocs.push({
                documentId: chunk.documentId,
                id: chunk.id,
                score: Math.round(chunk.score * 1_000_000) / 1_000_000,
                text: chunk.text,
                title: chunk.title
              })
            const context = buildContext(chunks)
            let generated = '[ERROR: timeout]'
            try {
              const response = await generateText({
                abortSignal: AbortSignal.timeout(240_000),
                model: provider.chatModel(chatModel),
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
      }
    )

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
