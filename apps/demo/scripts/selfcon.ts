/* eslint-disable no-await-in-loop,max-statements */
/** biome-ignore-all lint/performance/noAwaitInLoops: x */
import type { SearchMode, SearchResult } from 'ragts'

import { generateText } from 'ai'
import { writeFileSync } from 'node:fs'
import { buildContext, createProvider, dedupSubstrings, rerankChunks } from 'ragts'
import { getDbUrl, loadJsonArray, parseArgs, runWithRag } from 'ragts/utils'

import type { EvalEntry } from './utils'

import { BASE_URL, CHAT_MODEL, EMBEDDING_DIMENSION, EMBEDDING_MODEL, RERANK_MODEL } from './constants'
import { formatRetrievedDocs, getNumFlag, getStringFlag, loadSystemPrompt } from './utils'
import { extractVerdict, verdictsMatch } from './verdict'

interface SelfConEntry {
  allGenerated: string[]
  allVerdicts: string[]
  generated: string
  gold?: string
  majorityVerdict: string | undefined
  question: string
  retrievedDocs: EvalEntry['retrievedDocs']
}

const tallyVerdicts = (verdicts: (string | undefined)[]): Map<string, number> => {
    const counts = new Map<string, number>()
    for (const v of verdicts)
      if (v !== undefined) {
        let found = false
        for (const [existing, count] of counts)
          if (verdictsMatch(existing, v)) {
            counts.set(existing, count + 1)
            found = true
            break
          }
        if (!found) counts.set(v, 1)
      }

    return counts
  },
  pickMajority = (counts: Map<string, number>): { count: number; verdict: string | undefined } => {
    let best: string | undefined,
      maxCount = 0
    for (const [v, count] of counts)
      if (count > maxCount) {
        maxCount = count
        best = v
      }
    return { count: maxCount, verdict: best }
  },
  pickBestGenerated = (
    allGenerated: string[],
    allVerdicts: (string | undefined)[],
    majority: string | undefined
  ): string => {
    const fallback = allGenerated[0] ?? ''
    if (majority === undefined) return fallback
    for (let g = 0; g < allGenerated.length; g += 1) {
      const v = allVerdicts[g]
      if (v !== undefined && verdictsMatch(v, majority)) return allGenerated[g] ?? fallback
    }
    return fallback
  },
  args = process.argv.slice(2),
  main = async () => {
    const qnaFile = args.find(a => !a.startsWith('--'))
    if (!qnaFile)
      throw new Error(
        'Usage: bun selfcon.ts <qna.json> --system <prompt> --runs N [--limit N] [--rerank-top N] [--temperature F] [--output path]'
      )

    const parsed = parseArgs(args),
      systemPrompt = loadSystemPrompt(parsed),
      outputPath = getStringFlag(parsed, '--output') ?? 'out-selfcon.json',
      limit = getNumFlag(parsed, '--limit') ?? 50,
      modeFlag = getStringFlag(parsed, '--mode'),
      mode: SearchMode = modeFlag === 'vector' || modeFlag === 'bm25' ? modeFlag : 'hybrid',
      chatModel = getStringFlag(parsed, '--model') ?? CHAT_MODEL,
      rerankTop = getNumFlag(parsed, '--rerank-top') ?? 30,
      runs = getNumFlag(parsed, '--runs') ?? 5,
      temperature = getNumFlag(parsed, '--temperature') ?? 0.3,
      rrfK = getNumFlag(parsed, '--rrf-k'),
      qna = loadJsonArray<{ answer?: string; question: string }>(qnaFile),
      provider = createProvider({ baseURL: BASE_URL }),
      embedFn = provider.embedFn(EMBEDDING_MODEL),
      chat = provider.chatModel(chatModel),
      reranker = provider.rerankingModel(RERANK_MODEL),
      textCfg = getStringFlag(parsed, '--text-config'),
      results: SelfConEntry[] = []

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

            let chunks: SearchResult[] = await rag.retrieve({ embed: embedFn, limit, mode, query: pair.question, rrfK })

            process.stderr.write(`  Reranking ${String(chunks.length)} chunks → top ${String(rerankTop)}...\n`)
            chunks = await rerankChunks(pair.question, chunks, { model: reranker, topN: rerankTop })

            const before = chunks.length
            chunks = dedupSubstrings(chunks, { prefixLength: 100 })
            if (chunks.length < before)
              process.stderr.write(
                `  Substring dedup: ${String(before)} → ${String(chunks.length)} (removed ${String(before - chunks.length)})\n`
              )

            const context = buildContext(chunks),
              retrievedDocs = formatRetrievedDocs(chunks),
              allGenerated: string[] = [],
              allVerdicts: (string | undefined)[] = []
            for (let r = 0; r < runs; r += 1) {
              process.stderr.write(`  Run ${String(r + 1)}/${String(runs)}...`)
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
                process.stderr.write(` FAILED: ${genError instanceof Error ? genError.message : 'unknown'}\n`)
              }
              allGenerated.push(generated)
              const firstLine = generated.split('\n')[0]?.replaceAll('*', '').trim() ?? '',
                v = extractVerdict(firstLine)
              allVerdicts.push(v)
              process.stderr.write(` → ${v ?? '???'}\n`)
            }

            const counts = tallyVerdicts(allVerdicts),
              { count: maxCount, verdict: majorityVerdict } = pickMajority(counts),
              bestGenerated = pickBestGenerated(allGenerated, allVerdicts, majorityVerdict)

            process.stderr.write(`  Majority: ${majorityVerdict ?? '???'} (${String(maxCount)}/${String(runs)})\n`)
            results.push({
              allGenerated,
              allVerdicts: allVerdicts.map(v => v ?? 'unknown'),
              generated: bestGenerated,
              gold: pair.answer,
              majorityVerdict,
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
