/* eslint-disable no-await-in-loop,max-statements */
/** biome-ignore-all lint/performance/noAwaitInLoops: x */
import type { SearchMode } from 'ragts'

import { createOpenAICompatible } from '@ai-sdk/openai-compatible'
import { generateText } from 'ai'
import { readFileSync, writeFileSync } from 'node:fs'
import { buildContext } from 'ragts'

import { BASE_URL, CHAT_MODEL } from './constants'
import { createEmbedFn, parseArgs, rerankChunks, runWithRag } from './utils'

interface EvalEntry {
  generated: string
  gold?: string
  question: string
  retrievedDocs: { documentId: number; id: number; score: number; text: string; title: string }[]
}

interface QnaPair {
  answer?: string
  question: string
}

const args = process.argv.slice(2),
  loadQna = (filePath: string): QnaPair[] => {
    const raw = readFileSync(filePath, 'utf8'),
      parsed = JSON.parse(raw) as unknown
    if (!Array.isArray(parsed)) throw new Error(`Expected JSON array in ${filePath}`)
    return parsed as QnaPair[]
  },
  // eslint-disable-next-line max-statements
  main = async () => {
    const qnaFile = args.find(a => !a.startsWith('--'))
    if (!qnaFile)
      throw new Error(
        'Usage: bun eval.ts <qna.json> --system <prompt> [--limit N] [--mode hybrid|vector|bm25] [--model name] [--output path] [--rerank-top N]'
      )

    const parsed = parseArgs(args),
      systemFlag = parsed.get('--system'),
      systemPrompt = typeof systemFlag === 'string' ? systemFlag : '',
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
      qna = loadQna(qnaFile),
      embedFn = createEmbedFn(),
      provider = createOpenAICompatible({ apiKey: 'none', baseURL: BASE_URL, name: 'mlx' }),
      results: EvalEntry[] = []

    await runWithRag(parsed, async rag => {
      for (let i = 0; i < qna.length; i += 1) {
        const pair = qna[i]
        if (pair) {
          process.stderr.write(`[${String(i + 1)}/${String(qna.length)}] ${pair.question.slice(0, 80)}...\n`)
          let chunks = await rag.retrieve({ embed: embedFn, limit, mode, query: pair.question })

          if (!noRerank) {
            process.stderr.write(`  Reranking ${String(chunks.length)} chunks → top ${String(rerankTop)}...\n`)
            chunks = await rerankChunks(pair.question, chunks, rerankTop)
          }

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
              system: systemPrompt
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
    })

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
