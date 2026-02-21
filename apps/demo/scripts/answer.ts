/* eslint-disable max-statements */
import { createOpenAICompatible } from '@ai-sdk/openai-compatible'
import { generateText } from 'ai'
import { existsSync, readFileSync } from 'node:fs'
import {
  buildContext,
  createEmbedFn,
  dedupSubstrings,
  getDbUrl,
  getRequiredString,
  parseArgs,
  rerankChunks,
  runWithRag
} from 'ragts'

import { BASE_URL, CHAT_MODEL, EMBEDDING_DIMENSION, EMBEDDING_MODEL } from './constants'
import { createRerankingModel, expandQuery, mergeRetrievals } from './utils'

const args = process.argv.slice(2),
  main = async () => {
    const parsed = parseArgs(args),
      query = getRequiredString(parsed, '--query'),
      systemRaw = getRequiredString(parsed, '--system'),
      systemPrompt = existsSync(systemRaw) ? readFileSync(systemRaw, 'utf8') : systemRaw,
      embedFn = createEmbedFn({ baseURL: BASE_URL, model: EMBEDDING_MODEL }),
      provider = createOpenAICompatible({ apiKey: 'none', baseURL: BASE_URL, name: 'mlx' }),
      noRerank = parsed.get('--no-rerank') === true,
      expand = parsed.get('--expand') === true,
      textCfg = parsed.get('--text-config')
    await runWithRag(
      {
        connectionString: getDbUrl(parsed),
        dimension: EMBEDDING_DIMENSION,
        textConfig: typeof textCfg === 'string' ? textCfg : 'simple'
      },
      async rag => {
        const queries = expand ? await expandQuery(query, provider.chatModel(CHAT_MODEL)) : [query],
          resultSets = await Promise.all(
            queries.map(async q => rag.retrieve({ embed: embedFn, limit: 20, mode: 'hybrid', query: q }))
          )
        let chunks = mergeRetrievals(resultSets)

        if (!noRerank) chunks = await rerankChunks(query, chunks, { model: createRerankingModel(), topN: 10 })
        chunks = dedupSubstrings(chunks, { prefixLength: 100 })

        const context = buildContext(chunks),
          response = await generateText({
            model: provider.chatModel(CHAT_MODEL),
            prompt: `Question:\n${query}\n\nContext:\n${context}`,
            system: systemPrompt
          })
        console.log(response.text)
      }
    )
  }
try {
  await main()
} catch (error) {
  console.error(error)
  process.exitCode = 1
}
