import { createOpenAICompatible } from '@ai-sdk/openai-compatible'
import { generateText } from 'ai'
import { buildContext } from 'ragts'

import { BASE_URL, CHAT_MODEL } from './constants'
import { createEmbedFn, getRequiredString, parseArgs, rerankChunks, runWithRag } from './utils'

const args = process.argv.slice(2),
  main = async () => {
    const parsed = parseArgs(args),
      query = getRequiredString(parsed, '--query'),
      systemPrompt = getRequiredString(parsed, '--system'),
      embedFn = createEmbedFn(),
      provider = createOpenAICompatible({ apiKey: 'none', baseURL: BASE_URL, name: 'mlx' }),
      noRerank = parsed.get('--no-rerank') === true
    await runWithRag(parsed, async rag => {
      let chunks = await rag.retrieve({
        embed: embedFn,
        limit: 20,
        mode: 'hybrid',
        query
      })

      if (!noRerank) chunks = await rerankChunks(query, chunks, 10)

      const context = buildContext(chunks),
        response = await generateText({
          model: provider.chatModel(CHAT_MODEL),
          prompt: `Question:\n${query}\n\nContext:\n${context}`,
          system: systemPrompt
        })
      console.log(response.text)
    })
  }
try {
  await main()
} catch (error) {
  console.error(error)
  process.exitCode = 1
}
