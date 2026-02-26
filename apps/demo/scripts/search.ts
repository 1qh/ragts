import type { SearchConfig, SearchMode } from 'ragts'

import { createEmbedFn } from 'ragts'
import { getDbUrl, getRequiredString, parseArgs, runWithRag } from 'ragts/utils'

import { BASE_URL, EMBEDDING_DIMENSION, EMBEDDING_MODEL } from './constants'

const args = process.argv.slice(2),
  parseMode = (value: string): SearchMode => {
    if (value === 'bm25' || value === 'hybrid' || value === 'vector') return value
    throw new Error('Invalid --mode, expected hybrid|vector|bm25')
  },
  parseLimit = (value: string) => {
    const parsed = Number.parseInt(value, 10)
    if (Number.isNaN(parsed) || parsed <= 0) throw new Error('Invalid --limit, expected positive integer')
    return parsed
  },
  main = async () => {
    const parsed = parseArgs(args),
      query = getRequiredString(parsed, '--query'),
      modeFlag = parsed.get('--mode'),
      limitFlag = parsed.get('--limit'),
      mode = typeof modeFlag === 'string' ? parseMode(modeFlag) : 'hybrid',
      limit = typeof limitFlag === 'string' ? parseLimit(limitFlag) : 10,
      embedFn = createEmbedFn({ baseURL: BASE_URL, model: EMBEDDING_MODEL }),
      textCfg = parsed.get('--text-config')
    await runWithRag(
      {
        connectionString: getDbUrl(parsed),
        dimension: EMBEDDING_DIMENSION,
        textConfig: typeof textCfg === 'string' ? textCfg : 'simple'
      },
      async rag => {
        const config: SearchConfig = {
            embed: embedFn,
            limit,
            mode,
            query
          },
          results = await rag.retrieve(config),
          rows: {
            id: number
            mode: string
            preview: string
            score: string
            title: string
          }[] = []
        for (const result of results)
          rows.push({
            id: result.id,
            mode: result.mode,
            preview: result.text.slice(0, 90).replaceAll(/\s+/gu, ' '),
            score: result.score.toFixed(6),
            title: result.title
          })
        console.table(rows)
      }
    )
  }
try {
  await main()
} catch (error) {
  console.error(error)
  process.exitCode = 1
}
