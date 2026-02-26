import { getDbUrl, parseArgs, runWithRag } from 'ragts/utils'

import { EMBEDDING_DIMENSION } from './constants'

const args = process.argv.slice(2),
  main = async () => {
    const parsed = parseArgs(args),
      yesFlag = parsed.get('--yes'),
      textCfg = parsed.get('--text-config')
    if (yesFlag !== true) throw new Error('Refusing to wipe without --yes')
    await runWithRag(
      {
        connectionString: getDbUrl(parsed),
        dimension: EMBEDDING_DIMENSION,
        textConfig: typeof textCfg === 'string' ? textCfg : 'simple'
      },
      async rag => {
        await rag.drop()
        console.log('Dropped tables')
      }
    )
  }
try {
  await main()
} catch (error) {
  console.error(error)
  process.exitCode = 1
}
