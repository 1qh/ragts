import { parseArgs, runWithRag } from './utils'

const args = process.argv.slice(2),
  main = async () => {
    const parsed = parseArgs(args),
      yesFlag = parsed.get('--yes')
    if (yesFlag !== true) throw new Error('Refusing to wipe without --yes')
    await runWithRag(parsed, async rag => {
      await rag.drop()
      console.log('Dropped tables')
    })
  }
try {
  await main()
} catch (error) {
  console.error(error)
  process.exitCode = 1
}
