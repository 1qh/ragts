import { getRequiredString, parseArgs, runWithRag } from './utils'

const args = process.argv.slice(2),
  main = async () => {
    const parsed = parseArgs(args),
      command = args.find(a => !a.startsWith('--'))
    if (command === 'export') {
      const outputPath = getRequiredString(parsed, '--output')
      await runWithRag(parsed, async rag => {
        const result = await rag.exportBackup(outputPath)
        console.log(JSON.stringify(result, null, 2))
      })
    } else if (command === 'import') {
      const filePath = getRequiredString(parsed, '--file')
      await runWithRag(parsed, async rag => {
        const result = await rag.importBackup(filePath)
        console.log(JSON.stringify(result, null, 2))
      })
    } else throw new Error('Usage: bun backup.ts export --output <path> | bun backup.ts import --file <path>')
  }
try {
  await main()
} catch (error) {
  console.error(error)
  process.exitCode = 1
}
