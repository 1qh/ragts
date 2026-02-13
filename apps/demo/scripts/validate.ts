import { validateBackup } from 'ragts'

import { getRequiredString, parseArgs } from './utils'

const args = process.argv.slice(2),
  main = () => {
    const parsed = parseArgs(args),
      filePath = getRequiredString(parsed, '--file'),
      report = validateBackup(filePath)
    console.log(
      JSON.stringify(
        {
          dimensions: [...report.dimensions],
          duplicateHashes: report.duplicateHashes,
          errors: report.errors,
          totalChunks: report.totalChunks,
          totalDocuments: report.totalDocuments,
          valid: report.valid
        },
        null,
        2
      )
    )
  }
try {
  main()
} catch (error) {
  console.error(error)
  process.exitCode = 1
}
