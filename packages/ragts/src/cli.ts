#!/usr/bin/env bun
import { resolve } from 'node:path'

import { generateCompose } from './compose'

const [command, outputArg] = process.argv.slice(2),
  run = () => {
    if (command === 'gen-compose') {
      const outputDir = outputArg ?? '.',
        resolved = resolve(outputDir)
      generateCompose(resolved)
      process.stdout.write(`Generated docker-compose.yml and init.sql in ${resolved}\n`)
      return
    }

    process.stderr.write(
      'Usage: ragts <command>\n\nCommands:\n  gen-compose [dir]  Generate docker-compose.yml + init.sql (default: current dir)\n'
    )
    process.exit(1)
  }

run()
