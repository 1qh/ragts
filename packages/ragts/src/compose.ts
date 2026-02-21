import { copyFileSync, mkdirSync } from 'node:fs'
import { dirname, join } from 'node:path'

const dockerDir = join(dirname(import.meta.dirname), 'docker'),
  generateCompose = (outputDir: string) => {
    mkdirSync(outputDir, { recursive: true })
    copyFileSync(join(dockerDir, 'docker-compose.yml'), join(outputDir, 'docker-compose.yml'))
    copyFileSync(join(dockerDir, 'init.sql'), join(outputDir, 'init.sql'))
  }

export { generateCompose }
