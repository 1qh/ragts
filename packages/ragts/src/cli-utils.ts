import { readdirSync, readFileSync } from 'node:fs'
import { basename, extname, join } from 'node:path'

import type { Doc, RagtsConfig } from './types'

import { Rag } from './index'

const getDbUrl = (flags: Map<string, boolean | string>): string => {
    const value = flags.get('--db')
    return typeof value === 'string' && value.length > 0 ? value : 'postgresql://postgres:postgres@localhost:5432/postgres'
  },
  getRequiredString = (flags: Map<string, boolean | string>, name: string): string => {
    const value = flags.get(name)
    if (typeof value !== 'string' || value.length === 0) throw new Error(`Missing required flag: ${name}`)
    return value
  },
  loadDocsFromFolder = (folderPath: string): Doc[] => {
    const names = readdirSync(folderPath).toSorted((a, b) => a.localeCompare(b)),
      docs: Doc[] = []
    for (const fileName of names)
      if (extname(fileName) === '.txt' || extname(fileName) === '.md') {
        const fullPath = join(folderPath, fileName),
          content = readFileSync(fullPath, 'utf8'),
          title = basename(fileName, extname(fileName))
        docs.push({ content, title })
      }
    return docs
  },
  loadJsonArray = <T>(filePath: string): T[] => {
    const raw = readFileSync(filePath, 'utf8'),
      parsed = JSON.parse(raw) as unknown
    if (!Array.isArray(parsed)) throw new Error(`Expected JSON array in ${filePath}`)
    return parsed as T[]
  },
  parseArgs = (values: string[]): Map<string, boolean | string> => {
    const flags = new Map<string, boolean | string>()
    for (let i = 0; i < values.length; i += 1) {
      const token = values[i]
      if (token?.startsWith('--')) {
        const nextValue = values[i + 1]
        if (!nextValue || nextValue.startsWith('--')) flags.set(token, true)
        else {
          flags.set(token, nextValue)
          i += 1
        }
      }
    }
    return flags
  },
  runWithRag = async (config: RagtsConfig, fn: (rag: Rag) => Promise<void>): Promise<void> => {
    const rag = new Rag(config)
    try {
      await fn(rag)
    } finally {
      await rag.close()
    }
  }

export { getDbUrl, getRequiredString, loadDocsFromFolder, loadJsonArray, parseArgs, runWithRag }
