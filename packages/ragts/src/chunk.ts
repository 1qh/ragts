import { RecursiveChunker, TokenChunker } from '@chonkiejs/core'

import type { ChunkConfig } from './types'

interface ChunkOutput {
  endIndex: number
  startIndex: number
  text: string
  tokenCount: number
}

const toOutputs = (result: { endIndex: number; startIndex: number; text: string; tokenCount: number }[]) => {
    const outputs: ChunkOutput[] = []
    for (const c of result)
      outputs.push({ endIndex: c.endIndex, startIndex: c.startIndex, text: c.text, tokenCount: c.tokenCount })
    return outputs
  },
  chunkText = async (text: string, config?: ChunkConfig): Promise<ChunkOutput[]> => {
    const mode = config?.mode ?? 'recursive',
      chunkSize = config?.chunkSize ?? 512,
      chunkOverlap = config?.chunkOverlap ?? 50

    if (mode === 'token') {
      const chunker = await TokenChunker.create({ chunkOverlap, chunkSize })
      return toOutputs(await chunker.chunk(text))
    }

    const chunker = await RecursiveChunker.create({
      chunkSize,
      minCharactersPerChunk: Math.min(24, chunkOverlap)
    })
    return toOutputs(await chunker.chunk(text))
  }

export { chunkText }
