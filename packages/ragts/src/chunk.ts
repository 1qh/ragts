import type { ChunkConfig } from './types'

interface ChunkOutput {
  endIndex: number
  startIndex: number
  text: string
  tokenCount: number
}

const RE_OCR_GARBAGE = /\S{200,}/u,
  RE_HEADER = /^#{1,6}\s/mu,
  RE_LIST = /^\s*[-*>|]|\d+\.\s/u,
  SPLIT_LEVELS: RegExp[] = [/(?=\n#{1,6}\s)/u, /\n\n+/u, /(?<=[.!?])\s+/u, /(?<=[;,])\s+/u, /\n/u, /\s+/u],
  isStructuralBreak = (line: string, next: string): boolean =>
    line.trim() === '' || next.trim() === '' || RE_HEADER.test(next) || RE_LIST.test(next),
  unwrapHardBreaks = (text: string): string => {
    const lines = text.split('\n'),
      parts: string[] = []
    for (let i = 0; i < lines.length; i += 1) {
      const line = lines[i] ?? '',
        next = lines[i + 1]
      parts.push(line)
      if (next === undefined) break
      parts.push(isStructuralBreak(line, next) ? '\n' : ' ')
    }
    return parts.join('')
  },
  splitAtLevel = (text: string, maxSize: number, level: number): string[] => {
    if (text.length <= maxSize) return [text]
    const re = SPLIT_LEVELS[level]
    if (!re) return [text]
    const pieces = text.split(re)
    if (pieces.length <= 1) return splitAtLevel(text, maxSize, level + 1)
    const result: string[] = []
    for (const p of pieces)
      if (p && p.length <= maxSize) result.push(p)
      else if (p) result.push(...splitAtLevel(p, maxSize, level + 1))

    return result
  },
  mergeSep = (cur: string, next: string): string => (cur.endsWith('\n') || next.startsWith('#') ? '\n' : ' '),
  mergePieces = (pieces: string[], maxSize: number): string[] => {
    const merged: string[] = []
    let cur = ''
    for (const p of pieces) {
      const t = p.trim()
      if (t && !cur) cur = t
      else if (t && `${cur}${mergeSep(cur, t)}${t}`.length <= maxSize) cur = `${cur}${mergeSep(cur, t)}${t}`
      else if (t) {
        merged.push(cur)
        cur = t
      }
    }
    if (cur) merged.push(cur)
    return merged
  },
  buildOutputs = (chunks: string[], source: string): ChunkOutput[] => {
    const outputs: ChunkOutput[] = []
    let offset = 0
    for (const chunk of chunks) {
      const needle = chunk.slice(0, Math.min(80, chunk.length)),
        idx = source.indexOf(needle, Math.max(0, offset - 10)),
        start = idx === -1 ? offset : idx
      outputs.push({ endIndex: start + chunk.length, startIndex: start, text: chunk, tokenCount: chunk.length })
      offset = start + chunk.length
    }
    return outputs
  },
  addOverlap = (chunks: string[], overlapSize: number): string[] => {
    if (overlapSize <= 0 || chunks.length <= 1) return chunks
    const result: string[] = [chunks[0] ?? '']
    for (let i = 1; i < chunks.length; i += 1) {
      const prev = chunks[i - 1] ?? '',
        cur = chunks[i] ?? '',
        tail = prev.slice(-overlapSize),
        sep = tail.endsWith('\n') || cur.startsWith('#') ? '\n' : ' '
      result.push(`${tail}${sep}${cur}`)
    }
    return result
  },
  chunkText = (text: string, config?: ChunkConfig): ChunkOutput[] => {
    const maxSize = config?.chunkSize ?? 2048,
      overlapSize = config?.overlap ?? 0,
      preprocessed = config?.normalize ? config.normalize(text) : text,
      normalized = unwrapHardBreaks(preprocessed),
      pieces = splitAtLevel(normalized, maxSize, 0),
      merged = mergePieces(pieces, maxSize),
      withOverlap = overlapSize > 0 ? addOverlap(merged, overlapSize) : merged,
      filtered: string[] = []
    for (const chunk of withOverlap) if (chunk.length >= 50 && !RE_OCR_GARBAGE.test(chunk)) filtered.push(chunk)

    return buildOutputs(filtered, normalized)
  }

export { chunkText }
