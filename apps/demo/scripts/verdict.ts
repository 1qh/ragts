const RE_HEADING = /^#+\s*/u,
  RE_STARS = /\*+/gu,
  RE_TRAILING_PUNCT = /[.!]+$/u,
  RE_TRAILING_DOT = /\.$/u,
  RE_QUOTED = /"[^"]*"/gu,
  RE_PREFIX = /^(?:đáp án|nhận định|trả lời|kết luận|câu trả lời|câu hỏi)\s*:?\s*/iu,
  RE_WHITESPACE = /\s+/u,
  RE_SEPARATOR = /[—→\-:]\s*/u,
  POSITIVE = new Set(['Có', 'Đúng', 'Được', 'Phải']),
  NEGATIVE = new Set(['Không', 'Sai']),
  VERDICTS_ORDERED = ['Không', 'Đúng', 'Sai', 'Có', 'Được', 'Phải'] as const,
  normalizeVietnamese = (text: string): string =>
    text.replace(RE_HEADING, '').replaceAll(RE_STARS, '').replaceAll(RE_QUOTED, '').trim().replace(RE_TRAILING_PUNCT, ''),
  findVerdict = (text: string): string | undefined => {
    const upper = text.toUpperCase()
    if (upper.startsWith('KHÔNG') || upper.includes('KHÔNG THỂ') || upper.includes('KHÔNG ĐƯỢC')) return 'Không'
    for (const v of VERDICTS_ORDERED) {
      const vUp = v.toUpperCase()
      if (upper.startsWith(vUp) || upper.startsWith(`ÔNG ${vUp}`) || upper.startsWith(`BÀ ${vUp}`)) return v
      const idx = upper.indexOf(vUp)
      if (idx !== -1 && idx < 30) return v
    }
  },
  extractVerdict = (text: string): string | undefined => {
    const norm = normalizeVietnamese(text),
      stripped = norm.replace(RE_PREFIX, ''),
      direct = findVerdict(stripped.slice(0, 120))
    if (direct) return direct

    const sepMatch = RE_SEPARATOR.exec(stripped)
    if (sepMatch) {
      const afterSep = stripped.slice(sepMatch.index + sepMatch[0].length).trim(),
        sepVerdict = findVerdict(afterSep.slice(0, 60))
      if (sepVerdict) return sepVerdict
    }

    const words = stripped.split(RE_WHITESPACE)
    for (let i = 0; i < Math.min(words.length, 20); i += 1) {
      const w = (words[i] ?? '').replace(RE_TRAILING_PUNCT, '')
      for (const v of VERDICTS_ORDERED) if (w.toUpperCase() === v.toUpperCase()) return v
    }
  },
  verdictsMatch = (a: string, b: string): boolean => {
    if (a === b) return true
    if (POSITIVE.has(a) && POSITIVE.has(b)) return true
    if (NEGATIVE.has(a) && NEGATIVE.has(b)) return true
    return false
  }

export { extractVerdict, RE_TRAILING_DOT, verdictsMatch }
