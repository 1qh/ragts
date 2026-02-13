/* eslint-disable max-statements,complexity */
import { readFileSync } from 'node:fs'

interface EvalEntry {
  generated: string
  gold?: string
  question: string
  retrievedDocs: { documentId: number; id: number; score: number; text: string; title: string }[]
}

type ScoreCategory = 'cant_score' | 'correct' | 'no_gold' | 'timeout' | 'wrong'

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
      stripped = norm.replace(RE_PREFIX, '')
    const direct = findVerdict(stripped.slice(0, 120))
    if (direct) return direct

    const sepMatch = RE_SEPARATOR.exec(stripped)
    if (sepMatch) {
      const afterSep = stripped.slice(sepMatch.index + sepMatch[0].length).trim()
      const sepVerdict = findVerdict(afterSep.slice(0, 60))
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
  },
  scoreEntry = (entry: EvalEntry, idx: number): { category: ScoreCategory; detail: string } => {
    const gold = entry.gold?.trim() ?? ''
    if (!gold) return { category: 'no_gold', detail: `Q${String(idx)}` }
    if (entry.generated.startsWith('[ERROR')) return { category: 'timeout', detail: `Q${String(idx)}: TIMEOUT` }

    const firstLine = entry.generated.split('\n')[0]?.replaceAll('*', '').trim().replace(RE_TRAILING_DOT, '') ?? '',
      genV = extractVerdict(firstLine),
      goldFirst = gold.split('.')[0] ?? gold,
      goldV = extractVerdict(goldFirst) ?? extractVerdict(gold)

    if (genV === undefined || goldV === undefined) {
      const qShort = entry.question.slice(0, 60)
      return {
        category: 'cant_score',
        detail: `Q${String(idx)}: gold_v=${goldV ?? 'none'} gen_v=${genV ?? 'none'} | gold=${gold.slice(0, 60)} | ${qShort}`
      }
    }

    if (verdictsMatch(genV, goldV)) return { category: 'correct', detail: `Q${String(idx)}` }

    const qShort = entry.question.slice(0, 80)
    return { category: 'wrong', detail: `Q${String(idx)}: gold=${goldV} gen=${genV} | ${qShort}` }
  },
  printResults = (results: { category: ScoreCategory; detail: string }[]) => {
    let correct = 0,
      wrong = 0,
      timeoutCount = 0,
      noGold = 0,
      cantScore = 0
    const wrongList: string[] = [],
      cantScoreList: string[] = [],
      correctList: string[] = []

    for (const r of results)
      if (r.category === 'correct') {
        correct += 1
        correctList.push(r.detail)
      } else if (r.category === 'wrong') {
        wrong += 1
        wrongList.push(r.detail)
      } else if (r.category === 'timeout') {
        timeoutCount += 1
        wrongList.push(r.detail)
      } else if (r.category === 'cant_score') {
        cantScore += 1
        cantScoreList.push(r.detail)
      } else noGold += 1

    const scoreable = correct + wrong
    process.stdout.write(`Scoreable: ${String(scoreable)}/${String(scoreable + timeoutCount + cantScore)}\n`)
    process.stdout.write(
      `Correct: ${String(correct)}/${String(scoreable)} (${((100 * correct) / scoreable).toFixed(1)}%)\n`
    )
    process.stdout.write(`Wrong: ${String(wrong)}/${String(scoreable)}\n`)
    process.stdout.write(`Timeout: ${String(timeoutCount)}\n`)
    process.stdout.write(`Cant score (special format): ${String(cantScore)}\n`)
    process.stdout.write(`No gold: ${String(noGold)}\n\n`)
    process.stdout.write(`CORRECT: ${correctList.join(', ')}\n\n`)
    if (wrongList.length > 0) {
      process.stdout.write('WRONG:\n')
      for (const w of wrongList) process.stdout.write(`  ${w}\n`)
    }
    if (cantScoreList.length > 0) {
      process.stdout.write('\nCANT SCORE:\n')
      for (const c of cantScoreList) process.stdout.write(`  ${c}\n`)
    }
  },
  main = () => {
    const filePath = process.argv.at(2)
    if (!filePath) throw new Error('Usage: bun score.ts <eval-output.json>')

    const data = JSON.parse(readFileSync(filePath, 'utf8')) as EvalEntry[],
      results: { category: ScoreCategory; detail: string }[] = []
    for (let i = 0; i < data.length; i += 1) {
      const entry = data[i]
      if (entry) results.push(scoreEntry(entry, i + 1))
    }
    printResults(results)
  }

main()
