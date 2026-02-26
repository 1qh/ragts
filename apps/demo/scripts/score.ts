/* eslint-disable max-statements */
import { readFileSync } from 'node:fs'

import { extractVerdict, RE_TRAILING_DOT, verdictsMatch } from './verdict'

interface EvalEntry {
  generated: string
  gold?: string
  question: string
  retrievedDocs: { documentId: number; id: number; score: number; text: string; title: string }[]
}

type ScoreCategory = 'cant_score' | 'correct' | 'no_gold' | 'timeout' | 'wrong'

const RE_NUMBER = /\d[\d.,]*/gu,
  RE_CHOICE_PREFIX = /^(?<letter>[a-d])\.\s*/iu,
  extractNumbers = (text: string): Set<string> => {
    const nums = new Set<string>()
    for (const m of text.matchAll(RE_NUMBER)) nums.add(m[0].replaceAll('.', '').replaceAll(',', ''))
    return nums
  },
  tryNumericMatch = (gold: string, generated: string): boolean => {
    const goldNums = extractNumbers(gold)
    if (goldNums.size === 0) return false
    const genNums = extractNumbers(generated)
    for (const n of goldNums) if (!genNums.has(n)) return false
    return true
  },
  tryListMatch = (gold: string, generated: string): boolean => {
    const items = gold.split(',')
    if (items.length < 2) return false
    const genLower = generated.toLowerCase()
    for (const item of items) {
      const trimmed = item.trim().toLowerCase()
      if (trimmed.length > 0 && !genLower.includes(trimmed)) return false
    }
    return true
  },
  tryChoiceMatch = (gold: string, generated: string): boolean => {
    const m = RE_CHOICE_PREFIX.exec(gold)
    if (!m) return false
    const letter = (m.groups?.letter ?? '').toLowerCase(),
      genLower = generated.toLowerCase()
    return genLower.includes(`${letter}.`) || genLower.includes(`${letter})`)
  },
  scoreEntry = (entry: EvalEntry, idx: number): { category: ScoreCategory; detail: string } => {
    const gold = entry.gold?.trim() ?? ''
    if (!gold) return { category: 'no_gold', detail: `Q${String(idx)}` }
    if (entry.generated.startsWith('[ERROR')) return { category: 'timeout', detail: `Q${String(idx)}: TIMEOUT` }

    const firstLine = entry.generated.split('\n')[0]?.replaceAll('*', '').trim().replace(RE_TRAILING_DOT, '') ?? '',
      genV = extractVerdict(firstLine),
      goldFirst = gold.split('.')[0] ?? gold,
      goldV = extractVerdict(goldFirst) ?? extractVerdict(gold)

    if (genV !== undefined && goldV !== undefined) {
      if (verdictsMatch(genV, goldV)) return { category: 'correct', detail: `Q${String(idx)}` }
      const qShort = entry.question.slice(0, 80)
      return { category: 'wrong', detail: `Q${String(idx)}: gold=${goldV} gen=${genV} | ${qShort}` }
    }

    const genFull = entry.generated.toLowerCase().replaceAll(/\s+/gu, ' '),
      goldNorm = gold.toLowerCase().replaceAll(/\s+/gu, ' ').trim()
    if (goldNorm.length >= 3 && genFull.includes(goldNorm))
      return { category: 'correct', detail: `Q${String(idx)} (text-match)` }

    if (tryNumericMatch(gold, entry.generated)) return { category: 'correct', detail: `Q${String(idx)} (numeric-match)` }

    if (tryListMatch(gold, entry.generated)) return { category: 'correct', detail: `Q${String(idx)} (list-match)` }

    if (tryChoiceMatch(gold, entry.generated)) return { category: 'correct', detail: `Q${String(idx)} (choice-match)` }

    const qShort = entry.question.slice(0, 60)
    return {
      category: 'cant_score',
      detail: `Q${String(idx)}: gold_v=${goldV ?? 'none'} gen_v=${genV ?? 'none'} | gold=${gold.slice(0, 60)} | ${qShort}`
    }
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
