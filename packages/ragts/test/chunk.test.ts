import { describe, expect, test } from 'bun:test'
import { readFileSync } from 'node:fs'
import { join } from 'node:path'

import { chunkText } from '../src/chunk'

const EXP_DATA = join(import.meta.dir, '..', '..', '..', 'exp', 'data')

describe('Vietnamese Unicode', () => {
  test('all Vietnamese diacritics preserved exactly', () => {
    const diacritics = 'ắằẳẵặđơưêôâăẫễỗộợửữừứ ẮẰẲẴẶĐƠƯÊÔÂĂẪỄỖỘỢỬỮỪỨ và các ký tự đặc biệt khác trong tiếng Việt',
      chunks = chunkText(diacritics, { chunkSize: 512 })
    expect(chunks.length).toBe(1)
    expect(chunks[0]?.text).toBe(diacritics)
  })

  test('short Vietnamese text returns single chunk unchanged', () => {
    const text = 'Luật Tham gia lực lượng gìn giữ hòa bình của Liên hợp quốc',
      chunks = chunkText(text, { chunkSize: 512 })
    expect(chunks.length).toBe(1)
    expect(chunks[0]?.text).toBe(text)
  })

  test('long Vietnamese text splits without corrupting characters', () => {
    const sentence = 'Nghị định này quy định chi tiết và biện pháp thi hành một số điều của Luật. ',
      longText = sentence.repeat(200),
      chunks = chunkText(longText, { chunkSize: 256 })
    expect(chunks.length).toBeGreaterThan(1)
    const joined = chunks.map(c => c.text).join(' ')
    expect(joined).not.toContain('\uFFFD')
    for (const c of chunks) expect(c.text.length).toBeGreaterThan(0)
  })

  test('mixed Vietnamese and ASCII text works', () => {
    const text = 'Article 1. Điều 1: Phạm vi điều chỉnh - Scope of regulation. Test 123.',
      chunks = chunkText(text, { chunkSize: 512 })
    expect(chunks.length).toBe(1)
    expect(chunks[0]?.text).toContain('Điều 1')
    expect(chunks[0]?.text).toContain('Article 1')
  })

  test('Vietnamese punctuation marks preserved', () => {
    const text = 'Điều 1: \u201CQuy định\u201D \u2014 (thi hành); khoản 4, Điều 9\u2026 Luật này quy định chi tiết.',
      chunks = chunkText(text, { chunkSize: 512 })
    expect(chunks.length).toBe(1)
    expect(chunks[0]?.text).toContain('\u2014')
    expect(chunks[0]?.text).toContain('\u201C')
    expect(chunks[0]?.text).toContain('\u2026')
  })
})

describe('hard break unwrapping', () => {
  test('single hard breaks within sentences are joined with spaces', () => {
    const text = 'Luật Tham\ngia lực\nlượng gìn giữ hòa bình của Liên hợp quốc theo quy định',
      chunks = chunkText(text, { chunkSize: 512 })
    expect(chunks.length).toBe(1)
    expect(chunks[0]?.text).toContain('Tham gia')
    expect(chunks[0]?.text).toContain('lực lượng')
  })

  test('double newlines (paragraph breaks) are preserved', () => {
    const text = 'Paragraph one content here.\n\nParagraph two content here.',
      chunks = chunkText(text, { chunkSize: 512 })
    expect(chunks.length).toBe(1)
    expect(chunks[0]?.text).toContain('\n')
  })

  test('hard breaks before markdown headers preserved', () => {
    const text = 'Some text here.\n## Chapter II\nMore content after header.',
      chunks = chunkText(text, { chunkSize: 512 })
    expect(chunks.length).toBe(1)
    expect(chunks[0]?.text).toContain('## Chapter II')
  })

  test('hard breaks before list items preserved', () => {
    const text = 'Introduction:\n- Item one\n- Item two\n* Star item\n1. Numbered item',
      chunks = chunkText(text, { chunkSize: 512 })
    expect(chunks.length).toBe(1)
    const allText = chunks.map(c => c.text).join('')
    expect(allText).toContain('- Item one')
    expect(allText).toContain('- Item two')
    expect(allText).toContain('* Star item')
    expect(allText).toContain('1. Numbered item')
  })

  test('mix of hard breaks and structural breaks in same text', () => {
    const text =
        'First sentence that\ncontinues here.\n\nSecond paragraph that\nalso continues.\n## Header\nContent under header.',
      chunks = chunkText(text, { chunkSize: 2048 })
    expect(chunks.length).toBe(1)
    const allText = chunks.map(c => c.text).join('')
    expect(allText).toContain('sentence that continues')
    expect(allText).toContain('paragraph that also continues')
  })
})

describe('markdown structure', () => {
  test('headers are used as split points', () => {
    const sections: string[] = []
    for (let i = 1; i <= 10; i += 1) sections.push(`## Section ${i}\n\n${'Content for this section. '.repeat(30)}`)

    const text = sections.join('\n\n'),
      chunks = chunkText(text, { chunkSize: 300 })
    expect(chunks.length).toBeGreaterThan(1)
    const allText = chunks.map(c => c.text).join('\n---\n')
    expect(allText).toContain('## Section')
  })

  test('lists are not split from their content', () => {
    const text = '## Items\n\n- First item description\n- Second item description\n- Third item description',
      chunks = chunkText(text, { chunkSize: 512 })
    expect(chunks.length).toBe(1)
    const allText = chunks.map(c => c.text).join('')
    expect(allText).toContain('- First item')
    expect(allText).toContain('- Third item')
  })

  test('nested headers produce correct chunks', () => {
    const text =
        '# Title\n\nIntro paragraph.\n\n## Chapter 1\n\nChapter 1 content here.\n\n#### Article 1\n\nArticle 1 details.\n\n## Chapter 2\n\nChapter 2 content here.',
      chunks = chunkText(text, { chunkSize: 512 })
    expect(chunks.length).toBeGreaterThanOrEqual(1)
    const allText = chunks.map(c => c.text).join(' ')
    expect(allText).toContain('Title')
    expect(allText).toContain('Chapter 1')
    expect(allText).toContain('Article 1')
    expect(allText).toContain('Chapter 2')
  })

  test('code blocks stay together', () => {
    const text = '## Example\n\n```typescript\nconst x = 1\nconst y = 2\nconst z = x + y\n```\n\nAfter code.',
      chunks = chunkText(text, { chunkSize: 512 }),
      allText = chunks.map(c => c.text).join('\n')
    expect(allText).toContain('const x = 1')
    expect(allText).toContain('const z = x + y')
  })
})

describe('chunk size limits', () => {
  test('all chunks <= chunkSize for normal text', () => {
    const text = 'The quick brown fox jumps over the lazy dog. '.repeat(100),
      chunks = chunkText(text, { chunkSize: 200 })
    for (const c of chunks) expect(c.text.length).toBeLessThanOrEqual(200)
  })

  test('all chunks <= chunkSize for very long text', () => {
    const text = 'The quick brown fox jumps over the lazy dog. '.repeat(5000),
      chunks = chunkText(text, { chunkSize: 200 })
    expect(chunks.length).toBeGreaterThan(50)
    for (const c of chunks) expect(c.text.length).toBeLessThanOrEqual(200)
  })

  test('default chunkSize is 2048 when not specified', () => {
    const text = 'A sentence that is pretty long for testing default size. '.repeat(200),
      chunks = chunkText(text)
    expect(chunks.length).toBeGreaterThan(1)
    for (const c of chunks) expect(c.text.length).toBeLessThanOrEqual(2048)
  })

  test('chunkSize=2048 works on large text', () => {
    const text = 'Vietnamese legal text: Điều khoản quy định chi tiết. '.repeat(2000),
      chunks = chunkText(text, { chunkSize: 2048 })
    expect(chunks.length).toBeGreaterThan(1)
    for (const c of chunks) expect(c.text.length).toBeLessThanOrEqual(2048)
  })

  test('single word longer than chunkSize does not crash', () => {
    const longWord = 'a'.repeat(1000),
      chunks = chunkText(longWord, { chunkSize: 100 })
    expect(chunks.length).toBe(0)
  })

  test('empty text returns empty array', () => {
    const chunks = chunkText('', { chunkSize: 512 })
    expect(chunks.length).toBe(0)
  })

  test('whitespace-only text returns empty array', () => {
    const chunks = chunkText('   \n\n  \t  ', { chunkSize: 512 })
    expect(chunks.length).toBe(0)
  })
})

describe('split quality', () => {
  test('prefers splitting at sentence boundaries', () => {
    const text =
        'First long sentence here with enough content to pass filter. Second long sentence here with enough content as well. Third long sentence to fill more content. Fourth long sentence with plenty of detail.',
      chunks = chunkText(text, { chunkSize: 80 })
    expect(chunks.length).toBeGreaterThan(1)
    for (const c of chunks) expect(c.text.length).toBeLessThanOrEqual(80)
  })

  test('prefers splitting at paragraph boundaries', () => {
    const paragraphs: string[] = []
    for (let i = 1; i <= 5; i += 1) paragraphs.push(`Paragraph ${i} has some content here that is meaningful.`)

    const text = paragraphs.join('\n\n'),
      chunks = chunkText(text, { chunkSize: 120 })
    expect(chunks.length).toBeGreaterThan(1)
  })

  test('clause-level splits at comma and semicolon work', () => {
    const clauses: string[] = []
    for (let i = 1; i <= 20; i += 1) clauses.push(`clause number ${i} with details`)

    const text = clauses.join(', '),
      chunks = chunkText(text, { chunkSize: 150 })
    expect(chunks.length).toBeGreaterThan(1)
    for (const c of chunks) expect(c.text.length).toBeLessThanOrEqual(150)
  })

  test('no trailing or leading whitespace on chunks', () => {
    const text = 'First sentence. Second sentence. Third sentence. Fourth sentence. '.repeat(50),
      chunks = chunkText(text, { chunkSize: 100 })
    for (const c of chunks) expect(c.text).toBe(c.text.trim())
  })
})

describe('chunk metadata', () => {
  test('startIndex and endIndex are valid', () => {
    const text = 'The quick brown fox jumps over the lazy dog. '.repeat(100),
      chunks = chunkText(text, { chunkSize: 200 })
    for (const c of chunks) {
      expect(c.startIndex).toBeGreaterThanOrEqual(0)
      expect(c.endIndex).toBeGreaterThan(c.startIndex)
    }
  })

  test('startIndex values are monotonically increasing', () => {
    const text = 'Sentence one. Sentence two. Sentence three. Sentence four. '.repeat(100),
      chunks = chunkText(text, { chunkSize: 200 })
    expect(chunks.length).toBeGreaterThan(1)
    const starts = chunks.map(c => c.startIndex),
      sorted = [...starts].toSorted((a, b) => a - b)
    expect(starts).toEqual(sorted)
    const unique = new Set(starts)
    expect(unique.size).toBe(starts.length)
  })

  test('tokenCount equals text length', () => {
    const text = 'Some text for testing metadata. '.repeat(50),
      chunks = chunkText(text, { chunkSize: 100 })
    for (const c of chunks) expect(c.tokenCount).toBe(c.text.length)
  })
})

describe('real legal document', () => {
  test('small legal doc chunks within size limit', () => {
    const docPath = join(EXP_DATA, '001_01-2026-ND-CP_01-2026-ND-CP.md'),
      content = readFileSync(docPath, 'utf8'),
      chunks = chunkText(content, { chunkSize: 2048 })
    expect(chunks.length).toBeGreaterThan(1)
    for (const c of chunks) {
      expect(c.text.length).toBeLessThanOrEqual(2048)
      expect(c.text.length).toBeGreaterThan(0)
    }
  })

  test('no chunk has corrupted Vietnamese (NFC normalization)', () => {
    const docPath = join(EXP_DATA, '001_01-2026-ND-CP_01-2026-ND-CP.md'),
      content = readFileSync(docPath, 'utf8'),
      chunks = chunkText(content, { chunkSize: 2048 })
    for (const c of chunks) expect(c.text).toBe(c.text.normalize('NFC'))
  })

  test('big doc (1.6MB) all chunks within limit and completes fast', () => {
    const docPath = join(EXP_DATA, '465_26-2023-ND-CP_122-2016-ND-CP.md'),
      content = readFileSync(docPath, 'utf8')
    expect(content.length).toBeGreaterThan(1_000_000)
    const start = performance.now(),
      chunks = chunkText(content, { chunkSize: 2048 }),
      elapsed = performance.now() - start
    expect(elapsed).toBeLessThan(500)
    expect(chunks.length).toBeGreaterThan(10)
    for (const c of chunks) {
      expect(c.text.length).toBeLessThanOrEqual(2048)
      expect(c.text.length).toBeGreaterThan(0)
    }
  })
})

describe('chunkText post-chunk filters', () => {
  test('drops chunks shorter than 50 chars', () => {
    const text = 'Tiny.\n\nThis is a longer paragraph with more than fifty characters of real content here.',
      chunks = chunkText(text, { chunkSize: 512 })
    for (const c of chunks) expect(c.text.length).toBeGreaterThanOrEqual(50)
  })

  test('drops OCR garbage chunks (200+ consecutive non-whitespace)', () => {
    const garbage = 'a'.repeat(250),
      real = 'Điều 1. Luật này quy định chi tiết và biện pháp thi hành các điều khoản.',
      text = `${garbage}\n\n${real}`,
      chunks = chunkText(text, { chunkSize: 512 })
    for (const c of chunks) expect(c.text).not.toContain(garbage)
  })

  test('keeps chunks with long words but spaces between them', () => {
    const word = 'a'.repeat(100),
      text = `${word} ${word} and more content to fill fifty chars easily enough.`,
      chunks = chunkText(text, { chunkSize: 512 })
    expect(chunks.length).toBeGreaterThanOrEqual(1)
  })

  test('empty text still returns empty array', () => {
    expect(chunkText('', { chunkSize: 512 })).toEqual([])
  })

  test('text that is all tiny chunks returns empty', () => {
    const text = 'Hi.\n\nBye.\n\nOk.',
      chunks = chunkText(text, { chunkSize: 512 })
    expect(chunks.length).toBe(0)
  })
})
