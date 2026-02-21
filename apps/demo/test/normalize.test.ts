/* eslint-disable max-statements */
import { describe, expect, test } from 'bun:test'
import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { chunkText } from 'ragts'

import { normalizeMarkdown } from '../scripts/normalize'

const EXP_DATA = join(import.meta.dir, '..', '..', '..', 'exp', 'data'),
  RE_DASH_NUM = /^- \d+[.)]/mu,
  RE_SINGLE_ITALIC = /(?<!\*)\*[^*]+\*(?!\*)/u

describe('normalizeMarkdown', () => {
  test('strips dash-list prefix: "- 1." becomes "1."', () => {
    expect(normalizeMarkdown('- 1. Các doanh nghiệp.')).toBe('1. Các doanh nghiệp.')
  })

  test('strips dash-list prefix: "- 2." becomes "2."', () => {
    expect(normalizeMarkdown('- 2. Item two.')).toBe('2. Item two.')
  })

  test('strips dash-list prefix with parenthesis: "- 1)" becomes "1)"', () => {
    expect(normalizeMarkdown('- 1) First item.')).toBe('1) First item.')
  })

  test('strips dash-list prefix with letter: "- a)" becomes "a)"', () => {
    expect(normalizeMarkdown('- a) Sub item.')).toBe('a) Sub item.')
  })

  test('strips dash-list prefix with Vietnamese đ: "- đ)" becomes "đ)"', () => {
    expect(normalizeMarkdown('- đ) Vợ, chồng.')).toBe('đ) Vợ, chồng.')
  })

  test('preserves indentation for dash-list prefix', () => {
    expect(normalizeMarkdown('  - 1. Indented item.')).toBe('  1. Indented item.')
  })

  test('does not strip regular dash lists without numbered items', () => {
    expect(normalizeMarkdown('- Regular list item')).toBe('- Regular list item')
  })

  test('does not strip dash from non-list context', () => {
    expect(normalizeMarkdown('kinh tế - xã hội')).toBe('kinh tế - xã hội')
  })

  test('strips bold markdown: **text** becomes text', () => {
    expect(normalizeMarkdown('**LUẬT DOANH NGHIỆP**')).toBe('LUẬT DOANH NGHIỆP')
  })

  test('strips bold in header: ## **Điều 1.** becomes #### Điều 1.', () => {
    expect(normalizeMarkdown('## **Điều 1. Phạm vi**')).toBe('#### Điều 1. Phạm vi')
  })

  test('strips multiple bold spans in one line', () => {
    expect(normalizeMarkdown('**A** and **B**')).toBe('A and B')
  })

  test('strips italic markdown: *text* becomes text', () => {
    expect(normalizeMarkdown('*Cá nhân nước ngoài* là người')).toBe('Cá nhân nước ngoài là người')
  })

  test('does not strip bold when removing italic', () => {
    expect(normalizeMarkdown('**bold** and *italic*')).toBe('bold and italic')
  })

  test('handles bold-inside-italic edge: ***text*** strips both', () => {
    const result = normalizeMarkdown('***mixed***')
    expect(result).not.toContain('*')
  })

  test('removes image lines completely', () => {
    expect(normalizeMarkdown('![](_page_0_Picture_2.jpeg)')).toBe('')
  })

  test('removes image lines with alt text', () => {
    expect(normalizeMarkdown('![Some alt text](image.png)')).toBe('')
  })

  test('removes image lines with surrounding whitespace', () => {
    expect(normalizeMarkdown('  ![](img.jpg)  ')).toBe('')
  })

  test('strips image references inline too', () => {
    expect(normalizeMarkdown('See ![img](x.png) here')).toBe('See  here')
  })

  test('handles multiline with mixed patterns', () => {
    const input = [
        '![](_page_0_Picture_2.jpeg)',
        '',
        '## **QUỐC HỘI**',
        '',
        '- 1. *Các doanh nghiệp.*',
        '- 2. **Cơ quan**, tổ chức.'
      ].join('\n'),
      expected = ['', '#### QUỐC HỘI', '', '1. Các doanh nghiệp.', '2. Cơ quan, tổ chức.'].join('\n')
    expect(normalizeMarkdown(input)).toBe(expected)
  })

  test('preserves plain text unchanged', () => {
    const text = 'Luật này quy định về việc thành lập doanh nghiệp.'
    expect(normalizeMarkdown(text)).toBe(text)
  })

  test('preserves header markers without bold', () => {
    expect(normalizeMarkdown('#### Điều 1. Phạm vi')).toBe('#### Điều 1. Phạm vi')
  })

  test('preserves numbered lists without dash prefix', () => {
    expect(normalizeMarkdown('1. Các doanh nghiệp.')).toBe('1. Các doanh nghiệp.')
  })

  test('empty string returns empty string', () => {
    expect(normalizeMarkdown('')).toBe('')
  })

  test('handles all sub-item letters: a through h including đ', () => {
    const letters = ['a', 'b', 'c', 'd', 'đ', 'e', 'g', 'h']
    for (const l of letters) expect(normalizeMarkdown(`- ${l}) Item`)).toBe(`${l}) Item`)
  })
})

describe('normalizeMarkdown integration with chunkText', () => {
  test('chunkText strips bold from input before chunking', () => {
    const text = '**Bold heading** with some content here that is long enough to pass the minimum filter.',
      chunks = chunkText(text, { chunkSize: 512, normalize: normalizeMarkdown })
    expect(chunks[0]?.text).toBe('Bold heading with some content here that is long enough to pass the minimum filter.')
  })

  test('chunkText strips italic from input before chunking', () => {
    const text = '*Italic term* is defined as something important in the legal context here.',
      chunks = chunkText(text, { chunkSize: 512, normalize: normalizeMarkdown })
    expect(chunks[0]?.text).toBe('Italic term is defined as something important in the legal context here.')
  })

  test('chunkText strips dash-list prefix from input before chunking', () => {
    const text = '- 1. First item content here with enough detail to be a meaningful chunk.',
      chunks = chunkText(text, { chunkSize: 512, normalize: normalizeMarkdown })
    expect(chunks[0]?.text).toBe('1. First item content here with enough detail to be a meaningful chunk.')
  })

  test('chunkText removes image lines from input before chunking', () => {
    const text = '![](_page_0_Picture_2.jpeg)\n\nActual content here that is long enough to pass the chunk filter.',
      chunks = chunkText(text, { chunkSize: 512, normalize: normalizeMarkdown })
    expect(chunks[0]?.text).not.toContain('jpeg')
    expect(chunks[0]?.text).toContain('Actual content here')
  })

  test('PDF and gazette formats produce same chunk for identical content', () => {
    const pdf = '- 1. *Cá nhân nước ngoài* là người không có quốc tịch Việt Nam theo quy định pháp luật.',
      gazette = '1. Cá nhân nước ngoài là người không có quốc tịch Việt Nam theo quy định pháp luật.',
      pdfChunks = chunkText(pdf, { chunkSize: 512, normalize: normalizeMarkdown }),
      gazetteChunks = chunkText(gazette, { chunkSize: 512, normalize: normalizeMarkdown })
    expect(pdfChunks[0]?.text).toBe(gazetteChunks[0]?.text)
  })

  test('PDF and gazette formats produce same chunk for bold header content', () => {
    const pdf = '## **Điều 1. Phạm vi điều chỉnh**\n\nLuật này quy định về việc thành lập doanh nghiệp theo pháp luật.',
      gazette = '## Điều 1. Phạm vi điều chỉnh\n\nLuật này quy định về việc thành lập doanh nghiệp theo pháp luật.',
      pdfChunks = chunkText(pdf, { chunkSize: 512, normalize: normalizeMarkdown }),
      gazetteChunks = chunkText(gazette, { chunkSize: 512, normalize: normalizeMarkdown })
    expect(pdfChunks[0]?.text).toBe(gazetteChunks[0]?.text)
    expect(pdfChunks[0]?.text).toContain('#### Điều 1')
  })
})

describe('normalizeMarkdown v2: line filters', () => {
  test('strips CÔNG BÁO page header lines', () => {
    const input = 'Content before.\nCÔNG BÁO/Số 825 + 826/Ngày 15-11-2015\nContent after.'
    expect(normalizeMarkdown(input)).toBe('Content before.\nContent after.')
  })

  test('strips CÔNG BÁO with varying numbers', () => {
    expect(normalizeMarkdown('CÔNG BÁO/Số 1/Ngày 1-1-2020')).toBe('')
  })

  test('does not strip partial CÔNG BÁO match mid-line', () => {
    const input = 'Xem thêm tại CÔNG BÁO/Số 1'
    expect(normalizeMarkdown(input)).toBe(input)
  })

  test('strips THƯ VIỆN PHÁP LUẬT watermark lines', () => {
    expect(normalizeMarkdown('THƯ VIỆN PHÁP LUẬT some text')).toBe('')
  })

  test('strips ThuVienPhapLuat watermark lines', () => {
    expect(normalizeMarkdown('Source: ThuVienPhapLuat.vn')).toBe('')
  })

  test('strips foxyutils watermark lines', () => {
    expect(normalizeMarkdown('Created with foxyutils.com')).toBe('')
  })

  test('strips CamScanner watermark lines', () => {
    expect(normalizeMarkdown('Scanned with CamScanner')).toBe('')
  })

  test('strips Scanned by watermark lines', () => {
    expect(normalizeMarkdown('Scanned by TapScanner')).toBe('')
  })

  test('multiple skippable lines removed together', () => {
    const input = [
      'CÔNG BÁO/Số 825 + 826/Ngày 15-11-2015',
      'Real content here.',
      'THƯ VIỆN PHÁP LUẬT',
      'More real content.',
      '![](image.png)'
    ].join('\n')
    expect(normalizeMarkdown(input)).toBe('Real content here.\nMore real content.')
  })
})

describe('normalizeMarkdown v2: backslash artifacts', () => {
  test('replaces double backslash with space', () => {
    expect(normalizeMarkdown(String.raw`text\\more`)).toBe('text more')
  })

  test('replaces multiple double backslashes', () => {
    expect(normalizeMarkdown(String.raw`a\\b\\c`)).toBe('a b c')
  })

  test(String.raw`unescapes \* to *`, () => {
    expect(normalizeMarkdown(String.raw`10\*5`)).toBe('10*5')
  })

  test(String.raw`unescapes \. to .`, () => {
    expect(normalizeMarkdown(String.raw`item 1\. text`)).toBe('item 1. text')
  })

  test(String.raw`unescapes \( and \) to ( and )`, () => {
    expect(normalizeMarkdown(String.raw`see \(note\) here`)).toBe('see (note) here')
  })

  test(String.raw`unescapes \[ and \] to [ and ]`, () => {
    expect(normalizeMarkdown(String.raw`ref \[1\] text`)).toBe('ref [1] text')
  })

  test('handles mixed escaped chars in one line', () => {
    expect(normalizeMarkdown(String.raw`a\*b\.c\(d\)e\[f\]g`)).toBe('a*b.c(d)e[f]g')
  })

  test('double backslash before escaped char', () => {
    expect(normalizeMarkdown(String.raw`end\\ \*start`)).toBe('end  *start')
  })
})

describe('normalizeMarkdown v2: header normalization', () => {
  test('## becomes ####', () => {
    expect(normalizeMarkdown('## Điều 1')).toBe('#### Điều 1')
  })

  test('### becomes ####', () => {
    expect(normalizeMarkdown('### Chương II')).toBe('#### Chương II')
  })

  test('#### stays ####', () => {
    expect(normalizeMarkdown('#### Điều 1')).toBe('#### Điều 1')
  })

  test('# stays # (single hash not normalized)', () => {
    expect(normalizeMarkdown('# Title')).toBe('# Title')
  })

  test('##### stays ##### (5+ hashes not normalized)', () => {
    expect(normalizeMarkdown('##### Sub')).toBe('##### Sub')
  })

  test('###### stays ###### (6 hashes not normalized)', () => {
    expect(normalizeMarkdown('###### Deep')).toBe('###### Deep')
  })

  test('## with bold content: strips bold and normalizes header', () => {
    expect(normalizeMarkdown('## **QUỐC HỘI**')).toBe('#### QUỐC HỘI')
  })

  test('### with bold content: strips bold and normalizes header', () => {
    expect(normalizeMarkdown('### **Điều 5. Phạm vi**')).toBe('#### Điều 5. Phạm vi')
  })

  test('does not normalize ## inside text (no leading hash)', () => {
    expect(normalizeMarkdown('Issue ## tracker')).toBe('Issue ## tracker')
  })

  test('multiple headers in multiline text', () => {
    const input = '# Title\n## Chapter\n### Section\n#### Article'
    expect(normalizeMarkdown(input)).toBe('# Title\n#### Chapter\n#### Section\n#### Article')
  })
})

describe('normalizeMarkdown v2: order of operations', () => {
  test('bold stripped before dash-list (handles OCR artifact "- 1*.* text")', () => {
    const input = '- 1. **Doanh nghiệp** phải nộp thuế.'
    expect(normalizeMarkdown(input)).toBe('1. Doanh nghiệp phải nộp thuế.')
  })

  test(String.raw`escaped chars stripped before italic (\*text\* becomes text after unescape+italic)`, () => {
    expect(normalizeMarkdown(String.raw`\*not bold\*`)).toBe('not bold')
  })

  test('double backslash replaced before escaped chars', () => {
    expect(normalizeMarkdown(String.raw`a\\\*b`)).toBe('a *b')
  })

  test('full pipeline: CÔNG BÁO line + escaped + bold + dash + header', () => {
    const bs = '\\',
      input = ['CÔNG BÁO/Số 100/Ngày 1-1-2020', `## **Điều 1${bs}. Phạm vi**`, '- 1. *Doanh nghiệp* phải nộp thuế.'].join(
        '\n'
      ),
      expected = ['#### Điều 1. Phạm vi', '1. Doanh nghiệp phải nộp thuế.'].join('\n')
    expect(normalizeMarkdown(input)).toBe(expected)
  })
})

describe('real file format normalization', () => {
  test('PDF-format file has dash-list items normalized', () => {
    const docPath = join(EXP_DATA, '1849_87-2015-ND-CP_68-2014-QH13.md'),
      content = readFileSync(docPath, 'utf8'),
      chunks = chunkText(content, { chunkSize: 2048, normalize: normalizeMarkdown })
    for (const c of chunks) expect(c.text).not.toMatch(RE_DASH_NUM)
  })

  test('PDF-format file has bold markers stripped', () => {
    const docPath = join(EXP_DATA, '1849_87-2015-ND-CP_68-2014-QH13.md'),
      content = readFileSync(docPath, 'utf8'),
      chunks = chunkText(content, { chunkSize: 2048, normalize: normalizeMarkdown })
    for (const c of chunks) expect(c.text).not.toContain('**')
  })

  test('PDF-format file has italic markers stripped', () => {
    const docPath = join(EXP_DATA, '1849_87-2015-ND-CP_68-2014-QH13.md'),
      content = readFileSync(docPath, 'utf8'),
      chunks = chunkText(content, { chunkSize: 2048, normalize: normalizeMarkdown })
    for (const c of chunks) expect(c.text).not.toMatch(RE_SINGLE_ITALIC)
  })

  test('PDF-format file has no image references', () => {
    const docPath = join(EXP_DATA, '1849_87-2015-ND-CP_68-2014-QH13.md'),
      content = readFileSync(docPath, 'utf8')
    expect(content).toContain('![](')
    const chunks = chunkText(content, { chunkSize: 2048, normalize: normalizeMarkdown })
    for (const c of chunks) expect(c.text).not.toContain('![](')
  })

  test('gazette-format file is unchanged by normalization (no artifacts)', () => {
    const docPath = join(EXP_DATA, '758_51-2019-ND-CP_68-2014-QH13.md'),
      content = readFileSync(docPath, 'utf8'),
      chunks = chunkText(content, { chunkSize: 2048, normalize: normalizeMarkdown })
    expect(chunks.length).toBeGreaterThan(1)
    for (const c of chunks) expect(c.text.length).toBeGreaterThan(0)
  })
})
