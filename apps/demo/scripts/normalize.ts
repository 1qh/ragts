const RE_IMG_LINE = /^!\\?\[.*?\\?\]\(.*?\)\s*$/u,
  RE_IMG_INLINE = /!\\?\[.*?\\?\]\(.*?\)/gu,
  RE_CONG_BAO = /^(?:\d+\s+)?CÔNG BÁO\/Số\s/u,
  RE_TVPL = /THƯ VIỆN PHÁP LUẬ|ThuVienPhapLuat/u,
  RE_WATERMARK = /foxyutils\.com|CamScanner|Scanned by/u,
  RE_DOUBLE_BACKSLASH = /\\\\/gu,
  RE_ESCAPED_SPECIAL = /\\(?<ch>[*._()[\]])/gu,
  RE_HEADER_NORMALIZE = /^#{2,3}(?=\s)/u,
  RE_DASH_LIST = /^(?<indent>\s*)- (?<item>\d+[.)]|[a-zđ][.)])/u,
  RE_BOLD = /\*\*(?<bold>[^*]+)\*\*/gu,
  RE_ITALIC = /(?<!\*)\*(?<italic>[^*]+)\*(?!\*)/gu,
  isSkippableLine = (trimmed: string): boolean =>
    RE_IMG_LINE.test(trimmed) || RE_CONG_BAO.test(trimmed) || RE_TVPL.test(trimmed) || RE_WATERMARK.test(trimmed),
  normalizeLine = (line: string): string => {
    let normalized = line
      .replace(RE_DOUBLE_BACKSLASH, ' ')
      .replace(RE_ESCAPED_SPECIAL, '$<ch>')
      .replace(RE_IMG_INLINE, '')
      .replace(RE_BOLD, '$<bold>')
      .replace(RE_ITALIC, '$<italic>')
    const dashMatch = RE_DASH_LIST.exec(normalized)
    if (dashMatch) {
      const indent = dashMatch.groups?.indent ?? ''
      normalized = `${indent}${normalized.slice(indent.length + 2)}`
    }
    if (RE_HEADER_NORMALIZE.test(normalized)) normalized = normalized.replace(RE_HEADER_NORMALIZE, '####')

    return normalized
  },
  normalizeMarkdown = (text: string): string => {
    const lines = text.split('\n'),
      out: string[] = []
    for (const line of lines) if (!isSkippableLine(line.trim())) out.push(normalizeLine(line))

    return out.join('\n')
  }

export { normalizeMarkdown }
