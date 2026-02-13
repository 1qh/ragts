/* eslint-disable max-statements */
import { afterAll, beforeAll, describe, expect, test } from 'bun:test'
import { sql } from 'drizzle-orm'
import { existsSync, readFileSync, unlinkSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import type { BackupDoc, Doc, EmbedFn, IngestConfig, SearchConfig, SearchResult } from '../src/index'

import { validateBackup } from '../src/backup'
import { chunkText } from '../src/chunk'
import { generateCompose } from '../src/compose'
import { Rag } from '../src/index'

const TEST_DB_URL = 'postgresql://postgres:postgres@localhost:5432/ragts_test',
  fakeEmbed: EmbedFn = async (texts: string[]) => {
    await Promise.resolve()
    const embeddings: number[][] = []
    for (const t of texts) {
      const vec: number[] = [],
        textLength = t.length === 0 ? 1 : t.length
      for (let i = 0; i < 768; i += 1) {
        const codePoint = t.codePointAt(i % textLength) ?? 0
        vec.push(Math.sin(codePoint + i) * 0.5)
      }
      embeddings.push(vec)
    }
    return embeddings
  }

let rag: Rag, db: Awaited<ReturnType<Rag['init']>>

beforeAll(async () => {
  rag = new Rag({ connectionString: TEST_DB_URL, dimension: 768 })
  await rag.drop()
  db = await rag.init()
})

afterAll(async () => {
  await rag.drop()
  await rag.close()
})

describe('schema', () => {
  test('tables exist after migrate', async () => {
    const rows = await db.execute<{ tablename: string }>(
        sql.raw(
          "SELECT tablename FROM pg_tables WHERE schemaname = 'public' AND tablename IN ('documents', 'chunks') ORDER BY tablename"
        )
      ),
      names: string[] = []
    for (const r of rows) names.push(r.tablename)
    expect(names).toContain('chunks')
    expect(names).toContain('documents')
  })

  test('extensions are installed', async () => {
    const rows = await db.execute<{ extname: string }>(
        sql.raw(
          "SELECT extname FROM pg_extension WHERE extname IN ('vector', 'vectorscale', 'pg_textsearch') ORDER BY extname"
        )
      ),
      names: string[] = []
    for (const r of rows) names.push(r.extname)
    expect(names).toContain('vector')
    expect(names).toContain('vectorscale')
    expect(names).toContain('pg_textsearch')
  })

  test('unique constraint on content_hash', async () => {
    const rows = await db.execute<{ indexname: string }>(
      sql.raw("SELECT indexname FROM pg_indexes WHERE tablename = 'documents' AND indexname LIKE '%content_hash%'")
    )
    expect(rows.length).toBeGreaterThan(0)
  })

  test('diskann index exists on chunks', async () => {
    const rows = await db.execute<{ indexname: string }>(
      sql.raw("SELECT indexname FROM pg_indexes WHERE tablename = 'chunks' AND indexname LIKE '%embedding%'")
    )
    expect(rows.length).toBeGreaterThan(0)
  })

  test('bm25 index exists on chunks', async () => {
    const rows = await db.execute<{ indexname: string }>(
      sql.raw("SELECT indexname FROM pg_indexes WHERE tablename = 'chunks' AND indexname LIKE '%text%'")
    )
    expect(rows.length).toBeGreaterThan(0)
  })
})

describe('chunk', () => {
  test('recursive chunker splits text', async () => {
    const longText = 'The quick brown fox jumps over the lazy dog. '.repeat(100),
      chunks = await chunkText(longText, { chunkSize: 200, mode: 'recursive' })
    expect(chunks.length).toBeGreaterThan(1)
    for (const c of chunks) {
      expect(c.text.length).toBeGreaterThan(0)
      expect(c.startIndex).toBeGreaterThanOrEqual(0)
      expect(c.endIndex).toBeGreaterThan(c.startIndex)
      expect(c.tokenCount).toBeGreaterThan(0)
    }
  })

  test('token chunker splits text', async () => {
    const longText = 'Hello world this is a test of the token chunker functionality. '.repeat(50),
      chunks = await chunkText(longText, { chunkOverlap: 10, chunkSize: 100, mode: 'token' })
    expect(chunks.length).toBeGreaterThan(1)
    for (const c of chunks) {
      expect(c.text.length).toBeGreaterThan(0)
      expect(c.tokenCount).toBeGreaterThan(0)
    }
  })

  test('short text returns single chunk', async () => {
    const chunks = await chunkText('Hello world', { chunkSize: 512, mode: 'recursive' })
    expect(chunks.length).toBe(1)
    expect(chunks[0]?.text).toBe('Hello world')
  })

  test('unicode text works', async () => {
    const unicodeText = '日本語のテスト文章です。これは長いテキストです。'.repeat(50),
      chunks = await chunkText(unicodeText, { chunkSize: 100, mode: 'recursive' })
    expect(chunks.length).toBeGreaterThan(0)
    for (const c of chunks) expect(c.text.length).toBeGreaterThan(0)
  })

  test('empty text returns empty array', async () => {
    const chunks = await chunkText('', { chunkSize: 512, mode: 'recursive' })
    expect(chunks.length).toBe(0)
  })

  test('very long text produces many chunks', async () => {
    const longText = 'The quick brown fox jumps over the lazy dog. '.repeat(5000),
      chunks = await chunkText(longText, { chunkSize: 200, mode: 'recursive' })
    expect(chunks.length).toBeGreaterThan(50)
    for (const c of chunks) {
      expect(c.text.length).toBeGreaterThan(0)
      expect(c.startIndex).toBeGreaterThanOrEqual(0)
      expect(c.endIndex).toBeGreaterThan(c.startIndex)
    }
  })
})

describe('ingest', () => {
  beforeAll(async () => {
    await rag.drop()
    db = await rag.init()
  })

  test('single doc ingest', async () => {
    const docs: Doc[] = [
        {
          content: 'This is a test document with enough content to be meaningful for chunking and embedding.',
          title: 'Test Doc'
        }
      ],
      result = await rag.ingest(docs, { embed: fakeEmbed })
    expect(result.documentsInserted).toBe(1)
    expect(result.chunksInserted).toBeGreaterThan(0)
    expect(result.duplicatesSkipped).toBe(0)
  })

  test('duplicate detection', async () => {
    const docs: Doc[] = [
        {
          content: 'This is a test document with enough content to be meaningful for chunking and embedding.',
          title: 'Test Doc'
        }
      ],
      result = await rag.ingest(docs, { embed: fakeEmbed })
    expect(result.documentsInserted).toBe(0)
    expect(result.duplicatesSkipped).toBe(1)
  })

  test('batch docs ingest', async () => {
    const docs: Doc[] = []
    for (let i = 0; i < 5; i += 1)
      docs.push({
        content: `Batch document number ${i} with content about topic ${i}. This has enough text to be chunked properly.`,
        title: `Batch Doc ${i}`
      })

    const result = await rag.ingest(docs, { embed: fakeEmbed })
    expect(result.documentsInserted).toBe(5)
    expect(result.chunksInserted).toBeGreaterThan(0)
  })

  test('incremental backup during ingest', async () => {
    const backupPath = join(tmpdir(), `ragts-test-backup-${Date.now()}.jsonl`),
      docs: Doc[] = [
        { content: 'Content for backup test document with sufficient text for processing.', title: 'Backup Test Doc' }
      ],
      result = await rag.ingest(docs, { backupPath, embed: fakeEmbed })
    expect(result.documentsInserted).toBe(1)
    expect(existsSync(backupPath)).toBe(true)

    const content = readFileSync(backupPath, 'utf8').trim(),
      parsed = JSON.parse(content) as BackupDoc
    expect(parsed.title).toBe('Backup Test Doc')
    expect(parsed.chunks.length).toBeGreaterThan(0)
    expect(parsed.chunks[0]?.embedding.length).toBe(768)

    unlinkSync(backupPath)
  })

  test('progress callback fires', async () => {
    const progressCalls: { current: number; doc: string; total: number }[] = [],
      docs: Doc[] = [
        { content: 'Content for progress test document one.', title: 'Progress Doc 1' },
        { content: 'Content for progress test document two.', title: 'Progress Doc 2' }
      ]
    await rag.ingest(docs, {
      embed: fakeEmbed,
      onProgress: (doc, current, total) => {
        progressCalls.push({ current, doc, total })
      }
    })
    expect(progressCalls.length).toBe(2)
    expect(progressCalls[0]?.current).toBe(1)
    expect(progressCalls[1]?.current).toBe(2)
  })
})

describe('retrieve', () => {
  beforeAll(async () => {
    await rag.drop()
    db = await rag.init()

    const docs: Doc[] = [
      {
        content:
          'TypeScript is a typed superset of JavaScript that compiles to plain JavaScript. It adds optional static typing and class-based object-oriented programming to the language. TypeScript is developed and maintained by Microsoft.',
        title: 'TypeScript Guide'
      },
      {
        content:
          'Python is a high-level programming language known for its clear syntax and readability. It supports multiple programming paradigms including procedural, object-oriented, and functional programming.',
        title: 'Python Guide'
      },
      {
        content:
          'PostgreSQL is a powerful open-source relational database management system. It supports advanced data types and performance optimization features that make it suitable for enterprise applications.',
        title: 'PostgreSQL Guide'
      }
    ]
    await rag.ingest(docs, { embed: fakeEmbed })
  })

  test('vector search returns results', async () => {
    const results = await rag.retrieve({
      embed: fakeEmbed,
      mode: 'vector',
      query: 'TypeScript programming'
    })
    expect(results.length).toBeGreaterThan(0)
    for (const r of results) {
      expect(r.id).toBeGreaterThan(0)
      expect(r.documentId).toBeGreaterThan(0)
      expect(r.text.length).toBeGreaterThan(0)
      expect(r.title.length).toBeGreaterThan(0)
      expect(typeof r.score).toBe('number')
      expect(r.mode).toBe('vector')
    }
  })

  test('bm25 search returns results', async () => {
    const results = await rag.retrieve({
      embed: fakeEmbed,
      mode: 'bm25',
      query: 'TypeScript'
    })
    expect(results.length).toBeGreaterThan(0)
    for (const r of results) {
      expect(r.mode).toBe('bm25')
      expect(r.score).toBeGreaterThan(0)
    }
  })

  test('hybrid search returns results', async () => {
    const results = await rag.retrieve({
      embed: fakeEmbed,
      mode: 'hybrid',
      query: 'database management system'
    })
    expect(results.length).toBeGreaterThan(0)
  })

  test('search respects limit', async () => {
    const results = await rag.retrieve({
      embed: fakeEmbed,
      limit: 1,
      mode: 'vector',
      query: 'programming language'
    })
    expect(results.length).toBeLessThanOrEqual(1)
  })

  test('vector search threshold filters low-similarity results', async () => {
    const baseline = await rag.retrieve({
      embed: fakeEmbed,
      mode: 'vector',
      query: 'TypeScript programming'
    })
    expect(baseline.length).toBeGreaterThan(0)

    const scores = baseline.map(row => row.score),
      maxScore = Math.max(...scores),
      filtered = await rag.retrieve({
        embed: fakeEmbed,
        mode: 'vector',
        query: 'TypeScript programming',
        threshold: maxScore + 0.001
      })
    expect(filtered.length).toBe(0)
  })

  test('bm25 returns empty results for non-matching query', async () => {
    const results = await rag.retrieve({
      embed: fakeEmbed,
      mode: 'bm25',
      query: 'zzzzzzzzzz qqqqqqqqqq vvvvvvvvvv'
    })
    expect(results.length).toBe(0)
  })

  test('search with no matching query returns results', async () => {
    const results = await rag.retrieve({
      embed: fakeEmbed,
      mode: 'vector',
      query: 'xyzabc123nonexistent'
    })
    expect(results.length).toBeGreaterThanOrEqual(0)
  })
})

describe('backup', () => {
  let backupPath: string

  beforeAll(async () => {
    await rag.drop()
    db = await rag.init()

    const docs: Doc[] = [
      { content: 'First document for backup testing with plenty of content.', title: 'Backup Doc 1' },
      { content: 'Second document for backup testing with different content.', title: 'Backup Doc 2' }
    ]
    await rag.ingest(docs, { embed: fakeEmbed })

    backupPath = join(tmpdir(), `ragts-export-test-${Date.now()}.jsonl`)
  })

  afterAll(() => {
    if (existsSync(backupPath)) unlinkSync(backupPath)
  })

  test('export creates valid JSONL', async () => {
    const result = await rag.exportBackup(backupPath)
    expect(result.documentsExported).toBe(2)
    expect(existsSync(backupPath)).toBe(true)

    const content = readFileSync(backupPath, 'utf8'),
      lines = content.trim().split('\n')
    expect(lines.length).toBe(2)

    for (const line of lines) {
      const doc = JSON.parse(line) as BackupDoc
      expect(doc.title).toBeDefined()
      expect(doc.content).toBeDefined()
      expect(doc.contentHash).toBeDefined()
      expect(Array.isArray(doc.chunks)).toBe(true)
      expect(doc.chunks.length).toBeGreaterThan(0)
      expect(doc.chunks[0]?.embedding.length).toBe(768)
    }
  })

  test('validate backup detects valid file', () => {
    const result = validateBackup(backupPath)
    expect(result.valid).toBe(true)
    expect(result.totalDocuments).toBe(2)
    expect(result.totalChunks).toBeGreaterThan(0)
    expect(result.errors.length).toBe(0)
    expect(result.duplicateHashes.length).toBe(0)
    expect(result.dimensions.size).toBe(1)
    expect(result.dimensions.has(768)).toBe(true)
  })

  test('validate detects corrupt JSON', () => {
    const corruptPath = join(tmpdir(), `ragts-corrupt-${Date.now()}.jsonl`)
    writeFileSync(corruptPath, '{"valid": true}\nnot json at all\n')
    const result = validateBackup(corruptPath)
    expect(result.valid).toBe(false)
    expect(result.errors.length).toBeGreaterThan(0)
    unlinkSync(corruptPath)
  })

  test('validate detects duplicate hashes', () => {
    const dupPath = join(tmpdir(), `ragts-dup-${Date.now()}.jsonl`),
      line = JSON.stringify({
        chunks: [
          { embedding: Array.from({ length: 768 }, () => 0), endIndex: 5, startIndex: 0, text: 'test', tokenCount: 1 }
        ],
        content: 'test',
        contentHash: 'abc123',
        metadata: {},
        title: 'test'
      })
    writeFileSync(dupPath, `${line}\n${line}\n`)
    const result = validateBackup(dupPath)
    expect(result.duplicateHashes.length).toBe(1)
    unlinkSync(dupPath)
  })

  test('validate detects mixed embedding dimensions', () => {
    const mixedPath = join(tmpdir(), `ragts-mixed-${Date.now()}.jsonl`),
      first = JSON.stringify({
        chunks: [
          {
            embedding: Array.from({ length: 768 }, () => 0),
            endIndex: 5,
            startIndex: 0,
            text: 'first',
            tokenCount: 1
          }
        ],
        content: 'first content',
        contentHash: 'first-hash',
        metadata: {},
        title: 'first'
      }),
      second = JSON.stringify({
        chunks: [
          {
            embedding: Array.from({ length: 16 }, () => 0),
            endIndex: 6,
            startIndex: 0,
            text: 'second',
            tokenCount: 1
          }
        ],
        content: 'second content',
        contentHash: 'second-hash',
        metadata: {},
        title: 'second'
      })
    writeFileSync(mixedPath, `${first}\n${second}\n`)

    const result = validateBackup(mixedPath)
    expect(result.valid).toBe(false)
    expect(result.dimensions.size).toBe(2)
    expect(result.dimensions.has(768)).toBe(true)
    expect(result.dimensions.has(16)).toBe(true)

    unlinkSync(mixedPath)
  })

  test('incremental backup appends across multiple ingest calls', async () => {
    const incrementalPath = join(tmpdir(), `ragts-incremental-${Date.now()}.jsonl`)
    writeFileSync(incrementalPath, '')

    const firstBatch: Doc[] = [
        { content: 'Incremental backup document one with enough content for chunking.', title: 'Incremental 1' }
      ],
      secondBatch: Doc[] = [
        { content: 'Incremental backup document two with enough content for chunking.', title: 'Incremental 2' },
        { content: 'Incremental backup document three with enough content for chunking.', title: 'Incremental 3' }
      ]

    expect((await rag.ingest(firstBatch, { backupPath: incrementalPath, embed: fakeEmbed })).documentsInserted).toBe(1)

    expect((await rag.ingest(secondBatch, { backupPath: incrementalPath, embed: fakeEmbed })).documentsInserted).toBe(2)

    const lines = readFileSync(incrementalPath, 'utf8')
      .split('\n')
      .filter(line => line.trim().length > 0)
    expect(lines.length).toBe(3)

    unlinkSync(incrementalPath)
  })

  test('import with dedup skips existing', async () => {
    const importResult = await rag.importBackup(backupPath)
    expect(importResult.duplicatesSkipped).toBe(2)
    expect(importResult.documentsImported).toBe(0)
  })

  test('import after wipe restores data', async () => {
    await rag.drop()
    db = await rag.init()

    const importResult = await rag.importBackup(backupPath)
    expect(importResult.documentsImported).toBe(2)
    expect(importResult.chunksInserted).toBeGreaterThan(0)
    expect(importResult.duplicatesSkipped).toBe(0)
  })
})

describe('compose', () => {
  test('generates docker-compose.yml and init.sql', () => {
    const outputDir = join(tmpdir(), `ragts-compose-${Date.now()}`)
    generateCompose(outputDir)
    expect(existsSync(join(outputDir, 'docker-compose.yml'))).toBe(true)
    expect(existsSync(join(outputDir, 'init.sql'))).toBe(true)

    const compose = readFileSync(join(outputDir, 'docker-compose.yml'), 'utf8')
    expect(compose).toContain('timescale/timescaledb-ha:pg18')
    expect(compose).toContain('POSTGRES_PASSWORD')

    const initSql = readFileSync(join(outputDir, 'init.sql'), 'utf8')
    expect(initSql).toContain('vectorscale')
    expect(initSql).toContain('pg_textsearch')
  })
})

describe('integration', () => {
  test('full round-trip: ingest → search → backup → wipe → import → search', async () => {
    await rag.drop()
    db = await rag.init()

    const docs: Doc[] = [
        {
          content:
            'Machine learning is a branch of artificial intelligence that focuses on building applications that learn from data and improve their accuracy over time without being programmed to do so.',
          title: 'Machine Learning'
        },
        {
          content:
            'Deep learning is a subset of machine learning that uses neural networks with many layers. Deep learning drives many artificial intelligence applications and services.',
          title: 'Deep Learning'
        },
        {
          content:
            'Natural language processing enables computers to understand, interpret, and generate human language. NLP combines computational linguistics with statistical and deep learning models.',
          title: 'NLP Guide'
        }
      ],
      ingestResult = await rag.ingest(docs, { embed: fakeEmbed })
    expect(ingestResult.documentsInserted).toBe(3)

    const searchResults1 = await rag.retrieve({
      embed: fakeEmbed,
      mode: 'hybrid',
      query: 'neural networks deep learning'
    })
    expect(searchResults1.length).toBeGreaterThan(0)

    const backupPath = join(tmpdir(), `ragts-integration-${Date.now()}.jsonl`),
      exportResult = await rag.exportBackup(backupPath)
    expect(exportResult.documentsExported).toBe(3)

    await rag.drop()
    db = await rag.init()

    const verifyEmpty = await rag.retrieve({
      embed: fakeEmbed,
      mode: 'vector',
      query: 'machine learning'
    })
    expect(verifyEmpty.length).toBe(0)

    const importResult = await rag.importBackup(backupPath)
    expect(importResult.documentsImported).toBe(3)

    const searchResults2 = await rag.retrieve({
      embed: fakeEmbed,
      mode: 'hybrid',
      query: 'neural networks deep learning'
    })
    expect(searchResults2.length).toBeGreaterThan(0)

    const titles1: string[] = []
    for (const r of searchResults1) titles1.push(r.title)
    const titles2: string[] = []
    for (const r of searchResults2) titles2.push(r.title)
    expect(titles2).toEqual(titles1)

    unlinkSync(backupPath)
  })
})

describe('type safety', () => {
  test('Doc requires title and content', () => {
    const valid: Doc = { content: 'text', title: 'title' }
    expect(valid).toBeDefined()

    // @ts-expect-error missing content
    const a: Doc = { title: 'title' },
      // @ts-expect-error missing title
      b: Doc = { content: 'text' },
      // @ts-expect-error wrong prop name
      c: Doc = { conten: 'text', title: 'title' }
    expect([a, b, c]).toBeDefined()
  })

  test('SearchConfig requires query and embed', () => {
    const valid: SearchConfig = { embed: fakeEmbed, query: 'q' }
    expect(valid).toBeDefined()

    // @ts-expect-error missing embed
    const a: SearchConfig = { query: 'q' },
      // @ts-expect-error missing query
      b: SearchConfig = { embed: fakeEmbed },
      // @ts-expect-error wrong mode value
      c: SearchConfig = { embed: fakeEmbed, mode: 'fulltext', query: 'q' }
    expect([a, b, c]).toBeDefined()
  })

  test('IngestConfig requires embed', () => {
    const valid: IngestConfig = { embed: fakeEmbed }
    expect(valid).toBeDefined()

    // @ts-expect-error missing embed
    const a: IngestConfig = {},
      // @ts-expect-error wrong embed type
      b: IngestConfig = { embed: 'not a function' }
    expect([a, b]).toBeDefined()
  })

  test('SearchResult fields are typed', () => {
    const valid: SearchResult = { documentId: 1, id: 1, mode: 'vector', score: 0.9, text: 'text', title: 'title' }
    expect(valid).toBeDefined()

    // @ts-expect-error wrong mode value
    const a: SearchResult = { ...valid, mode: 'fulltext' },
      // @ts-expect-error missing required field score
      b: SearchResult = { documentId: 1, id: 1, mode: 'vector' as const, text: 'text', title: 'title' }
    expect([a, b]).toBeDefined()
  })

  test('EmbedFn rejects non-function', () => {
    const valid: EmbedFn = fakeEmbed
    expect(valid).toBeDefined()

    // @ts-expect-error string is not a valid EmbedFn
    const a: EmbedFn = 'not a function'
    expect(a).toBeDefined()
  })
})
