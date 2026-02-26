/* eslint-disable max-statements */
import { afterAll, beforeAll, describe, expect, test } from 'bun:test'
import { sql } from 'drizzle-orm'
import { existsSync, readFileSync, unlinkSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import type {
  BackupDoc,
  Doc,
  EmbedFn,
  GraphRelation,
  IngestConfig,
  RelationTarget,
  SearchConfig,
  SearchResult
} from '../src/index'

import { validateBackup } from '../src/backup'
import { chunkText } from '../src/chunk'
import { generateCompose } from '../src/compose'
import { buildContext, buildGraphContext, Rag } from '../src/index'

const RE_COMMUNITY_TITLE = /^_ragts_community_/u,
  TEST_DB_URL = 'postgresql://postgres:postgres@localhost:5432/ragts_test',
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
          "SELECT tablename FROM pg_tables WHERE schemaname = 'public' AND tablename IN ('documents', 'chunks', 'chunk_sources') ORDER BY tablename"
        )
      ),
      names: string[] = []
    for (const r of rows) names.push(r.tablename)
    expect(names).toContain('chunks')
    expect(names).toContain('chunk_sources')
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
      sql.raw("SELECT indexname FROM pg_indexes WHERE tablename = 'chunks' AND indexname LIKE '%text_idx%'")
    )
    expect(rows.length).toBeGreaterThan(0)
  })

  test('unique index on chunks text_hash', async () => {
    const rows = await db.execute<{ indexname: string }>(
      sql.raw("SELECT indexname FROM pg_indexes WHERE tablename = 'chunks' AND indexname LIKE '%text_hash%'")
    )
    expect(rows.length).toBeGreaterThan(0)
  })

  test('chunk_sources indexes exist', async () => {
    const rows = await db.execute<{ indexname: string }>(
        sql.raw("SELECT indexname FROM pg_indexes WHERE tablename = 'chunk_sources' ORDER BY indexname")
      ),
      names: string[] = []
    for (const r of rows) names.push(r.indexname)
    expect(names.some(n => n.includes('chunk_id'))).toBe(true)
    expect(names.some(n => n.includes('document_id'))).toBe(true)
  })
})

describe('chunk', () => {
  test('chunker splits text', () => {
    const longText = 'The quick brown fox jumps over the lazy dog. '.repeat(100),
      chunks = chunkText(longText, { chunkSize: 200 })
    expect(chunks.length).toBeGreaterThan(1)
    for (const c of chunks) {
      expect(c.text.length).toBeGreaterThan(0)
      expect(c.startIndex).toBeGreaterThanOrEqual(0)
      expect(c.endIndex).toBeGreaterThan(c.startIndex)
      expect(c.tokenCount).toBeGreaterThan(0)
    }
  })

  test('chunker splits text with small size', () => {
    const longText = 'Hello world this is a test of the token chunker functionality. '.repeat(50),
      chunks = chunkText(longText, { chunkSize: 100 })
    expect(chunks.length).toBeGreaterThan(1)
    for (const c of chunks) {
      expect(c.text.length).toBeGreaterThan(0)
      expect(c.tokenCount).toBeGreaterThan(0)
    }
  })

  test('short text below 50 chars is filtered out', () => {
    const chunks = chunkText('Hello world', { chunkSize: 512 })
    expect(chunks.length).toBe(0)
  })

  test('unicode text works', () => {
    const unicodeText = 'Nghị định này quy định chi tiết và biện pháp thi hành một số điều của Luật. '.repeat(50),
      chunks = chunkText(unicodeText, { chunkSize: 512 })
    expect(chunks.length).toBeGreaterThan(0)
    for (const c of chunks) expect(c.text.length).toBeGreaterThanOrEqual(50)
  })

  test('empty text returns empty array', () => {
    const chunks = chunkText('', { chunkSize: 512 })
    expect(chunks.length).toBe(0)
  })

  test('very long text produces many chunks', () => {
    const longText = 'The quick brown fox jumps over the lazy dog. '.repeat(5000),
      chunks = chunkText(longText, { chunkSize: 200 })
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
    expect(result.chunksReused).toBe(0)
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

describe('dedup', () => {
  beforeAll(async () => {
    await rag.drop()
    db = await rag.init()
  })

  test('two docs with identical chunk text produce 1 chunk row and 2 junction rows', async () => {
    const sharedContent =
        'Shared legal text about property rights in civil law that appears in multiple documents verbatim.',
      docA: Doc = { content: sharedContent, title: 'Law Doc A' },
      docB: Doc = { content: sharedContent, title: 'Law Doc B' },
      r1 = await rag.ingest([docA], { embed: fakeEmbed })
    expect(r1.documentsInserted).toBe(1)
    expect(r1.chunksInserted).toBeGreaterThan(0)
    expect(r1.chunksReused).toBe(0)

    const r2 = await rag.ingest([docB], { embed: fakeEmbed })
    expect(r2.documentsInserted).toBe(1)
    expect(r2.chunksReused).toBeGreaterThan(0)

    const chunkRows = await db.execute<{ count: string }>(sql.raw('SELECT COUNT(*) as count FROM chunks')),
      sourceRows = await db.execute<{ count: string }>(sql.raw('SELECT COUNT(*) as count FROM chunk_sources')),
      chunkCount = Number(chunkRows[0]?.count),
      sourceCount = Number(sourceRows[0]?.count)
    expect(sourceCount).toBeGreaterThan(chunkCount)
  })

  test('incremental ingest reuses existing chunks', async () => {
    await rag.drop()
    db = await rag.init()

    const shared = 'This exact paragraph about database indexing strategies appears across multiple technical documents.',
      doc1: Doc = {
        content: `${shared} Plus some unique content only in document one about query optimization.`,
        title: 'Tech Doc 1'
      },
      doc2: Doc = {
        content: `${shared} Plus different unique content only in document two about caching layers.`,
        title: 'Tech Doc 2'
      },
      r1 = await rag.ingest([doc1], { embed: fakeEmbed })
    expect(r1.chunksInserted).toBeGreaterThan(0)

    const chunksBefore = await db.execute<{ count: string }>(sql.raw('SELECT COUNT(*) as count FROM chunks')),
      countBefore = Number(chunksBefore[0]?.count),
      r2 = await rag.ingest([doc2], { embed: fakeEmbed })
    expect(r2.documentsInserted).toBe(1)

    const chunksAfter = await db.execute<{ count: string }>(sql.raw('SELECT COUNT(*) as count FROM chunks')),
      countAfter = Number(chunksAfter[0]?.count)
    expect(countAfter).toBeLessThanOrEqual(countBefore + r2.chunksInserted)
    expect(r2.chunksInserted + r2.chunksReused).toBeGreaterThan(0)
  })

  test('search returns correct title for deduped chunks', async () => {
    const results = await rag.retrieve({
      embed: fakeEmbed,
      mode: 'vector',
      query: 'database indexing strategies'
    })
    expect(results.length).toBeGreaterThan(0)
    for (const r of results) {
      expect(r.title.length).toBeGreaterThan(0)
      expect(r.documentId).toBeGreaterThan(0)
    }
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

describe('graph', () => {
  beforeAll(async () => {
    await rag.drop()
    db = await rag.init()
  })

  test('ingest with relations inserts document_relations rows', async () => {
    const docs: Doc[] = [
        { content: 'Decree 01 regulates implementation details of the parent law on construction.', title: 'decree-01' },
        { content: 'Law 13 establishes the legal framework for urban construction and planning.', title: 'law-13' },
        { content: 'Circular 07 provides guidance on applying decree 01 in practice.', title: 'circular-07' }
      ],
      relations: Record<string, string[]> = {
        'decree-01': ['law-13', 'circular-07'],
        'law-13': ['decree-01']
      },
      result = await rag.ingest(docs, { embed: fakeEmbed, relations })
    expect(result.documentsInserted).toBe(3)
    expect(result.relationsInserted).toBe(3)
    expect(result.unresolvedRelations.length).toBe(0)

    const relRows = await db.execute<{ count: string }>(sql.raw('SELECT COUNT(*) as count FROM document_relations'))
    expect(Number(relRows[0]?.count)).toBe(3)
  })

  test('ingest without relations returns zero relationsInserted', async () => {
    const docs: Doc[] = [{ content: 'Standalone document with no relations to any other document.', title: 'standalone' }],
      result = await rag.ingest(docs, { embed: fakeEmbed })
    expect(result.relationsInserted).toBe(0)
    expect(result.unresolvedRelations.length).toBe(0)
  })

  test('dangling references are tracked in unresolvedRelations', async () => {
    await rag.drop()
    db = await rag.init()

    const docs: Doc[] = [
        { content: 'Document that references something that does not exist in database.', title: 'ref-doc' }
      ],
      relations: Record<string, string[]> = {
        'ref-doc': ['nonexistent-doc']
      },
      result = await rag.ingest(docs, { embed: fakeEmbed, relations })
    expect(result.relationsInserted).toBe(0)
    expect(result.unresolvedRelations).toContain('nonexistent-doc')
  })

  test('self-references are skipped', async () => {
    await rag.drop()
    db = await rag.init()

    const docs: Doc[] = [
        { content: 'A document that references itself should not create a relation.', title: 'self-ref' }
      ],
      relations: Record<string, string[]> = {
        'self-ref': ['self-ref']
      },
      result = await rag.ingest(docs, { embed: fakeEmbed, relations })
    expect(result.relationsInserted).toBe(0)
  })

  test('empty relations arrays are no-ops', async () => {
    await rag.drop()
    db = await rag.init()

    const docs: Doc[] = [{ content: 'A document with explicitly empty relations list provided.', title: 'empty-rel' }],
      relations: Record<string, string[]> = {
        'empty-rel': []
      },
      result = await rag.ingest(docs, { embed: fakeEmbed, relations })
    expect(result.relationsInserted).toBe(0)
  })

  test('forward references within same batch are resolved', async () => {
    await rag.drop()
    db = await rag.init()

    const docs: Doc[] = [
        { content: 'First document in batch that references the second document.', title: 'batch-a' },
        { content: 'Second document in batch that is referenced by the first.', title: 'batch-b' }
      ],
      relations: Record<string, string[]> = {
        'batch-a': ['batch-b']
      },
      result = await rag.ingest(docs, { embed: fakeEmbed, relations })
    expect(result.relationsInserted).toBe(1)
  })

  test('retrieve with graphHops expands to related document chunks', async () => {
    await rag.drop()
    db = await rag.init()

    const docs: Doc[] = [
        {
          content: 'Document Alpha contains information about TypeScript generics and advanced type system features.',
          title: 'alpha'
        },
        {
          content: 'Document Beta contains information about JavaScript closures and prototype chain mechanisms.',
          title: 'beta'
        },
        { content: 'Document Gamma contains information about Rust ownership and borrowing memory model.', title: 'gamma' }
      ],
      relations: Record<string, string[]> = {
        alpha: ['beta']
      }
    await rag.ingest(docs, { embed: fakeEmbed, relations })

    const withoutGraph = await rag.retrieve({
      embed: fakeEmbed,
      limit: 1,
      mode: 'vector',
      query: 'TypeScript type system'
    })
    expect(withoutGraph.length).toBe(1)

    const withGraph = await rag.retrieve({
      embed: fakeEmbed,
      graphHops: 1,
      limit: 1,
      mode: 'vector',
      query: 'TypeScript type system'
    })
    expect(withGraph.length).toBeGreaterThan(1)

    const graphResults = withGraph.filter(r => r.mode === 'graph')
    expect(graphResults.length).toBeGreaterThan(0)
    for (const r of graphResults) {
      expect(r.score).toBeGreaterThan(0)
      expect(r.title).toBe('beta')
    }
  })

  test('bidirectional traversal: B found when querying A, and A found when querying B', async () => {
    await rag.drop()
    db = await rag.init()

    const docs: Doc[] = [
        { content: 'Document A about distributed systems and consensus algorithms in cloud computing.', title: 'doc-a' },
        { content: 'Document B about database replication and partition tolerance strategies.', title: 'doc-b' }
      ],
      relations: Record<string, string[]> = {
        'doc-a': ['doc-b']
      }
    await rag.ingest(docs, { embed: fakeEmbed, relations })

    const resultsFromA = await rag.retrieve({
        embed: fakeEmbed,
        graphHops: 1,
        limit: 1,
        mode: 'bm25',
        query: 'distributed systems consensus'
      }),
      graphFromA = resultsFromA.filter(r => r.mode === 'graph')
    expect(graphFromA.some(r => r.title === 'doc-b')).toBe(true)

    const resultsFromB = await rag.retrieve({
        embed: fakeEmbed,
        graphHops: 1,
        limit: 1,
        mode: 'bm25',
        query: 'database replication partition'
      }),
      graphFromB = resultsFromB.filter(r => r.mode === 'graph')
    expect(graphFromB.some(r => r.title === 'doc-a')).toBe(true)
  })

  test('graphHops=0 behaves like no graph expansion', async () => {
    const results = await rag.retrieve({
        embed: fakeEmbed,
        graphHops: 0,
        mode: 'vector',
        query: 'distributed systems'
      }),
      graphResults = results.filter(r => r.mode === 'graph')
    expect(graphResults.length).toBe(0)
  })

  test('retrieve without graphHops has no graph-mode results', async () => {
    const results = await rag.retrieve({
        embed: fakeEmbed,
        mode: 'vector',
        query: 'distributed systems'
      }),
      graphResults = results.filter(r => r.mode === 'graph')
    expect(graphResults.length).toBe(0)
  })

  test('circular references do not cause infinite recursion', async () => {
    await rag.drop()
    db = await rag.init()

    const docs: Doc[] = [
        { content: 'Circular A references B which references C which references A again.', title: 'circ-a' },
        { content: 'Circular B references C which references A which references B again.', title: 'circ-b' },
        { content: 'Circular C references A which references B which references C again.', title: 'circ-c' }
      ],
      relations: Record<string, string[]> = {
        'circ-a': ['circ-b'],
        'circ-b': ['circ-c'],
        'circ-c': ['circ-a']
      }
    await rag.ingest(docs, { embed: fakeEmbed, relations })

    const results = await rag.retrieve({
      embed: fakeEmbed,
      graphHops: 5,
      limit: 1,
      mode: 'bm25',
      query: 'Circular A references'
    })
    expect(results.length).toBeGreaterThan(0)
    const graphResults = results.filter(r => r.mode === 'graph')
    expect(graphResults.length).toBeGreaterThan(0)
  })

  test('document_relations table exists after init', async () => {
    const rows = await db.execute<{ tablename: string }>(
      sql.raw("SELECT tablename FROM pg_tables WHERE schemaname = 'public' AND tablename = 'document_relations'")
    )
    expect(rows.length).toBe(1)
  })

  test('document_relations table is dropped with rag.drop()', async () => {
    await rag.drop()
    const { db: freshDb } = await import('../src/db').then(m => {
        const result = m.createDb(TEST_DB_URL)
        return result
      }),
      rows = await freshDb.execute<{ tablename: string }>(
        sql.raw("SELECT tablename FROM pg_tables WHERE schemaname = 'public' AND tablename = 'document_relations'")
      )
    expect(rows.length).toBe(0)

    db = await rag.init()
  })

  test('backup includes relations and import restores them', async () => {
    await rag.drop()
    db = await rag.init()

    const docs: Doc[] = [
        { content: 'Backup test doc one with enough content for chunking and relation testing.', title: 'bk-doc-1' },
        { content: 'Backup test doc two with enough content for chunking and relation testing.', title: 'bk-doc-2' }
      ],
      relations: Record<string, string[]> = {
        'bk-doc-1': ['bk-doc-2']
      }
    await rag.ingest(docs, { embed: fakeEmbed, relations })

    const backupPath = join(tmpdir(), `ragts-graph-backup-${Date.now()}.jsonl`)
    await rag.exportBackup(backupPath)

    const backupContent = readFileSync(backupPath, 'utf8'),
      [firstDoc] = backupContent
        .trim()
        .split('\n')
        .map(l => JSON.parse(l) as BackupDoc)
    expect(firstDoc?.relations?.length).toBeGreaterThan(0)

    await rag.drop()
    db = await rag.init()
    await rag.importBackup(backupPath)

    const relRows = await db.execute<{ count: string }>(sql.raw('SELECT COUNT(*) as count FROM document_relations'))
    expect(Number(relRows[0]?.count)).toBeGreaterThan(0)

    unlinkSync(backupPath)
  })
})

describe('graph-phase2', () => {
  beforeAll(async () => {
    await rag.drop()
    db = await rag.init()
  })

  test('ingest with enriched RelationTarget (type and weight)', async () => {
    const docs: Doc[] = [
        {
          content: 'Law 50 establishes the regulatory framework for environmental protection standards.',
          title: 'law-50'
        },
        { content: 'Decree 10 implements specific provisions of law 50 for industrial emissions.', title: 'decree-10' },
        { content: 'Circular 03 provides operational guidance for decree 10 enforcement.', title: 'circular-03' }
      ],
      relations: Record<string, RelationTarget[]> = {
        'decree-10': [
          { title: 'law-50', type: 'implements', weight: 0.9 },
          { title: 'circular-03', type: 'guided_by', weight: 0.5 }
        ]
      },
      result = await rag.ingest(docs, { embed: fakeEmbed, relations })
    expect(result.documentsInserted).toBe(3)
    expect(result.relationsInserted).toBe(2)

    const relRows = await db.execute<{ rel_type: null | string; weight: number }>(
      sql.raw('SELECT rel_type, weight FROM document_relations ORDER BY id')
    )
    expect(relRows.length).toBe(2)
    expect(relRows[0]?.rel_type).toBe('implements')
    expect(relRows[0]?.weight).toBeCloseTo(0.9)
    expect(relRows[1]?.rel_type).toBe('guided_by')
    expect(relRows[1]?.weight).toBeCloseTo(0.5)
  })

  test('mixed string and enriched RelationTarget backward compat', async () => {
    await rag.drop()
    db = await rag.init()

    const docs: Doc[] = [
        { content: 'Document X about protocol specifications and networking standards.', title: 'doc-x' },
        { content: 'Document Y about implementation details of networking protocols.', title: 'doc-y' },
        { content: 'Document Z about testing network protocol implementations.', title: 'doc-z' }
      ],
      relations: Record<string, RelationTarget[]> = {
        'doc-x': ['doc-y', { title: 'doc-z', type: 'tested_by' }]
      },
      result = await rag.ingest(docs, { embed: fakeEmbed, relations })
    expect(result.relationsInserted).toBe(2)

    const relRows = await db.execute<{ rel_type: null | string; weight: number }>(
      sql.raw('SELECT rel_type, weight FROM document_relations ORDER BY id')
    )
    expect(relRows[0]?.rel_type).toBeNull()
    expect(relRows[0]?.weight).toBeCloseTo(1)
    expect(relRows[1]?.rel_type).toBe('tested_by')
  })

  test('graph expansion uses weight for scoring', async () => {
    await rag.drop()
    db = await rag.init()

    const docs: Doc[] = [
        { content: 'Main document about distributed computing and consensus protocols.', title: 'main-doc' },
        { content: 'High weight related document about Raft consensus algorithm details.', title: 'high-weight' },
        { content: 'Low weight related document about eventual consistency patterns.', title: 'low-weight' }
      ],
      relations: Record<string, RelationTarget[]> = {
        'main-doc': [
          { title: 'high-weight', type: 'core', weight: 0.95 },
          { title: 'low-weight', type: 'peripheral', weight: 0.1 }
        ]
      }
    await rag.ingest(docs, { embed: fakeEmbed, relations })

    const results = await rag.retrieve({
        embed: fakeEmbed,
        graphHops: 1,
        limit: 1,
        mode: 'bm25',
        query: 'distributed computing consensus'
      }),
      graphResults = results.filter(r => r.mode === 'graph')
    expect(graphResults.length).toBe(2)

    const highWeightResult = graphResults.find(r => r.title === 'high-weight'),
      lowWeightResult = graphResults.find(r => r.title === 'low-weight')
    expect(highWeightResult).toBeDefined()
    expect(lowWeightResult).toBeDefined()
    // oxlint-disable-next-line jest/no-conditional-in-test
    expect(highWeightResult?.score).toBeGreaterThan(lowWeightResult?.score ?? 0)
    expect(highWeightResult?.relationType).toBe('core')
    expect(lowWeightResult?.relationType).toBe('peripheral')
  })

  test('communityId is set after ingest with relations', async () => {
    await rag.drop()
    db = await rag.init()

    const docs: Doc[] = [
        { content: 'Community doc A about machine learning classification algorithms.', title: 'comm-a' },
        { content: 'Community doc B about neural network training procedures.', title: 'comm-b' },
        { content: 'Isolated doc C about unrelated gardening techniques and tips.', title: 'comm-c' }
      ],
      relations: Record<string, RelationTarget[]> = {
        'comm-a': ['comm-b']
      },
      result = await rag.ingest(docs, { embed: fakeEmbed, relations })
    expect(result.communitiesDetected).toBeGreaterThanOrEqual(2)

    const communityRows = await db.execute<{ community_id: null | number; title: string }>(
        sql.raw('SELECT title, community_id FROM documents ORDER BY title')
      ),
      docA = communityRows.find(r => r.title === 'comm-a'),
      docB = communityRows.find(r => r.title === 'comm-b'),
      docC = communityRows.find(r => r.title === 'comm-c')
    expect(docA?.community_id).toBe(docB?.community_id)
    expect(docA?.community_id).not.toBe(docC?.community_id)
  })

  test('detectCommunities standalone method', async () => {
    const count = await rag.detectCommunities()
    expect(count).toBeGreaterThanOrEqual(2)
  })

  test('buildCommunitySummaries generates summary docs', async () => {
    await rag.drop()
    db = await rag.init()

    const docs: Doc[] = [
        { content: 'Summary source A about climate change impact on agriculture.', title: 'src-a' },
        { content: 'Summary source B about agricultural policy reforms and subsidies.', title: 'src-b' },
        { content: 'Unrelated summary source C about quantum computing basics.', title: 'src-c' }
      ],
      relations: Record<string, RelationTarget[]> = {
        'src-a': ['src-b']
      }
    await rag.ingest(docs, { embed: fakeEmbed, relations })

    const summaryResult = await rag.buildCommunitySummaries({
      embed: fakeEmbed,
      minCommunitySize: 2,
      // eslint-disable-next-line @typescript-eslint/require-await
      summarize: async summaryDocs => {
        const titles: string[] = []
        for (const d of summaryDocs) titles.push(d.title)
        return `Summary of ${titles.join(', ')}: combined knowledge.`
      }
    })
    expect(summaryResult.communitiesProcessed).toBe(1)
    expect(summaryResult.summariesGenerated).toBe(1)

    const summaryRows = await db.execute<{ metadata: Record<string, unknown>; title: string }>(
      sql.raw("SELECT title, metadata FROM documents WHERE metadata->>'_ragts_type' = 'community_summary'")
    )
    expect(summaryRows.length).toBe(1)
    expect(summaryRows[0]?.title).toMatch(RE_COMMUNITY_TITLE)
    expect(summaryRows[0]?.metadata._ragts_member_titles).toBeDefined()
  })

  test('search results include communityId', async () => {
    const results = await rag.retrieve({
        embed: fakeEmbed,
        mode: 'vector',
        query: 'climate change agriculture'
      }),
      withCommunity = results.filter(r => r.communityId !== undefined)
    expect(withCommunity.length).toBeGreaterThan(0)
  })

  test('fetchRelations returns graph relations', async () => {
    await rag.drop()
    db = await rag.init()

    const docs: Doc[] = [
        { content: 'Relation fetch doc A about software engineering methodologies.', title: 'rel-a' },
        { content: 'Relation fetch doc B about agile development practices and sprints.', title: 'rel-b' }
      ],
      relations: Record<string, RelationTarget[]> = {
        'rel-a': [{ title: 'rel-b', type: 'references' }]
      }
    await rag.ingest(docs, { embed: fakeEmbed, relations })

    const docRows = await db.execute<{ id: number }>(sql.raw('SELECT id FROM documents ORDER BY id')),
      ids: number[] = []
    for (const r of docRows) ids.push(r.id)
    const graphRels = await rag.fetchRelations(ids)
    expect(graphRels.length).toBe(1)
    expect(graphRels[0]?.sourceTitle).toBe('rel-a')
    expect(graphRels[0]?.targetTitle).toBe('rel-b')
    expect(graphRels[0]?.type).toBe('references')
  })

  test('buildGraphContext includes relation section', () => {
    const results = [
        { text: 'Content about TypeScript', title: 'ts-guide' },
        { text: 'Content about JavaScript', title: 'js-guide' }
      ],
      relations: GraphRelation[] = [{ sourceTitle: 'ts-guide', targetTitle: 'js-guide', type: 'extends' }],
      ctx = buildGraphContext(results, relations)
    expect(ctx).toContain('=== Document Relations ===')
    expect(ctx).toContain('ts-guide → js-guide [extends]')
    expect(ctx).toContain('[1] ts-guide')
    expect(ctx).toContain('[2] js-guide')
  })

  test('buildGraphContext without relations matches buildContext', () => {
    const results = [
        { text: 'Some content here', title: 'doc-1' },
        { text: 'Other content here', title: 'doc-2' }
      ],
      graphCtx = buildGraphContext(results, []),
      normalCtx = buildContext(results)
    expect(graphCtx).toBe(normalCtx)
  })

  test('globalQuery with map-reduce over communities', async () => {
    await rag.drop()
    db = await rag.init()

    const docs: Doc[] = [
        { content: 'Global query doc A about renewable energy solar panels and wind turbines.', title: 'gq-a' },
        { content: 'Global query doc B about solar energy storage battery technology advancements.', title: 'gq-b' },
        { content: 'Global query doc C about nuclear fusion reactor design and engineering.', title: 'gq-c' },
        { content: 'Global query doc D about nuclear waste management disposal strategies.', title: 'gq-d' }
      ],
      relations: Record<string, RelationTarget[]> = {
        'gq-a': ['gq-b'],
        'gq-c': ['gq-d']
      }
    await rag.ingest(docs, { embed: fakeEmbed, relations })

    await rag.buildCommunitySummaries({
      embed: fakeEmbed,
      minCommunitySize: 2,
      // eslint-disable-next-line @typescript-eslint/require-await
      summarize: async summaryDocs => {
        const titles: string[] = []
        for (const d of summaryDocs) titles.push(d.title)
        return `Summary covering ${titles.join(' and ')}.`
      }
    })

    const result = await rag.globalQuery({
      embed: fakeEmbed,
      // eslint-disable-next-line @typescript-eslint/require-await
      generate: async (context, query) => `Answer for '${query}' based on: ${context.slice(0, 50)}`,
      query: 'energy technologies'
    })
    expect(result.answer.length).toBeGreaterThan(0)
    expect(result.partialAnswers.length).toBeGreaterThanOrEqual(0)
  })

  test('backup with enriched relations preserves type and weight', async () => {
    await rag.drop()
    db = await rag.init()

    const docs: Doc[] = [
        { content: 'Backup enriched doc A about data warehousing star schema design.', title: 'be-a' },
        { content: 'Backup enriched doc B about ETL pipeline orchestration tools.', title: 'be-b' }
      ],
      relations: Record<string, RelationTarget[]> = {
        'be-a': [{ title: 'be-b', type: 'feeds_into', weight: 0.7 }]
      }
    await rag.ingest(docs, { embed: fakeEmbed, relations })

    const backupPath = join(tmpdir(), `ragts-enriched-backup-${Date.now()}.jsonl`)
    await rag.exportBackup(backupPath)

    const content = readFileSync(backupPath, 'utf8'),
      lines = content.trim().split('\n'),
      allDocs: BackupDoc[] = []
    for (const line of lines) allDocs.push(JSON.parse(line) as BackupDoc)
    const beaDoc = allDocs.find(d => d.title === 'be-a')
    expect(beaDoc).toBeDefined()
    expect(beaDoc?.relations?.length).toBeGreaterThan(0)
    // oxlint-disable-next-line jest/no-conditional-in-test
    const [rel] = beaDoc?.relations ?? []
    expect(rel?.title).toBe('be-b')
    expect(rel?.type).toBe('feeds_into')
    expect(rel?.weight).toBeCloseTo(0.7)

    await rag.drop()
    db = await rag.init()
    await rag.importBackup(backupPath)

    const relRows = await db.execute<{ rel_type: null | string; weight: number }>(
      sql.raw('SELECT rel_type, weight FROM document_relations LIMIT 1')
    )
    expect(relRows[0]?.rel_type).toBe('feeds_into')
    expect(relRows[0]?.weight).toBeCloseTo(0.7)

    unlinkSync(backupPath)
  })

  test('backup with communityId preserves it', async () => {
    await rag.drop()
    db = await rag.init()

    const docs: Doc[] = [
        { content: 'Community backup A about microservice architecture patterns.', title: 'cb-a' },
        { content: 'Community backup B about service mesh networking proxies.', title: 'cb-b' }
      ],
      relations: Record<string, RelationTarget[]> = {
        'cb-a': ['cb-b']
      }
    await rag.ingest(docs, { embed: fakeEmbed, relations })

    const backupPath = join(tmpdir(), `ragts-community-backup-${Date.now()}.jsonl`)
    await rag.exportBackup(backupPath)

    const content = readFileSync(backupPath, 'utf8'),
      [firstLine] = content.trim().split('\n')
    expect(firstLine).toBeDefined()
    const firstDoc = JSON.parse(String(firstLine)) as BackupDoc
    expect(firstDoc.communityId).toBeDefined()

    await rag.drop()
    db = await rag.init()
    await rag.importBackup(backupPath)

    const communityRows = await db.execute<{ community_id: null | number }>(
      sql.raw('SELECT community_id FROM documents WHERE community_id IS NOT NULL LIMIT 1')
    )
    expect(communityRows.length).toBeGreaterThan(0)

    unlinkSync(backupPath)
  })

  test('query with graphHops produces graph context with relations', async () => {
    await rag.drop()
    db = await rag.init()

    const docs: Doc[] = [
        { content: 'Graph context doc A about compiler optimization techniques and passes.', title: 'gc-a' },
        { content: 'Graph context doc B about intermediate representation lowering passes.', title: 'gc-b' }
      ],
      relations: Record<string, RelationTarget[]> = {
        'gc-a': [{ title: 'gc-b', type: 'compiles_to' }]
      }
    await rag.ingest(docs, { embed: fakeEmbed, relations })

    const { context } = await rag.query({
      embed: fakeEmbed,
      graphHops: 1,
      limit: 1,
      mode: 'bm25',
      query: 'compiler optimization'
    })
    expect(context).toContain('gc-a')
  })

  test('graph results use RRF-style scoring and are sorted by score', async () => {
    await rag.drop()
    db = await rag.init()

    const docs: Doc[] = [
        { content: 'RRF main about functional programming paradigms and monads in Haskell.', title: 'rrf-main' },
        { content: 'RRF related about lambda calculus foundations and type theory proofs.', title: 'rrf-related' }
      ],
      relations: Record<string, string[]> = {
        'rrf-main': ['rrf-related']
      }
    await rag.ingest(docs, { embed: fakeEmbed, relations })

    const results = await rag.retrieve({
        embed: fakeEmbed,
        graphHops: 1,
        limit: 1,
        mode: 'bm25',
        query: 'functional programming paradigms monads Haskell'
      }),
      graphResults = results.filter(r => r.mode === 'graph')
    expect(graphResults.length).toBeGreaterThan(0)
    for (const r of graphResults) {
      expect(r.score).toBeGreaterThan(0)
      expect(r.score).toBeLessThan(0.1)
    }

    for (let i = 1; i < results.length; i += 1)
      // oxlint-disable-next-line jest/no-conditional-in-test
      expect(results[i - 1]?.score ?? 0).toBeGreaterThanOrEqual(results[i]?.score ?? 0)
  })

  test('graphDecay reduces score with hop distance', async () => {
    await rag.drop()
    db = await rag.init()

    const docs: Doc[] = [
        { content: 'Decay root about network security protocols and advanced firewalls.', title: 'decay-root' },
        { content: 'Decay hop1 about intrusion detection systems and real-time monitoring.', title: 'decay-hop1' },
        { content: 'Decay hop2 about threat intelligence feeds and forensic analysis.', title: 'decay-hop2' }
      ],
      relations: Record<string, string[]> = {
        'decay-hop1': ['decay-hop2'],
        'decay-root': ['decay-hop1']
      }
    await rag.ingest(docs, { embed: fakeEmbed, relations })

    const results = await rag.retrieve({
        embed: fakeEmbed,
        graphDecay: 0.5,
        graphHops: 2,
        limit: 1,
        mode: 'bm25',
        query: 'network security protocols advanced firewalls'
      }),
      graphResults = results.filter(r => r.mode === 'graph')
    expect(graphResults.length).toBeGreaterThan(0)
    const hop1 = graphResults.find(r => r.title === 'decay-hop1'),
      hop2 = graphResults.find(r => r.title === 'decay-hop2')
    expect(hop1).toBeDefined()
    expect(hop2).toBeDefined()
    // oxlint-disable-next-line jest/no-conditional-in-test
    expect(hop1?.score ?? 0).toBeGreaterThan(hop2?.score ?? 0)
  })

  test('graphWeight controls relative importance of graph results', async () => {
    await rag.drop()
    db = await rag.init()

    const docs: Doc[] = [
        { content: 'GW main about distributed database sharding strategies and techniques.', title: 'gw-main' },
        { content: 'GW related about partition tolerance and CAP theorem implications.', title: 'gw-related' }
      ],
      relations: Record<string, string[]> = {
        'gw-main': ['gw-related']
      }
    await rag.ingest(docs, { embed: fakeEmbed, relations })

    const lowWeight = await rag.retrieve({
        embed: fakeEmbed,
        graphHops: 1,
        graphWeight: 0.1,
        limit: 1,
        mode: 'bm25',
        query: 'distributed database sharding'
      }),
      highWeight = await rag.retrieve({
        embed: fakeEmbed,
        graphHops: 1,
        graphWeight: 2,
        limit: 1,
        mode: 'bm25',
        query: 'distributed database sharding'
      }),
      lowGraphResults = lowWeight.filter(r => r.mode === 'graph'),
      highGraphResults = highWeight.filter(r => r.mode === 'graph')
    expect(lowGraphResults.length).toBeGreaterThan(0)
    expect(highGraphResults.length).toBeGreaterThan(0)
    // oxlint-disable-next-line jest/no-conditional-in-test
    expect(highGraphResults[0]?.score ?? 0).toBeGreaterThan(lowGraphResults[0]?.score ?? 0)
  })

  test('communityBoost adds results from dominant community', async () => {
    await rag.drop()
    db = await rag.init()

    const docs: Doc[] = [
        { content: 'Boost doc A about machine learning gradient descent optimization algorithms.', title: 'boost-a' },
        { content: 'Boost doc B about neural network backpropagation training procedures.', title: 'boost-b' },
        { content: 'Boost doc C about unrelated quantum entanglement physics experiments.', title: 'boost-c' }
      ],
      relations: Record<string, string[]> = {
        'boost-a': ['boost-b']
      }
    await rag.ingest(docs, { embed: fakeEmbed, relations })

    const results = await rag.retrieve({
        communityBoost: 0.5,
        embed: fakeEmbed,
        limit: 1,
        mode: 'bm25',
        query: 'machine learning gradient descent optimization'
      }),
      communityResults = results.filter(r => r.mode === 'community')
    expect(communityResults.length).toBeGreaterThan(0)
    // oxlint-disable-next-line jest/no-conditional-in-test
    expect(communityResults.some(r => r.title === 'boost-b')).toBe(true)
  })

  test('communityBoost results have RRF-style scores', async () => {
    await rag.drop()
    db = await rag.init()

    const docs: Doc[] = [
        { content: 'CBoost score A about compiler design and lexical analysis phases.', title: 'cbs-a' },
        { content: 'CBoost score B about syntax parsing and abstract syntax tree generation.', title: 'cbs-b' }
      ],
      relations: Record<string, string[]> = {
        'cbs-a': ['cbs-b']
      }
    await rag.ingest(docs, { embed: fakeEmbed, relations })

    const results = await rag.retrieve({
        communityBoost: 0.8,
        embed: fakeEmbed,
        limit: 1,
        mode: 'bm25',
        query: 'compiler design lexical analysis'
      }),
      communityResults = results.filter(r => r.mode === 'community')
    for (const r of communityResults) {
      expect(r.score).toBeGreaterThan(0)
      expect(r.score).toBeLessThan(0.1)
    }
  })
})
