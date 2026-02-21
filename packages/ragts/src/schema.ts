import {
  bigint,
  bigserial,
  index,
  integer,
  jsonb,
  pgTable,
  text,
  timestamp,
  uniqueIndex,
  vector
} from 'drizzle-orm/pg-core'

const DEFAULT_DIMENSION = 2048,
  createSchema = (dimension = DEFAULT_DIMENSION) => {
    const docs = pgTable(
        'documents',
        {
          content: text().notNull(),
          contentHash: text('content_hash').notNull(),
          createdAt: timestamp('created_at', { withTimezone: true }).defaultNow(),
          id: bigserial({ mode: 'number' }).primaryKey(),
          metadata: jsonb().$type<Record<string, unknown>>().default({}),
          title: text().notNull()
        },
        t => [uniqueIndex('documents_content_hash_idx').on(t.contentHash)]
      ),
      chnks = pgTable('chunks', {
        createdAt: timestamp('created_at', { withTimezone: true }).defaultNow(),
        embedding: vector({ dimensions: dimension }).notNull(),
        id: bigserial({ mode: 'number' }).primaryKey(),
        text: text().notNull(),
        textHash: text('text_hash').notNull(),
        tokenCount: integer('token_count').notNull()
      }),
      chnkSources = pgTable(
        'chunk_sources',
        {
          chunkId: bigint('chunk_id', { mode: 'number' })
            .notNull()
            .references(() => chnks.id, { onDelete: 'cascade' }),
          documentId: bigint('document_id', { mode: 'number' })
            .notNull()
            .references(() => docs.id, { onDelete: 'cascade' }),
          endIndex: integer('end_index').notNull(),
          id: bigserial({ mode: 'number' }).primaryKey(),
          startIndex: integer('start_index').notNull()
        },
        t => [index('chunk_sources_chunk_id_idx').on(t.chunkId), index('chunk_sources_document_id_idx').on(t.documentId)]
      )

    return { chunks: chnks, chunkSources: chnkSources, documents: docs }
  },
  { chunks, chunkSources, documents } = createSchema()

export { chunks, chunkSources, createSchema, documents }
