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

const documents = pgTable(
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
  chunks = pgTable(
    'chunks',
    {
      createdAt: timestamp('created_at', { withTimezone: true }).defaultNow(),
      documentId: bigint('document_id', { mode: 'number' })
        .notNull()
        .references(() => documents.id, { onDelete: 'cascade' }),
      embedding: vector({ dimensions: 768 }).notNull(),
      endIndex: integer('end_index').notNull(),
      id: bigserial({ mode: 'number' }).primaryKey(),
      startIndex: integer('start_index').notNull(),
      text: text().notNull(),
      tokenCount: integer('token_count').notNull()
    },
    t => [
      index('chunks_document_id_idx').on(t.documentId),
      index('chunks_embedding_idx').using('diskann', t.embedding.op('vector_cosine_ops')),
      index('chunks_text_idx').using('bm25', t.text)
    ]
  )

export { chunks, documents }
