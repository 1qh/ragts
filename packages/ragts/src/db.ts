import type { PostgresJsDatabase } from 'drizzle-orm/postgres-js'

import { sql } from 'drizzle-orm'
import { drizzle } from 'drizzle-orm/postgres-js'
import postgres from 'postgres'

import * as schema from './schema'

type DrizzleDb = PostgresJsDatabase<typeof schema>

const initTables = async (db: DrizzleDb, dimension: number) => {
    await db.execute(
      sql.raw(`
      CREATE TABLE IF NOT EXISTS documents (
        id BIGSERIAL PRIMARY KEY,
        title TEXT NOT NULL,
        content TEXT NOT NULL,
        content_hash TEXT NOT NULL,
        metadata JSONB DEFAULT '{}',
        community_id INTEGER,
        created_at TIMESTAMPTZ DEFAULT NOW()
      )
    `)
    )
    await db.execute(
      sql.raw(`
      CREATE TABLE IF NOT EXISTS chunks (
        id BIGSERIAL PRIMARY KEY,
        text TEXT NOT NULL,
        text_hash TEXT NOT NULL,
        token_count INTEGER NOT NULL,
        embedding VECTOR(${String(dimension)}) NOT NULL,
        created_at TIMESTAMPTZ DEFAULT NOW()
      )
    `)
    )
    await db.execute(sql`
      CREATE TABLE IF NOT EXISTS chunk_sources (
        id BIGSERIAL PRIMARY KEY,
        chunk_id BIGINT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
        document_id BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
        start_index INTEGER NOT NULL,
        end_index INTEGER NOT NULL
      )
    `)
    await db.execute(sql`
      CREATE TABLE IF NOT EXISTS document_relations (
        id BIGSERIAL PRIMARY KEY,
        source_id BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
        target_id BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
        rel_type TEXT,
        weight REAL DEFAULT 1.0,
        UNIQUE(source_id, target_id)
      )
    `)
  },
  initIndexes = async (db: DrizzleDb, textConfig: string) => {
    await db.execute(sql`CREATE UNIQUE INDEX IF NOT EXISTS documents_content_hash_idx ON documents (content_hash)`)
    await db.execute(sql`CREATE UNIQUE INDEX IF NOT EXISTS chunks_text_hash_idx ON chunks (text_hash)`)
    await db.execute(sql`CREATE INDEX IF NOT EXISTS chunk_sources_chunk_id_idx ON chunk_sources (chunk_id)`)
    await db.execute(sql`CREATE INDEX IF NOT EXISTS chunk_sources_document_id_idx ON chunk_sources (document_id)`)
    await db.execute(sql`CREATE INDEX IF NOT EXISTS document_relations_source_id_idx ON document_relations (source_id)`)
    await db.execute(sql`CREATE INDEX IF NOT EXISTS document_relations_target_id_idx ON document_relations (target_id)`)
    await db.execute(sql`CREATE INDEX IF NOT EXISTS documents_community_id_idx ON documents (community_id)`)
    await db.execute(sql`
      CREATE INDEX IF NOT EXISTS chunks_embedding_idx
      ON chunks USING diskann (embedding vector_cosine_ops)
    `)
    await db.execute(
      sql.raw(`
      CREATE INDEX IF NOT EXISTS chunks_text_idx
      ON chunks USING bm25 (text) WITH (text_config = '${textConfig.replaceAll("'", "''")}')
    `)
    )
  },
  createDb = (connectionString: string) => {
    const client = postgres(connectionString),
      db: DrizzleDb = drizzle(client, { schema })
    return { client, db }
  },
  initSchema = async (db: DrizzleDb, dimension: number, textConfig: string) => {
    await db.execute(sql`CREATE EXTENSION IF NOT EXISTS vectorscale CASCADE`)
    await db.execute(sql`CREATE EXTENSION IF NOT EXISTS pg_textsearch`)
    await initTables(db, dimension)
    await initIndexes(db, textConfig)
  },
  dropSchema = async (connectionString: string) => {
    const client = postgres(connectionString),
      db: DrizzleDb = drizzle(client, { schema })
    await db.execute(sql`DROP TABLE IF EXISTS document_relations CASCADE`)
    await db.execute(sql`DROP TABLE IF EXISTS chunk_sources CASCADE`)
    await db.execute(sql`DROP TABLE IF EXISTS chunks CASCADE`)
    await db.execute(sql`DROP TABLE IF EXISTS documents CASCADE`)
    await client.end()
  }

export { createDb, dropSchema, initSchema }
export type { DrizzleDb }
