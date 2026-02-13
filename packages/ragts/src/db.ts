import type { PostgresJsDatabase } from 'drizzle-orm/postgres-js'

import { sql } from 'drizzle-orm'
import { drizzle } from 'drizzle-orm/postgres-js'
import postgres from 'postgres'

import * as schema from './schema'

type DrizzleDb = PostgresJsDatabase<typeof schema>

const createDb = (connectionString: string) => {
    const client = postgres(connectionString),
      db: DrizzleDb = drizzle(client, { schema })
    return { client, db }
  },
  initSchema = async (db: DrizzleDb, dimension: number, textConfig: string) => {
    await db.execute(sql`CREATE EXTENSION IF NOT EXISTS vectorscale CASCADE`)
    await db.execute(sql`CREATE EXTENSION IF NOT EXISTS pg_textsearch`)

    await db.execute(sql`
    CREATE TABLE IF NOT EXISTS documents (
      id BIGSERIAL PRIMARY KEY,
      title TEXT NOT NULL,
      content TEXT NOT NULL,
      metadata JSONB DEFAULT '{}',
      content_hash TEXT NOT NULL,
      created_at TIMESTAMPTZ DEFAULT NOW()
    )
  `)

    await db.execute(sql`
    CREATE UNIQUE INDEX IF NOT EXISTS documents_content_hash_idx
    ON documents (content_hash)
  `)

    await db.execute(
      sql.raw(`
    CREATE TABLE IF NOT EXISTS chunks (
      id BIGSERIAL PRIMARY KEY,
      document_id BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
      text TEXT NOT NULL,
      start_index INTEGER NOT NULL,
      end_index INTEGER NOT NULL,
      token_count INTEGER NOT NULL,
      embedding VECTOR(${String(dimension)}) NOT NULL,
      created_at TIMESTAMPTZ DEFAULT NOW()
    )
  `)
    )

    await db.execute(sql`
    CREATE INDEX IF NOT EXISTS chunks_document_id_idx
    ON chunks (document_id)
  `)

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
  dropSchema = async (connectionString: string) => {
    const client = postgres(connectionString),
      db: DrizzleDb = drizzle(client, { schema })
    await db.execute(sql`DROP TABLE IF EXISTS chunks CASCADE`)
    await db.execute(sql`DROP TABLE IF EXISTS documents CASCADE`)
    await client.end()
  }

export { createDb, dropSchema, initSchema }
export type { DrizzleDb }
