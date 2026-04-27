CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS listings (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    price INTEGER,
    description TEXT,
    url TEXT,
    location TEXT,
    source_id TEXT UNIQUE NOT NULL,
    source_timestamp BIGINT,
    is_stale BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

ALTER TABLE listings ADD COLUMN IF NOT EXISTS is_stale BOOLEAN NOT NULL DEFAULT FALSE;

CREATE TABLE IF NOT EXISTS listing_images (
    id SERIAL PRIMARY KEY,
    listing_id INTEGER NOT NULL REFERENCES listings(id) ON DELETE CASCADE,
    image_url TEXT NOT NULL,
    local_path TEXT,
    embedding vector(768)
);

CREATE INDEX IF NOT EXISTS listing_images_embedding_idx
ON listing_images
USING hnsw (embedding vector_cosine_ops);
