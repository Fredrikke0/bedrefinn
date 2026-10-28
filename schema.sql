CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS categories (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL
);

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
ALTER TABLE listings ADD COLUMN IF NOT EXISTS category_id TEXT REFERENCES categories(id);

CREATE INDEX IF NOT EXISTS listings_category_id_idx ON listings (category_id);

CREATE TABLE IF NOT EXISTS listing_images (
    id SERIAL PRIMARY KEY,
    listing_id INTEGER NOT NULL REFERENCES listings(id) ON DELETE CASCADE,
    image_url TEXT NOT NULL,
    local_path TEXT,
    embedding vector
);

CREATE INDEX IF NOT EXISTS listing_images_listing_id_idx
ON listing_images (listing_id);
