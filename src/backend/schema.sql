CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS products (
    platform TEXT NOT NULL,
    goods_no TEXT NOT NULL,
    goods_name TEXT NOT NULL,
    brand_name TEXT NOT NULL DEFAULT '',
    price INTEGER,
    product_url TEXT NOT NULL,
    image_path TEXT NOT NULL,
    embedding vector(512) NOT NULL,
    embedding_model TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (platform, goods_no)
);

CREATE INDEX IF NOT EXISTS products_embedding_hnsw_idx
ON products USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS products_platform_idx ON products (platform);

CREATE TABLE IF NOT EXISTS search_events (
    id BIGSERIAL PRIMARY KEY,
    query_id UUID NOT NULL,
    platform_filter TEXT,
    result_count INTEGER NOT NULL,
    elapsed_ms INTEGER NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
