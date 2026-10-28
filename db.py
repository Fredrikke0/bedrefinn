from psycopg_pool import ConnectionPool
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from dotenv import load_dotenv
import os
from pathlib import Path
import numpy as np
from collections.abc import Sequence
from psycopg import Connection, sql
from embedder import MATCH_THRESHOLD

load_dotenv()

pool = ConnectionPool(
    os.getenv("DATABASE_URL"),
    configure=register_vector,
    min_size=1,
    max_size=10,
    open=False
)

def get_all_listings():
    with pool.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SELECT id, title, price FROM listings ORDER BY created_at DESC")
            return cur.fetchall()


def upsert_category(category_id: str, name: str) -> None:
    with pool.connection() as conn:
        conn.execute(
            "INSERT INTO categories (id, name) VALUES (%s, %s) "
            "ON CONFLICT (id) DO UPDATE SET name = EXCLUDED.name",
            (category_id, name),
        )


def get_categories():
    with pool.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SELECT id, name FROM categories ORDER BY name")
            return cur.fetchall()


def get_random_image(category_ids: Sequence[str] | None = None, free_only: bool = False):
    with pool.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                WITH seed AS (
                    SELECT l.id, l.title FROM listings l
                    WHERE (%s::text[] IS NULL OR l.category_id = ANY(%s))
                      AND (NOT %s OR l.price = 0)
                      AND EXISTS (
                          SELECT 1 FROM listing_images li
                          WHERE li.listing_id = l.id AND li.embedding IS NOT NULL
                      )
                    ORDER BY random() LIMIT 1
                )
                SELECT photo.id AS image_id, seed.title, photo.local_path, photo.image_url
                FROM seed CROSS JOIN LATERAL (
                    SELECT li.id, li.local_path, li.image_url FROM listing_images li
                    WHERE li.listing_id = seed.id AND li.embedding IS NOT NULL
                    ORDER BY random() LIMIT 1
                ) photo
                """,
                (list(category_ids) if category_ids else None,
                 list(category_ids) if category_ids else None, free_only),
            )
            return cur.fetchone()


def get_image_embedding(image_id: int):
    with pool.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT li.listing_id, li.embedding, l.url FROM listing_images li "
                "JOIN listings l ON l.id = li.listing_id "
                "WHERE li.id = %s AND li.embedding IS NOT NULL",
                (image_id,),
            )
            return cur.fetchone()


def search_by_embedding(
    embedding: np.ndarray,
    limit: int = 10,
    free_only: bool = False,
    min_score: float = MATCH_THRESHOLD,
    exclude_listing_id: int | None = None,
    text_embedding: np.ndarray | None = None,
    category_ids: Sequence[str] | None = None,
):
    """Find visual matches, optionally reranking up to 300 listings by text."""
    parameters = [embedding]
    if category_ids:
        parameters.append(list(category_ids))
    parameters.extend((min_score, 300 if text_embedding is not None else limit))
    if text_embedding is not None:
        parameters.extend((text_embedding, limit))
    with pool.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                sql.SQL("""
                WITH distances AS MATERIALIZED (
                    SELECT li.id AS image_id, li.listing_id,
                              li.embedding <=> %s AS distance
                    FROM listing_images li {price_join}
                          WHERE li.embedding IS NOT NULL {category_filter} {price_filter} {exclude_filter}
                ), best_images AS (
                    SELECT DISTINCT ON (listing_id) image_id, listing_id, distance
                    FROM distances ORDER BY listing_id, distance
                ), matches AS (
                    SELECT *, 1 - distance AS similarity FROM best_images
                    WHERE 1 - distance >= %s
                    ORDER BY similarity DESC LIMIT %s
                    )
                SELECT l.id, li.id AS image_id, l.title, li.image_url, li.local_path, l.price, l.url,
                      l.category_id, c.name AS category_name, matches.similarity {text_score}
                FROM matches JOIN listings l ON l.id = matches.listing_id
                JOIN listing_images li ON li.id = matches.image_id
                  LEFT JOIN categories c ON c.id = l.category_id
                ORDER BY {result_order} {result_limit};
                """).format(
                    text_score=sql.SQL(
                        ", 1 - (li.embedding <=> %s) AS text_similarity"
                        if text_embedding is not None else ""
                    ),
                    result_order=sql.SQL(
                        "text_similarity DESC, matches.similarity DESC, l.id"
                        if text_embedding is not None else "matches.similarity DESC"
                    ),
                    result_limit=sql.SQL("LIMIT %s" if text_embedding is not None else ""),
                    price_join=sql.SQL(
                        "JOIN listings eligible ON eligible.id = li.listing_id"
                        if free_only or category_ids else ""
                    ),
                    category_filter=sql.SQL(
                        "AND eligible.category_id = ANY(%s)" if category_ids else ""
                    ),
                    price_filter=sql.SQL("AND eligible.price = 0" if free_only else ""),
                    exclude_filter=(
                        sql.SQL("AND li.listing_id <> {}").format(sql.Literal(exclude_listing_id))
                        if exclude_listing_id is not None else sql.SQL("")
                    ),
                ),
                parameters
            )
            return cur.fetchall()

def upsert_listing_and_embedding_status(
    title: str,
    price: int | None,
    description: str | None = None,
    url: str | None = None,
    location: str | None = None,
    source_id: str | None = None,
    source_timestamp: int | None = None,
    category_id: str | None = None,
) -> tuple[int, bool, bool] | None:
    """Insert listing if new; always return (listing_id, inserted, has_embeddings)."""
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                WITH ins AS (
                    INSERT INTO listings (
                        title,
                        price,
                        description,
                        url,
                        location,
                        source_id,
                        source_timestamp
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (source_id) DO NOTHING
                    RETURNING id
                )
                SELECT id, TRUE AS inserted, FALSE AS has_embeddings
                FROM ins
                UNION ALL
                SELECT
                    l.id,
                    FALSE AS inserted,
                    EXISTS(
                        SELECT 1
                        FROM listing_images li
                        WHERE li.listing_id = l.id
                          AND li.embedding IS NOT NULL
                        LIMIT 1
                    ) AS has_embeddings
                FROM listings l
                WHERE l.source_id = %s
                  AND NOT EXISTS (SELECT 1 FROM ins)
                LIMIT 1
                """,
                (
                    title,
                    price,
                    description,
                    url,
                    location,
                    source_id,
                    source_timestamp,
                    source_id,
                ),
            )
            row = cur.fetchone()
            if row is None:
                return None
            if category_id is not None:
                cur.execute(
                    "UPDATE listings SET category_id = %s WHERE id = %s",
                    (category_id, row[0]),
                )
            return int(row[0]), bool(row[1]), bool(row[2])

def delete_listing_images_for_listing(listing_id: int) -> int:
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM listing_images WHERE listing_id = %s", (listing_id,))
            return cur.rowcount

def insert_listing_images(
    rows: Sequence[tuple[int, str, str, np.ndarray]],
) -> int:
    """Insert multiple listing images in one transaction and return inserted row count."""
    if not rows:
        return 0
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO listing_images (listing_id, image_url, local_path, embedding) VALUES (%s, %s, %s, %s)",
                rows,
            )
    return len(rows)


def get_images_for_reembedding(conn: Connection) -> list[tuple[int, str | None]]:
    with conn.cursor() as cur:
        cur.execute("SELECT id, local_path FROM listing_images ORDER BY id")
        return cur.fetchall()


def update_image_embeddings(
    conn: Connection,
    rows: Sequence[tuple[np.ndarray, int]],
) -> int:
    with conn.cursor() as cur:
        cur.executemany(
            "UPDATE listing_images SET embedding = %s WHERE id = %s", rows
        )
        return cur.rowcount


def recreate_schema() -> None:
    """Drop and recreate tables from schema.sql."""
    schema_path = Path(__file__).with_name("schema.sql")
    schema_sql = schema_path.read_text(encoding="utf-8")
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS listing_images CASCADE")
            cur.execute("DROP TABLE IF EXISTS listings CASCADE")
            cur.execute(schema_sql)


def ensure_schema() -> None:
    with pool.connection() as conn:
        conn.execute(Path(__file__).with_name("schema.sql").read_text(encoding="utf-8"))


def mark_stale_listings_since_timestamp(
    seen_source_ids: Sequence[str],
    cutoff_source_timestamp: int,
    include_cutoff: bool = True,
    category_id: str | None = None,
) -> tuple[int, int]:
    """Mark unseen listings as stale for timestamps >= cutoff; revive seen listings.

    This supports partial full-scan runs: if the API is ordered newest->oldest and
    the scan has reached `cutoff_source_timestamp`, then any DB listing at or newer
    than that timestamp that was not seen can be treated as stale.

    If `include_cutoff` is False, only rows strictly newer than the cutoff are
    considered stale candidates (useful for interrupted scans).

    Returns (newly_stale_count, revived_count).
    """
    seen_list = list(seen_source_ids)
    cutoff_operator = ">=" if include_cutoff else ">"
    category_filter = " AND category_id = %s" if category_id is not None else ""
    category_parameters = (category_id,) if category_id is not None else ()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE listings SET is_stale = TRUE, updated_at = NOW() "
                "WHERE is_stale = FALSE "
                "AND source_timestamp IS NOT NULL "
                f"AND source_timestamp {cutoff_operator} %s "
                "AND NOT (source_id = ANY(%s))" + category_filter,
                (cutoff_source_timestamp, seen_list) + category_parameters,
            )
            newly_stale = cur.rowcount or 0
            cur.execute(
                "UPDATE listings SET is_stale = FALSE, updated_at = NOW() "
                "WHERE is_stale = TRUE AND source_id = ANY(%s)" + category_filter,
                (seen_list,) + category_parameters,
            )
            revived = cur.rowcount or 0
    return newly_stale, revived