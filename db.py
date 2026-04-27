from psycopg_pool import ConnectionPool
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from dotenv import load_dotenv
import os
import numpy as np
from collections.abc import Sequence
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

def get_max_source_timestamp() -> int | None:
    """Latest Finn API doc timestamp (ms) stored in DB, or None if empty / no values."""
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT MAX(source_timestamp) FROM listings WHERE source_timestamp IS NOT NULL"
            )
            row = cur.fetchone()
            if row is None or row[0] is None:
                return None
            return int(row[0])

def search_by_embedding(
    embedding: np.ndarray,
    limit: int = 10,
    free_only: bool = False,
    min_score: float = MATCH_THRESHOLD,
):
    with pool.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                WITH ranked AS (
                    SELECT
                        l.id, l.title, li.image_url, li.local_path, l.price, l.url,
                        1 - (li.embedding <=> %s) AS similarity,
                        ROW_NUMBER() OVER (
                        PARTITION BY l.id
                        ORDER BY li.embedding <=> %s
                        ) AS rn
                    FROM listing_images li
                    JOIN listings l ON l.id = li.listing_id
                    WHERE (%s = FALSE OR l.price = 0)
                    )
                    SELECT id, title, image_url, local_path, price, url, similarity
                    FROM ranked
                    WHERE rn = 1
                    AND similarity >= %s
                    ORDER BY similarity DESC
                    LIMIT %s;
                """,
                (embedding, embedding, free_only, min_score, limit)
            )
            return cur.fetchall()

def insert_listing(
    title: str,
    price: int,
    description: str | None = None,
    url: str | None = None,
    location: str | None = None,
    source_id: str | None = None,
    source_timestamp: int | None = None,
)->int:
    with pool.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "INSERT INTO listings (title, price, description, url, location, source_id, source_timestamp) VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT (source_id) DO NOTHING RETURNING id",
                (title, price, description, url, location, source_id, source_timestamp),
            )
            row = cur.fetchone()
            return row["id"] if row else None

def get_listing_id_by_source_id(source_id: str) -> int | None:
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM listings WHERE source_id = %s", (source_id,))
            row = cur.fetchone()
            return int(row[0]) if row and row[0] is not None else None

def count_listing_images_for_listing(listing_id: int) -> int:
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM listing_images WHERE listing_id = %s", (listing_id,))
            row = cur.fetchone()
            return int(row[0]) if row and row[0] is not None else 0

def delete_listing_images_for_listing(listing_id: int) -> int:
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM listing_images WHERE listing_id = %s", (listing_id,))
            return cur.rowcount

def get_listing_images_for_listing(listing_id: int) -> list[dict]:
    with pool.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT id, image_url, local_path FROM listing_images WHERE listing_id = %s ORDER BY id",
                (listing_id,),
            )
            return cur.fetchall()

def update_listing_image_local_path(image_id: int, local_path: str) -> None:
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE listing_images SET local_path = %s WHERE id = %s",
                (local_path, image_id),
            )

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

def mark_stale_listings(seen_source_ids: Sequence[str]) -> tuple[int, int]:
    """Flag listings missing from `seen_source_ids` as stale; revive any seen listings.

    Intended to run after a complete scan of the upstream catalog. Performs the
    update in a single transaction so the DB never exposes a half-applied state.

    Returns (newly_stale_count, revived_count).
    """
    seen_list = list(seen_source_ids)
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE listings SET is_stale = TRUE, updated_at = NOW() "
                "WHERE is_stale = FALSE AND NOT (source_id = ANY(%s))",
                (seen_list,),
            )
            newly_stale = cur.rowcount or 0
            cur.execute(
                "UPDATE listings SET is_stale = FALSE, updated_at = NOW() "
                "WHERE is_stale = TRUE AND source_id = ANY(%s)",
                (seen_list,),
            )
            revived = cur.rowcount or 0
    return newly_stale, revived


def mark_stale_listings_since_timestamp(
    seen_source_ids: Sequence[str],
    cutoff_source_timestamp: int,
    include_cutoff: bool = True,
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
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE listings SET is_stale = TRUE, updated_at = NOW() "
                "WHERE is_stale = FALSE "
                "AND source_timestamp IS NOT NULL "
                f"AND source_timestamp {cutoff_operator} %s "
                "AND NOT (source_id = ANY(%s))",
                (cutoff_source_timestamp, seen_list),
            )
            newly_stale = cur.rowcount or 0
            cur.execute(
                "UPDATE listings SET is_stale = FALSE, updated_at = NOW() "
                "WHERE is_stale = TRUE AND source_id = ANY(%s)",
                (seen_list,),
            )
            revived = cur.rowcount or 0
    return newly_stale, revived