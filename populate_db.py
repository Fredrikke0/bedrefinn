import argparse
import queue
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait
from io import BytesIO
from pathlib import Path
from time import sleep, time
from typing import Any

import db
import embedder
import imagehash
import requests
from PIL import Image
from tqdm import tqdm

URL = "https://www.finn.no/map/podium-resource/content/api/search/recommerce/SEARCH_ID_BAP_COMMON"
ROWS = 200
SUB_CATEGORY = "1.78.5196"
AREA_BBOX = "10.4155815206,59.6841435864,11.0874753077,60.1184878011"
IMAGE_DIR = "images"
MAX_IMAGES_PER_LISTING = 2
PHASH_DUPLICATE_THRESHOLD = 8
seen_hashes: set[imagehash.ImageHash] = set()
seen_hashes_lock = threading.Lock()
dedup_skipped_count = 0
dedup_skipped_lock = threading.Lock()


class RateLimiter:
    """Serialize HTTP calls with a minimum interval between requests."""

    def __init__(self, min_interval_s: float):
        self._min_interval_s = min_interval_s
        self._lock = threading.Lock()
        self._last = 0.0

    def wait_turn(self) -> None:
        with self._lock:
            now = time()
            wait_s = self._min_interval_s - (now - self._last)
            # Reserve this request slot while holding the lock, but sleep outside the lock
            # so other threads can compute their own wait windows concurrently.
            self._last = max(now, self._last) + max(wait_s, 0)
        if wait_s > 0:
            sleep(wait_s)


def http_get(rate: RateLimiter, url: str, **kwargs) -> requests.Response:
    rate.wait_turn()
    return requests.get(url, **kwargs)


def download_and_enqueue(
    listing_id: int,
    source_id: str,
    image_urls: list[str],
    image_dir: str,
    embed_queue: queue.Queue,
    shutdown: threading.Event,
) -> None:
    for idx, image_url in enumerate(image_urls):
        if shutdown.is_set():
            return
        try:
            resp = requests.get(image_url, timeout=10)
            resp.raise_for_status()
            img = Image.open(BytesIO(resp.content)).convert("RGB")
            if is_duplicate(img):
                increment_dedup_skipped_count()
                continue
            local_path = f"{image_dir}/{source_id}_{idx}.jpg"
            with open(local_path, "wb") as f:
                f.write(resp.content)
            embed_queue.put((listing_id, image_url, local_path, img))
        except Exception as e:
            print(f"Error downloading image {image_url}: {e}")


def is_duplicate(img: Image.Image, threshold: int = PHASH_DUPLICATE_THRESHOLD) -> bool:
    h = imagehash.phash(img)
    with seen_hashes_lock:
        for seen in seen_hashes:
            if h - seen < threshold:
                return True
        seen_hashes.add(h)
    return False


def increment_dedup_skipped_count() -> None:
    global dedup_skipped_count
    with dedup_skipped_lock:
        dedup_skipped_count += 1


def cleanup_listing_files(image_dir: str, source_id: str) -> None:
    pattern = f"{source_id}_*.jpg"
    for existing_file in Path(image_dir).glob(pattern):
        try:
            existing_file.unlink()
        except OSError as e:
            print(f"Error deleting stale image file {existing_file}: {e}")


def wipe_image_dir(image_dir: str) -> None:
    for entry in Path(image_dir).iterdir():
        try:
            if entry.is_file():
                entry.unlink()
        except OSError as e:
            print(f"Error deleting image file {entry}: {e}")


def redownload_existing_listing_images(
    source_id: str,
    image_urls: list[str],
    existing_rows: list[dict],
    image_dir: str,
    shutdown: threading.Event,
) -> None:
    rows_by_url: dict[str, list[dict]] = {}
    for row in existing_rows:
        image_url = row.get("image_url")
        if not image_url:
            continue
        rows_by_url.setdefault(image_url, []).append(row)

    for idx, image_url in enumerate(image_urls):
        if shutdown.is_set():
            return
        matching_rows = rows_by_url.get(image_url)
        if not matching_rows:
            continue
        row = matching_rows.pop()

        local_path = f"{image_dir}/{source_id}_{idx}.jpg"
        try:
            resp = requests.get(image_url, timeout=10)
            resp.raise_for_status()
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            with open(local_path, "wb") as f:
                f.write(resp.content)
            if row.get("local_path") != local_path:
                db.update_listing_image_local_path(row["id"], local_path)
        except Exception as e:
            print(f"Error re-downloading image {image_url}: {e}")


def embed_worker(
    embed_queue: queue.Queue,
    pbar: tqdm | None,
    pbar_lock: threading.Lock,
    batch_size: int,
) -> None:
    pending_items: list[tuple[int, str, str, Image.Image]] = []

    def flush_pending() -> None:
        if not pending_items:
            return
        try:
            images = [img for _, _, _, img in pending_items]
            embeddings = embedder.embed_images(images, batch_size=batch_size)
            rows = [
                (listing_id, image_url, local_path, embedding)
                for (listing_id, image_url, local_path, _), embedding in zip(
                    pending_items, embeddings, strict=True
                )
            ]
            db.insert_listing_images(rows)
        except Exception as e:
            print(f"Error embedding image batch ({len(pending_items)} images): {e}")
        finally:
            if pbar is not None:
                with pbar_lock:
                    pbar.update(len(pending_items))
            pending_items.clear()

    while True:
        item: Any = embed_queue.get()
        if item is None:
            flush_pending()
            break
        pending_items.append(item)
        if len(pending_items) >= batch_size:
            flush_pending()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Finn recommerce listings and embed images.")
    parser.add_argument(
        "--max-listings",
        type=int,
        default=None,
        help="Optional cap for new listings in this run (default: no cap).",
    )
    parser.add_argument(
        "--download-workers",
        type=int,
        default=4,
        help="Parallel download threads (default: 4).",
    )
    parser.add_argument(
        "--request-interval",
        type=float,
        default=0.5,
        help="Minimum seconds between outbound HTTP requests (default: 0.5).",
    )
    parser.add_argument(
        "--embed-queue-size",
        type=int,
        default=32,
        help="Max PIL images queued for embedding; bounds RAM (default: 32).",
    )
    parser.add_argument(
        "--embed-batch-size",
        type=int,
        default=32,
        help="Batch size for image embedding and DB image inserts (default: 32).",
    )
    parser.add_argument(
        "--full-scan",
        action="store_true",
        help="Ignore stored timestamps; fetch from newest until --max-listings new rows.",
    )
    parser.add_argument(
        "--wipe-images-and-redownload-from-db",
        action="store_true",
        help=(
            "Delete local files in images/ and force re-download using fresh listing image URLs for existing listings "
            "encountered in this run."
        ),
    )
    args = parser.parse_args()
    if args.embed_batch_size < 1:
        raise SystemExit("--embed-batch-size must be >= 1")

    shutdown = threading.Event()

    Path(IMAGE_DIR).mkdir(parents=True, exist_ok=True)
    force_redownload_existing = args.wipe_images_and_redownload_from_db
    if force_redownload_existing:
        wipe_image_dir(IMAGE_DIR)
        print(f"Wiped local image files in {IMAGE_DIR}/")

    try:
        db.pool.open()
        db.pool.wait()
    except Exception as e:
        print(f"Error opening database: {e}")
        raise SystemExit(1) from e

    since_ts: int | None = None
    if not args.full_scan and not force_redownload_existing:
        since_ts = db.get_max_source_timestamp()
        if since_ts is not None:
            print(f"Incremental mode: Finn doc timestamp > {since_ts} (newest stored in DB).")

    finn_rate = RateLimiter(args.request_interval)
    embed_queue: queue.Queue = queue.Queue(maxsize=args.embed_queue_size)
    pbar_lock = threading.Lock()
    listings_total = args.max_listings if args.max_listings is not None else None
    listings_pbar = tqdm(total=listings_total, desc="New listings", position=0)
    embed_pbar = tqdm(desc="Embedding images", position=1)
    embed_thread = threading.Thread(
        target=embed_worker,
        args=(embed_queue, embed_pbar, pbar_lock, args.embed_batch_size),
        daemon=True,
    )
    embed_thread.start()

    page = 1
    processed = 0
    stop_paging = False
    # For --full-scan we track the seen ids and oldest timestamp reached. Because
    # Finn is ordered newest->oldest, unseen DB rows newer than that cutoff can be
    # marked stale.
    seen_source_ids: set[str] | None = set() if args.full_scan else None
    oldest_seen_timestamp: int | None = None
    interrupted = False

    dl_pool: ThreadPoolExecutor | None = None
    try:
        dl_pool = ThreadPoolExecutor(max_workers=args.download_workers)
        while not stop_paging and not shutdown.is_set():
            params: dict = {
                "page": page,
                "rows": ROWS,
                "sub_category": SUB_CATEGORY,
                "sort": "NEWEST",
                "bbox": AREA_BBOX,
            }
            if since_ts is not None:
                # Finn may accept this for server-side filtering; client-side cutoff in any case.
                params["timestamp"] = since_ts

            r = http_get(finn_rate, URL, params=params, timeout=10)
            if shutdown.is_set():
                break
            r.raise_for_status()
            data = r.json()
            if page == 1 and args.max_listings is None:
                finn_match_count = (
                    (data.get("metadata") or {})
                    .get("result_size", {})
                    .get("match_count")
                )
                if isinstance(finn_match_count, int) and finn_match_count > 0:
                    with pbar_lock:
                        listings_pbar.total = finn_match_count
                        listings_pbar.refresh()
            docs = data.get("docs", [])
            if not docs:
                break

            page_futures = []
            reached_old = False
            for item in docs:
                if shutdown.is_set():
                    stop_paging = True
                    break
                item_ts = item.get("timestamp")
                if (
                    oldest_seen_timestamp is None
                    or (item_ts is not None and item_ts < oldest_seen_timestamp)
                ):
                    oldest_seen_timestamp = item_ts
                if (
                    since_ts is not None
                    and item_ts is not None
                    and item_ts <= since_ts
                ):
                    reached_old = True
                    break
                if args.max_listings is not None and processed >= args.max_listings:
                    stop_paging = True
                    break

                source_id = str(item.get("ad_id") or item.get("id"))
                if seen_source_ids is not None:
                    seen_source_ids.add(source_id)
                title = item.get("heading", "").strip()
                price = (item.get("price") or {}).get("amount")
                description = item.get("description")
                location = item.get("location")
                url = item.get("canonical_url")
                image_urls = item.get("image_urls") or (
                    [item["image"]["url"]] if item.get("image", {}).get("url") else []
                )
                image_urls = image_urls[:MAX_IMAGES_PER_LISTING]

                listing_id = db.insert_listing(
                    title=title,
                    price=price,
                    description=description,
                    url=url,
                    location=location,
                    source_id=source_id,
                    source_timestamp=item_ts,
                )
                should_enqueue_images = False
                if listing_id is None:
                    existing_listing_id = db.get_listing_id_by_source_id(source_id)
                    if existing_listing_id is None:
                        continue

                    if force_redownload_existing:
                        existing_rows = db.get_listing_images_for_listing(existing_listing_id)
                        existing_urls = [
                            row["image_url"] for row in existing_rows if row.get("image_url")
                        ]
                        if Counter(existing_urls) == Counter(image_urls):
                            cleanup_listing_files(IMAGE_DIR, source_id)
                            page_futures.append(
                                dl_pool.submit(
                                    redownload_existing_listing_images,
                                    source_id,
                                    image_urls,
                                    existing_rows,
                                    IMAGE_DIR,
                                    shutdown,
                                )
                            )
                            continue

                    if not db.listing_has_embeddings(existing_listing_id):
                        db.delete_listing_images_for_listing(existing_listing_id)
                        cleanup_listing_files(IMAGE_DIR, source_id)
                        listing_id = existing_listing_id
                        should_enqueue_images = True
                else:
                    processed += 1
                    should_enqueue_images = True
                    with pbar_lock:
                        listings_pbar.update(1)
                        listings_pbar.set_postfix_str(f"latest={source_id}", refresh=False)

                if should_enqueue_images:
                    with pbar_lock:
                        embed_pbar.total = (embed_pbar.total or 0) + len(image_urls)
                        embed_pbar.refresh()
                if image_urls:
                    if listing_id is not None and should_enqueue_images:
                        page_futures.append(
                            dl_pool.submit(
                                download_and_enqueue,
                                listing_id,
                                source_id,
                                image_urls,
                                IMAGE_DIR,
                                embed_queue,
                                shutdown,
                            )
                        )

            if shutdown.is_set():
                for future in page_futures:
                    future.cancel()
            else:
                wait(page_futures)

            if reached_old or stop_paging or (
                args.max_listings is not None and processed >= args.max_listings
            ):
                stop_paging = True
                break

            page += 1

    except KeyboardInterrupt:
        interrupted = True
        shutdown.set()
        print("Interrupted, shutting down...")

    finally:
        if dl_pool is not None:
            # Wait for in-flight downloads so the embedder is not stopped while work remains queued.
            dl_pool.shutdown(wait=True, cancel_futures=shutdown.is_set())
        embed_queue.put(None)
        embed_thread.join(timeout=600)
        if seen_source_ids is not None and oldest_seen_timestamp is not None:
            if seen_source_ids:
                newly_stale, revived = db.mark_stale_listings_since_timestamp(
                    list(seen_source_ids),
                    oldest_seen_timestamp,
                    include_cutoff=not interrupted,
                )
                cutoff_mode = ">=" if not interrupted else ">"
                print(
                    f"Stale check: {newly_stale} listing(s) marked stale, "
                    f"{revived} revived (saw {len(seen_source_ids)} active, "
                    f"cutoff{cutoff_mode}{oldest_seen_timestamp})."
                )
            else:
                # No listings returned by the API at all.
                print("Stale check skipped: scan returned 0 listings.")
        print(f"Perceptual dedup skipped {dedup_skipped_count} image(s).")
        db.pool.close()
        listings_pbar.close()
        embed_pbar.close()


if __name__ == "__main__":
    main()
