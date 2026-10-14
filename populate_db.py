import argparse
import os
import queue
import threading
from concurrent.futures import ThreadPoolExecutor, wait
from io import BytesIO
from pathlib import Path
from time import sleep, time
from typing import Any

import db
import embedder
import numpy as np
import requests
from PIL import Image
from tqdm import tqdm

URL = "https://www.finn.no/map/podium-resource/content/api/search/recommerce/SEARCH_ID_BAP_COMMON"
ROWS = 300
SUB_CATEGORY = "1.78.5196"
AREA_BBOX = "10.4155815206,59.6841435864,11.0874753077,60.1184878011"
IMAGE_DIR = "images"
MAX_IMAGES_PER_LISTING = 2
DOWNLOAD_WORKERS = int(os.getenv("DOWNLOAD_WORKERS", "4"))
REQUEST_INTERVAL = float(os.getenv("REQUEST_INTERVAL", "0.5"))
EMBED_QUEUE_SIZE = int(os.getenv("EMBED_QUEUE_SIZE", "64"))
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "32"))
blank_or_solid_skipped_count = 0
blank_or_solid_skipped_lock = threading.Lock()
_thread_local = threading.local()


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
    return get_thread_session().get(url, **kwargs)


def get_thread_session() -> requests.Session:
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = requests.Session()
        _thread_local.session = session
    return session


def download_and_enqueue(
    listing_id: int,
    source_id: str,
    image_urls: list[str],
    image_dir: str,
    embed_queue: queue.Queue,
    embed_pbar: tqdm | None,
    pbar_lock: threading.Lock,
    shutdown: threading.Event,
) -> None:
    for idx, image_url in enumerate(image_urls):
        if shutdown.is_set():
            return
        try:
            resp = get_thread_session().get(image_url, timeout=10)
            resp.raise_for_status()
            img = Image.open(BytesIO(resp.content)).convert("RGB")
            if is_blank_or_solid_color(img):
                increment_blank_or_solid_skipped_count()
                continue
            local_path = f"{image_dir}/{source_id}_{idx}.jpg"
            with open(local_path, "wb") as f:
                f.write(resp.content)
            embed_queue.put((listing_id, image_url, local_path, img))
        except Exception as e:
            print(f"Error downloading image {image_url}: {e}")
        finally:
            if embed_pbar is not None:
                with pbar_lock:
                    embed_pbar.update(1)


def is_blank_or_solid_color(img: Image.Image) -> bool:
    arr = np.array(img.convert("L"))
    return float(arr.std()) < 8.0


def increment_blank_or_solid_skipped_count() -> None:
    global blank_or_solid_skipped_count
    with blank_or_solid_skipped_lock:
        blank_or_solid_skipped_count += 1


def parse_listing_price(item: dict[str, Any]) -> int | None:
    trade_type = str(item.get("trade_type") or "").strip()
    raw_price = (item.get("price") or {}).get("amount")
    if trade_type == "Gis bort":
        return 0
    if raw_price in (None, 0):
        return None
    return int(raw_price)


def cleanup_listing_files(image_dir: str, source_id: str) -> None:
    image_root = Path(image_dir)
    for idx in range(MAX_IMAGES_PER_LISTING):
        existing_file = image_root / f"{source_id}_{idx}.jpg"
        try:
            existing_file.unlink(missing_ok=True)
        except OSError as e:
            print(f"Error deleting stale image file {existing_file}: {e}")


def wipe_image_dir(image_dir: str) -> None:
    for entry in Path(image_dir).iterdir():
        try:
            if entry.is_file():
                entry.unlink()
        except OSError as e:
            print(f"Error deleting image file {entry}: {e}")


def embed_worker(
    embed_queue: queue.Queue,
    batch_size: int,
) -> None:
    pending_items: list[tuple[int, str, str, Image.Image]] = []
    embedded_images = 0

    def flush_pending() -> None:
        nonlocal embedded_images
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
            embedded_images += db.insert_listing_images(rows)
        except Exception as e:
            print(f"Error embedding image batch ({len(pending_items)} images): {e}")
        finally:
            pending_items.clear()

    while True:
        item: Any = embed_queue.get()
        if item is None:
            flush_pending()
            break
        pending_items.append(item)
        if len(pending_items) >= batch_size:
            flush_pending()

    print(f"Embedding summary: stored={embedded_images} image(s)")


def reembed_saved_images(batch_size: int) -> None:
    with db.pool.connection() as conn:
        saved_images = db.get_images_for_reembedding(conn)
        if not saved_images:
            print("No saved image records to re-embed.")
            return
        missing_paths = [
            local_path for _, local_path in saved_images
            if not local_path or not Path(local_path).is_file()
        ]
        if missing_paths:
            raise FileNotFoundError(
                f"Cannot re-embed: {len(missing_paths)} image file(s) missing "
                f"(first: {missing_paths[0]!r}). Existing embeddings are unchanged."
            )
        with tqdm(total=len(saved_images), desc="Re-embedding images") as progress:
            for start in range(0, len(saved_images), batch_size):
                batch = saved_images[start : start + batch_size]
                images = []
                for _, local_path in batch:
                    with Image.open(local_path) as image:
                        images.append(image.convert("RGB"))
                embeddings = embedder.embed_images(images, batch_size=batch_size)
                rows = [
                    (embedding, image_id)
                    for (image_id, _), embedding in zip(batch, embeddings, strict=True)
                ]
                updated = db.update_image_embeddings(conn, rows)
                if updated != len(batch):
                    raise RuntimeError("Image records changed during re-embedding; rolling back.")
                progress.update(updated)
    print(f"Re-embedding complete: stored={len(saved_images)} image(s). Listings and files preserved.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Finn recommerce listings and embed images.")
    parser.add_argument(
        "--max-listings",
        type=int,
        default=None,
        help="Optional cap for total listings processed in this run (default: no cap).",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--clean-run",
        action="store_true",
        help="Delete all listings/images in DB and all local files in images/ before ingesting.",
    )
    mode.add_argument(
        "--reembed-only",
        action="store_true",
        help="Replace embeddings from saved image files without scraping or deleting listings/files.",
    )
    args = parser.parse_args()
    if args.reembed_only and args.max_listings is not None:
        parser.error("--max-listings cannot be combined with --reembed-only")
    if DOWNLOAD_WORKERS < 1:
        raise SystemExit("DOWNLOAD_WORKERS must be >= 1")
    if REQUEST_INTERVAL < 0:
        raise SystemExit("REQUEST_INTERVAL must be >= 0")
    if EMBED_QUEUE_SIZE < 1:
        raise SystemExit("EMBED_QUEUE_SIZE must be >= 1")
    if EMBED_BATCH_SIZE < 1:
        raise SystemExit("EMBED_BATCH_SIZE must be >= 1")

    shutdown = threading.Event()

    Path(IMAGE_DIR).mkdir(parents=True, exist_ok=True)
    try:
        db.pool.open()
        db.pool.wait()
    except Exception as e:
        print(f"Error opening database: {e}")
        raise SystemExit(1) from e

    if args.reembed_only:
        try:
            reembed_saved_images(EMBED_BATCH_SIZE)
        finally:
            db.pool.close()
        return

    if args.clean_run:
        db.recreate_schema()
        wipe_image_dir(IMAGE_DIR)
        print("Clean run: recreated DB tables and wiped local image files.")

    finn_rate = RateLimiter(REQUEST_INTERVAL)
    embed_queue: queue.Queue = queue.Queue(maxsize=EMBED_QUEUE_SIZE)
    pbar_lock = threading.Lock()
    listings_total = args.max_listings if args.max_listings is not None else None
    listings_pbar = tqdm(total=listings_total, desc="Listings", position=0)
    embed_pbar = tqdm(desc="Image downloads", position=1)
    embed_thread = threading.Thread(
        target=embed_worker,
        args=(embed_queue, EMBED_BATCH_SIZE),
        daemon=True,
    )
    embed_thread.start()

    processed = 0
    new_listings = 0
    stop_paging = False
    known_result_size: int | None = None
    newest_pass_source_ids: set[str] = set()
    sort_passes = ["NEWEST", "PUBLISHED_ASC"]
    # Track the seen ids and oldest timestamp reached. Because
    # Finn is ordered newest->oldest, unseen DB rows newer than that cutoff can be
    # marked stale.
    seen_source_ids: set[str] | None = set()
    oldest_seen_timestamp: int | None = None
    interrupted = False

    dl_pool: ThreadPoolExecutor | None = None
    try:
        dl_pool = ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS)
        for pass_idx, sort_key in enumerate(sort_passes):
            if stop_paging or shutdown.is_set():
                break
            pass_seen_first_ids: set[str] = set()
            reached_middle = False
            page = 1
            while page <= 50 and not stop_paging and not shutdown.is_set():
                params: list[tuple[str, Any]] = [
                    ("page", page),
                    ("rows", ROWS),
                    ("sub_category", SUB_CATEGORY),
                    ("sort", sort_key),
                    ("bbox", AREA_BBOX),
                    ("trade_type", 1),
                    ("trade_type", 2),
                ]
                r = http_get(finn_rate, URL, params=params, timeout=10)
                if shutdown.is_set():
                    break
                r.raise_for_status()
                data = r.json()
                finn_match_count = (data.get("metadata") or {}).get("result_size", {}).get("match_count")
                if isinstance(finn_match_count, int) and finn_match_count > 0:
                    known_result_size = finn_match_count
                if page == 1 and pass_idx == 0 and args.max_listings is None and known_result_size is not None:
                    with pbar_lock:
                        listings_pbar.total = known_result_size
                        listings_pbar.refresh()

                docs = data.get("docs", [])
                if not docs:
                    break

                first_source_id = str(docs[0].get("ad_id") or docs[0].get("id"))
                if first_source_id in pass_seen_first_ids:
                    print(
                        f"Paging stopped: sort={sort_key} page={page} repeated first listing id."
                    )
                    break
                pass_seen_first_ids.add(first_source_id)

                if sort_key == "PUBLISHED_ASC":
                    current_ids = {str(item.get("ad_id") or item.get("id")) for item in docs}
                    if current_ids & newest_pass_source_ids:
                        reached_middle = True

                page_futures = []
                download_jobs: list[tuple[int, str, list[str]]] = []
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
                    if args.max_listings is not None and processed >= args.max_listings:
                        stop_paging = True
                        break

                    source_id = str(item.get("ad_id") or item.get("id"))
                    if sort_key == "NEWEST":
                        newest_pass_source_ids.add(source_id)
                    if seen_source_ids is not None:
                        seen_source_ids.add(source_id)
                    title = item.get("heading", "").strip()
                    price = parse_listing_price(item)
                    description = item.get("description")
                    location = item.get("location")
                    url = item.get("canonical_url")
                    image_urls = item.get("image_urls") or (
                        [item["image"]["url"]] if item.get("image", {}).get("url") else []
                    )
                    image_urls = image_urls[:MAX_IMAGES_PER_LISTING]

                    processed += 1
                    with pbar_lock:
                        listings_pbar.update(1)
                        listings_pbar.set_postfix_str(
                            f"sort={sort_key} new={new_listings} latest={source_id}",
                            refresh=False,
                        )

                    listing_state = db.upsert_listing_and_embedding_status(
                        title=title,
                        price=price,
                        description=description,
                        url=url,
                        location=location,
                        source_id=source_id,
                        source_timestamp=item_ts,
                    )
                    if listing_state is None:
                        continue
                    listing_id, inserted, has_embeddings = listing_state
                    should_enqueue_images = False
                    if not inserted:
                        if not has_embeddings:
                            db.delete_listing_images_for_listing(listing_id)
                            cleanup_listing_files(IMAGE_DIR, source_id)
                            should_enqueue_images = True
                    else:
                        new_listings += 1
                        should_enqueue_images = True

                    if should_enqueue_images:
                        if image_urls and listing_id is not None:
                            download_jobs.append((listing_id, source_id, image_urls))

                for listing_id, source_id, image_urls in download_jobs:
                    page_futures.append(
                        dl_pool.submit(
                            download_and_enqueue,
                            listing_id,
                            source_id,
                            image_urls,
                            IMAGE_DIR,
                            embed_queue,
                            embed_pbar,
                            pbar_lock,
                            shutdown,
                        )
                    )

                if shutdown.is_set():
                    for future in page_futures:
                        future.cancel()
                else:
                    wait(page_futures)

                if stop_paging or (
                    args.max_listings is not None and processed >= args.max_listings
                ):
                    stop_paging = True
                    break

                if reached_middle:
                    print(
                        f"Reverse pass reached overlap with NEWEST at sort={sort_key} page={page}; stopping pass."
                    )
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
        print(f"Listings summary: scanned={processed}, new={new_listings}")
        print(
            "Image filter summary: "
            f"blank/solid={blank_or_solid_skipped_count}, "
            f"total={blank_or_solid_skipped_count}"
        )
        db.pool.close()
        listings_pbar.close()
        embed_pbar.close()


if __name__ == "__main__":
    main()
