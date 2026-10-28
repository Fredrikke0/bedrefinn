import argparse
import hashlib
import json
import os
import queue
import sys
import threading
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import closing, contextmanager
from pathlib import Path
from time import perf_counter, sleep
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import requests
from tqdm import tqdm

URL = "https://www.finn.no/map/podium-resource/content/api/search/recommerce/SEARCH_ID_BAP_COMMON"
ROWS = 300
CATEGORY = "0.78"
EXCLUDED_SUB_CATEGORIES = {"1.78.5222", "1.78.5223"}
MAX_PAGES = 50
AREA_BBOX = "10.4155815206,59.6841435864,11.0874753077,60.1184878011"
IMAGE_DIR = "images"
MAX_IMAGES_PER_LISTING = 2
DOWNLOAD_WORKERS = int(os.getenv("DOWNLOAD_WORKERS", "16"))
DOWNLOAD_MAX_PENDING = int(os.getenv("DOWNLOAD_MAX_PENDING", str(DOWNLOAD_WORKERS * 4)))
IMAGE_SESSION_MAX_REQUESTS = int(os.getenv("IMAGE_SESSION_MAX_REQUESTS", "0"))
IMAGE_RENDITION = os.getenv("IMAGE_RENDITION", "320w")
REQUEST_INTERVAL = float(os.getenv("REQUEST_INTERVAL", "0.25"))
EMBED_QUEUE_SIZE = int(os.getenv("EMBED_QUEUE_SIZE", "64"))
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "32"))
blank_or_solid_skipped_count = 0
blank_or_solid_skipped_lock = threading.Lock()
_thread_local = threading.local()
_profile = None


class IngestionProfile:
    SAMPLE_LIMIT = 4096
    HTTP_SAMPLE_LIMIT = 8
    HTTP_SLOW_HEADERS_SECONDS = 0.25
    HTTP_HEADERS = frozenset({
        "age", "cache-control", "cache-status", "cdn-cache-control",
        "cf-cache-status", "cf-ray", "connection", "content-encoding",
        "content-length", "content-type", "date", "expires", "retry-after",
        "server", "server-timing", "surrogate-control", "vary", "via",
        "x-amz-cf-pop", "x-cache", "x-cache-hits", "x-cdn", "x-proxy-cache",
        "x-served-by", "x-timer", "x-varnish",
    })

    def __init__(self, output: str):
        self.output = Path(output)
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.started = perf_counter()
        self.window_started = self.started
        self.lock = threading.Lock()
        self.totals = {}
        self.window = {}
        self.counters = {}
        self.window_counters = {}
        self.snapshots = []
        self.http_samples = {name: deque(maxlen=self.HTTP_SAMPLE_LIMIT) for name in ("fast", "slow", "error")}
        self.window_http_samples = {name: deque(maxlen=self.HTTP_SAMPLE_LIMIT) for name in self.http_samples}
        self.metadata = {
            "download_workers": DOWNLOAD_WORKERS,
            "download_max_pending": DOWNLOAD_MAX_PENDING,
            "image_session_max_requests": IMAGE_SESSION_MAX_REQUESTS,
            "image_rendition": IMAGE_RENDITION,
            "embed_batch_size": EMBED_BATCH_SIZE,
            "embed_queue_size": EMBED_QUEUE_SIZE,
            "request_interval_seconds": REQUEST_INTERVAL,
        }

    def record(self, stage: str, seconds: float) -> None:
        with self.lock:
            for target in (self.totals, self.window):
                entry = target.setdefault(stage, {
                    "count": 0, "total_seconds": 0.0, "max_seconds": 0.0,
                    "samples": deque(maxlen=self.SAMPLE_LIMIT),
                })
                entry["count"] += 1
                entry["total_seconds"] += seconds
                entry["max_seconds"] = max(entry["max_seconds"], seconds)
                entry["samples"].append(seconds)

    def increment(self, name: str, count: int = 1) -> None:
        with self.lock:
            for target in (self.counters, self.window_counters):
                target[name] = target.get(name, 0) + count

    def record_http_response(self, url, response, header_seconds, body_seconds, total_seconds, error_type):
        status = response.status_code if response is not None else None
        bucket = "error" if error_type or (status is not None and status >= 400) else (
            "slow" if header_seconds is not None and header_seconds >= self.HTTP_SLOW_HEADERS_SECONDS else "fast"
        )
        sample = {
            "elapsed_seconds": perf_counter() - self.started,
            "host": urlsplit(url).hostname or "unknown",
            "url_sha256": hashlib.sha256(url.encode()).hexdigest(),
            "status": status, "error_type": error_type,
            "header_seconds": header_seconds, "body_seconds": body_seconds,
            "total_seconds": total_seconds,
            "headers": {
                name.lower(): str(value)[:256]
                for name, value in (response.headers.items() if response is not None else ())
                if name.lower() in self.HTTP_HEADERS
            },
        }
        with self.lock:
            self.http_samples[bucket].append(sample)
            self.window_http_samples[bucket].append(sample)
            for counters in (self.counters, self.window_counters):
                key = f"image_http_class.{bucket}"
                counters[key] = counters.get(key, 0) + 1

    @staticmethod
    def summarize(stages):
        result = {}
        for name, entry in stages.items():
            samples = sorted(entry["samples"])
            middle = len(samples) // 2
            median = (samples[middle] + samples[(len(samples) - 1) // 2]) / 2
            result[name] = {
                key: entry[key] for key in ("count", "total_seconds", "max_seconds")
            } | {
                "median_seconds": median,
                "p95_seconds": samples[max(0, (95 * len(samples) + 99) // 100 - 1)],
                "percentile_sample_count": len(samples),
            }
        return result

    def snapshot(self, *, final: bool = False, **state) -> None:
        with self.lock:
            now = perf_counter()
            duration = now - self.window_started
            self.snapshots.append({
                "elapsed_seconds": now - self.started,
                "window_seconds": duration,
                "state": state,
                "counters": dict(self.window_counters),
                "rates_per_second": {
                    name: count / duration if duration else 0.0
                    for name, count in self.window_counters.items()
                    if name in ("listings", "downloaded_images", "stored_images")
                },
                "stages": self.summarize(self.window),
                "http_response_samples": {name: list(samples) for name, samples in self.window_http_samples.items()},
            })
            report = {
                "version": 1, "final": final,
                "elapsed_seconds": now - self.started,
                "metadata": self.metadata,
                "notes": [
                    "Concurrent and nested stages overlap; do not sum into wall-clock percentages.",
                    "Counts, totals and maxima are exact; percentiles use the latest 4096 observations per stage.",
                    "Windows attribute operations when they finish, not when they start.",
                    "Worker decode includes blank filtering; preprocessing prepares SigLIP2 image tensors.",
                    "First preparation submission includes lazy worker-pool startup; later submissions include dispatch/start acknowledgement.",
                    "DB call durations include connection acquisition and transaction commit.",
                    "HTTP durations include response-body download; errors are counted separately.",
                    "Image header wait includes connection setup, redirects and server wait, not an isolated DNS/TLS measurement.",
                    "HTTP response samples retain the latest 8 per fast/slow/error class, overall and per window; not a random sample.",
                    "Slow means header wait >= 0.25 seconds; errors include HTTP failures and body-read failures.",
                    "Only allowlisted response headers are retained, values truncated to 256 characters; image URLs are SHA256 hashes.",
                ],
                "counters": dict(self.counters),
                "stages": self.summarize(self.totals),
                "http_response_samples": {name: list(samples) for name, samples in self.http_samples.items()},
                "windows": list(self.snapshots),
            }
            self.window.clear()
            self.window_counters.clear()
            for samples in self.window_http_samples.values():
                samples.clear()
            self.window_started = now
        temporary = self.output.with_name(self.output.name + ".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.output)


@contextmanager
def profile_stage(name: str):
    profile = _profile
    if profile is None:
        yield
        return
    started = perf_counter()
    try:
        yield
    finally:
        profile.record(name, perf_counter() - started)


def profile_count(name: str, count: int = 1) -> None:
    if _profile is not None:
        _profile.increment(name, count)


def profile_call(stage: str, function, *args, **kwargs):
    with profile_stage(stage):
        return function(*args, **kwargs)


class DownloadScheduler:
    def __init__(self, executor, max_pending: int, shutdown: threading.Event):
        if max_pending < 1:
            raise ValueError("Download pending limit must be >= 1")
        self.executor = executor
        self.max_pending = max_pending
        self.shutdown = shutdown
        self.pending = set()

    def submit(self, *args) -> None:
        while not self.shutdown.is_set():
            self.pending = {future for future in self.pending if not future.done()}
            if len(self.pending) < self.max_pending:
                self.pending.add(self.executor.submit(*args))
                return
            profile_call("download_backpressure_wait", wait, self.pending, return_when=FIRST_COMPLETED)

    def drain_page(self) -> None:
        profile_call("page_download_jobs_wait", wait, self.pending)
        self.pending.clear()


class RateLimiter:
    """Serialize HTTP calls with a minimum interval between requests."""

    def __init__(self, min_interval_s: float):
        self._min_interval_s = min_interval_s
        self._lock = threading.Lock()
        self._last = 0.0

    def wait_turn(self) -> None:
        with self._lock:
            now = perf_counter()
            wait_s = self._min_interval_s - (now - self._last)
            # Reserve this request slot while holding the lock, but sleep outside the lock
            # so other threads can compute their own wait windows concurrently.
            self._last = now + max(wait_s, 0)
        if wait_s > 0:
            sleep(wait_s)


def http_get(rate: RateLimiter, url: str, **kwargs) -> requests.Response:
    profile_call("api_rate_limit_wait", rate.wait_turn)
    return profile_call("api_http", get_thread_session().get, url, **kwargs)


def get_thread_session(*, for_images: bool = False, profile: IngestionProfile | None = None) -> requests.Session:
    name = "image_session" if for_images else "session"
    session = getattr(_thread_local, name, None)
    if (for_images and session is not None and IMAGE_SESSION_MAX_REQUESTS > 0
            and getattr(_thread_local, "image_requests", 0) >= IMAGE_SESSION_MAX_REQUESTS):
        session.close()
        session = None
        if profile is not None:
            profile.increment("image_sessions_renewed")
    if session is None:
        session = requests.Session()
        setattr(_thread_local, name, session)
        if for_images:
            _thread_local.image_requests = 0
            if profile is not None:
                profile.increment("image_sessions_created")
    if for_images:
        _thread_local.image_requests += 1
    return session


def fetch_image_bytes(image_url: str, profile: IngestionProfile | None = None) -> bytes:
    profile = profile if profile is not None else _profile
    host = urlsplit(image_url).hostname or "unknown"
    started = perf_counter()
    response = None
    header_seconds = None
    body_seconds = None
    error_type = None
    try:
        response = get_thread_session(for_images=True, profile=profile).get(image_url, timeout=10, stream=True)
        header_seconds = perf_counter() - started
        if profile is not None:
            profile.record("image_http_headers", header_seconds)
            profile.record(f"image_http_host.{host}.headers", header_seconds)
            profile.increment(f"image_http_status.{response.status_code}")
            profile.increment("image_http_redirects", len(response.history))
            if response.headers.get("Connection", "").lower() == "close":
                profile.increment("image_http_connection_close")
            if response.headers.get("Retry-After"):
                profile.increment("image_http_retry_after")
        response.raise_for_status()
        body_started = perf_counter()
        try:
            return response.content
        finally:
            body_seconds = perf_counter() - body_started
            if profile is not None:
                profile.record("image_http_body", body_seconds)
                profile.record(f"image_http_host.{host}.body", body_seconds)
    except Exception as error:
        error_type = type(error).__name__
        if profile is not None:
            profile.increment(f"image_http_error.{type(error).__name__}")
            profile.increment(f"image_http_host.{host}.errors")
        raise
    finally:
        if response is not None:
            response.close()
        if profile is not None:
            elapsed = perf_counter() - started
            profile.record("image_http", elapsed)
            profile.record(f"image_http_host.{host}.total", elapsed)
            profile.record_http_response(image_url, response, header_seconds, body_seconds, elapsed, error_type)


def get_subcategories(rate: RateLimiter) -> list[dict[str, Any]]:
    response = http_get(
        rate, URL,
        params=[("category", CATEGORY), ("bbox", AREA_BBOX),
                ("trade_type", 1), ("trade_type", 2), ("rows", 1)],
        timeout=10,
    )
    response.raise_for_status()

    def find_category(nodes):
        for node in nodes:
            if node.get("name") == "category" and node.get("value") == CATEGORY:
                return node
            found = find_category(node.get("filter_items", []))
            if found is not None:
                return found
        return None

    category = find_category(profile_call("api_json", response.json).get("filters", []))
    if category is None:
        raise RuntimeError("FINN did not return the furniture category tree")
    categories = [
        node for node in category.get("filter_items", [])
        if node.get("name") == "sub_category"
        and node.get("value") not in EXCLUDED_SUB_CATEGORIES
    ]
    if not categories:
        raise RuntimeError("FINN did not return any eligible subcategories")
    return categories


def iter_category_pages(rate: RateLimiter, category_id: str, shutdown: threading.Event):
    newest_ids: set[str] = set()
    for sort_key in ("NEWEST", "PUBLISHED_ASC"):
        first_ids: set[str] = set()
        for page in range(1, MAX_PAGES + 1):
            if shutdown.is_set():
                return
            response = http_get(
                rate, URL,
                params=[("page", page), ("rows", ROWS),
                        ("sub_category", category_id), ("sort", sort_key),
                        ("bbox", AREA_BBOX), ("trade_type", 1), ("trade_type", 2)],
                timeout=10,
            )
            response.raise_for_status()
            data = profile_call("api_json", response.json)
            docs = data.get("docs", [])
            if not docs:
                return
            first_id = str(docs[0].get("ad_id") or docs[0].get("id"))
            if first_id in first_ids:
                raise RuntimeError(
                    f"Incomplete scan: category={category_id} sort={sort_key} "
                    f"page={page} repeated its first listing"
                )
            first_ids.add(first_id)
            source_ids = {str(item.get("ad_id") or item.get("id")) for item in docs}
            overlap = sort_key == "PUBLISHED_ASC" and bool(source_ids & newest_ids)
            if sort_key == "NEWEST":
                newest_ids.update(source_ids)
            yield sort_key, docs
            if shutdown.is_set():
                return
            total = (data.get("metadata") or {}).get("result_size", {}).get("match_count")
            if overlap or len(docs) < ROWS:
                return
            if sort_key == "NEWEST" and isinstance(total, int) and len(newest_ids) >= total:
                return
    raise RuntimeError(
        f"Incomplete scan: category={category_id} reached the pagination limit "
        "without overlap; split this category into smaller searches"
    )


def image_download_url(url: str) -> str:
    if IMAGE_RENDITION not in ("default", "320w"):
        raise ValueError("IMAGE_RENDITION must be default or 320w")
    if IMAGE_RENDITION == "default":
        return url
    parsed = urlsplit(url)
    prefix = "/dynamic/default/"
    if parsed.scheme != "https" or parsed.netloc != "images.finncdn.no" or not parsed.path.startswith(prefix):
        return url
    return urlunsplit(parsed._replace(path=f"/dynamic/{IMAGE_RENDITION}/" + parsed.path[len(prefix):]))


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
            download_url = image_download_url(image_url)
            if download_url != image_url:
                profile_count("image_rendition_rewrites")
            image_url = download_url
            content = fetch_image_bytes(image_url)
            local_path = f"{image_dir}/{source_id}_{idx}.jpg"
            with profile_stage("image_file_write"):
                with open(local_path, "wb") as f:
                    f.write(content)
            profile_count("downloaded_images")
            profile_count("downloaded_bytes", len(content))
            profile_call("download_queue_put_wait", embed_queue.put, (listing_id, image_url, local_path))
        except Exception as e:
            profile_count("download_errors")
            tqdm.write(f"Error downloading image {image_url}: {e}", file=sys.stderr)
        finally:
            if embed_pbar is not None:
                with pbar_lock:
                    embed_pbar.update(1)


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
            tqdm.write(f"Error deleting stale image file {existing_file}: {e}", file=sys.stderr)


def wipe_image_dir(image_dir: str) -> None:
    for entry in Path(image_dir).iterdir():
        try:
            if entry.is_file():
                entry.unlink()
        except OSError as e:
            tqdm.write(f"Error deleting image file {entry}: {e}", file=sys.stderr)


def embed_worker(
    embed_queue: queue.Queue,
    batch_size: int,
) -> None:
    with profile_stage("embedding_imports"):
        import db
        import embedder

    if _profile is not None:
        _profile.metadata.update({
            "preprocess_workers": embedder.PREPROCESS_WORKERS,
            "model_id": embedder.model_id,
            "coreml_compute_units": os.getenv("BEDREFIN_COREML_COMPUTE_UNITS", "ALL"),
        })

    pending_items: list[tuple[int, str, str]] = []
    prepared_batches = deque()
    embedded_images = 0
    first_submission = True

    def submit_pending() -> None:
        nonlocal first_submission
        if not pending_items:
            return
        try:
            paths = [local_path for _, _, local_path in pending_items]
            stage = "prepare_first_submit_and_startup" if first_submission else "prepare_submit"
            first_submission = False
            future = profile_call(stage, embedder.submit_image_paths,
                paths, skip_blank=True,
            )
            prepared_batches.append((list(pending_items), future))
        except Exception as error:
            profile_count("preparation_errors")
            tqdm.write(f"Error preparing image batch ({len(pending_items)} images): {error}", file=sys.stderr)
        finally:
            pending_items.clear()

    def flush_prepared() -> None:
        nonlocal embedded_images
        batch, future = prepared_batches.popleft()
        try:
            prepared = profile_call("prepare_result_wait", future.result)
            pixel_values, _, _, kept_indices, blank_indices, decode_errors = prepared
            if _profile is not None:
                _profile.record("worker_decode_and_filter", prepared[1])
                _profile.record("worker_preprocess", prepared[2])
            profile_count("blank_images", len(blank_indices))
            profile_count("decode_errors", len(decode_errors))
            for message in decode_errors:
                tqdm.write(message, file=sys.stderr)
            kept = set(kept_indices)
            for _ in blank_indices:
                increment_blank_or_solid_skipped_count()
            for index, (_, _, local_path) in enumerate(batch):
                if index not in kept:
                    try:
                        profile_call("rejected_file_cleanup", Path(local_path).unlink, missing_ok=True)
                    except OSError as error:
                        tqdm.write(f"Error removing rejected image {local_path}: {error}", file=sys.stderr)
            if not kept_indices:
                return
            embeddings = profile_call("siglip2_inference", embedder.embed_prepared_images, pixel_values)
            rows = [
                (listing_id, image_url, local_path, embedding)
                for (listing_id, image_url, local_path), embedding in zip(
                    (batch[index] for index in kept_indices), embeddings, strict=True
                )
            ]
            stored = profile_call("image_db_insert", db.insert_listing_images, rows)
            embedded_images += stored
            profile_count("stored_images", stored)
            profile_count("embedded_batches")
        except Exception as e:
            profile_count("embedding_errors")
            tqdm.write(f"Error embedding image batch ({len(batch)} images): {e}", file=sys.stderr)

    while True:
        try:
            item: Any = profile_call("embed_queue_get_wait", embed_queue.get, block=not prepared_batches)
        except queue.Empty:
            flush_prepared()
            continue
        if item is None:
            submit_pending()
            break
        pending_items.append(item)
        if len(pending_items) >= batch_size:
            submit_pending()
            if len(prepared_batches) >= embedder.PREPROCESS_WORKERS + 1:
                flush_prepared()

    while prepared_batches:
        flush_prepared()

    tqdm.write(f"Embedding summary: stored={embedded_images} image(s)", file=sys.stderr)


def reembed_saved_images(batch_size: int) -> None:
    import db
    import embedder

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
        paths = [local_path for _, local_path in saved_images]
        with (
            tqdm(total=len(saved_images), desc="Re-embedding images") as progress,
            closing(embedder.iter_prepared_image_batches(paths, batch_size)) as batches,
        ):
            for batch_index, prepared in enumerate(batches):
                start = batch_index * batch_size
                batch = saved_images[start : start + batch_size]
                embeddings = embedder.embed_prepared_images(prepared.pixel_values)
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
        "--profile-output", metavar="PATH",
        help="Write per-page ingestion timings and final totals as JSON (normal downloads only).",
    )
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
        parser.error("--max-listings cannot be combined with embedding-only modes")
    if args.profile_output and args.reembed_only:
        parser.error("--profile-output is only supported for normal downloads")
    if DOWNLOAD_WORKERS < 1:
        raise SystemExit("DOWNLOAD_WORKERS must be >= 1")
    if DOWNLOAD_MAX_PENDING < 1:
        raise SystemExit("DOWNLOAD_MAX_PENDING must be >= 1")
    if IMAGE_SESSION_MAX_REQUESTS < 0:
        raise SystemExit("IMAGE_SESSION_MAX_REQUESTS must be >= 0")
    if IMAGE_RENDITION not in ("default", "320w"):
        raise SystemExit("IMAGE_RENDITION must be default or 320w")
    if REQUEST_INTERVAL < 0:
        raise SystemExit("REQUEST_INTERVAL must be >= 0")
    if EMBED_QUEUE_SIZE < 1:
        raise SystemExit("EMBED_QUEUE_SIZE must be >= 1")
    if EMBED_BATCH_SIZE < 1:
        raise SystemExit("EMBED_BATCH_SIZE must be >= 1")

    global _profile
    _profile = IngestionProfile(args.profile_output) if args.profile_output else None
    try:
        if _profile is not None:
            _profile.snapshot(phase="startup")
        run_ingestion(args)
    finally:
        if _profile is not None:
            try:
                outcome = "error" if sys.exc_info()[0] else "completed"
                if _profile.counters.get("interrupts"):
                    outcome = "interrupted"
                if _profile.counters.get("embedding_drain_timeouts"):
                    outcome = "embedding_drain_timeout"
                _profile.snapshot(final=True, phase="finished", outcome=outcome)
                tqdm.write(f"Ingestion profile: {_profile.output}", file=sys.stderr)
            finally:
                _profile = None


def run_ingestion(args) -> None:
    with profile_stage("model_and_db_import"):
        import db

    shutdown = threading.Event()

    Path(IMAGE_DIR).mkdir(parents=True, exist_ok=True)
    try:
        with profile_stage("db_startup"):
            db.pool.open()
            db.pool.wait()
            db.ensure_schema()
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
    try:
        categories = get_subcategories(finn_rate)
        for category in categories:
            profile_call("category_db_upsert", db.upsert_category, category["value"], category["display_name"])
    except Exception:
        db.pool.close()
        raise
    print(f"Scanning {len(categories)} subcategories; kitchenware and decorations excluded.")
    embed_queue: queue.Queue = queue.Queue(maxsize=EMBED_QUEUE_SIZE)
    pbar_lock = threading.Lock()
    listings_total = args.max_listings
    if listings_total is None and all(isinstance(category.get("hits"), int) for category in categories):
        listings_total = sum(category["hits"] for category in categories)
    listings_pbar = tqdm(
        total=listings_total, desc="Listings", position=0,
        dynamic_ncols=True, leave=False, disable=None, file=sys.stderr,
    )
    embed_pbar = tqdm(
        desc="Image downloads", position=1,
        dynamic_ncols=True, leave=False, disable=None, file=sys.stderr,
    )
    embed_thread = threading.Thread(
        target=embed_worker,
        args=(embed_queue, EMBED_BATCH_SIZE),
        daemon=True,
    )
    embed_thread.start()

    processed = 0
    new_listings = 0
    stop_paging = False
    seen_run_ids: set[str] = set()
    completed_scans: list[tuple[str, set[str], int | None]] = []

    dl_pool: ThreadPoolExecutor | None = None
    try:
        dl_pool = ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS)
        downloads = DownloadScheduler(dl_pool, DOWNLOAD_MAX_PENDING, shutdown)
        for category in categories:
            if stop_paging or shutdown.is_set():
                break
            category_id = category["value"]
            seen_source_ids: set[str] = set()
            oldest_seen_timestamp: int | None = None
            tqdm.write(f"Scanning {category['display_name']} ({category_id})", file=sys.stderr)
            for sort_key, docs in iter_category_pages(finn_rate, category_id, shutdown):
                for item in docs:
                    if shutdown.is_set():
                        stop_paging = True
                        break
                    if args.max_listings is not None and processed >= args.max_listings:
                        stop_paging = True
                        break

                    source_id = str(item.get("ad_id") or item.get("id"))
                    seen_source_ids.add(source_id)
                    if source_id in seen_run_ids:
                        continue
                    seen_run_ids.add(source_id)
                    item_ts = item.get("timestamp")
                    if item_ts is not None and (
                        oldest_seen_timestamp is None or item_ts < oldest_seen_timestamp
                    ):
                        oldest_seen_timestamp = item_ts
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
                    profile_count("listings")
                    with pbar_lock:
                        listings_pbar.update(1)
                        listings_pbar.set_postfix_str(
                            f"category={category_id} sort={sort_key} new={new_listings}",
                            refresh=False,
                        )

                    listing_state = profile_call("listing_db_upsert", db.upsert_listing_and_embedding_status,
                        title=title,
                        price=price,
                        description=description,
                        url=url,
                        location=location,
                        source_id=source_id,
                        source_timestamp=item_ts,
                        category_id=category_id,
                    )
                    if listing_state is None:
                        continue
                    listing_id, inserted, has_embeddings = listing_state
                    if inserted:
                        new_listings += 1
                        profile_count("new_listings")
                    elif not has_embeddings:
                        profile_call("listing_db_image_delete", db.delete_listing_images_for_listing, listing_id)
                        profile_call("listing_file_cleanup", cleanup_listing_files, IMAGE_DIR, source_id)

                    if image_urls and (inserted or not has_embeddings):
                        downloads.submit(
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

                if getattr(args, "download_page_barrier", False):
                    downloads.drain_page()

                if _profile is not None:
                    _profile.snapshot(
                        phase="page", category_id=category_id, sort=sort_key,
                        scanned=processed, new_listings=new_listings,
                        queue_depth=embed_queue.qsize(),
                        download_pending=sum(not future.done() for future in downloads.pending),
                    )

                if stop_paging or (
                    args.max_listings is not None and processed >= args.max_listings
                ):
                    stop_paging = True
                    break

            if not stop_paging and not shutdown.is_set():
                completed_scans.append((category_id, seen_source_ids, oldest_seen_timestamp))

    except KeyboardInterrupt:
        shutdown.set()
        profile_count("interrupts")
        tqdm.write("Interrupted, shutting down...", file=sys.stderr)

    finally:
        if dl_pool is not None:
            # Wait for in-flight downloads so the embedder is not stopped while work remains queued.
            profile_call("download_shutdown_wait", dl_pool.shutdown, wait=True, cancel_futures=shutdown.is_set())
        profile_call("shutdown_queue_put_wait", embed_queue.put, None)
        profile_call("embedding_drain_wait", embed_thread.join, timeout=600)
        if embed_thread.is_alive():
            profile_count("embedding_drain_timeouts")
        embed_pbar.close()
        listings_pbar.close()
        for category_id, seen_source_ids, oldest_seen_timestamp in completed_scans:
            if seen_source_ids and oldest_seen_timestamp is not None:
                newly_stale, revived = profile_call("stale_db_check", db.mark_stale_listings_since_timestamp,
                    list(seen_source_ids),
                    oldest_seen_timestamp,
                    category_id=category_id,
                )
                print(
                    f"Stale check ({category_id}): {newly_stale} marked stale, "
                    f"{revived} revived (saw {len(seen_source_ids)} active)."
                )
        print(f"Listings summary: scanned={processed}, new={new_listings}")
        print(
            "Image filter summary: "
            f"blank/solid={blank_or_solid_skipped_count}, "
            f"total={blank_or_solid_skipped_count}"
        )
        profile_call("db_close", db.pool.close)


if __name__ == "__main__":
    main()
