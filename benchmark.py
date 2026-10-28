"""Saved-image benchmarks and bounded live-ingestion experiments."""

import argparse
import hashlib
import json
import os
import platform
import shutil
import statistics
import sys
import threading
import types
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from unittest.mock import patch
from uuid import uuid4

import psycopg
from dotenv import load_dotenv
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict
from psycopg.types.json import Jsonb

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "benchmark_data"
TEST_DATABASE = "bedrefinn_benchmark"


def connection_settings() -> tuple[dict, dict]:
    load_dotenv(ROOT / ".env")
    source = conninfo_to_dict(os.getenv(
        "DATABASE_URL", "postgresql://postgres:dev@localhost:5432/bedrefinn"
    ))
    if source.get("dbname") == TEST_DATABASE:
        raise ValueError("DATABASE_URL must point to the source, not the benchmark database")
    source["connect_timeout"] = "5"
    target = {**source, "dbname": TEST_DATABASE}
    return source, target


def create_dataset(limit: int) -> None:
    source_settings, target_settings = connection_settings()
    if ARTIFACTS.exists():
        raise FileExistsError(f"Refusing to overwrite {ARTIFACTS}")
    with psycopg.connect(**source_settings) as source:
        source.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
        listings = source.execute(
            "SELECT l.* FROM listings l WHERE EXISTS ("
            "SELECT 1 FROM listing_images li WHERE li.listing_id = l.id "
            "AND li.local_path IS NOT NULL) ORDER BY l.id LIMIT %s", (limit,)
        )
        listing_columns = [column.name for column in listings.description]
        listing_rows = listings.fetchall()
        if len(listing_rows) != limit:
            raise ValueError(f"Requested {limit} listings; found only {len(listing_rows)}")
        listing_ids = [row[listing_columns.index("id")] for row in listing_rows]
        image_rows = source.execute(
            "SELECT id, listing_id, image_url, local_path FROM listing_images "
            "WHERE listing_id = ANY(%s) ORDER BY listing_id, id", (listing_ids,)
        ).fetchall()
        paths = [ROOT / row[3] if row[3] else None for row in image_rows]
        missing = [path for path in paths if path is None or not path.is_file()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} source images missing; first: {missing[0]}")
        with psycopg.connect(**source_settings, autocommit=True) as admin:
            admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(TEST_DATABASE)))
        ARTIFACTS.mkdir()
        copied_rows = []
        digest = hashlib.sha256()
        for row, path in zip(image_rows, paths, strict=True):
            destination = ARTIFACTS / "images" / f"{row[0]}{path.suffix}"
            destination.parent.mkdir(exist_ok=True)
            shutil.copyfile(path, destination)
            digest.update(str(row[:3]).encode())
            digest.update(destination.read_bytes())
            copied_rows.append((*row[:3], str(destination.relative_to(ROOT))))
        metadata = {
            "source_database": source.info.dbname,
            "listings": len(listing_rows),
            "images": len(copied_rows),
            "sha256": digest.hexdigest(),
        }
        with psycopg.connect(**target_settings) as target:
            target.execute((ROOT / "schema.sql").read_text(encoding="utf-8"))
            with target.cursor() as cursor:
                cursor.executemany(
                    sql.SQL("INSERT INTO listings ({}) VALUES ({})").format(
                        sql.SQL(", ").join(map(sql.Identifier, listing_columns)),
                        sql.SQL(", ").join(sql.Placeholder() for _ in listing_columns),
                    ), listing_rows,
                )
                cursor.executemany(
                    "INSERT INTO listing_images (id, listing_id, image_url, local_path) "
                    "VALUES (%s, %s, %s, %s)", copied_rows,
                )
            for table in ("listings", "listing_images"):
                target.execute(sql.SQL(
                    "SELECT setval(pg_get_serial_sequence(%s, 'id'), "
                    "(SELECT max(id) FROM {}))"
                ).format(sql.Identifier(table)), (table,))
            target.execute("CREATE TABLE benchmark_dataset (metadata jsonb NOT NULL)")
            target.execute("INSERT INTO benchmark_dataset VALUES (%s)", (Jsonb(metadata),))
            target.execute("ANALYZE")
    print(json.dumps({"database": TEST_DATABASE, **metadata}, indent=2))


def run_benchmark(
    batch_size: int, repeats: int, max_images: int | None, output: Path,
    prefetch: bool = True,
    profile_stages: bool = False,
    coreml_batch_size: int = 32,
) -> None:
    if profile_stages:
        raise ValueError("PyTorch stage profiling is not available for Core ML")
    if coreml_batch_size not in (32, 64, 128):
        raise ValueError("Core ML batch size must be 32, 64, or 128")
    _, target_settings = connection_settings()
    with psycopg.connect(**target_settings) as connection:
        metadata = connection.execute("SELECT metadata FROM benchmark_dataset").fetchone()[0]
        if connection.execute(
            "SELECT count(*) FROM listing_images WHERE embedding IS NOT NULL"
        ).fetchone()[0]:
            raise ValueError("Expected an embedding-free benchmark database")
        rows = connection.execute(
            "SELECT id, local_path FROM listing_images ORDER BY listing_id, id LIMIT %s",
            (max_images,),
        ).fetchall()
        if not rows:
            raise ValueError("No images to benchmark")
        for _, local_path in rows:
            if not local_path or not (ROOT / local_path).is_file():
                raise FileNotFoundError(local_path)

        print(f"Loading model; {len(rows)} images, batch size {batch_size}", flush=True)
        started = perf_counter()
        import embedder
        import numpy as np
        import torch
        import transformers
        from pgvector.psycopg import register_vector

        embed_batch = embedder.embed_prepared_images
        if coreml_batch_size != 32:
            native_model = embedder._load_model(f"vision_{coreml_batch_size}.mlpackage")

            def embed_batch(pixel_values):
                if len(pixel_values) <= 32:
                    return embedder.embed_prepared_images(pixel_values)
                outputs = []
                for offset in range(0, len(pixel_values), coreml_batch_size):
                    batch = np.ascontiguousarray(
                        pixel_values[offset:offset + coreml_batch_size], dtype=np.float32,
                    )
                    if len(batch) <= 32:
                        outputs.append(embedder.embed_prepared_images(batch))
                    else:
                        padded = batch
                        if len(batch) < coreml_batch_size:
                            padded = np.pad(batch, (
                                (0, coreml_batch_size - len(batch)), (0, 0), (0, 0), (0, 0),
                            ))
                        outputs.append(embedder._normalize(native_model.predict({
                            "pixel_values": padded,
                        })["embeddings"][:len(batch)]))
                return np.concatenate(outputs)

        load_seconds = perf_counter() - started
        register_vector(connection)

        preparation_started = perf_counter()
        paths = [ROOT / local_path for _, local_path in rows]
        warmup_count = batch_size * (embedder.PREPROCESS_WORKERS if prefetch else 1)
        with closing(embedder.iter_prepared_image_batches(
            paths[:warmup_count], batch_size, prefetch=prefetch,
        )) as batches:
            warmup_batches = list(batches)
        preparation_startup_seconds = perf_counter() - preparation_started
        started = perf_counter()
        for warmup_batch in warmup_batches:
            embed_batch(warmup_batch.pixel_values)
        warmup_seconds = perf_counter() - started
        connection.rollback()
        measurements = []
        for repeat in range(repeats):
            decode_seconds = 0.0
            preprocess_seconds = 0.0
            prepare_wait_seconds = 0.0
            embed_seconds = 0.0
            write_seconds = 0.0
            batch_seconds = []
            started = perf_counter()
            try:
                with (
                    connection.cursor() as cursor,
                    closing(embedder.iter_prepared_image_batches(
                        paths, batch_size, prefetch=prefetch,
                    )) as batches,
                ):
                    for offset in range(0, len(rows), batch_size):
                        batch = rows[offset:offset + batch_size]
                        batch_started = perf_counter()
                        phase_started = perf_counter()
                        prepared = next(batches)
                        prepare_wait_seconds += perf_counter() - phase_started
                        decode_seconds += prepared.decode_seconds
                        preprocess_seconds += prepared.preprocess_seconds
                        phase_started = perf_counter()
                        embeddings = embed_batch(prepared.pixel_values)
                        embed_seconds += perf_counter() - phase_started
                        if (embeddings.shape != (len(batch), embedder.EMBEDDING_DIM)
                                or not np.isfinite(embeddings).all()
                                or not np.allclose(np.linalg.norm(embeddings, axis=1), 1, atol=1e-3)):
                            raise ValueError("Invalid embeddings: shape, finiteness, or normalization")
                        phase_started = perf_counter()
                        cursor.executemany(
                            "UPDATE listing_images SET embedding = %s WHERE id = %s",
                            [(embedding, row[0]) for embedding, row in zip(embeddings, batch, strict=True)],
                        )
                        if cursor.rowcount != len(batch):
                            raise RuntimeError("Image records changed during benchmark")
                        write_seconds += perf_counter() - phase_started
                        batch_seconds.append(perf_counter() - batch_started)
                elapsed = perf_counter() - started
            finally:
                connection.rollback()
            measurement = {
                "repeat": repeat + 1,
                "images": len(rows),
                "seconds": elapsed,
                "images_per_second": len(rows) / elapsed,
                "decode_seconds": decode_seconds,
                "preprocess_seconds": preprocess_seconds,
                "prepare_wait_seconds": prepare_wait_seconds,
                "embed_seconds": embed_seconds,
                "write_seconds": write_seconds,
                "median_batch_seconds": statistics.median(batch_seconds),
                "p95_batch_seconds": float(np.percentile(batch_seconds, 95)),
            }
            measurements.append(measurement)
            print(json.dumps(measurement), flush=True)
        result = {
            "dataset": metadata,
            "model": embedder.model_id,
            "device": embedder.device,
            "dtype": embedder.INFERENCE_DTYPE,
            "coreml_compute_units": embedder.compute_units,
            "coreml_inference_batch_size": coreml_batch_size,
            "mps_kernel_settings": None,
            "embedding_dim": embedder.EMBEDDING_DIM,
            "batch_size": batch_size,
            "prefetch": prefetch,
            "profile_stages": False,
            "profiling_note": None,
            "phase_timing": "decode/preprocess are CPU work times and may overlap inference; "
                            "prepare_wait is consumer wait; embed excludes preprocessing",
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "dependency_versions": {
                package: version(package)
                for package in (
                    "coremltools", "numpy", "pillow", "torch", "transformers",
                    "psycopg", "psycopg-pool", "pgvector", "tokenizers",
                )
            },
            "torch_threads": torch.get_num_threads(),
            "model_load_seconds": load_seconds,
            "preparation_startup_seconds": preparation_startup_seconds,
            "prefetch_executor": "persistent process" if prefetch else "none",
            "preprocess_workers": embedder.PREPROCESS_WORKERS if prefetch else 0,
            "warmup_seconds": warmup_seconds,
            "median_images_per_second": statistics.median(
                measurement["images_per_second"] for measurement in measurements
            ),
            "database_writes": "rolled back after each repeat; commit not measured",
            "measurements": measurements,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"Results: {output}")


def run_scaling_benchmark(listings: int, repeats: int, output: Path) -> None:
    import numpy as np
    from pgvector.psycopg import register_vector
    from psycopg_pool import ConnectionPool

    with patch.dict(sys.modules, {"embedder": types.SimpleNamespace(MATCH_THRESHOLD=0.0)}):
        import db

    _, target_settings = connection_settings()
    generator = np.random.default_rng(20261017)
    embeddings = generator.standard_normal((listings * 2, 768), dtype=np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    seed_embedding = embeddings[0]
    measurements = []
    with psycopg.connect(**target_settings, autocommit=True) as admin:
        for repeat in range(repeats):
            for existing in ((0, 10000) if repeat % 2 == 0 else (10000, 0)):
                schema = f"scaling_{uuid4().hex}"
                admin.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
                try:
                    settings = {**target_settings, "options": f"-c search_path={schema},public"}
                    with ConnectionPool(
                        kwargs=settings, configure=register_vector,
                        min_size=1, max_size=1,
                    ) as pool, patch.object(db, "pool", pool):
                        pool.wait()
                        with pool.connection() as connection:
                            connection.execute((ROOT / "schema.sql").read_text(encoding="utf-8"))
                            connection.execute("INSERT INTO categories VALUES ('scaling', 'Scaling test')")
                            connection.execute(
                                "INSERT INTO listings (title, price, source_id, category_id) "
                                "SELECT 'Synthetic listing ' || number, 100, 'seed-' || number, 'scaling' "
                                "FROM generate_series(1, %s) AS number", (existing,),
                            )
                            connection.execute(
                                "INSERT INTO listing_images (listing_id, image_url, embedding) "
                                "SELECT listings.id, 'https://example.invalid/image', %s::vector "
                                "FROM listings CROSS JOIN generate_series(1, 2)", (seed_embedding,),
                            )
                            connection.execute("ANALYZE listings")
                            connection.execute("ANALYZE listing_images")
                            actual_counts = connection.execute(
                                "SELECT (SELECT count(*) FROM listings), "
                                "(SELECT count(*) FROM listing_images)"
                            ).fetchone()
                            if actual_counts != (existing, existing * 2):
                                raise RuntimeError("Synthetic seed counts do not match")

                        upsert_seconds = 0.0
                        image_write_seconds = 0.0
                        upsert_latencies = []
                        started = perf_counter()
                        for start in range(0, listings, 300):
                            image_rows = []
                            for index in range(start, min(start + 300, listings)):
                                phase_started = perf_counter()
                                state = db.upsert_listing_and_embedding_status(
                                    title=f"New synthetic listing {index}", price=100,
                                    source_id=f"new-{index}", category_id="scaling",
                                )
                                latency = perf_counter() - phase_started
                                upsert_seconds += latency
                                upsert_latencies.append(latency)
                                if state is None or state[1:] != (True, False):
                                    raise RuntimeError(f"Incorrect new-listing status: {state}")
                                for image_index in range(2):
                                    image_rows.append((
                                        state[0], "https://example.invalid/image", None,
                                        embeddings[index * 2 + image_index],
                                    ))
                            for offset in range(0, len(image_rows), 32):
                                batch = image_rows[offset:offset + 32]
                                phase_started = perf_counter()
                                if db.insert_listing_images(batch) != len(batch):
                                    raise RuntimeError("Incorrect image insert count")
                                image_write_seconds += perf_counter() - phase_started
                        new_seconds = perf_counter() - started

                        started = perf_counter()
                        for index in range(listings):
                            source_id = (
                                f"seed-{index % existing + 1}" if existing else f"new-{index}"
                            )
                            state = db.upsert_listing_and_embedding_status(
                                title="Existing synthetic listing", price=100,
                                source_id=source_id, category_id="scaling",
                            )
                            if state is None or state[1:] != (False, True):
                                raise RuntimeError(f"Incorrect existing-listing status: {state}")
                        revisit_seconds = perf_counter() - started
                        with pool.connection() as connection:
                            final_counts = connection.execute(
                                "SELECT (SELECT count(*) FROM listings), "
                                "(SELECT count(*) FROM listing_images)"
                            ).fetchone()
                            if final_counts != (existing + listings, (existing + listings) * 2):
                                raise RuntimeError("Final counts do not match the workload")
                        measurement = {
                            "repeat": repeat + 1, "existing_listings": existing,
                            "existing_images": existing * 2, "new_listings": listings,
                            "new_seconds": new_seconds,
                            "new_listings_per_second": listings / new_seconds,
                            "upsert_seconds": upsert_seconds,
                            "upsert_median_ms": statistics.median(upsert_latencies) * 1000,
                            "upsert_p95_ms": float(np.percentile(upsert_latencies, 95)) * 1000,
                            "image_write_seconds": image_write_seconds,
                            "revisit_seconds": revisit_seconds,
                            "revisit_listings_per_second": listings / revisit_seconds,
                        }
                        measurements.append(measurement)
                        print(json.dumps(measurement), flush=True)
                finally:
                    admin.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))
        summaries = []
        for existing in (0, 10000):
            matching = [row for row in measurements if row["existing_listings"] == existing]
            summaries.append({
                "existing_listings": existing,
                **{key: statistics.median(row[key] for row in matching) for key in (
                    "new_listings_per_second", "upsert_median_ms", "upsert_p95_ms",
                    "image_write_seconds", "revisit_listings_per_second",
                )},
            })
        result = {
            "database": TEST_DATABASE, "embedding_dim": 768, "images_per_listing": 2,
            "scope": "Actual ingestion DB functions, per-listing commits, 32-image batch commits; "
                     "300-listing pages; preseeded logged tables with current schema/indexes and ANALYZE; "
                     "alternating empty/10000 order; no HTTP, decoding, inference, or queue waits. "
                     "Existing checks revisit seed rows, or newly inserted rows in the empty control. "
                     "Not a simulation of long-running process state or concurrent DB writers.",
            "isolation": "Unique disposable schemas in benchmark database; production not accessed; "
                         "schemas dropped after each measurement",
            "summaries": summaries, "measurements": measurements,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summaries, indent=2), flush=True)
        print(f"Results: {output}")


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def run_download_benchmark(max_images: int, repeats: int, output: Path) -> None:
    import requests
    import populate_db

    source_settings, _ = connection_settings()
    with psycopg.connect(**source_settings) as source:
        source.execute("SET TRANSACTION READ ONLY")
        urls = [row[0] for row in source.execute(
            "SELECT image_url FROM listing_images WHERE image_url IS NOT NULL "
            "GROUP BY image_url ORDER BY max(id) DESC LIMIT %s", (max_images,),
        ).fetchall()]
    if not urls:
        raise ValueError("No saved image URLs available for the download benchmark")
    if any(not url.startswith("https://") for url in urls):
        raise ValueError("The download benchmark requires HTTPS image URLs")
    result = {
        "scope": "HTTP-only replay of the same recent saved image URLs; no API scan, "
                 "file writes, embedding or database writes. Fresh thread pools per round; "
                 "sessions reused within each round. Repeated URLs can warm CDN caches. "
                 "Header wait combines DNS/connect/TLS/redirect/server latency; it does not isolate them. "
                 "No retries. Stop further requests/rounds on HTTP 429 or 503.",
        "images": len(urls), "requested_images": max_images, "repeats": repeats,
        "image_session_max_requests": populate_db.IMAGE_SESSION_MAX_REQUESTS,
        "url_sha256": hashlib.sha256("\n".join(urls).encode()).hexdigest(),
        "measurements": [], "summary": {}, "stopped_for_server_backoff": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = output.with_name(output.name + ".tmp")
        temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        temporary.replace(output)

    save()
    for repeat in range(repeats):
        for workers in ((4, 8) if repeat % 2 == 0 else (8, 4)):
            profile = populate_db.IngestionProfile(str(output))
            stop = threading.Event()

            def download(url):
                if stop.is_set():
                    profile.increment("skipped_after_backoff")
                    return
                try:
                    content = populate_db.fetch_image_bytes(url, profile)
                    profile.increment("downloaded_images")
                    profile.increment("downloaded_bytes", len(content))
                except requests.RequestException as error:
                    profile.increment("download_errors")
                    if error.response is not None and error.response.status_code in (429, 503):
                        stop.set()

            started = perf_counter()
            with ThreadPoolExecutor(max_workers=workers) as executor:
                list(executor.map(download, urls))
            elapsed = perf_counter() - started
            measurement = {
                "repeat": repeat + 1, "workers": workers, "seconds": elapsed,
                "images_per_second": profile.counters.get("downloaded_images", 0) / elapsed,
                "bytes_per_second": profile.counters.get("downloaded_bytes", 0) / elapsed,
                "counters": dict(profile.counters),
                "stages": profile.summarize(profile.totals),
                "http_response_samples": {name: list(samples) for name, samples in profile.http_samples.items()},
            }
            result["measurements"].append(measurement)
            result["stopped_for_server_backoff"] = stop.is_set()
            for worker_count in (4, 8):
                samples = [row for row in result["measurements"] if row["workers"] == worker_count]
                if samples:
                    result["summary"][str(worker_count)] = {
                        "rounds": len(samples),
                        "median_images_per_second": statistics.median(row["images_per_second"] for row in samples),
                        "download_errors": sum(row["counters"].get("download_errors", 0) for row in samples),
                    }
            save()
            print(json.dumps({key: measurement[key] for key in (
                "repeat", "workers", "seconds", "images_per_second", "counters",
            )}), flush=True)
            if stop.is_set():
                print(f"Server backoff detected; stopped. Results: {output}", flush=True)
                return
    print(json.dumps(result["summary"], indent=2), flush=True)
    print(f"Results: {output}", flush=True)


def run_live_download_benchmark(category_id: str, pages: int, output: Path) -> None:
    import populate_db
    import requests

    profile = populate_db.IngestionProfile(str(output))
    measurements = []
    experiment = {
        "category_id": category_id, "requested_download_heavy_pages": pages,
        "minimum_downloads_per_measured_page": 200,
        "scope": "Normal live ingestion with real file and database writes. "
                 "Different pages/images per treatment, not a matched URL comparison. "
                 "One persistent eight-thread executor; a per-page semaphore limits "
                 "active listing download jobs to four or eight. No session renewal. "
                 "Stops scheduling on HTTP 429/503 or after the requested heavy pages. "
                 "No stale-listing updates during this experiment.",
        "measurements": measurements,
    }
    profile.metadata.update({
        "live_download_experiment": experiment,
        "image_session_max_requests": 0, "download_workers": 8,
    })
    original_categories = populate_db.get_subcategories
    original_pages = populate_db.iter_category_pages
    original_download = populate_db.download_and_enqueue
    original_fetch = populate_db.fetch_image_bytes
    original_snapshot = profile.snapshot
    current = {"workers": 4, "gate": threading.BoundedSemaphore(4), "shutdown": None}

    def categories(rate):
        found = [category for category in original_categories(rate) if category["value"] == category_id]
        if not found:
            raise ValueError(f"Unknown ingestion category: {category_id}")
        return found

    def snapshot(**state):
        if state.get("phase") == "page":
            state["download_workers"] = current["workers"]
        original_snapshot(**state)

    def limited_download(*args, **kwargs):
        with current["gate"]:
            return original_download(*args, **kwargs)

    def fetch(url, profile=None):
        try:
            return original_fetch(url, profile)
        except requests.RequestException as error:
            if error.response is not None and error.response.status_code in (429, 503):
                experiment["server_backoff"] = True
                current["shutdown"].set()
            raise

    def category_pages(rate, selected_category, shutdown):
        current["shutdown"] = shutdown
        try:
            for sort_key, docs in original_pages(rate, selected_category, shutdown):
                if shutdown.is_set() or len(measurements) >= pages:
                    return
                current["workers"] = (4, 8, 8, 4)[len(measurements) % 4]
                current["gate"] = threading.BoundedSemaphore(current["workers"])
                yield sort_key, docs
                window = profile.snapshots[-1]
                downloaded = window["counters"].get("downloaded_images", 0)
                if downloaded >= 200:
                    measurement = {
                        "window_index": len(profile.snapshots) - 1,
                        "workers": current["workers"],
                        "seconds": window["window_seconds"],
                        "downloaded_images": downloaded,
                        "new_listings": window["counters"].get("new_listings", 0),
                        "images_per_second": downloaded / window["window_seconds"],
                        "download_errors": window["counters"].get("download_errors", 0),
                    }
                    measurements.append(measurement)
                    print(json.dumps(measurement), flush=True)
                if len(measurements) >= pages:
                    shutdown.set()
                    return
        finally:
            shutdown.set()

    args = types.SimpleNamespace(
        reembed_only=False, clean_run=False, max_listings=None,
        download_page_barrier=True,
    )
    with (
        patch.object(populate_db, "_profile", profile),
        patch.object(populate_db, "IMAGE_SESSION_MAX_REQUESTS", 0),
        patch.object(populate_db, "get_subcategories", categories),
        patch.object(populate_db, "iter_category_pages", category_pages),
        patch.object(populate_db, "download_and_enqueue", limited_download),
        patch.object(populate_db, "fetch_image_bytes", fetch),
        patch.object(populate_db, "ThreadPoolExecutor", side_effect=lambda **kwargs: ThreadPoolExecutor(max_workers=8)),
        patch.object(profile, "snapshot", snapshot),
    ):
        profile.snapshot(phase="startup")
        try:
            populate_db.run_ingestion(args)
        finally:
            experiment["completed_requested_pages"] = len(measurements) >= pages
            experiment["summary"] = {
                str(workers): {
                    "pages": len(samples),
                    "median_images_per_second": statistics.median(row["images_per_second"] for row in samples),
                    "download_errors": sum(row["download_errors"] for row in samples),
                }
                for workers in (4, 8)
                if (samples := [row for row in measurements if row["workers"] == workers])
            }
            outcome = "completed" if experiment["completed_requested_pages"] else "insufficient_new_images"
            if experiment.get("server_backoff"):
                outcome = "server_backoff"
            if profile.counters.get("interrupts"):
                outcome = "interrupted"
            if sys.exc_info()[0]:
                outcome = "error"
            profile.snapshot(final=True, phase="finished", outcome=outcome)
    print(json.dumps(experiment["summary"], indent=2), flush=True)
    print(f"Results: {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create", help="Copy listings and image files, without embeddings")
    create.add_argument("--listings", type=positive_int, default=1000)
    run = commands.add_parser("run", help="Benchmark decode, embedding, and rolled-back DB writes")
    run.add_argument("--batch-size", type=positive_int, default=32)
    run.add_argument("--repeats", type=positive_int, default=3)
    run.add_argument("--max-images", type=positive_int)
    run.add_argument("--no-prefetch", action="store_true", help="Prepare batches sequentially")
    run.add_argument(
        "--coreml-batch-size", type=int, choices=(32, 64, 128), default=32,
        help="Inference batch shape; requires the matching converted vision package",
    )
    run.add_argument(
        "--profile-stages", action="store_true",
        help="Time transfer, forward, normalization, and export with extra GPU barriers",
    )
    run.add_argument("--output", type=Path, default=ARTIFACTS / "baseline.json")
    scaling = commands.add_parser("scaling", help="Compare ingestion DB writes with 0 or 10000 seed listings")
    scaling.add_argument("--listings", type=positive_int, default=1000)
    scaling.add_argument("--repeats", type=positive_int, default=3)
    scaling.add_argument("--output", type=Path, default=ARTIFACTS / "ingestion_db_scaling.json")
    downloads = commands.add_parser("downloads", help="Compare 4/8 HTTP workers on the same recent image URLs (source DB read-only)")
    downloads.add_argument("--max-images", type=positive_int, default=128)
    downloads.add_argument("--repeats", type=positive_int, default=3)
    downloads.add_argument("--output", type=Path, default=ARTIFACTS / "download_workers.json")
    live = commands.add_parser("live-downloads", help="Compare 4/8 workers during real ingestion (writes listings/images)")
    live.add_argument("--category-id", required=True)
    live.add_argument("--pages", type=positive_int, default=6)
    live.add_argument("--output", type=Path, default=ARTIFACTS / "download_live_workers.json")
    args = parser.parse_args()
    if args.command == "create":
        create_dataset(args.listings)
    elif args.command == "scaling":
        run_scaling_benchmark(args.listings, args.repeats, args.output)
    elif args.command == "downloads":
        run_download_benchmark(args.max_images, args.repeats, args.output)
    elif args.command == "live-downloads":
        run_live_download_benchmark(args.category_id, args.pages, args.output)
    else:
        run_benchmark(
            args.batch_size, args.repeats, args.max_images, args.output,
            prefetch=not args.no_prefetch,
            profile_stages=args.profile_stages,
            coreml_batch_size=args.coreml_batch_size,
        )


if __name__ == "__main__":
    main()