import importlib
import json
import os
import queue
import signal
import subprocess
import sys
import textwrap
import threading
import types
import unittest
from contextlib import ExitStack, nullcontext, redirect_stdout
from concurrent.futures import Future, wait
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import numpy as np
import psycopg
import populate_db
from fastapi.testclient import TestClient
from pgvector import Vector
from pgvector.psycopg import register_vector


fake_embedder = types.ModuleType("embedder")
fake_embedder.MATCH_THRESHOLD = 0.0
fake_embedder.embed_text = Mock(return_value=np.array([0, 1], dtype=np.float32))
fake_embedder.prob_to_cosine = Mock(return_value=0.25)
fake_embedder.cosine_to_prob = Mock(return_value=0.73)
with patch.dict(sys.modules, {"embedder": fake_embedder}):
    db = importlib.import_module("db")
    main = importlib.import_module("main")


def initialize_interrupt_test_worker(started, release):
    import image_preprocessing

    global _interrupt_started, _interrupt_release
    _interrupt_started, _interrupt_release = started, release
    processor = Mock(side_effect=lambda images, **kwargs: {
        "pixel_values": np.ones((len(images), 3, 2, 2), dtype=np.float32),
    })
    with patch.object(image_preprocessing, "load_processor", return_value=processor):
        image_preprocessing.initialize_worker(Mock(), "test-model")


def prepare_interrupt_test_batch(path):
    import image_preprocessing

    _interrupt_started.set()
    if not _interrupt_release.wait(15):
        raise TimeoutError("Interrupt test did not release worker")
    return image_preprocessing.prepare_paths([path], "test-model", skip_blank=True)


class RenditionStudyTests(unittest.TestCase):
    def test_preparation_copies_sources_and_keeps_all_artifacts_in_study(self):
        import rendition_comparison as study
        from PIL import Image

        with TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            source = root / "production.png"
            with Image.new("RGB", (8, 8), "red") as image:
                image.save(source)
            before = source.read_bytes()
            items = [dict(id=index, title=str(index), price=0, category_id="chairs",
                          category_name="Chairs", url="https://example.com", local_path=str(source),
                          image_url=f"https://images.finncdn.no/dynamic/default/item/{index}/photo")
                     for index in (1, 2)]
            encoder = types.SimpleNamespace(
                model_id="test-study", model_dir=root,
                embed_images=Mock(side_effect=lambda images: np.tile(np.array([1, 0], dtype=np.float32), (len(images), 1))),
            )
            stack.enter_context(patch.dict(sys.modules, {"embedder": encoder}))
            stack.enter_context(patch.object(study, "select_items", return_value=items))
            fetch = stack.enter_context(patch.object(populate_db, "fetch_image_bytes", return_value=before))
            stack.enter_context(patch("builtins.print"))
            destination = root / "study"
            study.prepare_study(destination, 2, 2)
            manifest = study.read_study(destination)
            self.assertTrue(manifest["ready"])
            self.assertEqual(manifest["listings"], 2)
            self.assertEqual(len(list((destination / "images").glob("*/*.img"))), 4)
            self.assertEqual(fetch.call_count, 2)
            self.assertTrue(all("/dynamic/320w/" in call.args[0] for call in fetch.call_args_list))
            self.assertEqual(source.read_bytes(), before)
            with np.load(destination / "vectors.npz") as vectors:
                self.assertEqual(vectors["siglip2_default"].shape, (2, 2))
                self.assertEqual(vectors["siglip2_320w"].shape, (2, 2))
            study.prepare_study(destination, 2, 2)
            self.assertEqual(fetch.call_count, 2)

    def test_selection_opens_read_only_transaction(self):
        import benchmark
        import rendition_comparison as study

        connection = Mock()
        connection.execute.return_value.fetchall.return_value = [{"id": 1, "sample_rank": 1}]
        with patch.object(benchmark, "connection_settings", return_value=({}, {})):
            with patch.object(psycopg, "connect") as connect:
                connect.return_value.__enter__.return_value = connection
                self.assertEqual(study.select_items(1), [{"id": 1}])
        self.assertEqual(connection.execute.call_args_list[0].args,
                         ("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY",))
        self.assertEqual(connection.execute.call_count, 2)

    def test_blinded_api_uses_paired_vectors_and_saves_judgments(self):
        import rendition_comparison as study
        from PIL import Image

        with TemporaryDirectory() as directory, ExitStack() as stack:
            path = Path(directory)
            items = [dict(id=index, title=f"Listing {index}", price=0, category_id="chairs",
                          category_name="Chairs", url="https://example.com") for index in (1, 2, 3)]
            study.write_json(path / "study.json", {"kind": study.STUDY_KIND, "ready": True,
                "items": items, "model_id": "test-study", "coreml_directory": directory, "quality": {}})
            baseline = np.array([[1, 0], [0.8, 0.6], [0, 1]], dtype=np.float32)
            smaller = baseline[::-1].copy()
            np.savez(path / "vectors.npz", siglip2_default=baseline, siglip2_320w=smaller)
            for rendition in study.RENDITIONS:
                (path / "images" / rendition).mkdir(parents=True)
                with Image.new("RGB", (8, 8), "green") as image:
                    image.save(path / "images" / rendition / "1.img", format="PNG")
            encoder = types.SimpleNamespace(model_id="test-study", embed_text=Mock(return_value=np.array([1, 0])))
            stack.enter_context(patch.dict(os.environ))
            stack.enter_context(patch.dict(sys.modules, {"embedder": encoder}))
            stack.enter_context(patch.object(study, "STUDY_ROOT", path))
            with TestClient(study.create_app(path)) as client:
                response = client.get("/search", params={"query": "chair", "seed": "test"})
                self.assertEqual(response.status_code, 200)
                result = response.json()
                default_side = next(side for side, rendition in result["renditions"].items() if rendition == "default")
                small_side = "B" if default_side == "A" else "A"
                self.assertEqual(result["results"][default_side][0]["id"], 1)
                self.assertEqual(result["results"][small_side][0]["id"], 3)
                image_search = client.get("/search", params={"image_id": 1}).json()
                for matches in image_search["results"].values():
                    self.assertNotIn(1, [item["id"] for item in matches])
                refined = client.get("/search", params={"image_id": 1, "query": "wooden"})
                self.assertEqual(refined.status_code, 200)
                self.assertTrue(all("text_score" in match for matches in refined.json()["results"].values() for match in matches))
                self.assertEqual(client.get("/photo/reference/1").headers["content-type"], "image/png")
                self.assertEqual(client.get("/search", params={"image_id": 999}).status_code, 404)
                self.assertEqual(client.get("/search", params={"query": "chair", "limit": 100}).status_code, 422)
                vote = client.post("/judgments", json={"key": result["key"], "choice": default_side, "query": "chair"})
                self.assertEqual(vote.json(), {"saved": 1})
                self.assertEqual(client.get("/judgments").json()[0]["selected_rendition"], "default")
                with self.assertRaises(RuntimeError):
                    study.cleanup_study(path)

    def test_same_corpus_ranking_filters_and_source_exclusion(self):
        import rendition_comparison as study

        items = [dict(id=index, title=str(index), price=0 if index != 2 else 100,
                      category_id="chairs" if index != 3 else "tables",
                      category_name="Furniture", url="https://example.com") for index in (1, 2, 3)]
        vectors = np.array([[1, 0], [0.8, 0.6], [0, 1]])
        self.assertEqual([item["id"] for item in study.rank_results(items, vectors, [1, 0])], [1, 2, 3])
        self.assertEqual(study.rank_results(items, vectors, [1, 0], excluded_id=1,
                                           category="chairs", free_only=True), [])
        self.assertEqual([item["id"] for item in study.rank_results(items, vectors, [1, 0], limit=1)], [1])
        with self.assertRaises(ValueError):
            study.rank_results(items, vectors, [1, 0, 0])

    def test_study_url_boundaries_and_cleanup_marker(self):
        import rendition_comparison as study

        self.assertEqual(study.rendition_url("https://images.finncdn.no/dynamic/default/item/1/image", "320w"),
                         "https://images.finncdn.no/dynamic/320w/item/1/image")
        with self.assertRaises(ValueError):
            study.rendition_url("https://example.com/image", "320w")
        with TemporaryDirectory() as directory:
            path = Path(directory)
            with self.assertRaises(ValueError):
                study.cleanup_study(path)
            with patch.object(study, "STUDY_ROOT", path):
                study.write_json(path / "study.json", {"kind": "not-a-study"})
                with self.assertRaises(ValueError):
                    study.cleanup_study(path)
                study.write_json(path / "study.json", {"kind": study.STUDY_KIND})
                study.cleanup_study(path)
                self.assertFalse(path.exists())


class ImageRenditionTests(unittest.TestCase):
    def test_default_preserves_original_url(self):
        url = "https://images.finncdn.no/dynamic/default/item/123/photo?quality=80#image"
        with patch.object(populate_db, "IMAGE_RENDITION", "default"):
            self.assertEqual(populate_db.image_download_url(url), url)

    def test_opt_in_rewrites_only_default_finn_renditions(self):
        url = "https://images.finncdn.no/dynamic/default//item/123/photo?quality=80#image"
        with patch.object(populate_db, "IMAGE_RENDITION", "320w"):
            self.assertEqual(populate_db.image_download_url(url),
                             "https://images.finncdn.no/dynamic/320w//item/123/photo?quality=80#image")
            for unchanged in (
                "https://images.example/dynamic/default/photo.jpg",
                "https://images.finncdn.no.example/dynamic/default/photo.jpg",
                "http://images.finncdn.no/dynamic/default/photo.jpg",
                "https://images.finncdn.no/dynamic/480w/photo.jpg",
                "https://images.finncdn.no/dynamic/320w/photo.jpg",
                "https://images.finncdn.no/other/photo.jpg",
            ):
                self.assertEqual(populate_db.image_download_url(unchanged), unchanged)

    def test_download_stores_actual_rendition_url_and_profiles_rewrite(self):
        original = "https://images.finncdn.no/dynamic/default/item/123/photo"
        expected = "https://images.finncdn.no/dynamic/320w/item/123/photo"
        with TemporaryDirectory() as directory, ExitStack() as stack:
            stack.enter_context(patch.object(populate_db, "IMAGE_RENDITION", "320w"))
            profile = populate_db.IngestionProfile(f"{directory}/profile.json")
            source = queue.Queue()
            stack.enter_context(patch.object(populate_db, "_profile", profile))
            fetch = stack.enter_context(patch.object(populate_db, "fetch_image_bytes", return_value=b"image"))
            populate_db.download_and_enqueue(10, "source", [original], directory, source,
                                             None, threading.Lock(), threading.Event())
            fetch.assert_called_once_with(expected)
            self.assertEqual(source.get_nowait(), (10, expected, f"{directory}/source_0.jpg"))
            self.assertEqual((Path(directory) / "source_0.jpg").read_bytes(), b"image")
            self.assertEqual(profile.counters["image_rendition_rewrites"], 1)
            self.assertEqual(profile.counters["downloaded_images"], 1)
            self.assertEqual(profile.metadata["image_rendition"], "320w")

    def test_invalid_rendition_is_rejected_before_ingestion(self):
        with patch.object(populate_db, "IMAGE_RENDITION", "invalid"):
            with patch.object(sys, "argv", ["populate_db.py"]):
                with patch.object(populate_db, "run_ingestion") as ingest:
                    with self.assertRaisesRegex(SystemExit, "IMAGE_RENDITION"):
                        populate_db.main()
                    ingest.assert_not_called()


class RateLimiterTests(unittest.TestCase):
    def test_burst_reserves_evenly_spaced_slots(self):
        limiter = populate_db.RateLimiter(0.25)
        with patch.object(populate_db, "perf_counter", return_value=100.0):
            with patch.object(populate_db, "sleep") as sleep:
                for _ in range(4):
                    limiter.wait_turn()
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [0.25, 0.5, 0.75])

    def test_idle_time_resets_the_next_slot(self):
        limiter = populate_db.RateLimiter(0.25)
        with patch.object(populate_db, "perf_counter", side_effect=[100.0, 100.1, 101.0]):
            with patch.object(populate_db, "sleep") as sleep:
                for _ in range(3):
                    limiter.wait_turn()
        sleep.assert_called_once()
        self.assertAlmostEqual(sleep.call_args.args[0], 0.15)

    def test_zero_interval_does_not_sleep(self):
        limiter = populate_db.RateLimiter(0)
        with patch.object(populate_db, "perf_counter", return_value=100.0):
            with patch.object(populate_db, "sleep") as sleep:
                for _ in range(4):
                    limiter.wait_turn()
        sleep.assert_not_called()


class DownloadSchedulerTests(unittest.TestCase):
    def test_bound_waits_for_any_completed_job_not_the_oldest(self):
        first, second, third = Future(), Future(), Future()
        executor = Mock()
        executor.submit.side_effect = [first, second, third]
        scheduler = populate_db.DownloadScheduler(executor, 2, threading.Event())
        scheduler.submit(Mock(), 1)
        scheduler.submit(Mock(), 2)
        waiting = threading.Event()

        def bounded_wait(*args, **kwargs):
            waiting.set()
            return wait(*args, **kwargs)

        with patch.object(populate_db, "wait", side_effect=bounded_wait):
            worker = threading.Thread(target=scheduler.submit, args=(Mock(), 3))
            worker.start()
            try:
                self.assertTrue(waiting.wait(2))
                self.assertEqual(executor.submit.call_count, 2)
                second.set_result(None)
                worker.join(2)
                self.assertFalse(worker.is_alive())
                self.assertEqual(executor.submit.call_count, 3)
                self.assertEqual(scheduler.pending, {first, third})
                self.assertFalse(first.done())
            finally:
                if not first.done():
                    first.set_result(None)
                if not third.done():
                    third.set_result(None)
                worker.join(2)

    def test_shutdown_stops_submission_and_invalid_limit_is_rejected(self):
        shutdown = threading.Event()
        shutdown.set()
        executor = Mock()
        populate_db.DownloadScheduler(executor, 1, shutdown).submit(Mock())
        executor.submit.assert_not_called()
        with self.assertRaises(ValueError):
            populate_db.DownloadScheduler(executor, 0, shutdown)


class DownloadBenchmarkTests(unittest.TestCase):
    def test_live_experiment_alternates_heavy_pages_and_caps_without_stale_updates(self):
        import benchmark

        shutdown = threading.Event()
        def ingest(args):
            categories = populate_db.get_subcategories(Mock())
            self.assertEqual([category["value"] for category in categories], ["chairs"])
            for sort_key, docs in populate_db.iter_category_pages(Mock(), "chairs", shutdown):
                populate_db._profile.increment("downloaded_images", 200)
                populate_db._profile.increment("new_listings", 100)
                populate_db._profile.snapshot(phase="page", category_id="chairs")
            self.assertTrue(shutdown.is_set())

        with TemporaryDirectory() as directory, ExitStack() as stack:
            output = Path(directory) / "live.json"
            stack.enter_context(patch.object(populate_db, "get_subcategories", return_value=[
                {"value": "chairs"}, {"value": "sofas"},
            ]))
            stack.enter_context(patch.object(populate_db, "iter_category_pages", return_value=iter(
                [("NEWEST", [])] * 10,
            )))
            stack.enter_context(patch.object(populate_db, "run_ingestion", side_effect=ingest))
            stack.enter_context(patch("builtins.print"))
            benchmark.run_live_download_benchmark("chairs", 6, output)
            report = json.loads(output.read_text())
            experiment = report["metadata"]["live_download_experiment"]
            self.assertEqual([row["workers"] for row in experiment["measurements"]], [4, 8, 8, 4, 4, 8])
            self.assertEqual(experiment["summary"]["4"]["pages"], 3)
            self.assertEqual(experiment["summary"]["8"]["pages"], 3)
            self.assertEqual(report["metadata"]["image_session_max_requests"], 0)
            self.assertEqual(report["counters"]["downloaded_images"], 1200)
            self.assertTrue(report["final"])

    def test_alternates_workers_replays_urls_and_reads_database_only(self):
        import benchmark

        with TemporaryDirectory() as directory, ExitStack() as stack:
            output = Path(directory) / "downloads.json"
            urls = ["https://images.example/one.jpg", "https://images.example/two.jpg"]
            connection = Mock()
            connection.execute.return_value.fetchall.return_value = [(url,) for url in urls]
            connect = stack.enter_context(patch.object(benchmark.psycopg, "connect"))
            connect.return_value.__enter__.return_value = connection
            stack.enter_context(patch.object(benchmark, "connection_settings", return_value=({}, {})))
            fetch = stack.enter_context(patch.object(populate_db, "fetch_image_bytes", return_value=b"image"))
            stack.enter_context(patch("builtins.print"))
            benchmark.run_download_benchmark(2, 2, output)
            report = json.loads(output.read_text())
            self.assertEqual([row["workers"] for row in report["measurements"]], [4, 8, 8, 4])
            self.assertCountEqual([call.args[0] for call in fetch.call_args_list], urls * 4)
            self.assertEqual(connection.execute.call_args_list[0].args, ("SET TRANSACTION READ ONLY",))
            self.assertEqual(connection.execute.call_count, 2)
            self.assertEqual(report["summary"]["4"]["rounds"], 2)
            self.assertEqual(report["summary"]["8"]["download_errors"], 0)
            self.assertEqual(report["measurements"][0]["counters"]["downloaded_bytes"], 10)
            self.assertNotIn(urls[0], output.read_text())

    def test_stops_on_server_backoff(self):
        import benchmark

        with TemporaryDirectory() as directory, ExitStack() as stack:
            output = Path(directory) / "downloads.json"
            connection = Mock()
            connection.execute.return_value.fetchall.return_value = [("https://images.example/one.jpg",)]
            connect = stack.enter_context(patch.object(benchmark.psycopg, "connect"))
            connect.return_value.__enter__.return_value = connection
            stack.enter_context(patch.object(benchmark, "connection_settings", return_value=({}, {})))
            response = populate_db.requests.Response()
            response.status_code = 429
            stack.enter_context(patch.object(populate_db, "fetch_image_bytes", side_effect=populate_db.requests.HTTPError(response=response)))
            stack.enter_context(patch("builtins.print"))
            benchmark.run_download_benchmark(1, 3, output)
            report = json.loads(output.read_text())
            self.assertTrue(report["stopped_for_server_backoff"])
            self.assertEqual(len(report["measurements"]), 1)
            self.assertEqual(report["summary"]["4"]["download_errors"], 1)


class IngestionProfileTests(unittest.TestCase):
    def test_cdn_samples_classify_bound_redact_and_reset_windows(self):
        with TemporaryDirectory() as directory:
            profile = populate_db.IngestionProfile(f"{directory}/profile.json")
            response = populate_db.requests.Response()
            response.status_code = 200
            response.headers.update({
                "Age": "120", "X-Cache": "HIT", "Cache-Status": "cdn; hit",
                "Set-Cookie": "secret-cookie", "Authorization": "secret-token",
                "X-Unlisted": "private", "Server-Timing": "x" * 300,
            })
            url = "https://images.example/photo.jpg?token=secret-url"
            for index in range(20):
                profile.record_http_response(url, response, 0.01, 0.003, 0.013, None)
            profile.record_http_response(url, response, 0.25, 0.003, 0.253, None)
            profile.record_http_response(url, None, None, None, 10.0, "ReadTimeout")
            profile.snapshot()
            profile.snapshot(final=True)
            text = profile.output.read_text()
            report = json.loads(text)
            samples = report["http_response_samples"]
            self.assertEqual(len(samples["fast"]), 8)
            self.assertEqual(len(samples["slow"]), 1)
            self.assertEqual(len(samples["error"]), 1)
            self.assertEqual(samples["fast"][0]["headers"]["x-cache"], "HIT")
            self.assertEqual(samples["slow"][0]["headers"]["age"], "120")
            self.assertEqual(len(samples["slow"][0]["headers"]["server-timing"]), 256)
            self.assertEqual(samples["error"][0]["error_type"], "ReadTimeout")
            self.assertIsNone(samples["error"][0]["header_seconds"])
            self.assertEqual(report["counters"]["image_http_class.fast"], 20)
            self.assertEqual(len(report["windows"][0]["http_response_samples"]["fast"]), 8)
            self.assertEqual(report["windows"][1]["http_response_samples"], {"fast": [], "slow": [], "error": []})
            for sensitive in ("secret-url", "secret-token", "secret-cookie", "photo.jpg", "set-cookie", "authorization", "x-unlisted"):
                self.assertNotIn(sensitive, text)

    def test_cdn_samples_capture_body_failures_and_http_errors(self):
        with TemporaryDirectory() as directory:
            profile = populate_db.IngestionProfile(f"{directory}/profile.json")
            broken = populate_db.requests.Response()
            broken.status_code = 200
            broken.headers["X-Cache"] = "MISS"
            broken.raw = Mock()
            broken.raw.stream.side_effect = populate_db.requests.exceptions.ChunkedEncodingError("broken body")
            limited = populate_db.requests.Response()
            limited.status_code = 429
            limited.raw = Mock()
            limited.headers["Retry-After"] = "10"
            session = Mock()
            session.get.side_effect = [broken, limited]
            with patch.object(populate_db, "get_thread_session", return_value=session):
                for error in (populate_db.requests.exceptions.ChunkedEncodingError, populate_db.requests.HTTPError):
                    with self.assertRaises(error):
                        populate_db.fetch_image_bytes("https://images.example/photo.jpg", profile)
            samples = list(profile.http_samples["error"])
            self.assertEqual([sample["status"] for sample in samples], [200, 429])
            self.assertEqual(samples[0]["error_type"], "ChunkedEncodingError")
            self.assertIsNotNone(samples[0]["body_seconds"])
            self.assertEqual(samples[0]["headers"]["x-cache"], "MISS")
            self.assertIsNone(samples[1]["body_seconds"])
            self.assertEqual(samples[1]["headers"]["retry-after"], "10")
            broken.raw.close.assert_called_once()

    def test_session_renewal_replaces_real_keepalive_connection(self):
        peers = []

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self):
                peers.append(self.client_address)
                self.send_response(200)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"ok")

            def log_message(self, *args):
                pass

        with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
            worker = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
            worker.start()
            try:
                for limit in (0, 2):
                    peers.clear()
                    with TemporaryDirectory() as directory, ExitStack() as stack:
                        profile = populate_db.IngestionProfile(f"{directory}/profile.json")
                        local = threading.local()
                        stack.enter_context(patch.object(populate_db, "_thread_local", local))
                        stack.enter_context(patch.object(populate_db, "IMAGE_SESSION_MAX_REQUESTS", limit))
                        try:
                            for _ in range(3):
                                self.assertEqual(populate_db.fetch_image_bytes(
                                    f"http://127.0.0.1:{server.server_port}/image", profile,
                                ), b"ok")
                            self.assertEqual(peers[0], peers[1])
                            if limit:
                                self.assertNotEqual(peers[1], peers[2])
                                self.assertEqual(profile.counters["image_sessions_renewed"], 1)
                            else:
                                self.assertEqual(peers[1], peers[2])
                        finally:
                            local.image_session.close()
            finally:
                server.shutdown()
                worker.join()

    @unittest.skipUnless(os.name == "posix", "requires process-group signals")
    def test_real_process_group_interrupt_preserves_inflight_and_tail_preparation(self):
        from PIL import Image

        program = textwrap.dedent("""
            import json
            import multiprocessing
            import os
            import signal
            import sys
            from concurrent.futures import ProcessPoolExecutor
            from image_preprocessing import prepare_paths
            from test_search import initialize_interrupt_test_worker, prepare_interrupt_test_batch

            context = multiprocessing.get_context('spawn')
            started, release = context.Event(), context.Event()
            with ProcessPoolExecutor(max_workers=1, mp_context=context,
                    initializer=initialize_interrupt_test_worker, initargs=(started, release)) as pool:
                inflight = pool.submit(prepare_interrupt_test_batch, sys.argv[1])
                if not started.wait(20):
                    raise TimeoutError('Worker did not start')
                try:
                    os.killpg(os.getpgrp(), signal.SIGINT)
                except KeyboardInterrupt:
                    pass
                release.set()
                first = inflight.result(timeout=20)
                tail = pool.submit(prepare_paths, [sys.argv[1]], 'test-model', None, True).result(timeout=20)
                print(json.dumps({'first': first[3], 'tail': tail[3]}))
        """)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "pattern.png"
            pattern = np.tile(np.array([0, 255], dtype=np.uint8), (8, 4))
            with Image.fromarray(pattern) as image:
                image.save(path)
            process = subprocess.Popen(
                [sys.executable, "-c", program, str(path)],
                cwd=Path(__file__).parent, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, start_new_session=True,
            )
            try:
                stdout, stderr = process.communicate(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.communicate()
                raise
            self.assertEqual(process.returncode, 0, stderr)
            self.assertEqual(json.loads(stdout), {"first": [0], "tail": [0]})

    def test_image_session_renewal_is_opt_in_and_does_not_reset_api_session(self):
        for limit in (0, 2):
            with self.subTest(limit=limit), TemporaryDirectory() as directory, ExitStack() as stack:
                profile = populate_db.IngestionProfile(f"{directory}/profile.json")
                sessions = [Mock() for _ in range(3)]
                stack.enter_context(patch.object(populate_db, "_thread_local", threading.local()))
                stack.enter_context(patch.object(populate_db, "IMAGE_SESSION_MAX_REQUESTS", limit))
                factory = stack.enter_context(patch.object(populate_db.requests, "Session", side_effect=sessions))
                api_session = populate_db.get_thread_session()
                used = [populate_db.get_thread_session(for_images=True, profile=profile) for _ in range(3)]
                self.assertIs(populate_db.get_thread_session(), api_session)
                self.assertIs(used[0], used[1])
                if limit:
                    self.assertIs(used[2], sessions[2])
                    sessions[1].close.assert_called_once()
                    self.assertEqual(profile.counters["image_sessions_renewed"], 1)
                    self.assertEqual(factory.call_count, 3)
                else:
                    self.assertIs(used[2], used[0])
                    sessions[1].close.assert_not_called()
                    self.assertNotIn("image_sessions_renewed", profile.counters)
                    self.assertEqual(factory.call_count, 2)
                sessions[0].close.assert_not_called()

    def test_failed_image_attempts_count_toward_renewal(self):
        with TemporaryDirectory() as directory, ExitStack() as stack:
            profile = populate_db.IngestionProfile(f"{directory}/profile.json")
            first, second = Mock(), Mock()
            first.get.side_effect = populate_db.requests.ReadTimeout("timeout")
            second.get.return_value = Mock(content=b"ok", status_code=200, history=[], headers={})
            stack.enter_context(patch.object(populate_db, "_thread_local", threading.local()))
            stack.enter_context(patch.object(populate_db, "IMAGE_SESSION_MAX_REQUESTS", 1))
            stack.enter_context(patch.object(populate_db.requests, "Session", side_effect=[first, second]))
            with self.assertRaises(populate_db.requests.ReadTimeout):
                populate_db.fetch_image_bytes("https://images.example/one.jpg", profile)
            self.assertEqual(populate_db.fetch_image_bytes("https://images.example/two.jpg", profile), b"ok")
            first.close.assert_called_once()
            self.assertEqual(profile.counters["image_sessions_created"], 2)
            self.assertEqual(profile.counters["image_sessions_renewed"], 1)

    def test_preparation_workers_ignore_terminal_interrupts_before_loading(self):
        import image_preprocessing

        ready = Mock()
        processor = Mock()
        events = []
        with patch.object(image_preprocessing.signal, "signal", side_effect=lambda *args: events.append("ignore")) as handler:
            with patch.object(image_preprocessing, "load_processor", side_effect=lambda *args, **kwargs: events.append("load") or processor):
                with patch.object(image_preprocessing, "_ready"), patch.object(image_preprocessing, "_processor"):
                    image_preprocessing.initialize_worker(ready, "test-model")
                    self.assertIs(image_preprocessing._processor, processor)
        handler.assert_called_once_with(image_preprocessing.signal.SIGINT, image_preprocessing.signal.SIG_IGN)
        self.assertEqual(events, ["ignore", "load"])
        ready.put.assert_called_once_with(None)

    def test_http_diagnostics_and_response_cleanup(self):
        with TemporaryDirectory() as directory:
            profile = populate_db.IngestionProfile(f"{directory}/profile.json")
            response = Mock(content=b"image", status_code=200, history=[])
            response.headers = {"Connection": "close", "Retry-After": "10"}
            session = Mock()
            session.get.return_value = response
            with patch.object(populate_db, "get_thread_session", return_value=session):
                content = populate_db.fetch_image_bytes("https://images.example/photo.jpg", profile)
            self.assertEqual(content, b"image")
            session.get.assert_called_once_with("https://images.example/photo.jpg", timeout=10, stream=True)
            response.close.assert_called_once()
            for stage in ("image_http", "image_http_headers", "image_http_body", "image_http_host.images.example.total"):
                self.assertEqual(profile.totals[stage]["count"], 1)
            self.assertEqual(profile.counters["image_http_status.200"], 1)
            self.assertEqual(profile.counters["image_http_connection_close"], 1)
            self.assertEqual(profile.counters["image_http_retry_after"], 1)

    def test_http_failures_are_classified_and_closed(self):
        with TemporaryDirectory() as directory:
            profile = populate_db.IngestionProfile(f"{directory}/profile.json")
            response = Mock(status_code=429, history=[])
            response.headers = {"Retry-After": "30"}
            response.raise_for_status.side_effect = populate_db.requests.HTTPError("429")
            session = Mock()
            session.get.side_effect = [response, populate_db.requests.ReadTimeout("timeout")]
            with patch.object(populate_db, "get_thread_session", return_value=session):
                for error in (populate_db.requests.HTTPError, populate_db.requests.ReadTimeout):
                    with self.assertRaises(error):
                        populate_db.fetch_image_bytes("https://images.example/photo.jpg", profile)
            response.close.assert_called_once()
            self.assertEqual(profile.counters["image_http_status.429"], 1)
            self.assertEqual(profile.counters["image_http_error.HTTPError"], 1)
            self.assertEqual(profile.counters["image_http_error.ReadTimeout"], 1)
            self.assertEqual(profile.totals["image_http"]["count"], 2)
            self.assertNotIn("image_http_body", profile.totals)

    def test_profiled_download_run_with_real_threads(self):
        with patch.object(populate_db, "EMBED_BATCH_SIZE", 2):
            with TemporaryDirectory() as directory, ExitStack() as stack:
                output = Path(directory) / "profile.json"
                response = Mock(content=b"image bytes", status_code=200, history=[])
                response.headers = {}
                session = Mock()
                download_started = threading.Event()
                next_page_requested = threading.Event()

                def get_image(url, **kwargs):
                    if url != "good":
                        return self.fail_download()
                    download_started.set()
                    if not next_page_requested.wait(2):
                        raise TimeoutError("Next API page waited for the previous download")
                    return response

                session.get.side_effect = get_image

                def pages(*args):
                    yield "NEWEST", [{"ad_id": 1, "timestamp": 100, "image_urls": ["bad", "good"]}]
                    self.assertTrue(download_started.wait(2))
                    next_page_requested.set()
                    yield "NEWEST", [{"ad_id": 2, "timestamp": 100}]

                def submit(paths, **kwargs):
                    future = Mock()
                    future.result.return_value = (
                        np.ones((len(paths), 2)), 0.125, 0.25,
                        list(range(len(paths))), [], [],
                    )
                    return future
                encoder = types.SimpleNamespace(
                    model_id="test-model", PREPROCESS_WORKERS=1,
                    submit_image_paths=submit,
                    embed_prepared_images=Mock(side_effect=lambda pixels: pixels),
                )
                database = types.SimpleNamespace(
                    pool=Mock(), ensure_schema=Mock(), upsert_category=Mock(),
                    upsert_listing_and_embedding_status=Mock(side_effect=[(1, True, False), (2, False, True)]),
                    insert_listing_images=Mock(side_effect=lambda rows, **kwargs: len(rows)),
                    mark_stale_listings_since_timestamp=Mock(return_value=(0, 0)),
                )
                stack.enter_context(patch.dict(sys.modules, {"db": database, "embedder": encoder}))
                stack.enter_context(patch.object(sys, "argv", ["populate_db.py", "--profile-output", str(output)]))
                stack.enter_context(patch.object(populate_db, "IMAGE_DIR", directory))
                stack.enter_context(patch.object(populate_db, "get_thread_session", return_value=session))
                stack.enter_context(patch.object(populate_db, "get_subcategories", return_value=[{
                    "value": "chairs", "display_name": "Chairs",
                }]))
                stack.enter_context(patch.object(populate_db, "iter_category_pages", side_effect=pages))
                stack.enter_context(patch.object(populate_db, "tqdm"))
                stack.enter_context(patch("builtins.print"))
                populate_db.main()
                report = json.loads(output.read_text())
                self.assertTrue(report["final"])
                self.assertEqual(report["windows"][-1]["state"]["outcome"], "completed")
                self.assertEqual(len(report["windows"]), 4)
                self.assertEqual(report["counters"]["listings"], 2)
                self.assertEqual(report["counters"]["new_listings"], 1)
                self.assertEqual(report["counters"]["download_errors"], 1)
                self.assertEqual(report["counters"]["stored_images"], 1)
                self.assertTrue(next_page_requested.is_set())
                self.assertNotIn("page_download_jobs_wait", report["stages"])
                self.assertEqual(report["metadata"]["download_max_pending"], populate_db.DOWNLOAD_MAX_PENDING)
                self.assertEqual(report["counters"]["downloaded_bytes"], len(response.content))
                self.assertEqual(report["stages"]["image_http"]["count"], 2)
                self.assertEqual(report["stages"]["worker_decode_and_filter"]["total_seconds"], 0.125)
                self.assertEqual(report["stages"]["worker_preprocess"]["total_seconds"], 0.25)
                for stage in ("listing_db_upsert", "image_file_write", "download_queue_put_wait",
                              "embed_queue_get_wait", "prepare_first_submit_and_startup",
                              "prepare_result_wait", "siglip2_inference", "image_db_insert",
                              "download_shutdown_wait", "embedding_drain_wait", "stale_db_check"):
                    self.assertIn(stage, report["stages"])
                self.assertIsNone(populate_db._profile)
                database.pool.close.assert_called_once()

    @staticmethod
    def fail_download():
        raise populate_db.requests.ReadTimeout("test timeout")

    def test_final_report_on_error_and_interrupt(self):
        for error, outcome in ((RuntimeError("failed"), "error"), (KeyboardInterrupt(), "interrupted")):
            with self.subTest(outcome=outcome), TemporaryDirectory() as directory, ExitStack() as stack:
                output = Path(directory) / "profile.json"
                database = types.SimpleNamespace(
                    pool=Mock(), ensure_schema=Mock(), upsert_category=Mock(),
                )
                stack.enter_context(patch.dict(sys.modules, {"db": database}))
                stack.enter_context(patch.object(sys, "argv", ["populate_db.py", "--profile-output", str(output)]))
                stack.enter_context(patch.object(populate_db, "IMAGE_DIR", directory))
                stack.enter_context(patch.object(populate_db, "get_subcategories", return_value=[{
                    "value": "chairs", "display_name": "Chairs",
                }]))
                stack.enter_context(patch.object(populate_db, "iter_category_pages", side_effect=error))
                thread = stack.enter_context(patch.object(populate_db.threading, "Thread"))
                thread.return_value.is_alive.return_value = False
                stack.enter_context(patch.object(populate_db, "ThreadPoolExecutor"))
                stack.enter_context(patch.object(populate_db, "tqdm"))
                stack.enter_context(patch("builtins.print"))
                if outcome == "error":
                    with self.assertRaises(RuntimeError):
                        populate_db.main()
                else:
                    populate_db.main()
                report = json.loads(output.read_text())
                self.assertTrue(report["final"])
                self.assertEqual(report["windows"][-1]["state"]["outcome"], outcome)
                self.assertIsNone(populate_db._profile)

    def test_profile_rejects_embedding_only_modes(self):
        with patch.object(sys, "argv", ["populate_db.py", "--reembed-only", "--profile-output", "unused.json"]):
            with patch.object(sys, "stderr", StringIO()), self.assertRaises(SystemExit) as error:
                populate_db.main()
            self.assertEqual(error.exception.code, 2)

    def test_concurrent_accounting_windows_and_bounded_samples(self):
        with TemporaryDirectory() as directory:
            profile = populate_db.IngestionProfile(f"{directory}/nested/profile.json")
            def record():
                for _ in range(1100):
                    profile.record("download", 0.25)
                    profile.increment("downloaded_images")
            workers = [threading.Thread(target=record) for _ in range(4)]
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join()
            profile.snapshot(queue_depth=2)
            profile.record("download", 1.0)
            profile.snapshot(final=True, queue_depth=0)
            report = json.loads(profile.output.read_text())
            self.assertTrue(report["final"])
            self.assertEqual(report["counters"]["downloaded_images"], 4400)
            self.assertEqual(report["stages"]["download"]["count"], 4401)
            self.assertEqual(report["stages"]["download"]["total_seconds"], 1101)
            self.assertEqual(report["stages"]["download"]["percentile_sample_count"], 4096)
            self.assertEqual(report["windows"][0]["stages"]["download"]["median_seconds"], 0.25)
            self.assertEqual(report["windows"][1]["stages"]["download"]["count"], 1)
            self.assertEqual(report["windows"][1]["counters"], {})
            self.assertFalse(profile.output.with_name("profile.json.tmp").exists())

    def test_stage_records_failure_and_disabled_mode(self):
        with TemporaryDirectory() as directory:
            profile = populate_db.IngestionProfile(f"{directory}/profile.json")
            with patch.object(populate_db, "_profile", profile):
                with self.assertRaises(ValueError), populate_db.profile_stage("failed"):
                    raise ValueError("expected")
            self.assertEqual(profile.totals["failed"]["count"], 1)
            with patch.object(populate_db, "_profile", None):
                self.assertEqual(populate_db.profile_call("disabled", lambda: 42), 42)


class TextEmbeddingTests(unittest.TestCase):
    def test_text_input_uses_exported_sequence_length(self):
        import image_preprocessing

        model = Mock()
        model.user_defined_metadata = {
            "model_id": "test-model", "embedding_dim": "2",
            "logit_scale": "1", "logit_bias": "0",
        }
        model.get_spec.return_value.description.input = [types.SimpleNamespace(
            type=types.SimpleNamespace(multiArrayType=types.SimpleNamespace(shape=[1, 64])),
        )]
        model.predict.return_value = {"embeddings": np.array([[3, 4]], dtype=np.float32)}
        processor = Mock(return_value={"input_ids": np.zeros((1, 64), dtype=np.int64)})
        coreml = types.SimpleNamespace(
            models=types.SimpleNamespace(MLModel=Mock(return_value=model)),
            ComputeUnit={"ALL": "ALL"},
        )
        spec = importlib.util.spec_from_file_location(
            "_test_embedder", Path(__file__).with_name("embedder.py"),
        )
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"coremltools": coreml}):
            with patch.dict(os.environ, {"BEDREFIN_MODEL_ID": "test-model", "BEDREFIN_COREML_COMPUTE_UNITS": "ALL"}):
                with patch.object(image_preprocessing, "load_processor", return_value=processor):
                    spec.loader.exec_module(module)
        for text in ("stol", "wooden chair " * 100):
            np.testing.assert_allclose(module.embed_text(text), [0.6, 0.8])
            processor.assert_called_with(
                text=[text], padding="max_length", truncation=True,
                max_length=64, return_tensors="np",
            )
            self.assertEqual(model.predict.call_args.args[0]["input_ids"].shape, (1, 64))
            self.assertEqual(model.predict.call_args.args[0]["input_ids"].dtype, np.int32)


class ApplicationLifecycleTests(unittest.TestCase):
    def test_pool_closes_on_normal_shutdown(self):
        with patch.object(db, "pool") as pool, patch.object(db, "ensure_schema") as ensure_schema:
            with TestClient(main.app):
                pool.open.assert_called_once_with()
                pool.wait.assert_called_once_with()
                ensure_schema.assert_called_once_with()
                pool.close.assert_not_called()
            pool.close.assert_called_once_with()

    def test_pool_closes_when_startup_fails(self):
        for stage in ("wait", "schema"):
            with self.subTest(stage=stage):
                with patch.object(db, "pool") as pool, patch.object(db, "ensure_schema") as ensure_schema:
                    failing_call = pool.wait if stage == "wait" else ensure_schema
                    failing_call.side_effect = RuntimeError("startup failed")
                    with self.assertRaisesRegex(RuntimeError, "startup failed"):
                        with TestClient(main.app):
                            self.fail("Startup must not complete")
                    pool.close.assert_called_once_with()

    def test_pool_closes_when_lifespan_body_fails(self):
        with patch.object(db, "pool") as pool, patch.object(db, "ensure_schema"):
            with self.assertRaisesRegex(RuntimeError, "application failed"):
                with TestClient(main.app):
                    raise RuntimeError("application failed")
            pool.close.assert_called_once_with()


class SearchAPITests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(main.app)
        self.source = Vector([1, 0])
        self.image_lookup = patch.object(
            db, "get_image_embedding",
            return_value={"embedding": self.source, "listing_id": 10,
                          "url": "https://www.finn.no/10"},
        )
        self.search_patch = patch.object(db, "search_by_embedding", return_value=[
            {"id": 20, "similarity": 0.8, "text_similarity": 0.3},
        ])
        self.lookup = self.image_lookup.start()
        self.search = self.search_patch.start()
        self.addCleanup(self.image_lookup.stop)
        self.addCleanup(self.search_patch.stop)
        fake_embedder.embed_text.reset_mock()

    def test_reverse_search_returns_source_listing_url(self):
        response = self.client.get("/search/", params={"image_id": 11})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["source_url"], "https://www.finn.no/10")
        response = self.client.get("/search/", params={"query": "chair"})
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.json()["source_url"])

    def test_random_image_uses_filters_and_returns_seed_metadata(self):
        with patch.object(db, "get_random_image", return_value={
            "image_id": 11, "title": "Chair", "local_path": "images/chair.jpg",
        }) as sample:
            response = self.client.get("/random-image/", params=[
                ("category_id", "chairs"), ("category_id", "sofas"), ("free_only", "true"),
            ])
        self.assertEqual(response.status_code, 200)
        sample.assert_called_once_with(category_ids=["chairs", "sofas"], free_only=True)
        self.assertEqual(response.json(), {
            "image_id": 11, "title": "Chair", "image_src": "/images/chair.jpg",
        })
        self.assertEqual(self.client.get("/search/", params={"image_id": 11}).status_code, 200)
        self.assertIs(self.search.call_args.args[0], self.source)
        self.assertEqual(self.search.call_args.kwargs["exclude_listing_id"], 10)
        fake_embedder.embed_text.assert_not_called()

    def test_random_image_empty_filters_and_remote_photo(self):
        with patch.object(db, "get_random_image", return_value=None) as sample:
            response = self.client.get("/random-image/")
        self.assertEqual(response.status_code, 404)
        sample.assert_called_once_with(category_ids=None, free_only=False)
        with patch.object(db, "get_random_image", return_value={
            "image_id": 11, "title": "Chair", "local_path": None, "image_url": "https://example.com/chair.jpg",
        }):
            self.assertEqual(self.client.get("/random-image/").json()["image_src"],
                             "https://example.com/chair.jpg")

    def test_combined_search_keeps_vectors_and_scores_separate(self):
        response = self.client.get("/search/", params={
            "image_id": 11, "query": " sculptural ",
            "free_only": True,
        })
        self.assertEqual(response.status_code, 200)
        self.assertIs(self.search.call_args.args[0], self.source)
        fake_embedder.embed_text.assert_called_once_with("a photo of sculptural")
        options = self.search.call_args.kwargs
        self.assertEqual(options["limit"], 60)
        self.assertTrue(options["free_only"])
        self.assertEqual(options["min_score"], 0.1)
        self.assertEqual(options["exclude_listing_id"], 10)
        np.testing.assert_array_equal(options["text_embedding"], [0, 1])
        np.testing.assert_array_equal(self.source.to_numpy(), [1, 0])
        result = response.json()["results"][0]
        self.assertEqual(result["match_pct"], 80)
        self.assertEqual(result["text_similarity"], 0.3)

    def test_category_filters_and_metadata(self):
        self.search.return_value = [{
            "id": 20, "similarity": 0.8, "category_id": "chairs", "category_name": "Chairs",
        }]
        response = self.client.get("/search/", params=[
            ("query", "sculptural"), ("category_id", "chairs"), ("category_id", "sofas"),
        ])
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.search.call_args.kwargs["category_ids"], ["chairs", "sofas"])
        self.assertEqual(response.json()["results"][0]["category_name"], "Chairs")
        with patch.object(db, "get_categories", return_value=[{"id": "chairs", "name": "Chairs"}]):
            self.assertEqual(self.client.get("/categories/").json(), {
                "categories": [{"id": "chairs", "name": "Chairs"}],
            })

    def test_image_only_search(self):
        response = self.client.get("/search/", params={"image_id": 11})
        self.assertEqual(response.status_code, 200)
        self.assertIs(self.search.call_args.args[0], self.source)
        self.assertIsNone(self.search.call_args.kwargs["text_embedding"])
        self.assertEqual(self.search.call_args.kwargs["min_score"], 0.1)
        fake_embedder.embed_text.assert_not_called()

    def test_explicit_match_threshold(self):
        response = self.client.get("/search/", params={"image_id": 11, "min_match_pct": 20})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.search.call_args.kwargs["min_score"], 0.2)

    def test_text_only_search(self):
        response = self.client.get("/search/", params={"query": "comfortable"})
        self.assertEqual(response.status_code, 200)
        fake_embedder.embed_text.assert_called_once_with("a photo of a comfortable")
        fake_embedder.prob_to_cosine.assert_called_with(0.1)
        self.assertEqual(self.search.call_args.kwargs["min_score"], 0.25)
        self.assertIsNone(self.search.call_args.kwargs["text_embedding"])
        self.assertEqual(response.json()["results"][0]["match_pct"], 73)

    def test_single_encoder_api_contract(self):
        response = self.client.get("/search/", params={"image_id": 11, "query": "wooden"})
        self.assertEqual(response.status_code, 200)
        self.lookup.assert_called_once_with(11)
        self.assertNotIn("image_model", self.search.call_args.kwargs)
        self.assertNotIn("image_model", response.json())

    def test_missing_input_and_image(self):
        self.assertEqual(self.client.get("/search/").status_code, 400)
        self.lookup.return_value = None
        response = self.client.get("/search/", params={"image_id": 999})
        self.assertEqual(response.status_code, 404)


class CategoryIngestionTests(unittest.TestCase):
    def response(self, source_ids, total=4):
        response = Mock()
        response.json.return_value = {
            "docs": [{"ad_id": source_id} for source_id in source_ids],
            "metadata": {"result_size": {"match_count": total}},
        }
        return response

    def test_discovery_excludes_only_requested_subcategories(self):
        response = Mock()
        children = [
            {"name": "sub_category", "value": category_id, "display_name": category_id}
            for category_id in ("1.78.5196", "1.78.7756", "1.78.5222", "1.78.5223")
        ]
        response.json.return_value = {"filters": [{"filter_items": [{
            "name": "category", "value": "0.78", "filter_items": children,
        }]}]}
        with patch.object(populate_db, "http_get", return_value=response):
            categories = populate_db.get_subcategories(Mock())
        self.assertEqual([category["value"] for category in categories], ["1.78.5196", "1.78.7756"])

    def test_missing_taxonomy_is_an_error(self):
        response = Mock()
        response.json.return_value = {}
        with patch.object(populate_db, "http_get", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "category tree"):
                populate_db.get_subcategories(Mock())

    def test_pagination_and_overlap_reset_per_category(self):
        responses = [self.response(ids) for ids in ([4, 3], [1, 3], [8, 7], [5, 7])]
        with patch.object(populate_db, "ROWS", 2), patch.object(populate_db, "MAX_PAGES", 1):
            with patch.object(populate_db, "http_get", side_effect=responses) as get:
                for category_id in ("tables", "sofas"):
                    pages = list(populate_db.iter_category_pages(Mock(), category_id, threading.Event()))
                    self.assertEqual(len(pages), 2)
                params = [dict(call.kwargs["params"]) for call in get.call_args_list]
        self.assertEqual(
            [(item["sub_category"], item["page"], item["sort"]) for item in params],
            [("tables", 1, "NEWEST"), ("tables", 1, "PUBLISHED_ASC"),
             ("sofas", 1, "NEWEST"), ("sofas", 1, "PUBLISHED_ASC")],
        )

    def test_incomplete_scan_is_not_silently_accepted(self):
        with patch.object(populate_db, "ROWS", 2), patch.object(populate_db, "MAX_PAGES", 1):
            with patch.object(populate_db, "http_get", side_effect=[self.response([4, 3]), self.response([1, 2])]):
                with self.assertRaisesRegex(RuntimeError, "without overlap"):
                    list(populate_db.iter_category_pages(Mock(), "large", threading.Event()))

    def test_repeated_page_is_an_error(self):
        with patch.object(populate_db, "ROWS", 2):
            with patch.object(populate_db, "http_get", side_effect=[self.response([4, 3]), self.response([4, 3])]):
                with self.assertRaisesRegex(RuntimeError, "repeated"):
                    list(populate_db.iter_category_pages(Mock(), "tables", threading.Event()))

    def test_small_category_does_not_need_reverse_pass(self):
        with patch.object(populate_db, "http_get", return_value=self.response([1], total=1)) as get:
            pages = list(populate_db.iter_category_pages(Mock(), "small", threading.Event()))
            self.assertEqual(len(pages), 1)
            self.assertEqual(get.call_count, 1)

    def run_ingestion(self, pages, maximum=None):
        stack = ExitStack()
        self.addCleanup(stack.close)
        categories = [{"value": category_id, "display_name": category_id} for category_id in ("tables", "sofas")]
        arguments = ["populate_db.py"]
        if maximum is not None:
            arguments += ["--max-listings", str(maximum)]
        stack.enter_context(patch.dict(sys.modules, {"db": db}))
        stack.enter_context(patch.object(sys, "argv", arguments))
        stack.enter_context(patch.object(populate_db, "get_subcategories", return_value=categories))
        stack.enter_context(patch.object(populate_db, "iter_category_pages", side_effect=pages))
        stack.enter_context(patch.object(populate_db.threading, "Thread"))
        stack.enter_context(patch.object(populate_db, "ThreadPoolExecutor"))
        self.output_events = []
        self.progress = stack.enter_context(patch.object(populate_db, "tqdm"))
        self.progress.return_value.close.side_effect = lambda: self.output_events.append(("close", ""))
        stack.enter_context(patch(
            "builtins.print",
            side_effect=lambda message: self.output_events.append(("message", message)),
        ))
        stack.enter_context(patch.multiple(
            db, pool=Mock(), ensure_schema=Mock(), upsert_category=Mock(),
            upsert_listing_and_embedding_status=Mock(return_value=(1, False, True)),
            mark_stale_listings_since_timestamp=Mock(return_value=(0, 0)),
            delete_listing_images_for_listing=Mock(),
        ))
        self.upsert = db.upsert_listing_and_embedding_status
        self.stale = db.mark_stale_listings_since_timestamp
        self.deleted = db.delete_listing_images_for_listing
        populate_db.main()

    def test_global_cap_and_only_completed_category_stale_updates(self):
        def pages(rate, category_id, shutdown):
            source_ids = [1, 2] if category_id == "tables" else [3, 4]
            return iter([("NEWEST", [
                {"ad_id": source_id, "heading": "Listing", "timestamp": 100}
                for source_id in source_ids
            ])])

        self.run_ingestion(pages, maximum=3)
        self.assertEqual(self.upsert.call_count, 3)
        self.assertEqual([call.kwargs["category_id"] for call in self.upsert.call_args_list], ["tables", "tables", "sofas"])
        self.stale.assert_called_once()
        self.assertEqual(self.stale.call_args.kwargs["category_id"], "tables")
        self.deleted.assert_not_called()

    def test_failed_category_is_not_marked_stale(self):
        def pages(rate, category_id, shutdown):
            yield "NEWEST", [{"ad_id": category_id, "heading": "Listing", "timestamp": 100}]
            if category_id == "sofas":
                raise RuntimeError("network failed")

        with self.assertRaisesRegex(RuntimeError, "network failed"):
            self.run_ingestion(pages)
        self.stale.assert_called_once()
        self.assertEqual(self.stale.call_args.kwargs["category_id"], "tables")

    def test_progress_bars_close_before_summaries(self):
        self.run_ingestion(lambda *args: iter(()))
        for call in self.progress.call_args_list:
            self.assertTrue(call.kwargs["dynamic_ncols"])
            self.assertFalse(call.kwargs["leave"])
            self.assertIsNone(call.kwargs["disable"])
            self.assertIs(call.kwargs["file"], sys.stderr)
        summary_index = next(
            index for index, (kind, message) in enumerate(self.output_events)
            if kind == "message" and message.startswith("Listings summary:")
        )
        self.assertEqual(
            [kind for kind, _ in self.output_events[:summary_index]].count("close"), 2,
        )
        self.assertNotIn("close", [kind for kind, _ in self.output_events[summary_index:]])


class ProgressOutputTests(unittest.TestCase):
    def test_concurrent_download_errors_do_not_interrupt_bars(self):
        console = StringIO()
        stdout = StringIO()
        session = Mock()
        session.get.side_effect = populate_db.requests.ReadTimeout("timed out")
        with patch.object(populate_db, "get_thread_session", return_value=session):
            with patch.object(sys, "stderr", console), redirect_stdout(stdout):
                with (
                    populate_db.tqdm(total=4, desc="Listings", position=0,
                                     file=console, ncols=100, leave=False) as listings,
                    populate_db.tqdm(total=4, desc="Image downloads", position=1,
                                     file=console, ncols=100, leave=False) as downloads,
                ):
                    lock = threading.Lock()
                    workers = [threading.Thread(
                        target=populate_db.download_and_enqueue,
                        args=(1, "source", ["url"], "images", queue.Queue(),
                              downloads, lock, threading.Event()),
                    ) for _ in range(4)]
                    for worker in workers:
                        worker.start()
                    for worker in workers:
                        worker.join()
                    listings.update(4)
                    self.assertEqual(downloads.n, 4)
        self.assertEqual(stdout.getvalue(), "")
        rendered_lines = console.getvalue().replace("\x1b[A", "").splitlines()
        self.assertEqual(rendered_lines.count(
            "Error downloading image url: timed out",
        ), 4)


class EmbeddingPipelineTests(unittest.TestCase):
    def run_worker(self, count, failed_batch=None, sparse=False, rejected_ids=()):
        events = []
        items = [(index, f"url-{index}", f"path-{index}") for index in range(count)]
        source = queue.Queue()
        for item in items:
            source.put(item)
        source.put(None)
        if sparse:
            original_get = source.get

            def get_item(*args, **kwargs):
                if kwargs.get("block") is False:
                    raise queue.Empty
                return original_get(*args, **kwargs)

            source.get = get_item

        def submit(paths, skip_blank=False):
            self.assertTrue(skip_blank)
            events.append(("submit", paths))
            future = Mock()
            if paths == failed_batch:
                future.result.side_effect = ValueError("bad image")
            else:
                indices = [index for index, path in enumerate(paths)
                           if int(path.split("-")[1]) not in rejected_ids]
                future.result.return_value = (
                    [paths[index] for index in indices], 0.0, 0.0, indices,
                    [index for index in range(len(paths)) if index not in indices],
                    [],
                )
            return future

        def embed(paths):
            events.append(("embed", paths))
            return [np.array([int(path.split("-")[1]), 1]) for path in paths]

        encoder = types.SimpleNamespace(
            PREPROCESS_WORKERS=2, submit_image_paths=submit,
            embed_prepared_images=embed,
        )
        database = types.SimpleNamespace(
            insert_listing_images=Mock(side_effect=lambda rows, **kwargs: len(rows)),
        )
        with patch.dict(sys.modules, {"db": database, "embedder": encoder}):
            with patch.object(populate_db.tqdm, "write"), patch.object(Path, "unlink") as unlink:
                with patch.object(populate_db, "increment_blank_or_solid_skipped_count") as skipped:
                    populate_db.embed_worker(source, 2)
            self.assertEqual(unlink.call_count, len(rejected_ids))
            self.assertEqual(skipped.call_count, len(rejected_ids))
        rows = [row for call in database.insert_listing_images.call_args_list
                for row in call.args[0]]
        return events, rows

    def test_prepares_ahead_and_preserves_order_and_tail(self):
        events, rows = self.run_worker(7)
        self.assertEqual([kind for kind, _ in events[:3]], ["submit"] * 3)
        self.assertEqual([row[0] for row in rows], list(range(7)))
        self.assertEqual([row[3][0] for row in rows], list(range(7)))
        self.assertEqual(events[-1], ("embed", ["path-6"]))
        outstanding = 0
        for kind, _ in events:
            outstanding += 1 if kind == "submit" else -1
            self.assertLessEqual(outstanding, 3)
        self.assertEqual(outstanding, 0)

    def test_empty_queue_finishes_without_submitting(self):
        self.assertEqual(self.run_worker(0), ([], []))

    def test_sparse_downloads_do_not_delay_submitted_batch(self):
        events, rows = self.run_worker(4, sparse=True)
        self.assertEqual([kind for kind, _ in events], ["submit", "embed"] * 2)
        self.assertEqual([row[0] for row in rows], list(range(4)))

    def test_failed_preparation_does_not_discard_later_batches(self):
        _, rows = self.run_worker(5, failed_batch=["path-0", "path-1"])
        self.assertEqual([row[0] for row in rows], [2, 3, 4])

    def test_blank_filter_preserves_metadata_and_skips_empty_batches(self):
        _, rows = self.run_worker(5, rejected_ids=(0, 2, 3))
        self.assertEqual([row[0] for row in rows], [1, 4])
        self.assertEqual([row[3][0] for row in rows], [1, 4])

    def test_preparation_filters_blank_images_before_processing(self):
        import image_preprocessing
        from PIL import Image

        pixels = np.ones((1, 3, 2, 2), dtype=np.float32)
        processor = Mock(return_value={"pixel_values": pixels})
        with TemporaryDirectory() as directory:
            blank_path = Path(directory) / "blank.png"
            patterned_path = Path(directory) / "patterned.png"
            with Image.new("RGB", (8, 8), "white") as image:
                image.save(blank_path)
            pattern = np.tile(np.array([0, 255], dtype=np.uint8), (8, 4))
            with Image.fromarray(pattern) as image:
                image.save(patterned_path)
            with patch.object(image_preprocessing, "_processor", processor):
                result = image_preprocessing.prepare_paths(
                    [blank_path, patterned_path], "test-model", skip_blank=True,
                )
                self.assertIs(result[0], pixels)
                self.assertEqual(result[3], [1])
                self.assertEqual(result[4], [0])
                processor.assert_called_once()
                self.assertEqual(len(processor.call_args.kwargs["images"]), 1)
                processor.reset_mock()
                result = image_preprocessing.prepare_paths(
                    [blank_path], "test-model", skip_blank=True,
                )
                self.assertEqual(len(result[0]), 0)
                self.assertEqual(result[3], [])
                processor.assert_not_called()
                with patch("builtins.print") as output:
                    result = image_preprocessing.prepare_paths(
                        [Path(directory) / "missing.png", patterned_path],
                        "test-model", skip_blank=True,
                    )
                output.assert_not_called()
                self.assertEqual(result[3], [1])
                self.assertEqual(result[4], [])
                self.assertEqual(len(result[5]), 1)
                self.assertIn("missing.png", result[5][0])
                processor.assert_called_once()

@unittest.skipUnless(os.getenv("BEDREFIN_TEST_DATABASE"), "database tests are opt-in")
class SearchDatabaseTests(unittest.TestCase):
    def setUp(self):
        self.conn = psycopg.connect(os.environ["DATABASE_URL"])
        self.addCleanup(self.conn.close)
        register_vector(self.conn)
        self.conn.execute("CREATE TEMP TABLE categories (id text PRIMARY KEY, name text) ON COMMIT DROP")
        self.conn.execute("INSERT INTO categories VALUES ('tables', 'Tables'), ('sofas', 'Sofas')")
        self.conn.execute("""
            CREATE TEMP TABLE listings (
            id serial PRIMARY KEY, title text, price integer, url text,
            description text, location text, source_id text UNIQUE,
            source_timestamp bigint, is_stale boolean DEFAULT FALSE,
            updated_at timestamptz DEFAULT NOW(), category_id text REFERENCES categories(id)
            ) ON COMMIT DROP
        """)
        self.conn.execute("""
            CREATE TEMP TABLE listing_images (
                id integer PRIMARY KEY, listing_id integer,
                embedding vector(2), image_url text, local_path text
            ) ON COMMIT DROP
        """)
        with self.conn.cursor() as cursor:
            cursor.executemany(
                "INSERT INTO listings (id, price) VALUES (%s, %s)",
                [(listing_id, 0 if listing_id % 2 == 0 else 100)
                 for listing_id in range(1, 307)] + [(1000, 0)],
            )
            cursor.executemany(
                "INSERT INTO listing_images (id, listing_id, embedding) VALUES (%s, %s, %s)",
                [(listing_id, listing_id, Vector([1, max(listing_id - 2, 0) / 1000]))
                 for listing_id in range(1, 307)]
                + [(1000, 1000, Vector([0, 1])), (9000, 2, Vector([0, 1]))],
            )
            self.conn.execute("UPDATE listings SET category_id = CASE WHEN id >= 302 THEN 'sofas' ELSE 'tables' END")
            self.conn.execute("SELECT setval(pg_get_serial_sequence('pg_temp.listings', 'id'), 1000)")
        self.pool_patch = patch.object(db, "pool")
        self.pool = self.pool_patch.start()
        self.addCleanup(self.pool_patch.stop)
        self.pool.connection.side_effect = lambda: nullcontext(self.conn)

    def search(self, **options):
        return db.search_by_embedding(
            Vector([1, 0]), exclude_listing_id=1,
            text_embedding=np.array([0, 1], dtype=np.float32), **options,
        )

    def test_random_image_filters_seed_listing_and_skips_missing_vectors(self):
        self.conn.execute("UPDATE listing_images SET embedding = NULL WHERE listing_id >= 302")
        self.assertIsNone(db.get_random_image(category_ids=["sofas"]))
        self.conn.execute("UPDATE listing_images SET embedding = %s WHERE id IN (302, 303)", (Vector([1, 0]),))
        for _ in range(3):
            self.assertEqual(db.get_random_image(category_ids=["sofas"], free_only=True)["image_id"], 302)
        self.assertIsNone(db.get_random_image(category_ids=["unknown"], free_only=True))
        self.assertIsNotNone(db.get_random_image())
        self.conn.execute("UPDATE listing_images SET embedding = NULL WHERE id = 302")
        self.assertIsNone(db.get_random_image(category_ids=["sofas"], free_only=True))
        self.assertEqual(db.get_random_image(category_ids=["sofas"])["image_id"], 303)

    def test_candidate_cap_and_same_image_reranking(self):
        results = self.search(limit=400, min_score=0.9)
        self.assertEqual(len(results), 300)
        self.assertEqual([row["id"] for row in results[:2]], [301, 300])
        self.assertEqual(results[-1]["id"], 2)
        self.assertEqual(results[-1]["image_id"], 2)
        self.assertAlmostEqual(results[-1]["text_similarity"], 0)

    def test_threshold_and_final_limit(self):
        results = self.search(limit=3, min_score=0.99)
        self.assertEqual([row["id"] for row in results], [144, 143, 142])
        self.assertTrue(all(row["similarity"] >= 0.99 for row in results))
        self.assertEqual(self.search(limit=3, min_score=1.01), [])

    def test_price_filter_applies_before_ranking(self):
        results = self.search(limit=3, min_score=0.9, free_only=True)
        self.assertEqual([row["id"] for row in results], [306, 304, 302])
        self.assertTrue(all(row["price"] == 0 for row in results))

    def test_single_vector_search_is_unchanged(self):
        results = db.search_by_embedding(
            Vector([1, 0]), limit=3, min_score=0.9, exclude_listing_id=1,
        )
        self.assertEqual([row["id"] for row in results], [2, 3, 4])

    def test_category_filter_precedes_candidate_limit(self):
        results = self.search(limit=10, min_score=0.9, category_ids=["sofas"])
        self.assertEqual([row["id"] for row in results], [306, 305, 304, 303, 302])
        self.assertTrue(all(row["category_name"] == "Sofas" for row in results))
        results = self.search(limit=3, min_score=0.9, category_ids=["sofas"], free_only=True)
        self.assertEqual([row["id"] for row in results], [306, 304, 302])
        self.assertEqual(self.search(limit=3, category_ids=["unknown"]), [])

    def test_category_assignment_preserves_embeddings_and_scopes_stale_checks(self):
        original = db.upsert_listing_and_embedding_status(
            "Existing", 100, source_id="existing", source_timestamp=100,
        )
        self.conn.execute(
            "INSERT INTO listing_images (id, listing_id, embedding) VALUES (10001, %s, %s)",
            (original[0], Vector([1, 0])),
        )
        assigned = db.upsert_listing_and_embedding_status(
            "Existing", 100, source_id="existing", source_timestamp=100, category_id="tables",
        )
        self.assertEqual(assigned, (original[0], False, True))
        other = db.upsert_listing_and_embedding_status(
            "Other", 100, source_id="other", source_timestamp=100, category_id="sofas",
        )
        self.assertEqual(db.mark_stale_listings_since_timestamp([], 100, category_id="tables"), (1, 0))
        self.assertFalse(self.conn.execute("SELECT is_stale FROM listings WHERE id = %s", (other[0],)).fetchone()[0])
        self.assertEqual(db.mark_stale_listings_since_timestamp(["existing"], 100, category_id="tables"), (0, 1))
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM listing_images WHERE id = 10001").fetchone()[0], 1)


if __name__ == "__main__":
    unittest.main()