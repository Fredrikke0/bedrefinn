import argparse
import hashlib
import json
import os
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit, urlunsplit

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
STUDY_ROOT = ROOT / "benchmark_data" / "rendition_comparison"
STUDY_KIND = "bedrefinn-rendition-comparison-v1"
RENDITIONS = ("default", "320w")


def write_json(path: Path, value) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_study(directory: Path):
    manifest = json.loads((directory / "study.json").read_text(encoding="utf-8"))
    if manifest.get("kind") != STUDY_KIND:
        raise ValueError("Not a rendition comparison study")
    return manifest


def rendition_url(url: str, rendition: str) -> str:
    parsed = urlsplit(url)
    prefix = "/dynamic/default/"
    if rendition not in RENDITIONS or parsed.scheme != "https" or parsed.netloc != "images.finncdn.no" or not parsed.path.startswith(prefix):
        raise ValueError("Expected a default FINN image URL and a study rendition")
    return urlunsplit(parsed._replace(path=f"/dynamic/{rendition}/" + parsed.path[len(prefix):]))


def rank_results(items, vectors, query_vector, *, excluded_id=None, category=None, free_only=False, limit=10):
    values = np.asarray(vectors, dtype=np.float32)
    query = np.asarray(query_vector, dtype=np.float32)
    if values.shape != (len(items), len(query)):
        raise ValueError("Study image/query vector dimensions differ")
    scores = values @ query
    eligible = [index for index, item in enumerate(items)
                if item["id"] != excluded_id
                and (not category or item["category_id"] == category)
                and (not free_only or item["price"] == 0)]
    eligible.sort(key=lambda index: (-float(scores[index]), items[index]["id"]))
    return [{"id": items[index]["id"], "title": items[index]["title"],
             "price": items[index]["price"], "category_name": items[index]["category_name"],
             "url": items[index]["url"], "score": float(scores[index])}
            for index in eligible[:limit]]


def cleanup_study(directory: Path) -> None:
    if directory.is_symlink() or directory.resolve() != STUDY_ROOT.resolve():
        raise ValueError("Cleanup is restricted to the dedicated rendition study directory")
    read_study(directory)
    with study_lock(directory):
        shutil.rmtree(directory)


def comparison_order(key: str):
    return RENDITIONS if int(hashlib.sha256(key.encode()).hexdigest()[0], 16) % 2 else tuple(reversed(RENDITIONS))


@contextmanager
def study_lock(directory: Path):
    import fcntl

    with (directory / ".lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("The study is in use; stop its server or preparation before cleanup") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def select_items(limit: int):
    import psycopg
    from psycopg.rows import dict_row
    from benchmark import connection_settings

    settings, _ = connection_settings()
    with psycopg.connect(**settings, row_factory=dict_row) as connection:
        connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
        rows = connection.execute("""
            WITH photos AS (
                SELECT DISTINCT ON (listing_id) listing_id, image_url, local_path
                FROM listing_images
                WHERE image_url LIKE 'https://images.finncdn.no/dynamic/default/%%'
                  AND local_path IS NOT NULL
                ORDER BY listing_id, id
            ), candidates AS (
                SELECT l.id, l.title, l.price, l.url, l.category_id,
                       COALESCE(c.name, 'Unknown') AS category_name,
                       p.image_url, p.local_path,
                       row_number() OVER (PARTITION BY l.category_id ORDER BY md5(l.id::text)) AS sample_rank
                FROM listings l JOIN photos p ON p.listing_id = l.id
                LEFT JOIN categories c ON c.id = l.category_id
            ) SELECT * FROM candidates ORDER BY sample_rank, category_id, id LIMIT %s
        """, (limit,)).fetchall()
    if len(rows) != limit:
        raise ValueError(f"Requested {limit} listings; only {len(rows)} eligible listings found")
    return [{key: value for key, value in row.items() if key != "sample_rank"} for row in rows]


def prepare_study(directory: Path, limit: int, workers: int) -> None:
    from PIL import Image
    import populate_db
    import requests

    directory.mkdir(parents=True, exist_ok=True)
    with study_lock(directory):
        path = directory / "study.json"
        if path.exists():
            manifest = read_study(directory)
            if manifest.get("requested_listings", len(manifest["items"])) != limit:
                raise ValueError("Existing study has a different size; clean it up first")
            if manifest.get("ready"):
                print(f"Study already ready: {directory}", flush=True)
                return
        else:
            if any(entry.name != ".lock" for entry in directory.iterdir()):
                raise FileExistsError("Refusing to reuse an unmarked study directory")
            manifest = {"kind": STUDY_KIND, "ready": False, "requested_listings": limit, "items": select_items(limit),
                        "scope": "One photo per listing, balanced by category from existing default-rendition records. Source read-only. Independent files and freshly computed vectors for both renditions. Not a download-speed benchmark or labeled relevance evaluation."}
            write_json(path, manifest)
        for rendition in RENDITIONS:
            (directory / "images" / rendition).mkdir(parents=True, exist_ok=True)
        stop = threading.Event()

        def prepare_item(item):
            for rendition in RENDITIONS:
                if stop.is_set():
                    raise RuntimeError("Study download stopped after server backoff")
                destination = directory / "images" / rendition / f"{item['id']}.img"
                if destination.exists():
                    try:
                        with Image.open(destination) as image:
                            image.verify()
                        continue
                    except (OSError, ValueError):
                        destination.unlink()
                original = (ROOT / item["local_path"]).resolve()
                if rendition == "default" and original.is_file():
                    content = original.read_bytes()
                else:
                    try:
                        content = populate_db.fetch_image_bytes(rendition_url(item["image_url"], rendition))
                    except requests.RequestException as error:
                        if error.response is not None and error.response.status_code in (429, 503):
                            stop.set()
                        raise
                from io import BytesIO
                with Image.open(BytesIO(content)) as image:
                    image.verify()
                temporary = destination.with_suffix(".tmp")
                temporary.write_bytes(content)
                temporary.replace(destination)
            return item["id"]

        def prepare_pair(item):
            try:
                return {"id": prepare_item(item)}
            except (requests.RequestException, OSError, ValueError) as error:
                if stop.is_set():
                    raise
                return {"id": item["id"], "error": type(error).__name__}

        completed, excluded = set(), []
        selected_count = len(manifest["items"])
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for count, result in enumerate(executor.map(prepare_pair, manifest["items"]), 1):
                if "error" in result:
                    excluded.append(result)
                else:
                    completed.add(result["id"])
                if count % 100 == 0 or count == selected_count:
                    print(f"Image pairs: {count}/{selected_count}; unavailable={len(excluded)}", flush=True)
        manifest["items"] = [item for item in manifest["items"] if item["id"] in completed]
        manifest["excluded"] = manifest.get("excluded", []) + excluded
        limit = len(manifest["items"])
        if limit < 2:
            raise ValueError("Not enough complete image pairs to build the study")
        write_json(path, manifest)

        import embedder
        vectors = {}
        for rendition in RENDITIONS:
            siglip_batches = []
            for offset in range(0, limit, 32):
                images = []
                try:
                    for item in manifest["items"][offset:offset + 32]:
                        with Image.open(directory / "images" / rendition / f"{item['id']}.img") as image:
                            images.append(image.convert("RGB"))
                    siglip_batches.append(embedder.embed_images(images))
                finally:
                    for image in images:
                        image.close()
            vectors[f"siglip2_{rendition}"] = np.concatenate(siglip_batches)
            print(f"Embedded {limit} {rendition} images", flush=True)
        quality = {}
        for model in ("siglip2",):
            for rendition in RENDITIONS:
                values = vectors[f"{model}_{rendition}"]
                if not np.isfinite(values).all():
                    raise ValueError("Nonfinite study vectors")
                np.testing.assert_allclose(np.linalg.norm(values, axis=1), 1, atol=1e-5)
            cosine = np.sum(vectors[f"{model}_default"] * vectors[f"{model}_320w"], axis=1)
            quality[model] = {"median_cosine": float(np.median(cosine)), "min_cosine": float(cosine.min())}
        temporary = directory / "vectors.tmp.npz"
        np.savez(temporary, **vectors)
        temporary.replace(directory / "vectors.npz")
        manifest.update({"ready": True, "model_id": embedder.model_id,
                         "coreml_directory": str(embedder.model_dir.resolve()), "quality": quality,
                         "listings": limit, "image_files": limit * 2})
        write_json(path, manifest)
        print(json.dumps({"directory": str(directory), "listings": limit, "quality": quality}), flush=True)


class Judgment(BaseModel):
    key: str = Field(pattern=r"^[0-9a-f]{64}$")
    choice: Literal["A", "B", "tie"]
    query: str = Field(default="", max_length=500)
    image_id: int | None = None
    category: str | None = None
    free_only: bool = False
    revealed: bool = False


def create_app(directory: Path = STUDY_ROOT) -> FastAPI:
    manifest = read_study(directory)
    if not manifest.get("ready"):
        raise ValueError("Prepare the study before starting its server")
    items = manifest["items"]
    indices = {item["id"]: index for index, item in enumerate(items)}
    with np.load(directory / "vectors.npz", allow_pickle=False) as archive:
        vectors = {name: archive[name] for name in archive.files}
    os.environ["BEDREFIN_MODEL_ID"] = manifest["model_id"]
    os.environ["BEDREFIN_COREML_DIR"] = manifest["coreml_directory"]
    import embedder
    if embedder.model_id != manifest["model_id"]:
        raise ValueError("Study and text encoder checkpoints differ")
    vote_lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(app):
        with study_lock(directory):
            yield

    app = FastAPI(lifespan=lifespan)

    @app.get("/")
    def index():
        return FileResponse(ROOT / "static" / "rendition_comparison.html")

    @app.get("/metadata")
    def metadata():
        return {"listings": len(items), "model_id": manifest["model_id"],
                "categories": [{"id": key, "name": name} for key, name in sorted(
                    {(item["category_id"], item["category_name"]) for item in items}, key=lambda entry: entry[1])],
                "quality": manifest["quality"]}

    @app.get("/search")
    def search(query: str = "", image_id: int | None = None,
               category: str | None = None,
               free_only: bool = False, limit: int = Query(default=10, ge=1, le=20), seed: str = ""):
        query = query.strip()
        if len(query) > 500 or len(seed) > 100:
            raise HTTPException(400, "Query is too long")
        if image_id is not None and image_id not in indices:
            raise HTTPException(404, "Reference image not in study")
        if image_id is None and not query:
            raise HTTPException(400, "Provide a query or image")
        model = "siglip2"
        if image_id is None:
            reference = embedder.embed_text(f"a photo of a {query}")
        else:
            reference = vectors[f"{model}_default"][indices[image_id]]
        key = hashlib.sha256(json.dumps([query, image_id, model, category, free_only, limit, seed]).encode()).hexdigest()
        order = comparison_order(key)
        results = {}
        text = embedder.embed_text(f"a photo of {query}") if image_id is not None and query else None
        for side, rendition in zip(("A", "B"), order, strict=True):
            matches = rank_results(items, vectors[f"{model}_{rendition}"], reference,
                                   excluded_id=image_id, category=category, free_only=free_only,
                                   limit=300 if text is not None else limit)
            if text is not None:
                for match in matches:
                    match["text_score"] = float(vectors[f"siglip2_{rendition}"][indices[match["id"]]] @ text)
                matches.sort(key=lambda item: (-item["text_score"], -item["score"], item["id"]))
                matches = matches[:limit]
            for match in matches:
                match["image_src"] = f"/photo/{side}/{match['id']}?key={key}"
            results[side] = matches
        shared = len({item["id"] for item in results["A"]} & {item["id"] for item in results["B"]})
        return {"key": key, "model": model, "results": results, "shared": shared,
                "renditions": dict(zip(("A", "B"), order, strict=True))}

    @app.get("/photo/{side}/{listing_id}")
    def photo(side: Literal["A", "B", "reference"], listing_id: int, key: str = Query(default="", max_length=64)):
        from PIL import Image

        if listing_id not in indices:
            raise HTTPException(404, "Image not in study")
        rendition = "default" if side == "reference" else comparison_order(key)[0 if side == "A" else 1]
        path = directory / "images" / rendition / f"{listing_id}.img"
        with Image.open(path) as image:
            media_type = Image.MIME.get(image.format, "application/octet-stream")
        return FileResponse(path, media_type=media_type)

    @app.post("/judgments")
    def judge(judgment: Judgment):
        with vote_lock:
            path = directory / "judgments.json"
            votes = json.loads(path.read_text()) if path.exists() else []
            vote = judgment.model_dump()
            vote["created_at"] = datetime.now(timezone.utc).isoformat()
            vote["renditions"] = dict(zip(("A", "B"), comparison_order(judgment.key), strict=True))
            vote["selected_rendition"] = vote["renditions"].get(judgment.choice, "tie")
            votes.append(vote)
            write_json(path, votes)
        return {"saved": len(votes)}

    @app.get("/judgments")
    def judgments():
        path = directory / "judgments.json"
        return json.loads(path.read_text()) if path.exists() else []

    return app


def main():
    parser = argparse.ArgumentParser(description="Isolated default/320w relevance study; production is read-only")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--listings", type=int, default=1000)
    prepare.add_argument("--workers", type=int, default=8)
    serve = commands.add_parser("serve")
    serve.add_argument("--port", type=int, default=8005)
    commands.add_parser("cleanup", help="Remove the marked study directory after stopping its server")
    args = parser.parse_args()
    if args.command == "prepare":
        if args.listings < 1 or args.workers < 1:
            parser.error("Listings and workers must be positive")
        prepare_study(STUDY_ROOT, args.listings, args.workers)
    elif args.command == "cleanup":
        cleanup_study(STUDY_ROOT)
        print("Removed the rendition comparison study; production data unchanged.")
    else:
        import uvicorn
        uvicorn.run(create_app(), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()