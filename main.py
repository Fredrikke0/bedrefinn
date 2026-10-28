from fastapi import FastAPI, HTTPException, Query
from contextlib import asynccontextmanager
import db
import embedder
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.pool.open()
    try:
        db.pool.wait()
        db.ensure_schema()
        yield
    finally:
        db.pool.close()

app = FastAPI(lifespan=lifespan)
app.mount("/images", StaticFiles(directory="images"), name="images")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    return FileResponse("static/index.html")


@app.get("/search/")
def search(
    query: str | None = None,
    limit: int = 60,
    free_only: bool = False,
    min_match_pct: float = 10.0,
    image_id: int | None = None,
    category_id: list[str] | None = Query(default=None),
):
    query = query.strip() if query else ""
    exclude_listing_id = None
    source_url = None
    text_embedding = None
    if image_id is not None:
        image = db.get_image_embedding(image_id)
        if image is None:
            raise HTTPException(status_code=404, detail="Image embedding not found")
        query_emb = image["embedding"]
        exclude_listing_id = image["listing_id"]
        source_url = image.get("url")
        if query:
            text_embedding = embedder.embed_text(f"a photo of {query}")
    elif query:
        query_emb = embedder.embed_text(f"a photo of a {query}")
    else:
        raise HTTPException(status_code=400, detail="Provide query or image_id")
    min_prob = min(max(min_match_pct / 100.0, 0.0), 0.99)
    min_score = min_prob if image_id is not None else embedder.prob_to_cosine(min_prob)
    results = db.search_by_embedding(
        query_emb,
        limit=limit,
        free_only=free_only,
        min_score=min_score,
        exclude_listing_id=exclude_listing_id,
        text_embedding=text_embedding,
        category_ids=category_id,
    )

    formatted_results = []
    for row in results:
        local_path = row.get("local_path")
        image_src = f"/{local_path.lstrip('/')}" if local_path else row.get("image_url")
        similarity = float(row.get("similarity") or 0.0)
        match_score = (
            min(max(similarity, 0.0), 1.0)
            if image_id is not None else embedder.cosine_to_prob(similarity)
        )
        formatted_results.append(
            {
                "id": row.get("id"),
                "image_id": row.get("image_id"),
                "title": row.get("title"),
                "image_src": image_src,
                "url": row.get("url"),
                "price": row.get("price"),
                "category_id": row.get("category_id"),
                "category_name": row.get("category_name"),
                "similarity": row.get("similarity"),
                "text_similarity": row.get("text_similarity"),
                "match_pct": round(match_score * 100),
            }
        )
    return {"results": formatted_results, "source_url": source_url}


@app.get("/categories/")
def get_categories():
    return {"categories": db.get_categories()}


@app.get("/random-image/")
def get_random_image(
    category_id: list[str] | None = Query(default=None),
    free_only: bool = False,
):
    image = db.get_random_image(category_ids=category_id, free_only=free_only)
    if image is None:
        raise HTTPException(status_code=404, detail="No images match the selected filters")
    local_path = image.get("local_path")
    return {
        "image_id": image["image_id"],
        "title": image["title"],
        "image_src": f"/{local_path.lstrip('/')}" if local_path else image.get("image_url"),
    }


@app.get("/listings")
def get_listings():
    return {"listings": db.get_all_listings()}