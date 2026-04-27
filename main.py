from fastapi import FastAPI
from contextlib import asynccontextmanager
import db
import embedder
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.pool.open()
    db.pool.wait()
    yield
    db.pool.close()

app = FastAPI(lifespan=lifespan)
app.mount("/images", StaticFiles(directory="images"), name="images")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    return FileResponse("static/index.html")


@app.get("/search/")
def search(
    query: str,
    limit: int = 60,
    free_only: bool = False,
    min_score: float = 0.0,
):
    query_emb = embedder.embed_text(f"a photo of a {query}")
    results = db.search_by_embedding(
        query_emb,
        limit=limit,
        free_only=free_only,
        min_score=min_score,
    )

    formatted_results = []
    for row in results:
        local_path = row.get("local_path")
        image_src = f"/{local_path.lstrip('/')}" if local_path else row.get("image_url")
        formatted_results.append(
            {
                "id": row.get("id"),
                "title": row.get("title"),
                "image_src": image_src,
                "url": row.get("url"),
                "similarity": row.get("similarity"),
            }
        )
    return {"results": formatted_results}

@app.get("/listings")
def get_listings():
    listings = db.get_all_listings()

    return {"listings": listings}