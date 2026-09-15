from __future__ import annotations

import io
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from src.backend.config import get_settings
from src.backend.db import connect_db, count_products, initialize_database, search_products
from src.backend.ml import get_models


class SearchResult(BaseModel):
    platform: str
    goods_no: str
    goods_name: str
    brand_name: str
    price: int | None
    product_url: str
    image_url: str
    similarity: float


class SearchResponse(BaseModel):
    query_id: uuid.UUID
    used_top_mask: bool
    top_ratio: float
    elapsed_ms: int
    results: list[SearchResult]


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    settings.storage_root.mkdir(parents=True, exist_ok=True)
    initialize_database()
    yield


app = FastAPI(title="Fashion Similarity Search API", version="0.1.0", lifespan=lifespan)
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, object]:
    return {"status": "ok", "products": count_products()}


@app.post("/api/search", response_model=SearchResponse)
async def search(
    image: UploadFile = File(...),
    limit: int = Query(20, ge=1, le=50),
    platform: str | None = Query(None, pattern="^(musinsa|ably)$"),
) -> SearchResponse:
    started = time.perf_counter()
    body = await image.read(settings.max_upload_bytes + 1)
    if len(body) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="image is too large")
    try:
        source = Image.open(io.BytesIO(body)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="invalid image") from exc

    models = get_models()
    prepared = models.prepare_query(source)
    embedding = models.embed([prepared.image])[0].tolist()
    rows = search_products(embedding, limit, platform)
    query_id = uuid.uuid4()
    elapsed_ms = round((time.perf_counter() - started) * 1000)

    with connect_db() as connection:
        connection.execute(
            "INSERT INTO search_events (query_id, platform_filter, result_count, elapsed_ms) VALUES (%s, %s, %s, %s)",
            (query_id, platform, len(rows), elapsed_ms),
        )

    results = [SearchResult(
        platform=row["platform"], goods_no=row["goods_no"], goods_name=row["goods_name"],
        brand_name=row["brand_name"], price=row["price"], product_url=row["product_url"],
        image_url=f"/media/{row['platform']}/{row['goods_no']}", similarity=float(row["similarity"]),
    ) for row in rows]
    return SearchResponse(
        query_id=query_id, used_top_mask=prepared.used_top_mask,
        top_ratio=prepared.top_ratio, elapsed_ms=elapsed_ms, results=results,
    )


@app.get("/media/{platform}/{goods_no}")
def product_image(platform: str, goods_no: str) -> FileResponse:
    if platform not in {"musinsa", "ably"} or not goods_no.isdigit():
        raise HTTPException(status_code=404, detail="image not found")
    with connect_db() as connection:
        row = connection.execute(
            "SELECT image_path FROM products WHERE platform = %s AND goods_no = %s",
            (platform, goods_no),
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="image not found")
    path = Path(row["image_path"]).resolve()
    allowed_root = settings.storage_root.resolve()
    if not path.is_relative_to(allowed_root) or not path.is_file():
        raise HTTPException(status_code=404, detail="image not found")
    return FileResponse(path)
