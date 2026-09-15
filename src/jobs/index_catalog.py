from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

from PIL import Image

from src.backend.config import get_settings
from src.backend.db import connect_db, initialize_database, vector_literal
from src.backend.ml import get_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embed selected catalog images into pgvector")
    parser.add_argument("--platform", choices=["musinsa", "ably"], required=True)
    parser.add_argument("--products", type=Path)
    parser.add_argument("--selections", type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def defaults(args: argparse.Namespace) -> tuple[Path, Path]:
    base = Path("data") / args.platform / "tops"
    return args.products or base / "products.csv", args.selections or base / "selected/selections.jsonl"


def load_records(products_path: Path, selections_path: Path, platform: str) -> list[dict]:
    with products_path.open(encoding="utf-8-sig", newline="") as stream:
        products = {row["goods_no"]: row for row in csv.DictReader(stream)}
    records = []
    for line in selections_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        selection = json.loads(line)
        if selection.get("status") != "selected" or not selection.get("local_path"):
            continue
        product = products.get(str(selection["goods_no"]))
        source = Path(selection["local_path"])
        if product and source.is_file():
            records.append({"platform": platform, "product": product, "source": source})
    return records


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")
    products_path, selections_path = defaults(args)
    records = load_records(products_path, selections_path, args.platform)
    if args.limit is not None:
        records = records[:args.limit]
    if not records:
        print("No selected images to index.")
        return 0

    settings = get_settings()
    initialize_database()
    models = get_models()
    indexed = 0
    for start in range(0, len(records), args.batch_size):
        batch = records[start:start + args.batch_size]
        images = [Image.open(record["source"]).convert("RGB") for record in batch]
        embeddings = models.embed(images)
        with connect_db() as connection:
            for record, embedding in zip(batch, embeddings):
                product = record["product"]
                suffix = record["source"].suffix.lower() or ".jpg"
                destination = settings.storage_root / "products" / args.platform / f"{product['goods_no']}{suffix}"
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(record["source"], destination)
                connection.execute("""
                    INSERT INTO products (
                        platform, goods_no, goods_name, brand_name, price,
                        product_url, image_path, embedding, embedding_model
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector, %s)
                    ON CONFLICT (platform, goods_no) DO UPDATE SET
                        goods_name = EXCLUDED.goods_name,
                        brand_name = EXCLUDED.brand_name,
                        price = EXCLUDED.price,
                        product_url = EXCLUDED.product_url,
                        image_path = EXCLUDED.image_path,
                        embedding = EXCLUDED.embedding,
                        embedding_model = EXCLUDED.embedding_model,
                        updated_at = now()
                """, (
                    args.platform, product["goods_no"], product["goods_name"],
                    product.get("brand_name", ""), int(product["final_price"] or product["price"] or 0) or None,
                    product["product_url"], str(destination.resolve()),
                    vector_literal(embedding.tolist()), settings.fashion_clip_model,
                ))
        indexed += len(batch)
        print(f"Indexed {indexed}/{len(records)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
