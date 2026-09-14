#!/usr/bin/env python3
"""Crawl Musinsa tops product metadata.

The crawler starts from the public category page, extracts its embedded Next.js
data, and follows the signed ``nextPageUrl`` returned by Musinsa.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from src.common.http import Fetcher, NEXT_DATA_RE


CATEGORY_URL = "https://www.musinsa.com/category/001/goods?gf=A"
CSV_FIELDS = [
    "platform", "goods_no", "goods_name", "brand_id", "brand_name", "gender",
    "normal_price", "price", "final_price", "sale_rate", "sold_out",
    "review_count", "review_score", "product_url", "thumbnail_url",
    "crawled_at",
]


@dataclass(frozen=True)
class Page:
    products: list[dict[str, Any]]
    page: int
    total_pages: int
    total_count: int
    next_url: str | None


def _find_product_query(next_data: dict[str, Any]) -> dict[str, Any]:
    queries = next_data["props"]["pageProps"]["dehydratedState"]["queries"]
    for query in queries:
        pages = query.get("state", {}).get("data", {}).get("pages", [])
        if pages and "pagination" in pages[0].get("data", {}):
            return pages[0]["data"]
    raise ValueError("product list was not found in __NEXT_DATA__; page structure may have changed")


def parse_initial_page(body: bytes) -> Page:
    match = NEXT_DATA_RE.search(body)
    if not match:
        raise ValueError("__NEXT_DATA__ was not found; request may have been blocked")
    payload = _find_product_query(json.loads(match.group(1)))
    return page_from_payload(payload)


def parse_api_page(body: bytes) -> Page:
    document = json.loads(body)
    if document.get("meta", {}).get("result") not in (None, "SUCCESS"):
        raise ValueError(f"Musinsa API error: {document.get('meta')}")
    return page_from_payload(document["data"])


def page_from_payload(payload: dict[str, Any]) -> Page:
    pagination = payload["pagination"]
    # Promotional cards may appear in the list. A real product has these fields.
    products = [
        item for item in payload.get("list", [])
        if item.get("goodsNo") and item.get("goodsName") and item.get("thumbnail")
    ]
    return Page(
        products=products,
        page=int(pagination["page"]),
        total_pages=int(pagination["totalPages"]),
        total_count=int(pagination["totalCount"]),
        next_url=pagination.get("nextPageUrl") if pagination.get("hasNext") else None,
    )


def normalize(item: dict[str, Any]) -> dict[str, Any]:
    goods_no = str(item["goodsNo"])
    image_url = item["thumbnail"]
    return {
        "platform": "musinsa",
        "goods_no": goods_no,
        "goods_name": item.get("goodsName", ""),
        "brand_id": item.get("brand", ""),
        "brand_name": item.get("brandName", ""),
        "gender": item.get("displayGenderText", ""),
        "normal_price": item.get("normalPrice"),
        "price": item.get("price"),
        "final_price": item.get("finalPrice"),
        "sale_rate": item.get("finalDiscount", item.get("saleRate")),
        "sold_out": bool(item.get("isSoldOut")),
        "review_count": item.get("reviewCount"),
        "review_score": item.get("reviewScore"),
        "product_url": item.get("goodsLinkUrl", f"https://www.musinsa.com/products/{goods_no}"),
        "thumbnail_url": image_url,
        "crawled_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def load_seen(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        return {row["goods_no"] for row in csv.DictReader(stream) if row.get("goods_no")}


def append_rows(csv_path: Path, jsonl_path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", encoding="utf-8-sig", newline="") as csv_stream, \
            jsonl_path.open("a", encoding="utf-8") as jsonl_stream:
        writer = csv.DictWriter(csv_stream, fieldnames=CSV_FIELDS)
        if needs_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)
            jsonl_stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crawl Musinsa tops metadata")
    parser.add_argument("--output", type=Path, default=Path("data/musinsa/tops"))
    parser.add_argument("--max-pages", type=int, default=None, help="omit to crawl every page")
    parser.add_argument("--max-products", type=int, default=None)
    parser.add_argument("--delay", type=float, default=1.0, help="seconds between list requests")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.delay < 0:
        raise SystemExit("--delay must be >= 0")

    output = args.output.resolve()
    csv_path = output / "products.csv"
    jsonl_path = output / "products.jsonl"
    checkpoint_path = output / "checkpoint.json"
    output.mkdir(parents=True, exist_ok=True)

    fetcher = Fetcher(timeout=args.timeout, retries=args.retries)
    seen = load_seen(csv_path)
    saved_this_run = 0
    page_count = 0

    print(f"Fetching category: {CATEGORY_URL}", flush=True)
    page = parse_initial_page(fetcher.get(CATEGORY_URL))
    print(f"Musinsa reports {page.total_count:,} tops across {page.total_pages:,} pages", flush=True)

    while True:
        page_count += 1
        rows = [normalize(item) for item in page.products]
        rows = [row for row in rows if row["goods_no"] not in seen]
        if args.max_products is not None:
            rows = rows[: max(0, args.max_products - saved_this_run)]

        append_rows(csv_path, jsonl_path, rows)
        for row in rows:
            seen.add(row["goods_no"])
        saved_this_run += len(rows)

        checkpoint_path.write_text(json.dumps({
            "last_page": page.page,
            "next_url": page.next_url,
            "saved_this_run": saved_this_run,
            "total_unique": len(seen),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            f"page {page.page}/{page.total_pages}: +{len(rows)} products "
            f"({len(seen)} unique total)", flush=True
        )

        reached_limit = (
            (args.max_pages is not None and page_count >= args.max_pages)
            or (args.max_products is not None and saved_this_run >= args.max_products)
        )
        if reached_limit or not page.next_url:
            break
        time.sleep(args.delay)
        page = parse_api_page(fetcher.get(page.next_url, referer=CATEGORY_URL))

    print(f"Done. Metadata: {csv_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
