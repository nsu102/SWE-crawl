#!/usr/bin/env python3
"""Crawl Ably product metadata from a public listing page.

Ably's mobile web site does not expose Musinsa-style Next.js pagination data.
This crawler therefore discovers public ``/goods/<id>`` links from a listing
page and normalizes the Open Graph/product meta tags on every product page.
Pass the URL of a category, ranking, or search result with ``--listing-url``;
the default is the public recommendation page and is useful as a smoke test.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin, urlsplit, urlunsplit

from src.common.http import Fetcher


DEFAULT_LISTING_URL = "https://mobile.a-bly.com/"
GOODS_PATH_RE = re.compile(r"/goods/(?P<goods_no>\d+)(?:[/?#\"'])")
REVIEW_RE = re.compile(r"리뷰\s*([\d,]+)")
CHALLENGE_MARKERS = ("cf-mitigated", "challenge-error-text", "enable javascript and cookies")
CSV_FIELDS = [
    "platform", "goods_no", "goods_name", "brand_id", "brand_name", "gender",
    "normal_price", "price", "final_price", "sale_rate", "sold_out",
    "review_count", "review_score", "product_url", "thumbnail_url", "crawled_at",
]


class _MetaParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.meta: dict[str, str] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "meta":
            return
        values = dict(attrs)
        key = values.get("property") or values.get("name")
        content = values.get("content")
        if key and content and key not in self.meta:
            self.meta[key.lower()] = content.strip()


@dataclass(frozen=True)
class Product:
    goods_no: str
    goods_name: str
    brand_name: str
    price: int | None
    sold_out: bool
    review_count: int | None
    product_url: str
    thumbnail_url: str
    keywords: tuple[str, ...]


def _integer(value: str | None) -> int | None:
    if not value:
        return None
    digits = re.sub(r"[^0-9]", "", value)
    return int(digits) if digits else None


def _canonical_goods_url(url: str) -> str:
    parts = urlsplit(url)
    match = re.search(r"/goods/(\d+)", parts.path)
    if not match:
        raise ValueError(f"not an Ably product URL: {url}")
    return urlunsplit((parts.scheme or "https", parts.netloc or "mobile.a-bly.com", f"/goods/{match.group(1)}", "", ""))


def parse_listing_urls(body: bytes, listing_url: str) -> list[str]:
    """Return unique public product URLs in their listing order."""
    text = body.decode("utf-8", errors="replace")
    if any(marker in text.lower() for marker in CHALLENGE_MARKERS):
        raise RuntimeError(
            "Ably served a Cloudflare bot challenge, not a product listing. "
            "This crawler does not bypass access controls; use an authorized API/data source instead."
        )
    urls: list[str] = []
    for match in GOODS_PATH_RE.finditer(text):
        url = _canonical_goods_url(urljoin(listing_url, match.group(0)[:-1]))
        if url not in urls:
            urls.append(url)
    return urls


def parse_product_page(body: bytes, product_url: str) -> Product:
    parser = _MetaParser()
    html = body.decode("utf-8", errors="replace")
    if any(marker in html.lower() for marker in CHALLENGE_MARKERS):
        raise RuntimeError(
            "Ably served a Cloudflare bot challenge for a product page. "
            "This crawler does not bypass access controls; use an authorized API/data source instead."
        )
    parser.feed(html)
    meta = parser.meta
    goods_no_match = re.search(r"/goods/(\d+)", product_url)
    goods_no = meta.get("product:retailer_item_id") or (goods_no_match.group(1) if goods_no_match else None)
    name = meta.get("og:title", "").removesuffix(" - 에이블리").strip()
    thumbnail = meta.get("og:image", "")
    if not goods_no or not name or not thumbnail:
        raise ValueError("Ably product meta tags were not found; page structure may have changed or request was blocked")
    keywords = tuple(part.strip() for part in meta.get("keywords", "").split(",") if part.strip())
    review_match = REVIEW_RE.search(html)
    return Product(
        goods_no=str(goods_no),
        goods_name=name,
        brand_name=meta.get("product:brand", ""),
        price=_integer(meta.get("product:price:amount")),
        sold_out=meta.get("product:availability", "").lower() not in {"in stock", "instock", "available"},
        review_count=_integer(review_match.group(1)) if review_match else None,
        product_url=_canonical_goods_url(product_url),
        thumbnail_url=thumbnail,
        keywords=keywords,
    )


def normalize(product: Product) -> dict[str, object]:
    return {
        "platform": "ably", "goods_no": product.goods_no, "goods_name": product.goods_name,
        "brand_id": "", "brand_name": product.brand_name, "gender": "",
        "normal_price": None, "price": product.price, "final_price": product.price,
        "sale_rate": None, "sold_out": product.sold_out, "review_count": product.review_count,
        "review_score": None, "product_url": product.product_url, "thumbnail_url": product.thumbnail_url,
        "crawled_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def load_seen(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        return {row["goods_no"] for row in csv.DictReader(stream) if row.get("goods_no")}


def append_rows(csv_path: Path, jsonl_path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", encoding="utf-8-sig", newline="") as csv_stream, jsonl_path.open("a", encoding="utf-8") as jsonl_stream:
        writer = csv.DictWriter(csv_stream, fieldnames=CSV_FIELDS)
        if needs_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)
            jsonl_stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crawl Ably product metadata")
    parser.add_argument("--listing-url", default=DEFAULT_LISTING_URL, help="public Ably category/ranking/search URL")
    parser.add_argument("--output", type=Path, default=Path("data/ably/tops"))
    parser.add_argument("--max-products", type=int, default=None)
    parser.add_argument("--category-keyword", default="상의", help="only save pages whose keywords include this value; pass '' to disable")
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.delay < 0:
        raise SystemExit("--delay must be >= 0")
    output = args.output.resolve()
    csv_path, jsonl_path, checkpoint_path = output / "products.csv", output / "products.jsonl", output / "checkpoint.json"
    fetcher = Fetcher(timeout=args.timeout, retries=args.retries)
    seen = load_seen(csv_path)

    print(f"Fetching listing: {args.listing_url}", flush=True)
    urls = parse_listing_urls(fetcher.get(args.listing_url), args.listing_url)
    if not urls:
        raise RuntimeError("No product links found. Use a public listing URL; Ably may have served a bot challenge.")
    saved = 0
    for index, url in enumerate(urls, start=1):
        goods_no = re.search(r"/goods/(\d+)", url).group(1)
        if goods_no in seen:
            continue
        if args.max_products is not None and saved >= args.max_products:
            break
        if index > 1:
            time.sleep(args.delay)
        product = parse_product_page(fetcher.get(url, referer=args.listing_url), url)
        if args.category_keyword and args.category_keyword not in product.keywords:
            continue
        row = normalize(product)
        append_rows(csv_path, jsonl_path, [row])
        seen.add(product.goods_no)
        saved += 1
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_text(json.dumps({"last_goods_no": product.goods_no, "saved_this_run": saved, "total_unique": len(seen)}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{index}/{len(urls)}: +1 product ({len(seen)} unique total)", flush=True)
    print(f"Done. Metadata: {csv_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
