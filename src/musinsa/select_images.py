#!/usr/bin/env python3
"""Select the first image containing a top from each Musinsa product gallery.

The product page is fetched once. Candidate images are then downloaded in UI
order and discarded immediately unless selected. Results are resumable and can
optionally be uploaded to an S3-compatible bucket.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import mimetypes
import os
import time
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

from src.common.http import Fetcher, NEXT_DATA_RE
from src.common.human_parser import DEFAULT_MODEL, label_ids_for_tops, select_device


IMAGE_BASE = "https://image.msscdn.net"
def load_project_env(path: Path = Path(".env")) -> None:
    """Load simple KEY=VALUE entries without overwriting shell variables."""
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value:
            os.environ.setdefault(key, value)


def parse_args() -> argparse.Namespace:
    load_project_env()
    parser = argparse.ArgumentParser(description="Crawl and select Musinsa detail images")
    parser.add_argument("--products", type=Path, default=Path("data/musinsa/tops/products.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/musinsa/tops/selected"))
    parser.add_argument("--limit", type=int, help="maximum new products to process")
    parser.add_argument("--goods-no", help="process only one goods_no")
    parser.add_argument("--start-after", help="skip rows through this goods_no")
    parser.add_argument("--delay", type=float, default=1.0, help="delay between product pages")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--top-threshold", type=float, default=0.05)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--retry-no-match", action="store_true")
    parser.add_argument(
        "--delete-local-after-upload",
        action="store_true",
        help="remove the local selected image after a successful S3 upload",
    )
    parser.add_argument(
        "--s3-bucket",
        default=os.getenv("MUSINSA_S3_BUCKET") or os.getenv("AWS_BUCKET_NAME"),
    )
    parser.add_argument("--s3-prefix", default="musinsa/products")
    parser.add_argument("--s3-endpoint-url", default=os.getenv("S3_ENDPOINT_URL"))
    return parser.parse_args()


def parse_gallery_urls(
    body: bytes,
    thumbnail_url: str | None = None,
) -> list[str]:
    match = NEXT_DATA_RE.search(body)
    if not match:
        raise ValueError("__NEXT_DATA__ missing from product page")
    document = json.loads(match.group(1))
    page_props = document["props"]["pageProps"]
    product = page_props.get("meta", {}).get("data", {})
    gallery = product.get("goodsImages")
    if gallery is None:
        for query in page_props.get("dehydratedState", {}).get("queries", []):
            candidate = query.get("state", {}).get("data", {}).get("data", {})
            if candidate.get("goodsImages") is not None:
                gallery = candidate["goodsImages"]
                product = candidate
                break

    urls: list[str] = []
    if thumbnail_url:
        urls.append(thumbnail_url)
    for item in gallery or []:
        url = item.get("imageUrl") if isinstance(item, dict) else None
        if url:
            urls.append(urljoin(IMAGE_BASE, url))
    # Only use the representative thumbnail and top gallery. The lower
    # goodsContents (상품정보 보기) section is intentionally ignored.
    return list(dict.fromkeys(urls))


def predict_classes(
    image: Image.Image,
    processor: AutoImageProcessor,
    model: SegformerForSemanticSegmentation,
    device: torch.device,
) -> np.ndarray:
    inputs = processor(images=image, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        logits = model(**inputs).logits
        logits = torch.nn.functional.interpolate(
            logits, size=(image.height, image.width), mode="bilinear", align_corners=False
        )
    return logits.argmax(dim=1)[0].cpu().numpy()


def score_image(
    image: Image.Image,
    processor: AutoImageProcessor,
    model: SegformerForSemanticSegmentation,
    device: torch.device,
    top_ids: list[int],
) -> tuple[float, np.ndarray]:
    prediction = predict_classes(image, processor, model, device)
    top_mask = np.isin(prediction, top_ids)
    top_ratio = float(top_mask.mean())
    return top_ratio, top_mask


def mask_border_ratio(mask: np.ndarray) -> float:
    """Measure whether the detected garment is cut off by the crop boundary."""
    band = max(1, round(min(mask.shape) * 0.03))
    border = np.zeros_like(mask, dtype=bool)
    border[:band, :] = True
    border[-band:, :] = True
    border[:, :band] = True
    border[:, -band:] = True
    garment_pixels = int(mask.sum())
    return float((mask & border).sum() / garment_pixels) if garment_pixels else 1.0


def load_completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                completed.add(str(json.loads(line)["goods_no"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return completed


def load_statuses(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    statuses: dict[str, str] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("goods_no"):
                statuses[str(row["goods_no"])] = str(row.get("status", ""))
    return statuses


def extension_for(url: str, image: Image.Image) -> str:
    suffix = Path(url.split("?", 1)[0]).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".webp"}:
        return ".jpg" if suffix == ".jpeg" else suffix
    return ".png" if image.format == "PNG" else ".jpg"


def save_image(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    if path.suffix.lower() == ".png":
        image.save(temporary, format="PNG", optimize=True)
    else:
        image.convert("RGB").save(temporary, format="JPEG", quality=95)
    temporary.replace(path)


def get_s3_client(args: argparse.Namespace):
    if not args.s3_bucket:
        return None
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError("S3 upload requires: pip install boto3") from exc
    return boto3.client(
        "s3",
        endpoint_url=args.s3_endpoint_url,
        region_name=os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION"),
    )


def upload_s3(client, bucket: str, local_path: Path, key: str) -> None:
    content_type = mimetypes.guess_type(local_path.name)[0] or "application/octet-stream"
    client.upload_file(
        str(local_path), bucket, key,
        ExtraArgs={"ContentType": content_type, "CacheControl": "public,max-age=31536000,immutable"},
    )


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False) + "\n")


def save_result(path: Path, payload: dict[str, Any], overwrite: bool) -> None:
    if not overwrite or not path.exists():
        append_jsonl(path, payload)
        return
    goods_no = str(payload["goods_no"])
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(row.get("goods_no")) != goods_no:
                rows.append(row)
    rows.append(payload)
    temporary = path.with_suffix(".jsonl.part")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def process_product(
    row: dict[str, str], args: argparse.Namespace, fetcher: Fetcher,
    processor: AutoImageProcessor, model: SegformerForSemanticSegmentation,
    device: torch.device, top_ids: list[int], s3_client,
) -> dict[str, Any]:
    goods_no = row["goods_no"]
    product_url = row.get("product_url") or f"https://www.musinsa.com/products/{goods_no}"
    page_body = fetcher.get(product_url)
    urls = parse_gallery_urls(page_body, row.get("thumbnail_url"))
    if not urls:
        raise ValueError("product gallery is empty")

    candidates: list[dict[str, Any]] = []
    selected: tuple[int, str, Image.Image, float] | None = None

    for index, url in enumerate(urls):
        try:
            body = fetcher.get(url, referer=product_url)
            image = Image.open(io.BytesIO(body)).convert("RGB")
        except Exception as exc:
            candidates.append({"index": index, "url": url, "error": str(exc)})
            continue
        top_ratio, top_mask = score_image(image, processor, model, device, top_ids)
        candidates.append({
            "index": index,
            "url": url,
            "top_ratio": top_ratio,
            "width": image.width,
            "height": image.height,
            "top_border_ratio": mask_border_ratio(top_mask),
        })
        # A person may be present. Select the first gallery image with enough top pixels.
        if top_ratio >= args.top_threshold:
            selected = (index, url, image, top_ratio)
            break
    if selected is None:
        return {
            "platform": "musinsa",
            "goods_no": goods_no,
            "product_url": product_url,
            "status": "excluded",
            "exclude_reason": "no_top_image",
            "selected_index": None,
            "selected_url": None,
            "local_path": None,
            "s3_bucket": args.s3_bucket,
            "s3_key": None,
            "top_ratio": None,
            "mask_path": None,
            "checked_images": candidates,
            "processed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "model": args.model,
        }

    index, url, image, top_ratio = selected
    suffix = extension_for(url, image)
    image_hash = hashlib.sha256(image.tobytes()).hexdigest()[:12]
    local_path = args.output / "images" / goods_no / f"selected-{image_hash}{suffix}"
    save_image(image, local_path)

    s3_key = None
    if s3_client:
        s3_key = f"{args.s3_prefix.strip('/')}/{goods_no}/{local_path.name}"
        upload_s3(s3_client, args.s3_bucket, local_path, s3_key)

    local_path_value: str | None = str(local_path.resolve())
    if args.delete_local_after_upload and s3_key:
        local_path.unlink()
        local_path_value = None

    return {
        "platform": "musinsa",
        "goods_no": goods_no,
        "product_url": product_url,
        "status": "selected",
        "selected_index": index,
        "selected_url": url,
        "local_path": local_path_value,
        "s3_bucket": args.s3_bucket,
        "s3_key": s3_key,
        "top_ratio": top_ratio,
        "mask_path": None,
        "checked_images": candidates,
        "processed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": args.model,
    }


def main() -> int:
    args = parse_args()
    if (
        args.delay < 0
        or args.top_threshold < 0
    ):
        raise SystemExit("delay and thresholds must be non-negative")
    args.output = args.output.resolve()
    results_path = args.output / "selections.jsonl"
    errors_path = args.output / "errors.jsonl"
    completed = set() if args.overwrite else load_completed(results_path)
    statuses = load_statuses(results_path)
    fetcher = Fetcher(args.timeout, args.retries)
    s3_client = get_s3_client(args)

    device = select_device()
    print(f"Loading {args.model} on {device}...", flush=True)
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = SegformerForSemanticSegmentation.from_pretrained(args.model).to(device).eval()
    top_ids, top_labels = label_ids_for_tops(model.config.id2label)
    print(f"Top classes: {top_labels}", flush=True)

    processed = failed = 0
    start_reached = args.start_after is None
    with args.products.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            goods_no = row.get("goods_no", "")
            if args.goods_no and goods_no != args.goods_no:
                continue
            if args.retry_no_match and statuses.get(goods_no) not in {"no_match", "excluded"}:
                continue
            if not start_reached:
                if goods_no == args.start_after:
                    start_reached = True
                continue
            if not goods_no or goods_no in completed:
                continue
            if args.limit is not None and processed >= args.limit:
                break
            try:
                result = process_product(
                    row, args, fetcher, processor, model, device,
                    top_ids, s3_client,
                )
                save_result(results_path, result, args.overwrite)
                processed += 1
                print(
                    f"{goods_no}: " + (
                        f"selected image {result['selected_index']} "
                        f"(top={result['top_ratio']:.2%})"
                        if result["status"] == "selected" else "no top image"
                    ), flush=True,
                )
            except Exception as exc:
                failed += 1
                append_jsonl(errors_path, {
                    "goods_no": goods_no, "error": str(exc),
                    "failed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                })
                print(f"{goods_no}: FAILED: {exc}", flush=True)
            if args.delay:
                time.sleep(args.delay)

    print(f"Done: {processed} processed, {failed} failed. Results: {results_path}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
