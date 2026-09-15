#!/usr/bin/env python3
"""Upload locally selected catalog images to S3 and update their metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


def load_project_env(path: Path = Path(".env")) -> None:
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        if key.strip() and value:
            os.environ.setdefault(key.strip(), value)


def parse_args() -> argparse.Namespace:
    load_project_env()
    parser = argparse.ArgumentParser(description="Upload selected product images to S3")
    parser.add_argument(
        "--selections",
        type=Path,
        default=Path("data/musinsa/tops/selected/selections.jsonl"),
    )
    parser.add_argument("--bucket", default=os.getenv("AWS_BUCKET_NAME"))
    parser.add_argument("--prefix", default="musinsa/products")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--verify-samples", type=int, default=5)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def main() -> int:
    args = parse_args()
    if not args.bucket:
        raise SystemExit("AWS_BUCKET_NAME or --bucket is required")
    if args.workers < 1 or args.verify_samples < 0:
        raise SystemExit("workers must be positive and verify-samples non-negative")

    import boto3

    client = boto3.client(
        "s3",
        region_name=os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION"),
        endpoint_url=os.getenv("S3_ENDPOINT_URL"),
    )
    rows = read_jsonl(args.selections)
    targets: list[tuple[dict[str, Any], Path, str]] = []
    for row in rows:
        if row.get("status") != "selected" or not row.get("local_path"):
            continue
        local_path = Path(row["local_path"])
        if not local_path.is_file():
            raise FileNotFoundError(local_path)
        key = f"{args.prefix.strip('/')}/{row['goods_no']}/{local_path.name}"
        targets.append((row, local_path, key))

    def upload(target: tuple[dict[str, Any], Path, str]) -> tuple[dict[str, Any], str, int]:
        row, local_path, key = target
        content_type = mimetypes.guess_type(local_path.name)[0] or "application/octet-stream"
        client.upload_file(
            str(local_path),
            args.bucket,
            key,
            ExtraArgs={
                "ContentType": content_type,
                "CacheControl": "public,max-age=31536000,immutable",
            },
        )
        remote_size = int(client.head_object(Bucket=args.bucket, Key=key)["ContentLength"])
        if remote_size != local_path.stat().st_size:
            raise RuntimeError(f"size mismatch for {key}: local={local_path.stat().st_size}, remote={remote_size}")
        return row, key, remote_size

    total_bytes = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(upload, target) for target in targets]
        for index, future in enumerate(as_completed(futures), start=1):
            row, key, remote_size = future.result()
            row["s3_bucket"] = args.bucket
            row["s3_key"] = key
            total_bytes += remote_size
            if index % 10 == 0 or index == len(futures):
                print(f"uploaded={index}/{len(futures)}", flush=True)

    sample_count = min(args.verify_samples, len(targets))
    random.Random(20260915).shuffle(targets)
    verified = 0
    for _row, local_path, key in targets[:sample_count]:
        remote = client.get_object(Bucket=args.bucket, Key=key)["Body"].read()
        if sha256_bytes(local_path.read_bytes()) != sha256_bytes(remote):
            raise RuntimeError(f"SHA-256 mismatch for {key}")
        verified += 1

    write_jsonl_atomic(args.selections, rows)
    print(f"selected_rows={len(targets)}")
    print(f"uploaded_bytes={total_bytes}")
    print(f"head_verified={len(targets)}")
    print(f"sha256_verified_samples={verified}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
