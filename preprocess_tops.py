#!/usr/bin/env python3
"""Extract upper-clothing pixels from fashion images with human parsing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation


DEFAULT_MODEL = "fashn-ai/fashn-human-parser"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
TOP_LABEL_HINTS = {
    "upper-clothes", "upper_clothes", "upperclothes", "shirt", "top", "coat", "jacket"
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Segment and crop upper garments")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="one image")
    source.add_argument("--input-dir", type=Path, help="directory of images")
    parser.add_argument("--output-dir", type=Path, default=Path("data/musinsa_tops/processed"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--background", default="#D9D9D9", help="masked background color")
    parser.add_argument("--padding", type=float, default=0.08, help="crop padding ratio")
    parser.add_argument("--min-mask-ratio", type=float, default=0.015)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def label_ids_for_tops(id2label: dict[int | str, str]) -> tuple[list[int], dict[int, str]]:
    labels = {int(key): value for key, value in id2label.items()}
    chosen = [
        idx for idx, label in labels.items()
        if label.lower().replace(" ", "_") in TOP_LABEL_HINTS
        or any(hint in label.lower() for hint in ("upper", "shirt", "jacket", "coat"))
    ]
    if not chosen:
        raise RuntimeError(f"No upper-clothing class found in model labels: {labels}")
    return chosen, labels


def parse_color(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    if len(value) != 6:
        raise ValueError("--background must be a 6-digit hex color, e.g. #D9D9D9")
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))


def padded_box(mask: np.ndarray, padding: float) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    left, right = int(xs.min()), int(xs.max()) + 1
    top, bottom = int(ys.min()), int(ys.max()) + 1
    pad_x = round((right - left) * padding)
    pad_y = round((bottom - top) * padding)
    height, width = mask.shape
    return max(0, left - pad_x), max(0, top - pad_y), min(width, right + pad_x), min(height, bottom + pad_y)


def infer_mask(
    image: Image.Image,
    processor: AutoImageProcessor,
    model: SegformerForSemanticSegmentation,
    device: torch.device,
    class_ids: list[int],
) -> np.ndarray:
    inputs = processor(images=image, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        logits = model(**inputs).logits
        logits = F.interpolate(logits, size=(image.height, image.width), mode="bilinear", align_corners=False)
        prediction = logits.argmax(dim=1)[0].cpu().numpy()
    return np.isin(prediction, class_ids)


def write_outputs(
    image: Image.Image,
    mask: np.ndarray,
    output_dir: Path,
    stem: str,
    background: tuple[int, int, int],
    padding: float,
    metadata: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_image = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    # A tiny blur only softens jagged borders; threshold-free alpha preserves edge detail.
    alpha = mask_image.filter(ImageFilter.GaussianBlur(radius=0.6))
    rgba = image.convert("RGBA")
    rgba.putalpha(alpha)
    rgba.save(output_dir / f"{stem}_transparent.png")

    flat = Image.new("RGB", image.size, background)
    flat.paste(image, mask=alpha)
    flat.save(output_dir / f"{stem}_masked.jpg", quality=95)

    box = padded_box(mask, padding)
    flat.crop(box).save(output_dir / f"{stem}_crop.jpg", quality=95)
    mask_image.save(output_dir / f"{stem}_mask.png")

    metadata.update({
        "mask_ratio": float(mask.mean()),
        "crop_box": list(box),
        "outputs": {
            "mask": f"{stem}_mask.png",
            "transparent": f"{stem}_transparent.png",
            "masked": f"{stem}_masked.jpg",
            "crop": f"{stem}_crop.jpg",
        },
    })
    (output_dir / f"{stem}.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def iter_images(args: argparse.Namespace) -> list[Path]:
    if args.input:
        return [args.input]
    return sorted(path for path in args.input_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)


def main() -> int:
    args = parse_args()
    if not 0 <= args.padding <= 1:
        raise SystemExit("--padding must be between 0 and 1")
    background = parse_color(args.background)
    device = select_device()
    print(f"Loading {args.model} on {device}...", flush=True)
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = SegformerForSemanticSegmentation.from_pretrained(args.model).to(device).eval()
    class_ids, labels = label_ids_for_tops(model.config.id2label)
    print(f"Top classes: {[(idx, labels[idx]) for idx in class_ids]}", flush=True)

    completed = skipped = failed = 0
    for path in iter_images(args):
        marker = args.output_dir / f"{path.stem}.json"
        if marker.exists() and not args.overwrite:
            skipped += 1
            continue
        try:
            image = Image.open(path).convert("RGB")
            mask = infer_mask(image, processor, model, device, class_ids)
            ratio = float(mask.mean())
            if ratio < args.min_mask_ratio:
                raise ValueError(f"upper-clothing mask too small ({ratio:.2%})")
            write_outputs(
                image, mask, args.output_dir, path.stem, background, args.padding,
                {"source": str(path.resolve()), "model": args.model,
                 "class_ids": class_ids, "class_labels": [labels[i] for i in class_ids]},
            )
            completed += 1
            print(f"{path.name}: ok ({ratio:.1%} garment pixels)", flush=True)
        except Exception as exc:
            failed += 1
            print(f"{path.name}: FAILED: {exc}", flush=True)

    print(f"Done: {completed} processed, {skipped} skipped, {failed} failed", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
