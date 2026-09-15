from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, CLIPModel, CLIPProcessor, SegformerForSemanticSegmentation

from src.backend.config import get_settings
from src.common.human_parser import label_ids_for_tops, select_device


@dataclass(frozen=True)
class PreparedImage:
    image: Image.Image
    used_top_mask: bool
    top_ratio: float


class FashionModels:
    def __init__(self) -> None:
        settings = get_settings()
        self.device = select_device()
        self._lock = threading.Lock()
        self.clip_processor = CLIPProcessor.from_pretrained(settings.fashion_clip_model)
        self.clip_model = CLIPModel.from_pretrained(settings.fashion_clip_model).to(self.device).eval()
        self.human_parser_model = settings.human_parser_model
        self.parser_processor: AutoImageProcessor | None = None
        self.parser_model: SegformerForSemanticSegmentation | None = None
        self.top_ids: list[int] | None = None

    def _ensure_parser(self) -> None:
        if self.parser_model is not None:
            return
        self.parser_processor = AutoImageProcessor.from_pretrained(self.human_parser_model)
        self.parser_model = SegformerForSemanticSegmentation.from_pretrained(
            self.human_parser_model
        ).to(self.device).eval()
        self.top_ids, _ = label_ids_for_tops(self.parser_model.config.id2label)

    def prepare_query(self, image: Image.Image, min_top_ratio: float = 0.015) -> PreparedImage:
        image = image.convert("RGB")
        with self._lock:
            self._ensure_parser()
        assert self.parser_processor is not None
        assert self.parser_model is not None
        assert self.top_ids is not None
        with self._lock, torch.inference_mode():
            inputs = self.parser_processor(images=image, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            logits = self.parser_model(**inputs).logits
            logits = F.interpolate(
                logits, size=(image.height, image.width), mode="bilinear", align_corners=False
            )
            prediction = logits.argmax(dim=1)[0].cpu().numpy()
        mask = np.isin(prediction, self.top_ids)
        ratio = float(mask.mean())
        if ratio < min_top_ratio:
            return PreparedImage(image=image, used_top_mask=False, top_ratio=ratio)

        ys, xs = np.nonzero(mask)
        left, right = int(xs.min()), int(xs.max()) + 1
        top, bottom = int(ys.min()), int(ys.max()) + 1
        pad_x = round((right - left) * 0.08)
        pad_y = round((bottom - top) * 0.08)
        box = (
            max(0, left - pad_x), max(0, top - pad_y),
            min(image.width, right + pad_x), min(image.height, bottom + pad_y),
        )
        neutral = Image.new("RGB", image.size, (217, 217, 217))
        neutral.paste(image, mask=Image.fromarray((mask * 255).astype(np.uint8), mode="L"))
        return PreparedImage(image=neutral.crop(box), used_top_mask=True, top_ratio=ratio)

    def embed(self, images: list[Image.Image]) -> np.ndarray:
        with self._lock, torch.inference_mode():
            inputs = self.clip_processor(images=[image.convert("RGB") for image in images], return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)
            features = self.clip_model.get_image_features(pixel_values=pixel_values)
            if not isinstance(features, torch.Tensor):
                features = features.pooler_output
            features = F.normalize(features, p=2, dim=-1)
        return features.cpu().numpy().astype(np.float32)


_models: FashionModels | None = None
_models_lock = threading.Lock()


def get_models() -> FashionModels:
    global _models
    if _models is None:
        with _models_lock:
            if _models is None:
                _models = FashionModels()
    return _models
