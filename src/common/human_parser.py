from __future__ import annotations

import torch


DEFAULT_MODEL = "fashn-ai/fashn-human-parser"
TOP_LABEL_HINTS = {
    "upper-clothes", "upper_clothes", "upperclothes", "shirt", "top", "coat", "jacket"
}


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
