from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    database_url: str
    storage_root: Path
    fashion_clip_model: str
    human_parser_model: str
    max_upload_bytes: int
    cors_origins: tuple[str, ...]


@lru_cache
def get_settings() -> Settings:
    origins = tuple(
        value.strip() for value in os.getenv(
            "CORS_ORIGINS", "http://localhost:3000,http://localhost:5173"
        ).split(",") if value.strip()
    )
    return Settings(
        database_url=os.getenv(
            "DATABASE_URL", "postgresql://fashion:fashion@localhost:5432/fashion"
        ),
        storage_root=Path(os.getenv("LOCAL_STORAGE_ROOT", "storage")).resolve(),
        fashion_clip_model=os.getenv("FASHION_CLIP_MODEL", "patrickjohncyh/fashion-clip"),
        human_parser_model=os.getenv("HUMAN_PARSER_MODEL", "fashn-ai/fashn-human-parser"),
        max_upload_bytes=int(os.getenv("MAX_UPLOAD_MB", "10")) * 1024 * 1024,
        cors_origins=origins,
    )
