from __future__ import annotations

from pathlib import Path
from typing import Any

from psycopg import Connection, connect
from psycopg.rows import dict_row

from src.backend.config import get_settings


def connect_db() -> Connection:
    return connect(get_settings().database_url, row_factory=dict_row)


def initialize_database() -> None:
    schema = Path(__file__).with_name("schema.sql").read_text(encoding="utf-8")
    with connect_db() as connection:
        connection.execute(schema)


def vector_literal(values: list[float]) -> str:
    return "[" + ",".join(f"{value:.8f}" for value in values) + "]"


def search_products(
    embedding: list[float], limit: int, platform: str | None = None
) -> list[dict[str, Any]]:
    vector = vector_literal(embedding)
    where = "WHERE platform = %s" if platform else ""
    params: list[Any] = [vector]
    if platform:
        params.append(platform)
    params.extend([vector, limit])
    query = f"""
        SELECT platform, goods_no, goods_name, brand_name, price,
               product_url, image_path,
               1 - (embedding <=> %s::vector) AS similarity
        FROM products
        {where}
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """
    with connect_db() as connection:
        return list(connection.execute(query, params).fetchall())


def count_products() -> int:
    with connect_db() as connection:
        row = connection.execute("SELECT count(*) AS count FROM products").fetchone()
        return int(row["count"])
