import json
import tempfile
import unittest
from pathlib import Path

from src.musinsa.crawl_products import append_rows, load_seen, parse_api_page, parse_initial_page


PAYLOAD = {
    "list": [
        {"goodsNo": 123, "goodsName": "테스트 티셔츠", "thumbnail": "https://img/123.jpg"},
        {"id": 999, "title": "프로모션 카드", "url": "https://img/banner.jpg"},
    ],
    "pagination": {
        "page": 1, "totalPages": 3, "totalCount": 121,
        "hasNext": True, "nextPageUrl": "https://api.example/page=2",
    },
}


class ParserTest(unittest.TestCase):
    def test_api_page_filters_promotions(self):
        page = parse_api_page(json.dumps({"data": PAYLOAD, "meta": {"result": "SUCCESS"}}).encode())
        self.assertEqual([123], [item["goodsNo"] for item in page.products])
        self.assertEqual("https://api.example/page=2", page.next_url)

    def test_initial_next_data(self):
        next_data = {
            "props": {"pageProps": {"dehydratedState": {"queries": [
                {"state": {"data": {"pages": [{"data": PAYLOAD}]}}}
            ]}}}
        }
        html = (
            b'<html><script id="__NEXT_DATA__" type="application/json">'
            + json.dumps(next_data).encode()
            + b"</script></html>"
        )
        page = parse_initial_page(html)
        self.assertEqual(1, len(page.products))
        self.assertEqual(121, page.total_count)

    def test_csv_resume_reads_utf8_bom(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "products.csv"
            jsonl_path = Path(directory) / "products.jsonl"
            row = {
                "platform": "musinsa", "goods_no": "123", "goods_name": "티셔츠", "brand_id": "brand",
                "brand_name": "브랜드", "gender": "공용", "normal_price": 10000,
                "price": 9000, "final_price": 9000, "sale_rate": 10,
                "sold_out": False, "review_count": 0, "review_score": 0,
                "product_url": "https://example/product", "thumbnail_url": "https://example/image.jpg",
                "crawled_at": "2026-01-01T00:00:00+0900",
            }
            append_rows(csv_path, jsonl_path, [row])
            self.assertEqual({"123"}, load_seen(csv_path))


if __name__ == "__main__":
    unittest.main()
