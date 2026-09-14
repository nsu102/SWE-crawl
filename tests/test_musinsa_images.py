import json
import tempfile
import unittest
from pathlib import Path

from src.musinsa.select_images import parse_gallery_urls, save_result


class DetailParserTest(unittest.TestCase):
    def test_extracts_thumbnail_and_detail_images_in_order(self):
        document = {
            "props": {"pageProps": {"meta": {"data": {"goodsImages": [
                {"imageUrl": "/images/detail-1.jpg"},
                {"imageUrl": "https://image.msscdn.net/images/detail-2.jpg"},
            ], "goodsContents": (
                '<div><img src="//image.msscdn.net/images/content-1.jpg">'
                '<img data-src="/images/content-2.jpg"></div>'
            )}}}}
        }
        html = (
            b'<script id="__NEXT_DATA__" type="application/json">'
            + json.dumps(document).encode()
            + b"</script>"
        )
        self.assertEqual([
            "https://image.msscdn.net/images/main.jpg",
            "https://image.msscdn.net/images/detail-1.jpg",
            "https://image.msscdn.net/images/detail-2.jpg",
            "https://image.msscdn.net/images/content-1.jpg",
            "https://image.msscdn.net/images/content-2.jpg",
        ], parse_gallery_urls(html, "https://image.msscdn.net/images/main.jpg"))

    def test_deduplicates_urls(self):
        document = {"props": {"pageProps": {"meta": {"data": {"goodsImages": [
            {"imageUrl": "/images/main.jpg"}
        ]}}}}}
        html = (
            b'<script id="__NEXT_DATA__" type="application/json">'
            + json.dumps(document).encode()
            + b"</script>"
        )
        self.assertEqual(
            ["https://image.msscdn.net/images/main.jpg"],
            parse_gallery_urls(html, "https://image.msscdn.net/images/main.jpg"),
        )

    def test_overwrite_upserts_result(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selections.jsonl"
            save_result(path, {"goods_no": "1", "selected_index": 0}, False)
            save_result(path, {"goods_no": "2", "selected_index": 0}, False)
            save_result(path, {"goods_no": "1", "selected_index": 7}, True)
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(2, len(rows))
            self.assertEqual(7, next(row for row in rows if row["goods_no"] == "1")["selected_index"])


if __name__ == "__main__":
    unittest.main()
