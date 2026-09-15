import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

import numpy as np

from src.musinsa.select_images import analysis_views, mask_border_ratio, parse_gallery_urls, save_result


class DetailParserTest(unittest.TestCase):
    def test_extracts_thumbnail_and_detail_images_in_order(self):
        document = {
            "props": {"pageProps": {"meta": {"data": {"goodsImages": [
                {"imageUrl": "/images/detail-1.jpg"},
                {"imageUrl": "https://image.msscdn.net/images/detail-2.jpg"},
            ], "goodsContents": '<img src="//image.msscdn.net/images/detail-3.jpg">'}}}}
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
            "https://image.msscdn.net/images/detail-3.jpg",
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

    def test_splits_very_tall_detail_image(self):
        image = Image.new("RGB", (100, 1000))
        views = analysis_views(image)
        self.assertGreater(len(views), 1)
        self.assertEqual((100, 150), views[0][0].size)
        self.assertEqual((0, 850, 100, 1000), views[-1][1])

    def test_penalizes_garment_mask_touching_crop_edges(self):
        centered = np.zeros((100, 100), dtype=bool)
        centered[20:80, 20:80] = True
        clipped = np.zeros((100, 100), dtype=bool)
        clipped[:, 20:80] = True
        self.assertLess(mask_border_ratio(centered), mask_border_ratio(clipped))


if __name__ == "__main__":
    unittest.main()
