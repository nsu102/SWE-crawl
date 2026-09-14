import unittest

from src.ably.crawl_products import parse_listing_urls, parse_product_page


class AblyParserTest(unittest.TestCase):
    def test_listing_urls_are_canonical_and_unique(self):
        html = b'<a href="/goods/123?LIST_INDEX=0"></a><a href="/goods/123"></a><a href="/goods/456"></a>'
        self.assertEqual(
            ["https://mobile.a-bly.com/goods/123", "https://mobile.a-bly.com/goods/456"],
            parse_listing_urls(html, "https://mobile.a-bly.com/overview"),
        )

    def test_listing_challenge_is_reported(self):
        with self.assertRaisesRegex(RuntimeError, "Cloudflare bot challenge"):
            parse_listing_urls(b'<span id="challenge-error-text">Enable JavaScript and cookies</span>', "https://mobile.a-bly.com/")

    def test_product_meta_is_normalized(self):
        html = """
        <meta property="og:title" content="테스트 니트 - 에이블리">
        <meta property="og:image" content="https://img.example/123.jpg">
        <meta property="product:retailer_item_id" content="123">
        <meta property="product:brand" content="테스트 브랜드">
        <meta property="product:price:amount" content="24,610">
        <meta property="product:availability" content="in stock">
        <meta name="keywords" content="테스트 브랜드, 테스트 니트, 상의, 니트">
        <div>리뷰 1,234</div>
        """.encode()
        product = parse_product_page(html, "https://mobile.a-bly.com/goods/123?LIST_INDEX=0")
        self.assertEqual("123", product.goods_no)
        self.assertEqual("테스트 니트", product.goods_name)
        self.assertEqual(24610, product.price)
        self.assertEqual(1234, product.review_count)
        self.assertFalse(product.sold_out)
        self.assertIn("상의", product.keywords)


if __name__ == "__main__":
    unittest.main()
