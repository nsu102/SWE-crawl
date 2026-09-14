# Ably pipeline

`crawl_products.py`는 공개 목록 페이지에서 상품 URL을 찾고 각 상품의
공개 Open Graph/product 메타데이터를 `data/ably/tops/`에 누적 저장합니다.
무신사 DOM/API 구조는 공유하지 않으며 공통 HTTP 유틸리티만 `src.common`에서
가져옵니다.

```bash
# 기본 공개 추천 목록에서 상의만 수집 (연결 확인용)
python -m src.ably.crawl_products --max-products 20

# 에이블리에서 연 카테고리·랭킹·검색 결과 URL을 명시해 수집
python -m src.ably.crawl_products --listing-url 'https://mobile.a-bly.com/...' \
  --category-keyword 상의 --max-products 100
```

에이블리가 자동화 요청에 봇 차단 페이지를 돌려주면 우회하지 않습니다. 브라우저에서
접근 가능한 공개 목록 URL인지 확인한 뒤 잠시 후 다시 실행하세요.
