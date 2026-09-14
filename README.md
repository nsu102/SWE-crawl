# Fashion Similarity Data Pipeline

플랫폼별 DOM/API 차이를 분리한 상의 상품 수집 프로젝트입니다.

```text
src/
├── common/
│   ├── http.py                 # 재시도 HTTP 클라이언트
│   └── human_parser.py         # 공통 ML 라벨·디바이스 유틸리티
├── musinsa/
│   ├── crawl_products.py       # 무신사 상품 메타데이터
│   └── select_images.py        # 무신사 상세 이미지 선택·S3 업로드
└── ably/
    └── README.md               # 에이블리 파이프라인 위치

data/
├── musinsa/tops/
│   ├── products.csv
│   ├── products.jsonl
│   └── selected/
└── ably/tops/
```

## 설치

```bash
python3 -m venv work/.venv
work/.venv/bin/pip install -r requirements-ml.txt
```

## 무신사 실행

상품 목록은 메타데이터만 수집합니다. 대표 썸네일을 미리 모두 저장하는 이전 단계는 제거했습니다.

```bash
work/.venv/bin/python -m src.musinsa.crawl_products --max-products 1000
```

상세 이미지 선택기는 대표 썸네일, 상단 갤러리, 하단 상품 상세정보 순서로 이미지를 검사합니다. 사람 없는 상의를 찾으면 선택 이미지만 저장합니다.

```bash
# 신규 상품 전체
work/.venv/bin/python -m src.musinsa.select_images

# 테스트용 5개
work/.venv/bin/python -m src.musinsa.select_images --limit 5

# 특정 상품 재처리
work/.venv/bin/python -m src.musinsa.select_images --goods-no 7091145 --overwrite
```

S3를 사용할 때는 EC2 IAM Role 또는 AWS 자격증명을 설정합니다.

```bash
work/.venv/bin/pip install -r requirements-aws.txt
work/.venv/bin/python -m src.musinsa.select_images \
  --s3-bucket YOUR_BUCKET \
  --s3-prefix musinsa/products
```

에이블리는 `src/ably/`에 독립 구현하고 저장 경로와 S3 key는 각각 `data/ably/`, `ably/products/`를 사용합니다.
