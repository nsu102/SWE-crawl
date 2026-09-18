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
    ├── crawl_products.py       # 에이블리 상품 메타데이터
    └── README.md               # 에이블리 실행 방법

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

상세 이미지 선택기는 대표 썸네일과 상단 갤러리만 검사합니다. 세로로 긴 하단 상품 상세정보 이미지는 제외합니다. 사람 없는 상의를 찾으면 선택 이미지만 저장하고, 없으면 `status=no_match`로 기록합니다.

```bash
# 신규 상품 전체
work/.venv/bin/python -m src.musinsa.select_images

# 테스트용 5개
work/.venv/bin/python -m src.musinsa.select_images --limit 5

# 특정 상품 재처리
work/.venv/bin/python -m src.musinsa.select_images --goods-no 7091145 --overwrite

# 기존 no_match만 상품 상세 이미지(기본 최대 20장)까지 재검사하고 S3 업로드
work/.venv/bin/python -m src.musinsa.select_images --retry-no-match --overwrite
```

S3를 사용할 때는 EC2 IAM Role 또는 `.env`에 AWS 자격증명을 설정합니다. `.env`의
`AWS_BUCKET_NAME`이 설정되어 있으면 `--s3-bucket`은 생략할 수 있습니다.

```bash
work/.venv/bin/pip install -r requirements-aws.txt
work/.venv/bin/python -m src.musinsa.select_images --s3-prefix musinsa/products
```

이미 로컬에 선별된 결과를 S3에 일괄 동기화할 때는 모델을 다시 실행하지 않아도 됩니다.

```bash
work/.venv/bin/python -m src.jobs.upload_selected
```

무신사 상의 전체 수집은 메타데이터 수집을 완료한 뒤 ML 선별을 이어서 실행합니다.
S3 업로드 후 로컬 이미지를 삭제하므로 장기 실행 중 로컬 디스크를 채우지 않습니다.

```bash
scripts/run_musinsa_full.sh
```

에이블리는 `src/ably/`에 독립 구현하고 저장 경로와 S3 key는 각각 `data/ably/`, `ably/products/`를 사용합니다.

## 에이블리 실행

에이블리는 공개 목록의 상품 링크를 수집한 다음, 각 상품 페이지의 Open Graph/product
메타데이터를 정규화합니다. `--listing-url`에는 에이블리에서 확인한 카테고리·랭킹·검색
결과 URL을 넣을 수 있습니다.

```bash
work/.venv/bin/python -m src.ably.crawl_products --max-products 20
work/.venv/bin/python -m src.ably.crawl_products \
  --listing-url 'https://mobile.a-bly.com/...' --category-keyword 상의
```

## 로컬 검색 백엔드

현재 백엔드는 S3 대신 `storage/`를 사용하고, RDS 대신 Docker의 PostgreSQL + pgvector를 사용합니다. 배포할 때 `DATABASE_URL`과 저장소 구현만 교체할 수 있도록 분리되어 있습니다.

```text
src/backend/
├── app.py          # FastAPI 검색·이미지 API
├── config.py       # 환경변수 설정
├── db.py           # PostgreSQL/pgvector 접근
├── ml.py           # Human Parser + FashionCLIP
└── schema.sql      # DB 스키마와 HNSW 인덱스

src/jobs/
└── index_catalog.py # 선택 상품 이미지 임베딩·DB 적재
```

### 1. 설치 및 DB 실행

```bash
work/.venv/bin/pip install -r requirements-ml.txt -r requirements-backend.txt
docker compose up -d postgres
```

기본 접속 정보는 로컬 개발 전용입니다.

```text
postgresql://fashion:fashion@localhost:5432/fashion
```

다른 DB를 사용하려면 `.env.example`을 참고하여 `DATABASE_URL`을 설정합니다.

### 2. 상품 임베딩 적재

```bash
# 먼저 5개만 확인
work/.venv/bin/python -m src.jobs.index_catalog \
  --platform musinsa --limit 5

# 선택 완료된 무신사 상품 전체
work/.venv/bin/python -m src.jobs.index_catalog \
  --platform musinsa
```

선택 이미지는 `storage/products/{platform}/{goods_no}.jpg`로 복사되고, 512차원 FashionCLIP 벡터와 상품 정보는 PostgreSQL에 upsert됩니다.

### 3. API 실행

```bash
work/.venv/bin/uvicorn src.backend.app:app \
  --host 127.0.0.1 --port 8000 --reload
```

- API 문서: `http://127.0.0.1:8000/docs`
- 상태 확인: `GET /health`
- 이미지 검색: `POST /api/search`
- 상품 이미지: `GET /media/{platform}/{goods_no}`

검색 예시:

```bash
curl -X POST 'http://127.0.0.1:8000/api/search?limit=20&platform=musinsa' \
  -F 'image=@query.jpg'
```

검색 사진은 메모리에서 상의만 추출한 후 즉시 버려지며 로컬 디스크에 저장하지 않습니다. 상품 이미지만 `storage/`에 영구 저장됩니다.
