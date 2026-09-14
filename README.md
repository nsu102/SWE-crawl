# 무신사 상의 썸네일 크롤러

무신사 상의 카테고리(`001`)의 상품 메타데이터와 썸네일을 저장합니다. 카테고리 HTML에 포함된 공식 페이지네이션 URL을 따라가며, 홍보 카드 등 상품이 아닌 항목은 제외합니다.

## 빠른 실행

Python 3.10 이상만 있으면 별도 패키지 설치 없이 실행됩니다.

```bash
# 먼저 100개만 테스트
python3 musinsa_crawler.py --max-products 100

# 전체 상의 수집 (현재 약 35만 개이므로 오래 걸리고 용량이 큼)
python3 musinsa_crawler.py

# 메타데이터만 수집
python3 musinsa_crawler.py --metadata-only
```

기본 결과 위치는 `data/musinsa_tops/`입니다.

```text
data/musinsa_tops/
├── products.csv       # 엑셀에서도 열기 쉬운 UTF-8 CSV
├── products.jsonl     # ML 파이프라인용 메타데이터
├── images/            # {goods_no}.jpg 형식의 썸네일
├── checkpoint.json    # 마지막 처리 상태
└── failures.jsonl     # 실패한 이미지가 있을 때만 생성
```

## 주요 옵션

```text
--max-pages N         목록 N페이지만 수집
--max-products N      신규 상품을 최대 N개 수집
--workers N           이미지 병렬 다운로드 수 (기본 8)
--delay SEC           목록 API 요청 간격 (기본 1초)
--output PATH         저장 디렉터리
--metadata-only       이미지를 받지 않음
--overwrite-images    기존 이미지를 다시 받음
```

같은 출력 폴더로 다시 실행하면 `goods_no` 기준으로 이미 기록된 상품은 건너뜁니다. 중단된 페이지부터의 완전한 재개가 아니라 첫 페이지부터 빠르게 중복을 거르는 방식이라, 서명 URL이 만료되어도 안전하게 재실행할 수 있습니다.

## 데이터 컬럼

상품 번호, 상품명, 브랜드 ID/이름, 성별, 정상가/판매가/최종가, 할인율, 품절 여부, 리뷰 수/점수, 상품 URL, 썸네일 URL, 로컬 이미지 경로, 수집 시각을 저장합니다.

사이트 구조가 바뀌면 파서도 수정해야 합니다. 과도한 부하를 피하려면 `--delay`를 줄이지 말고, 전체 수집 전에 작은 `--max-products` 값으로 확인하세요.

## 사람이 포함된 썸네일에서 상의 추출

패션 human-parsing 모델로 상의 픽셀만 분할하고, 마스크·투명 PNG·회색 배경 이미지·상의 크롭을 생성합니다.

```bash
python3 -m venv work/.venv
work/.venv/bin/pip install -r requirements-ml.txt

# 이미지 하나
work/.venv/bin/python preprocess_tops.py \
  --input data/musinsa_tops/images/6876064.jpg

# 수집된 이미지 전체 (이미 처리된 상품은 자동으로 건너뜀)
work/.venv/bin/python preprocess_tops.py \
  --input-dir data/musinsa_tops/images
```

결과는 기본적으로 `data/musinsa_tops/processed/`에 생성됩니다. 검색 임베딩에는 `*_crop.jpg` 또는 `*_masked.jpg`를 사용하고 원본 임베딩도 함께 보관하는 것을 권장합니다.

## 상세 이미지에서 사람 없는 첫 상의 선택

상품 상세 페이지의 대표 이미지, 상단 `goodsImages`, 하단 상품 상세정보의 `goodsContents` 이미지를 순서대로 검사합니다. 사람 없는 상의 사진을 찾으면 이후 이미지는 다운로드하지 않으며, 전부 착용 사진이면 가장 점수가 좋은 사진과 상의 마스크를 저장합니다.

```bash
# 먼저 5개 상품으로 검증
work/.venv/bin/python crawl_detail_images.py --limit 5

# 특정 상품만 다시 처리
work/.venv/bin/python crawl_detail_images.py --goods-no 7091145 --overwrite

# products.csv의 미처리 상품 전체 실행
work/.venv/bin/python crawl_detail_images.py
```

결과는 `data/musinsa_tops/selected/` 아래에 저장됩니다.

```text
selected/
├── images/{goods_no}/selected-{hash}.jpg
├── masks/{goods_no}/top-mask.png   # 전부 착용 사진이었던 경우
├── selections.jsonl
└── errors.jsonl
```

### Amazon S3 동시 업로드

EC2 Instance Profile 또는 로컬 AWS 자격증명을 설정한 뒤 실행합니다. 액세스 키를 코드나 `.env`에 직접 저장하지 마세요.

```bash
work/.venv/bin/pip install -r requirements-aws.txt

work/.venv/bin/python crawl_detail_images.py \
  --s3-bucket YOUR_BUCKET_NAME \
  --s3-prefix products
```

선택 이미지는 `s3://YOUR_BUCKET_NAME/products/{goods_no}/selected-{hash}.jpg`에 올라가고 `selections.jsonl`에 bucket과 key가 기록됩니다.
