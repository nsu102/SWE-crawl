#!/bin/zsh
set -u

SCRIPT_DIR=${0:A:h}
PROJECT_DIR=${SCRIPT_DIR:h}
PYTHON_BIN="$PROJECT_DIR/work/.venv/bin/python"
STATUS_FILE="$PROJECT_DIR/logs/musinsa-full.status"

cd "$PROJECT_DIR" || exit 1
mkdir -p logs

set_status() {
  print -r -- "$(date '+%Y-%m-%dT%H:%M:%S%z') $1" > "$STATUS_FILE"
}

set_status "phase=metadata status=running"
"$PYTHON_BIN" -m src.musinsa.crawl_products --delay 1.0
metadata_exit=$?
if (( metadata_exit != 0 )); then
  set_status "phase=metadata status=failed exit_code=$metadata_exit"
  exit $metadata_exit
fi

set_status "phase=selection status=running"
"$PYTHON_BIN" -m src.musinsa.select_images \
  --delay 0.3 \
  --max-detail-images 50 \
  --delete-local-after-upload
selection_exit=$?
if (( selection_exit != 0 )); then
  set_status "phase=selection status=failed exit_code=$selection_exit"
  exit $selection_exit
fi

set_status "phase=complete status=complete"
