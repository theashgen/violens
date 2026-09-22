#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw
curl -L --fail --retry 5 --continue-at - \
  --output data/violence-dataset.zip \
  https://www.kaggle.com/api/v1/datasets/download/magicearth25/video-violence-detection-dataset
python3 -m zipfile -e data/violence-dataset.zip data/raw
echo 'Downloaded and extracted. Original archive retained in data/.'
