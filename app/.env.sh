#! /bin/bash

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." >/dev/null && pwd )"

export BUCKET_DATA=${BASE_DIR}/data
export BUCKET_MODEL=${BASE_DIR}/model
export MODEL_VERSION=v1
