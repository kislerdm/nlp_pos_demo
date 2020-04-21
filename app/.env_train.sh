#! /bin/bash

BASE_DIR="$( cd "$(dirname "$0")"/.. >/dev/null 2>&1 ; pwd -P )"

export BUCKET_DATA=${BASE_DIR}/data
export BUCKET_MODEL=${BASE_DIR}/model
export MODEL_VERSION=v1
