#! /bin/bash

BASE_DIR="$( cd "$(dirname "$0")"/.. >/dev/null 2>&1 ; pwd -P )"

export BUCKET_DATA_INPUT=${BASE_DIR}/data/prediction/input
export BUCKET_DATA_OUTPUT=${BASE_DIR}/data/prediction/output
export BUCKET_MODEL=${BASE_DIR}/model
export MODEL_VERSION=v1
