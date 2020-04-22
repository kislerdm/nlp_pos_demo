#! /bin/bash

BASE_DIR="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

if [[ ("$1" !=  "train") && ("$1" !=  "serve") ]]; then
  echo "Provide type of the serivce: train, or serve"
  exit 1
fi

docker-compose -f ${BASE_DIR}/app/compose-serivce-$1.yml run $1 "${@:2}"