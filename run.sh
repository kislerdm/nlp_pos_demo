#! /bin/bash

# tagger service runner/builer
# Dmitry Kisler © 2020-present
# www.dkisler.com

BASE_DIR="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

msg () {
    echo "$(date +"%Y-%m-%d %H:%M:%S") $1"
}

if [[ ("$1" !=  "train") && ("$1" !=  "serve") && ("$1" !=  "build") ]]; then
  msg "Provide type of the serivce: './run.sh train', or './run.sh serve'"
  msg "To rebuild the service, type: './run.sh build train MODEL_VERSION', or './run.sh build serve MODEL_VERSION'"
  msg "For example, ./run.sh build train v1"
  exit 1
fi

if [[ "$1" !=  "build" ]]; then 
  docker-compose -f ${BASE_DIR}/app/compose-serivce-$1.yml run $1 "${@:2}"
else 
  if [[ ("$2" ==  "train") || ("$2" ==  "serve") ]]; then
    COMPOSE_DOCKER_CLI_BUILD=1 BOCKER_BUILDKIT=1 MODEL_VERSION=$3 docker-compose -f ${BASE_DIR}/app/compose-serivce-$2.yml build $2
  else 
    msg "To rebuild the service, type: './run.sh build train', or './run.sh build train'"
    exit 1
  fi
fi
