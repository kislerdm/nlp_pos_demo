#! /bin/bash

# tagger service runner/builer
# Dmitry Kisler © 2020-present
# www.dkisler.com

BASE_DIR="$( cd "$(dirname "${0}")" >/dev/null 2>&1 ; pwd -P )"

msg () {
    echo "$(date +"%Y-%m-%d %H:%M:%S") ${1}"
}

if [[ ("${1}" !=  "train") && ("${1}" !=  "serve") && ("${1}" !=  "build") && ("${1}" !=  "test") ]]; then
  msg "Provide type of the serivce: './run.sh train MODEL_VERSION', or './run.sh serve MODEL_VERSION'"
  msg "To rebuild the app, type: './run.sh build MODEL_VERSION'"
  msg "For example, ./run.sh build train v1"
  msg "To perform tests, ./run.sh test. Note that all tests may take a while (up to 5-10 minutes range)."
  exit 1
fi

if [[ "${1}" ==  "test" ]]; then 
  
  COMPOSE_DOCKER_CLI_BUILD=1 BOCKER_BUILDKIT=1 docker-compose \
    -f ${BASE_DIR}/app/compose.e2e-test.yml up \
    --build

  COMPOSE_DOCKER_CLI_BUILD=1 BOCKER_BUILDKIT=1 docker-compose \
    -f ${BASE_DIR}/app/compose.e2e-test.yml down \
    --remove-orphans \
    --rmi 'all'

elif [[ "${1}" !=  "build" ]]; then
  
  MODEL_VERSION=${2} docker-compose -f ${BASE_DIR}/app/compose-app.yml run pos-tagger ${1} "${@:3}"

else 
  
    COMPOSE_DOCKER_CLI_BUILD=1 BOCKER_BUILDKIT=1 MODEL_VERSION=${2} docker-compose -f ${BASE_DIR}/app/compose-app.yml build

fi
