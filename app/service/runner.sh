#! /bin/bash

# tagger services router
# Dmitry Kisler © 2020-present
# www.dkisler.com

BASE_DIR="$( cd "$(dirname "${0}")" >/dev/null 2>&1 ; pwd -P )"

msg () {
    echo "$(date +"%Y-%m-%d %H:%M:%S") ${1}"
}

if [[ ("${1}" !=  "train") && ("${1}" !=  "serve") && ("${1}" !=  "evaluate") ]]; then
  msg "Provide type of the serivce: 'train', or 'serve', or 'evaluate'"
  msg "To call service helper, run 'train -h', or 'serve -h', or 'evaluate -h'"
  exit 0
fi

python ${1}/runner.py "${@:2}"