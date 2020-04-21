#! /bin/bash

# Dmitry Kisler © 2020-present
# www.dkisler.com

# script to install Linux dependencies 
# and optimized versions of py libs
# required by python package

# OS pkgs

pkgs="wget"

if [ ! -f /etc/os-release ]; then
  echo "Unrecognized OS"; exit 1
fi

OS_ID=$(cat /etc/os-release | grep -i "^id=" | awk -F "=" '{print $2}' | sed 's/"//g')

if [[ "$?" != "0" ]]; then
  echo "Unrecognized OS"; exit 1
fi

debian () {
  apt-get update -y \
  && apt-get install $1 -y
}

centos () {
  yum update -y \
  && yum install $1 -y
}

alpine () {
  apk update \
  && apk add $1
}

if [[ "${pkgs}" != "" ]]; then

  case "${OS_ID}" in
      alpine*)  alpine ${pkgs} ;;
      debian*)  debian ${pkgs} ;;
      ubuntu*)  debian ${pkgs} ;;
      centos*)  centos ${pkgs} ;;
      *)        echo "Unrecognized OS"; exit 1 ;;
  esac

fi

# py libs
# CPU optimized pytorch
pip install --no-cache-dir https://download.pytorch.org/whl/cpu/torch-1.4.0%2Bcpu-cp38-cp38-linux_x86_64.whl fastjsonschema flair==0.4.5
# fetch pre-trained library
wget -O /app/en-pos-ontonotes-fast-v0.4.pt "https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/models-v0.4/release-pos-fast-0/en-pos-ontonotes-fast-v0.4.pt"
