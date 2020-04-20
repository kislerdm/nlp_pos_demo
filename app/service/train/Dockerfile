FROM python:3.8.1-slim-buster AS base

LABEL maintainer="Dmitry Kisler" \
  email=admin@dkisler.com \
  web=www.dkisler.com

VOLUME /data /model

# model version
ARG MODEL_VERSION="v1"
ENV MODEL_VERSION ${MODEL_VERSION}

WORKDIR /app

# copy the service
# to be modified to "COPY . ." if using ci/cd 
# with the prior step of copying framework to the service dir, or artifacts
COPY service/train .
COPY framework framework

ARG PATH_MISC="framework/tagger_framework/tagger/pos/${MODEL_VERSION}/install_dependencies.sh"

RUN pip install --upgrade pip \
  # install custom packages and dependencies
  && pip install --no-cache-dir pytest -r requirements.txt \
  # install os dependencies if required by the model's py requirements
  && if [ -f ${PATH_MISC} ]; then chmod +x ${PATH_MISC} && ./${PATH_MISC}; fi \
  # test the model modules
  && pytest -W ignore -vv framework/tests/*.py \
  && if [ -d framework/tests/${MODEL_VERSION} ]; then pytest -W ignore -vv framework/tests/${MODEL_VERSION}/; fi \
  && pip uninstall pytest -y

ENTRYPOINT [ "python", "runner.py" ]