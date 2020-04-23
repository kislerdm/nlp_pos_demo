FROM python:3.8.2-slim-buster AS base

LABEL maintainer="Dmitry Kisler" \
  email=admin@dkisler.com \
  web=www.dkisler.com

VOLUME /data /model

# model version
ARG MODEL_VERSION="v1"
ENV MODEL_VERSION ${MODEL_VERSION}

# run tests on build (workaround solution)
# this step must be done prior build in CI pipeline!
ARG RUN_TESTS='y'

ENV BUCKET_DATA_EVAL=/data
ENV BUCKET_MODEL=/model

WORKDIR /app

# copy the service
# to be modified to "COPY . ." if using ci/cd 
# with the prior step of copying framework to the service dir, or artifacts
COPY service/serve .
COPY framework framework

ARG PATH_MISC="framework/tagger_framework/tagger/pos/${MODEL_VERSION}/install_dependencies.sh"

RUN pip install --upgrade pip \
  # install os dependencies if required by the model's py requirements
  && if [ -f ${PATH_MISC} ]; then chmod +x ${PATH_MISC} && ./${PATH_MISC}; fi \
  # install custom packages and dependencies
  && pip install --no-cache-dir -r requirements.txt \
  # test the model modules
  && if [ ${RUN_TESTS} = 'y' ]; then pip install pytest && pytest -W ignore -vv framework/tests/*.py; fi \
  && if [ ${RUN_TESTS} = 'y' ] && [ -d framework/tests/${MODEL_VERSION} ]; then pytest -W ignore -vv framework/tests/${MODEL_VERSION}/; fi \
  # remove pytest and tests
  && if [ ${RUN_TESTS} = 'y' ]; then pip uninstall pytest -y; fi \
  && rm -r framework/tests

ENTRYPOINT [ "python", "runner.py" ]