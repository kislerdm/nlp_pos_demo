# Dmitry Kisler © 2020-present
# www.dkisler.com

import os
import pathlib
import time
import importlib
import argparse
import json
from typing import Tuple, Union, List
from tagger_framework.utils.logger import getLogger
from tagger_framework.utils.io_fs import corpus_reader, prediction_writer
import warnings


warnings.simplefilter(action='ignore',
                      category=FutureWarning)


MODEL_PKG_NAME = "tagger_framework.tagger.pos"
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")

BUCKET_DATA_INPUT = os.getenv("BUCKET_DATA_INPUT", "/data/prediction/input")
BUCKET_DATA_OUTPUT = os.getenv("BUCKET_DATA_OUTPUT", "/data/prediction/output")
BUCKET_MODEL = os.getenv("BUCKET_MODEL", "/model")


def get_data_dir() -> str:
    """Generate date to store prediction data into."""
    return f"{BUCKET_DATA_OUTPUT}/{MODEL_VERSION}/{time.strftime('%Y/%m/%d', time.gmtime())}"


def get_args() -> argparse.Namespace:
    """Input parameters parser.

    Returns:
      Namespace of stdin parameters.
    """
    parser = argparse.ArgumentParser(description="Model trainer.")

    parser.add_argument('--path-model',
                        help="Path to the model.",
                        type=str,
                        default=None,
                        required=True)
    parser.add_argument('--path-input',
                        help="Path to the dataset input for prediction.",
                        type=str,
                        default=None,
                        required=True)
    parser.add_argument('--path-output',
                        help="""Path to store prediction output.
                        ! Note: file must have .json, or .json.gz extention""",
                        type=str,
                        default=None,
                        required=False)

    args = parser.parse_args()
    return args


def convert_data_out(sentences_predict: List[List[Tuple[str]]]) -> dict:
    """Output data converter.

    Args:
      sentences_predict: Prediction results.

    Returns:
      Prediction results in the output format.
    """
    output = []

    for sentence in sentences_predict:
        sentence_dict = dict(sentence)
        output.append({
            "form": list(sentence_dict.keys()),
            "upos": list(sentence_dict.values())
        })

    return output


if __name__ == "__main__":
    logs = getLogger(logger=f"service/serve/{MODEL_VERSION}")

    # link the model module
    try:
        model_definition = importlib.import_module(
            f"{MODEL_PKG_NAME}.{MODEL_VERSION}.model")
    except Exception as ex:
        logs.send(f"Model {MODEL_VERSION} is not defined in the package {MODEL_PKG_NAME}.\nError:{ex}",
                  lineno=logs.get_line(),
                  kill=True)

    args = get_args()

    path_model = f"{BUCKET_MODEL}/{args.path_model}"
    if not os.path.isfile(path_model):
        logs.send(f"Model file {path_model} not found.",
                  lineno=logs.get_line(),
                  kill=True)

    path_data_in = f"{BUCKET_DATA_INPUT}/{args.path_input}"
    if not os.path.isfile(path_data_in):
        logs.send(f"Input data file {path_data_in} not found.",
                  lineno=logs.get_line(),
                  kill=True)

    logs.send("Loading the model.", is_error=False, kill=False)

    t0 = time.time()
    try:
        model = model_definition.Model(path=path_model)
    except Exception as ex:
        logs.send(f"Fail to load the model from {path_model}.\nError: {ex}",
                  lineno=logs.get_line(),
                  kill=True)
    
    # sys.exit(0)
    logs.send("Read the input data.", is_error=False, kill=False)

    data_in_str, err = corpus_reader(path_data_in)
    if err:
        logs.send(f"Fail to read input data from {path_data_in}.\nError: {err}",
                  lineno=logs.get_line(),
                  kill=True)

    logs.send("Prepare/tokenize data.", is_error=False, kill=False)

    try:
        data_in_sentenses = model_definition.tokenization(data_in_str)
        del data_in_str
    except Exception as ex:
        logs.send(f"Fail to tokenize input data.\nError: {ex}",
                  lineno=logs.get_line(),
                  kill=True)
    
    logs.send("Run prediction.", is_error=False, kill=False)

    try:
        data_prediction = model.predict(data_in_sentenses)
    except Exception as ex:
        logs.send(f"Fail to run prediction.\nError: {ex}",
                  lineno=logs.get_line(),
                  kill=True)

    logs.send("Convert prediction to output format.",
              is_error=False, 
              kill=False)

    try:
        data_prediction_out = convert_data_out(data_prediction)
        del data_prediction
    except Exception as ex:
        logs.send(f"Fail to convert prediction results to the output format.\nError: {ex}",
                  lineno=logs.get_line(),
                  kill=True)

    path_data_out = f"{BUCKET_DATA_OUTPUT}/{args.path_output}" if args.path_output\
        else f"{get_data_dir()}/prediction_{time.strftime('%Y%m%dT%H%M%sZ', time.gmtime())}.json.gz"

    logs.send(f"Write prediction results to {path_data_out}", 
              is_error=False, 
              kill=False)

    try:
        dir_data_out = os.path.dirname(path_data_out)
        if not os.path.isdir(dir_data_out):
            os.makedirs(dir_data_out)

        prediction_writer(obj=data_prediction_out,
                          path=path_data_out)
    except Exception as ex:
        logs.send(f"Fail to write out the prediction results.\nError: {ex}",
                  lineno=logs.get_line(),
                  kill=True)

    logs.send(f"Prediction completed. Elapsed time {round(time.time() - t0, 2)} sec.",
              is_error=False,
              kill=False)
