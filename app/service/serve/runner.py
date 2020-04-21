# Dmitry Kisler © 2020-present
# www.dkisler.com

import os
import pathlib
import time
import importlib
import argparse
import json
from typing import Tuple, Union
from tagger_framework.utils.logger import getLogger
from tagger_framework.utils.io_fs import corpus_reader, prediction_writer
import warnings


warnings.simplefilter(action='ignore',
                      category=FutureWarning)


MODEL_PKG_NAME = "tagger_framework.tagger.pos"
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")

BUCKET_DATA = os.getenv("BUCKET_DATA", "/data")
BUCKET_MODEL = os.getenv("BUCKET_MODEL", "/model")


def get_data_dir() -> str:
    """Generate date to store prediction data into."""
    return f"{BUCKET_DATA}/{MODEL_VERSION}/{time.strftime('%Y/%m/%d', time.gmtime())}"


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
    parser.add_argument('--path-intput',
                        help="Path to the dataset input for prediction.",
                        type=str,
                        default=None,
                        required=True)
    parser.add_argument('--path-output',
                        help="Path to store prediction output.",
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
          "form": sentence_dict.keys(),
          "upos": sentence_dict.values()
        })
    
    return output
  

if __name__ == "__main__":
    logs = getLogger(logger=f"service/serve/{MODEL_VERSION}")
    
    # link the model module
    try:
        model_definition = importlib.import_module(f"{MODEL_PKG_NAME}.{MODEL_VERSION}.model")
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
    
    path_data_in = f"{BUCKET_MODEL}/{args.path_intput}"
    if not os.path.isfile(path_data_in):
        logs.send(f"Input data file {path_data_in} not found.",
                  lineno=logs.get_line(),
                  kill=True)
    
    logs.send("Loading the model.", is_error=False, kill=False)
    
    t0 = time.time()
    try:
        model = model_definition.Model(path=path_model)
    except Exception as ex:
        logs.send(f"Fail loading the model from {path_model}.",
                  lineno=logs.get_line(),
                  kill=True)
    
    logs.send("Read the input data.", is_error=False, kill=False)
    
    data_in_str, err = corpus_reader(path_data_in)
    if err:
        logs.send(f"Fail to read input data from {path_data_in}.",
                  lineno=logs.get_line(),
                  kill=True)
    
    logs.send("Prepare/tokenize data.", is_error=False, kill=False)
    
    try:
        data_in_sentenses = model_definition.tokenization(data_in_str)
        del data_in_str
    except Exception as ex:
        logs.send("Fail to tokenize input data.",
                  lineno=logs.get_line(),
                  kill=True)
    
    logs.send("Run prediction.", is_error=False, kill=False)
    
    try:
        data_prediction = model.predict(data_in_sentenses)
    except Exception as ex:
        logs.send("Fail to run prediction.",
                  lineno=logs.get_line(),
                  kill=True)
    
    logs.send("Convert prediction to output format.", is_error=False, kill=False)    
    
    try:
        data_prediction_out = convert_data_out(data_prediction)
        del data_prediction
    except Exception as ex:
        logs.send("Fail to convert prediction results to the output format.",
                  lineno=logs.get_line(),
                  kill=True)
    
    logs.send("Write prediction.", is_error=False, kill=False)
    
    try:
        prediction_writer(data_prediction_out)
    except Exception as ex:
        logs.send("Fail to write outs the prediction results.",
                  lineno=logs.get_line(),
                  kill=True)
    
    logs.send(f"Prediction completed. Elapsed time {round(time.time() - t0, 2)} sec.",
              is_error=False, 
              kill=False)
