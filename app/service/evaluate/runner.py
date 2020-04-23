# Dmitry Kisler © 2020-present
# www.dkisler.com

import os
import pathlib
import time
import importlib
import argparse
import json
from typing import Tuple, Union, List
from tagger_framework import __version__ as framework_version
from tagger_framework.utils.logger import getLogger
from tagger_framework.utils.io_fs import corpus_reader, prediction_writer
import warnings


warnings.simplefilter(action='ignore',
                      category=FutureWarning)


MODEL_PKG_NAME = "tagger_framework.tagger.pos"
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")

BUCKET_DATA_EVAL = os.getenv("BUCKET_DATA_EVAL", "/data")
BUCKET_MODEL = os.getenv("BUCKET_MODEL", "/model")


def get_data_dir() -> str:
    """Generate date to store evaluation results."""
    return f"{BUCKET_MODEL}/{MODEL_VERSION}/{time.strftime('%Y/%m/%d', time.gmtime())}/evaluation"    


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
                        help="Path to the conllu dataset for model evaluation.",
                        type=str,
                        default=None,
                        required=True)
    parser.add_argument('--dir-output',
                        help="Dir to store evaluation results.",
                        type=str,
                        default=None,
                        required=False)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    logs = getLogger(logger=f"service/evaluate/{MODEL_VERSION}")

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

    path_data = f"{BUCKET_DATA_EVAL}/{args.path_input}"
    if not os.path.isfile(path_data):
        logs.send(f"Input data file {path_data} not found.",
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
    
    logs.send("Reading data and preparing dataset.", is_error=False, kill=False)
    
    try:
        dataset = model_definition.Dataset(path=path_data)
    except Exception as ex:
        logs.send(f"Fail to load build evaluation dataset from {path_data}.\nError: {ex}",
                  lineno=logs.get_line(),
                  kill=True)
    
    logs.send("Starting model performance evaluation.", is_error=False, kill=False)
    
    try:
        evaluation_metrics = model.evaluate(dataset)
        
        # add metadata to output with metrics
        evaluation_metrics['meta'] = {
            "model": {
                "version": MODEL_VERSION,
                "framework_version": framework_version,
            },
            "data": {
                "relative_path": args.path_input,
                "volume": {
                    "sentences_count": len(dataset),
                    "tags_count": sum([len(tags_sentence) 
                                       for tags_sentence in dataset.get_tags()]),
                }
            }
        }
        
    except Exception as ex:
        logs.send(f"Fail to evaluate model.\nError: {ex}",
                  lineno=logs.get_line(),
                  kill=True)
    
    dir_data_out = f"{BUCKET_MODEL}/{args.dir_output}" if args.dir_output else get_data_dir()
    
    path_metrics = f"{dir_data_out}/metrics_{MODEL_VERSION}_{time.strftime('%Y%m%dT%H%M%sZ', time.gmtime())}.json"
    
    logs.send(f"Writing evaluation results to {dir_data_out}",
              is_error=False,
              kill=False)
    
    try:
        if not os.path.isdir(dir_data_out):
            os.makedirs(dir_data_out)

        prediction_writer(obj=evaluation_metrics,
                          path=path_metrics)
    except Exception as ex:
        logs.send(f"Fail to write out evaluation metrics.\nError: {ex}",
                  lineno=logs.get_line(),
                  kill=True)

    logs.send(f"Model evaluation completed. Elapsed time {round(time.time() - t0, 2)} sec.",
              is_error=False,
              kill=False)
