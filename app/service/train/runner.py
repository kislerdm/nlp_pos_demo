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
import warnings


warnings.simplefilter(action='ignore', 
                      category=FutureWarning)


MODEL_PKG_NAME = "tagger_framework.tagger.pos"
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")

BUCKET_DATA = os.getenv("BUCKET_DATA", "/data")
BUCKET_MODEL = os.getenv("BUCKET_MODEL", "/model")


def get_model_dir() -> str:
    """Generate date to store the model into."""
    return f"{BUCKET_MODEL}/{MODEL_VERSION}/{time.strftime('%Y/%m/%d', time.gmtime())}"


def get_args() -> argparse.Namespace:
    """Input parameters parser.
    
    Returns:
      Namespace of stdin parameters.
    """
    parser = argparse.ArgumentParser(description="Model trainer.")
    
    parser.add_argument('--path-train',
                        help="Path to train corpus.",
                        type=str,
                        default=None,
                        required=True)
    parser.add_argument('--path-dev',
                        help="Path to dev corpus.",
                        type=str,
                        default=None,
                        required=True)
    parser.add_argument('--path-test',
                        help="Path to test corpus.",
                        type=str,
                        default=None,
                        required=False)
    parser.add_argument('--path-model',
                        help="Path to pre-trained model.",
                        type=str,
                        default=None,
                        required=False)
    parser.add_argument('--path-model-out',
                        help="Path to store model into.",
                        type=str,
                        default=None,
                        required=False)
    parser.add_argument('--train-config',
                        help="""Configurations for model training as JSON.
                        
                        Example:
                            {"learning_rate": 0.00001}
                        """,
                        type=str,
                        default=None,
                        required=False)
    
    args = parser.parse_args()
    return args


def train_configs_parser(conf: str) -> Tuple[Union[dict, None], 
                                             Union[None, str]]:
    """Function to parse training parameters config.
    
    Args:
      conf: Input configs string.
    """
    try:
        return json.loads(conf), None
    except json.JSONDecodeError as ex:
        return None, ex
    

if __name__ == "__main__":
    logs = getLogger(logger=f"service/train/{MODEL_VERSION}")
    
    logs.send("Init.", is_error=False, kill=False)
    
    # link the model module
    try:
        model_module = importlib.import_module(f"{MODEL_PKG_NAME}.{MODEL_VERSION}.model")
    except Exception as ex:
        logs.send(f"Model {MODEL_VERSION} is not defined in the package {MODEL_PKG_NAME}.\nError:{ex}",
                  lineno=logs.get_line(),
                  kill=True)
    
    args = get_args()
    
    logs.send("Reading data and building corpus.", is_error=False, kill=False)
    
    # read train data corpus
    path_data_train = f"{BUCKET_DATA}/{args.path_train}"
    if not os.path.isfile(path_data_train):
        logs.send(f"Train data set {path_data_train} not found.",
                  lineno=logs.get_line(),
                  kill=True)
    
    path_data_dev = f"{BUCKET_DATA}/{args.path_dev}"
    if not os.path.isfile(path_data_dev):
        logs.send(f"Dev data set {path_data_dev} not found.",
                  lineno=logs.get_line(),
                  kill=True)
    
    path_data_test = None
    if args.path_test:
        path_data_test = f"{BUCKET_DATA}/{args.path_test}"
        path_data_test = path_data_test if os.path.isfile(path_data_test) else None
    
    # build corpus to train and eval model
    try:
        corpus = model_module.Corpus(path_train=path_data_train,
                                     path_dev=path_data_dev,
                                     path_test=path_data_test)
    except Exception as ex:
        logs.send(f"Corpus extraction error.\nError: {ex}",
                  lineno=logs.get_line(),
                  kill=True)
    
    # instantiate the model
    logs.send("Defining the model.", is_error=False, kill=False)      
    
    path_model = args.path_model
    if path_model:
        logs.send(f"Pre-trained model from {path_model} is being pre-loaded.", 
                  is_error=False, 
                  kill=False)
                
    model = model_module.Model(path=path_model)
    
    train_config = None
    if args.train_config:
        train_config, err = train_configs_parser(args.train_config)
            
    logs.send("Start model training", is_error=False, kill=False)
    t0 = time.time()
    train_metrics = model.train(corpus=corpus,
                                evaluate=True,
                                config=train_config)
    
    logs.send(f"Training completed. Elapsed time {round(time.time() - t0, 2)} sec.",
              is_error=False, 
              kill=False)
    
    logs.send(f"Model score:\n{json.dumps(train_metrics, indent=2)}",
              is_error=False,
              kill=False)
    
    if args.path_model_out:
        dir_model = BUCKET_MODEL
        path_model = f"{dir_model}/{args.path_model_out}"
    else:
        dir_model = get_model_dir()
        path_model = f"{dir_model}/{MODEL_VERSION}_{time.strftime('%Y%m%dT%H%M%sZ', time.gmtime())}.pt"
    
    logs.send(f"Saving model to {path_model}", is_error=False, kill=False)
    
    if not os.path.isdir(f"{dir_model}"):
        try:
            os.makedirs(dir_model)
        except Exception as ex:
            logs.send(f"Error when creating {dir_model}.\nError: {ex}", 
                      lineno=logs.get_line(),
                      kill=True)
            # further steps should be defined depending on the tool use case
    
    try:
        model.save(path_model)
    except Exception as ex:
        logs.send(f"Cannot save the model to {path_model}. Error:\n{ex}",
                  lineno=logs.get_line(),
                  kill=True)
