# Dmitry Kisler © 2020-present
# www.dkisler.com

import os
import pathlib
import time
import importlib
import argparse
import warnings
from framework.utils.logger import getLogger
from framework.utils.io_fs import corpus_reader
warnings.simplefilter(action='ignore', 
                      category=FutureWarning)


MODEL_PKG_NAME = "framework.tagger.pos"
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")

BUCKET_DATA = os.getenv("BUCKET_DATA", "/data")
BUCKET_MODEL = os.getenv("BUCKET_MODEL", "/model")


def get_model_dir() -> str:
    """Generate die to store the model into."""
    return f"{BUCKET_MODEL}/{MODEL_VERSION}/{time.strftime('%Y/%m/%d', time.gmtime())}"


def get_args() -> argparse.Namespace:
    """Input parameters parser.
    
    Returns:
      Namespace of stdin parameters.
    """
    parser = argparse.ArgumentParser(description="Model trainer.")
    
    parser.add_argument('--path_train',
                        help="Path to train corpus.",
                        type=str,
                        default=None,
                        required=True)
    parser.add_argument('--path_dev',
                        help="Path to dev corpus.",
                        type=str,
                        default=None,
                        required=False)
    parser.add_argument('--path_model',
                        help="Path to pre-trained model.",
                        type=str,
                        default=None,
                        required=False)
    
    args = parser.parse_args()
    return args
  

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
    
    str_train, err = corpus_reader(path_data_train)
    if err:
        logs.send(f"Error reading train data set {path_data_train}.\nError: {err}",
                  lineno=logs.get_line(),
                  kill=True)

    path_data_dev = None
    if args.path_dev:
        path_data_dev = f"{BUCKET_DATA}/{args.path_dev}"
        path_data_dev = path_data_dev if os.path.isfile(path_data_dev) else None
        
        if path_data_dev:
            str_dev, err = corpus_reader(path_data_dev)
            if err:
                logs.send(f"Error reading dev data set {path_data_train}.\nError: {err}",
                          lineno=logs.get_line(),
                          is_error=False,
                          kill=False)
    
    # build corpus to train and eval model
    try:
        corpus = model_module.Corpus(train=str_train,
                                     dev=str_dev)
    except Exception as ex:
        logs.send(f"Corpus extraction error.\nError: {ex}",
                  lineno=logs.get_line(),
                  kill=True)
    
    # instantiate the model
    logs.send("Defining the model.", is_error=False, kill=False)      
    
    path_model = args.path_model
    if path_model:
        logs.send(f"Pre-trained model from {path_model} is being pre-loaded.", is_error=False, kill=False)
                
    model = model_module.Model(path=path_model)
    
    logs.send("Start model training", is_error=False, kill=False)
    t0 = time.time()
    train_metrics = model.train(corpus=corpus,
                                evaluate=True)
    
    train_metrics = '\n'.join([str(i) for i in train_metrics])
    logs.send(f"Training completed. Elapsed time {round(time.time() - t0, 2)} sec. Model score:\n{train_metrics}",
              is_error=False, 
              kill=False)
    
    dir_model = get_model_dir()
    path_model = f"{dir_model}/{MODEL_VERSION}_{time.strftime('%Y%m%dT%H%M%sZ', time.gmtime())}.pt"
    
    logs.send(f"Saving model to {path_model}", is_error=False, kill=False)
    
    if os.path.isdir(f"{dir_model}"):
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
