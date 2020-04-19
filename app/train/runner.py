# Dmitry Kisler © 2020-present
# www.dkisler.com

from os
import pathlib
import time
import importlib
import argparse
import warnings
from framework.utils.logger import getLogger
from framework.utils.io import corpus_reader
warnings.simplefilter(action='ignore', 
                      category=FutureWarning)


MODEL_PKG_NAME = "framework.tagger.pos"
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")

BUCKET_DATA = os.getenv("BUCKET_DATA", "/data")
BUCKET_MODEL = os.getenv("BUCKET_MODEL", "/model")


def get_args() -> argparse:
    """Input parameter.
    
    Returns:
          
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
    
    args = parser.parse_args()
    return args
  

if __name__ == "__main__":
    logs = getLogger(logger=f"service/train/{MODEL_VERSION}")

    # link the model module
    try:
        model_definition = importlib.import_module(
            f"{MODEL_PKG_NAME}.{MODEL_VERSION}.model")
    except Exception as ex:
        logs.send(f"Model {MODEL_VERSION} is not defined in the package {MODEL_PKG_NAME}.\nError:{ex}",
                  lineno=logs.get_line(),
                  kill=True)
    
    args = get_args()
    
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
        path_data_dev = path_data_dev if os.path.file(path_data_dev) else None
        
        if path_data_dev:
            str_dev, err = corpus_reader(path_data_dev)
            if err:
                logs.send(f"Error reading dev data set {path_data_train}.\nError: {err}",
                          lineno=logs.get_line(),
                          is_error=False,
                          kill=False)
      
    
          
    
