# Dmitry Kisler © 2020-present
# www.dkisler.com

from gzip import open as gzip_open
from typing import Tuple, Union, Any
import pickle
import json


def corpus_reader(path: str) -> Union[Tuple[str, None],
                                      Tuple[None, str]]:
    """Function to read corpus text file.
    
    Args:
      path: Path to the file.
    
    Returns:
      Corpus text and error string in case of any.
    
    Raises:
      IOError: Occurred on reading/unpacking error.
    """
    try:
        if path.endswith(".gz"):
            with gzip_open(path, 'rb') as f:
                return f.read(), None
        else:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read(), None
    except IOError as ex:
        return None, ex
      

def prediction_writer(obj: dict,
                      path: str) -> None:
    """Function to write prediction as json file.
    
    Args:
      obj: Prediction object to write to.
      path: Path to the file.
    

    Raises:
      IOError: Occurred on writing error.
    """
    try:
        if path.endswith(".gz"):
            with gzip_open(path, 'wb') as f:
                json.dump(obj, f)
        else:
            with open(path, 'w') as f:
                json.dump(obj, f)
    except IOError as ex:
        raise ex


def load_obj_pkl(path: str) -> Tuple[Any, 
                                     Union[str, None]]:
    """Function to load and deserialize object from pickled file.

    Args:
      path: Path to object.

    Returns:
      Tuple with the object and error string in case of any.
    """
    try:
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        return obj, None
    except Exception as ex:
        return None, ex


def save_obj_pkl(obj: Any, 
                 path: str) -> None:
    """Function to serialize and save the object as pickled file.

    Args:
      obj: Python object to pickle.
      path: Path to store to.

    Raises:
      IOError: Raises when writing error.
    """
    try:
        with open(path, 'wb') as f:
            pickle.dump(obj, f)
    except IOError as ex:
        raise ex
