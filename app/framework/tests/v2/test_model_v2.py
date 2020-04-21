# Dmitry Kisler © 2020-present
# www.dkisler.com

import os
import pathlib
import pytest
import importlib.util
from types import ModuleType
import inspect
import warnings


warnings.simplefilter(action='ignore', 
                      category=FutureWarning)


DIR = pathlib.Path(__file__).absolute().parents
PACKAGE = "tagger_framework/tagger/pos/v2"
MODULE = "model"

FUNCTIONS = set(["tokenization"])
CLASSES = set(["Model", "Corpus"])

CLASS_MODEL_METHODS = set(['_model_definition', 
                           'evaluate',
                           'train', 'predict',
                           'save', 'load'])


CLASS_MODEL_EVAL_ELEMENTS = ['dataset', 'accuracy']

CLASS_CORPUS_METHODS = set(['_build_dataset', 'train', 'dev', 'test'])

DATA_DIR = f"{DIR[1]}/data"
DATASET_TRAIN = f"{DATA_DIR}/train.conllu"
DATASET_DEV = f"{DATA_DIR}/dev.conllu"
DATASET_TEST = f"{DATA_DIR}/test.conllu"


def load_module(module_name: str) -> ModuleType:
    """Function to load the module.
    Args:
        module_name: module name
    Returns:
        module object
    """
    file_path = f"{DIR[2]}/{PACKAGE}/{module_name}.py"
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_module_exists():
    try:
        _ = load_module(MODULE)
    except Exception as ex:
        raise ex
    return


module = load_module(MODULE)


def test_module_miss_classes() -> None:
    missing = CLASSES.difference(set(module.__dir__()))
    assert not missing, f"""Class(es) '{"', '".join(missing)}' is(are) missing."""
    return


def test_module_miss_functions() -> None:
    missing = FUNCTIONS.difference(set(module.__dir__()))
    assert not missing, f"""Function(s) '{"', '".join(missing)}' is(are) missing."""
    return
  

def test_class_corpus_miss_methods_attrs() -> None:
    members = module.Corpus(path_train=DATASET_DEV).__dir__()
    missing = CLASS_CORPUS_METHODS.difference(set(members))
    assert not missing, f"""Class Corpus Method(s) '{"', '".join(missing)}' is(are) missing."""
    return
  

def test_corpus_generation():
    try:
        corpus = module.Corpus(path_train=DATASET_TRAIN,
                               path_dev=DATASET_DEV,
                               path_test=DATASET_TEST)
    except Exception as ex:
        raise Exception(ex)
    
    assert (len(corpus.train), len(corpus.dev), len(corpus.test)) == (2, 1, 1),\
        "Corpus generation error (count of sentenses)"
    
    assert corpus.train == [
      [('Aesthetic', 'ADJ'), ('Appreciation', 'NOUN'), 
       ('and', 'CCONJ'), ('Spanish', 'ADJ'), 
       ('Art', 'NOUN'), (':', 'PUNCT')], 
      [('Insights', 'NOUN'), ('from', 'ADP'), 
       ('Eye-Tracking', 'NOUN')]
      ],\
        "Corpus generation error (token/tag split)"
    
    return


def test_class_model_miss_methods() -> None:
    model_members = inspect.getmembers(module.Model)
    missing = CLASS_MODEL_METHODS.difference(set([i[0] for i in model_members]))
    assert not missing, f"""Class Model Method(s) '{"', '".join(missing)}' is(are) missing."""
    return


def test_class_model_miss_eval():
    for i in CLASS_MODEL_EVAL_ELEMENTS:
        assert getattr(module.Model.model_eval, i),\
            f"Model eval metric is missing attribute {i}."
    return


model = module.Model()


def test_class_model_model_definition():
    assert str(type(model.model)) == "<class 'nltk.tag.sequential.UnigramTagger'>",\
        "Model definition error"
    return
  

corpus = module.Corpus(path_train=DATASET_TRAIN,
                       path_dev=DATASET_DEV,
                       path_test=DATASET_TEST)


def test_class_model_model_train():
    model_eval = model.train(corpus=corpus,
                             evaluate=True)
    assert (round(model_eval[0].accuracy, 1), 
            round(model_eval[1].accuracy, 1),
            round(model_eval[2].accuracy, 1)) == (1., 1., 1.),\
        "Model training error"
    return

  
def test_class_model_model_predict():
    assert model.predict([["Introduction"]]) == [[('Introduction', 'NOUN')]],\
        "Prediction error"
    return


PATH_MODEL_TEST = "/tmp/model_v2.pt"


def test_class_model_model_save():
    try:
        model.save(PATH_MODEL_TEST)
    except IOError as ex:
        assert f"Saving error: {ex}"


def test_class_model_model_load():
    model = module.Model()
    try:
        model.load(PATH_MODEL_TEST)
    except IOError as ex:
        assert f"Loading error: {ex}"
    
    if os.path.isfile(PATH_MODEL_TEST):
        os.remove(PATH_MODEL_TEST)
    return
