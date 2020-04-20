# Dmitry Kisler © 2020-present
# www.dkisler.com

import os
import pathlib
import pytest
import importlib.util
from types import ModuleType
import inspect
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


DIR = pathlib.Path(__file__).absolute().parents
PACKAGE = "tagger_framework/tagger/pos/v3"
MODULE = "model"

FUNCTIONS = set(["tokenization"])
CLASSES = set(["Model", "Dataset", "Corpus"])

CLASS_MODEL_METHODS = set(
    ["_model_definition", "evaluate", "train", "predict", "save", "load"]
)


CLASS_MODEL_EVAL_ELEMENTS = ["dataset", "accuracy"]

CLASS_CORPUS_METHODS = set(["train", "dev", "test"])

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
    assert (
        not missing
    ), f"""Class Corpus Method(s) '{"', '".join(missing)}' is(are) missing."""
    return


def test_corpus_generation():
    try:
        corpus = module.Corpus(
            path_train=DATASET_TRAIN, path_dev=DATASET_DEV, path_test=DATASET_TEST
        )
    except Exception as ex:
        raise Exception(ex)

    assert (len(corpus.train), len(corpus.dev), len(corpus.test)) == (2, 1,1 ),\
         "Corpus generation error (count of sentenses)"

    assert corpus.make_tag_dictionary("upos").get_items() == [
        "<unk>",
        "O",
        "ADJ",
        "NOUN",
        "CCONJ",
        "PUNCT",
        "ADP",
        "<START>",
        "<STOP>",
    ], "Corpus generation error (tags extraction)"
    return
