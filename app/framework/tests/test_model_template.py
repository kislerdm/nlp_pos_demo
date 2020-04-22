# Dmitry Kisler © 2020-present
# www.dkisler.com

import pathlib
import pytest
import importlib.util
from io import StringIO
from types import ModuleType


DIR = pathlib.Path(__file__).absolute().parents
PACKAGE = "tagger_framework/tagger/pos"
MODULE = "model_template"

FUNCTIONS = set(['tokenization'])
CLASSES = set(['Model', 'Dataset', 'Corpus'])

CLASS_DATASET_ATTS = set(['total_sentence_count', 'sentences',
                          '__len__', '__getitem__',
                          'get_tags', 'get_tokens'])


def load_module(module_name: str) -> ModuleType:
    """Function to load the module.
    Args:
        module_name: module name
    Returns:
        module object
    """
    file_path = f"{DIR[1]}/{PACKAGE}/{module_name}.py"
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


def test_class_dataset() -> None:
    dataset = module.Dataset(f"{DIR[0]}/data/test.conllu")
    assert len(dataset) == 1, "Dataset reading error: wrong dataset length."
    assert dataset.get_tags() == [
        ["NOUN"]], "Dataset reading error: wrong tags."
    assert dataset.get_tokens() == [["Summary"]],\
        "Dataset reading error: wrong tokens."
    return


def test_class_corpus() -> None:
    corpus = module.Corpus(path_train=f"{DIR[0]}/data/train.conllu",
                           path_dev=f"{DIR[0]}/data/dev.conllu",
                           path_test=f"{DIR[0]}/data/test.conllu")
    assert (
        len(corpus.train),
        len(corpus.dev),
        len(corpus.test)
    ) == (2, 1, 1), "Corpus reading error: wrong dataset length."
    return
