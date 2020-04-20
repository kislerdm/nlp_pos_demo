# Dmitry Kisler © 2020-present
# www.dkisler.com

import pathlib
import pytest
import importlib.util
from types import ModuleType


DIR = pathlib.Path(__file__).absolute().parents[1]
PACKAGE = "tagger_framework/tagger/pos"
MODULE = "model_template"

FUNCTIONS = set([])
CLASSES = set(['Model', 'Corpus'])


def load_module(module_name: str) -> ModuleType:
    """Function to load the module.
    Args:
        module_name: module name
    Returns:
        module object
    """
    file_path = f"{DIR}/{PACKAGE}/{module_name}.py"
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
