# Dmitry Kisler © 2020-present
# www.dkisler.com

import pathlib
import pytest
import importlib.util
from types import ModuleType


DIR = pathlib.Path(__file__).absolute().parents[1]
PACKAGE = "tagger_framework/tagger/pos"
MODULE = "evaluation"

FUNCTIONS = set(['accuracy'])


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


def test_module_miss_functions() -> None:
    missing = FUNCTIONS.difference(set(module.__dir__()))
    assert not missing, f"""Function(s) '{"', '".join(missing)}' is(are) missing."""
    return

  
def test_accuracy1() -> None:
  out = module.accuracy([], [])
  assert out is None, "Wrong logic to treat empty input."
  return


def test_accuracy2() -> None:
  out = module.accuracy(["NOUN"], ["NOUN"])
  assert out == 1.0, "Wrong logic to compute accuracy."
  return


def test_accuracy3() -> None:
  try:
      _ = module.accuracy(["NOUN", "X"], ["NOUN"])
  except ValueError as ex:
      assert str(ex) == "Lengths of input lists don't match.",\
        "Wrong logic to treat input of not matching length."
  return


def test_accuracy4() -> None:
  try:
      _ = module.accuracy([["NOUN"], ["X", "ADJ"]], [["NOUN"], ["X"]])
  except ValueError as ex:
      assert str(ex) == "Numper of tokens don't match between y_true and y_pred.",\
        "Wrong logic to treat input of not matching length."
  return
