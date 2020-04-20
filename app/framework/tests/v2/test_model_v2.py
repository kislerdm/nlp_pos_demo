# Dmitry Kisler © 2020-present
# www.dkisler.com

import pathlib
import pytest
import importlib.util
from types import ModuleType
import inspect
import warnings
warnings.simplefilter(action='ignore', 
                      category=FutureWarning)


DIR = pathlib.Path(__file__).absolute().parents[2]
PACKAGE = "tagger_framework/tagger/pos/v2"
MODULE = "model"

FUNCTIONS = set(["tokenization"])
CLASSES = set(["Model", "Corpus"])

CLASS_MODEL_METHODS = set(['_model_definition', 
                           'evaluate',
                           'train', 'predict',
                           'save', 'load'])

CLASS_MODEL_EVAL_ELEMENTS = ['dataset', 'accuracy']

CLASS_CORPUS_METHODS = set(['_build_corpus', 'train', 'dev'])

DATASET_TRAIN = """# newdoc id = GUM_academic_art
# sent_id = GUM_academic_art-1
# text = Aesthetic Appreciation and Spanish Art:
# s_type=frag
1	Aesthetic	aesthetic	ADJ	JJ	Degree=Pos	2	amod	_	_
2	Appreciation	appreciation	NOUN	NN	Number=Sing	0	root	_	_
3	and	and	CCONJ	CC	_	5	cc	_	_
4	Spanish	Spanish	ADJ	JJ	Degree=Pos	5	amod	_	_
5	Art	art	NOUN	NN	Number=Sing	2	conj	_	SpaceAfter=No
6	:	:	PUNCT	:	_	2	punct	_	_

# sent_id = GUM_academic_art-2
# text = Insights from Eye-Tracking
# s_type=frag
1	Insights	insight	NOUN	NNS	Number=Plur	0	root	_	_
2	from	from	ADP	IN	_	3	case	_	_
3	Eye-Tracking	eye-tracking	NOUN	NN	Number=Sing	1	nmod	_	_
"""

DATASET_DEV = """# newdoc id = GUM_academic_exposure
# sent_id = GUM_academic_exposure-1
# text = Introduction
# s_type=frag
1	Introduction	introduction	NOUN	NN	Number=Sing	0	root	_	_
"""


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
  

def test_class_corpus_miss_methods_attrs() -> None:
    members = module.Corpus("").__dir__()
    missing = CLASS_CORPUS_METHODS.difference(set(members))
    assert not missing, f"""Class Corpus Method(s) '{"', '".join(missing)}' is(are) missing."""
    return
  

def test_corpus_generation():
    try:
        corpus = module.Corpus(train=DATASET_TRAIN,
                               dev=DATASET_DEV)
    except Exception as ex:
        raise Exception(ex)
    
    assert (len(corpus.train), len(corpus.dev)) == (2, 1),\
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
    missing = CLASS_MODEL_METHODS.difference(set(model_members))
    assert not missing, f"""Class Model Method(s) '{"', '".join(missing)}' is(are) missing."""
    return


def test_class_model_miss_eval():
    for i in CLASS_MODEL_EVAL_ELEMENTS:
        assert getattr(module.Model.model_eval, i),\
            f"Model eval metric is missing attribute {i}."
    return


def test_class_model_model_definition():
    model = module.Model()
    assert str(type(model.model)) == "<class 'nltk.tag.sequential.UnigramTagger'>",\
      "Model definition error"
    return
  

corpus = module.Corpus(train=DATASET_TRAIN,
                       dev=DATASET_DEV)


def test_class_model_model_train():
    model = module.Model()
    model_eval = model.train(corpus=corpus,
                             evaluate=True)
    assert (round(model_eval[0].accuracy, 1), 
            round(model_eval[1].accuracy, 1)) == (1., 1.),\
              "Model training error"
    return
  
def test_class_model_model_predict():
    model = module.Model()
    model.train(corpus=corpus, evaluate=False)
    assert model.predict([["Introduction"]]) == [[('Introduction', 'NOUN')]],\
              "Prediction error"
    return
