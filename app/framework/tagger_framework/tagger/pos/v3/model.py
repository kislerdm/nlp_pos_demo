# Dmitry Kisler © 2020-present
# www.dkisler.com

import pathlib
import importlib.util
from typing import Tuple, List, Union, NamedTuple
import torch
from flair import device as torch_device
from flair.data import Sentence
from flair.data import Corpus as Corpus_flair
from flair.datasets import UniversalDependenciesDataset
from flair.trainers import ModelTrainer
from nltk.tokenize import regexp_tokenize as tokenizer
import warnings

warnings.filterwarnings("ignore")


# fix required for flair
# see the issue https://github.com/flairNLP/flair/issues/1528

torch.__version__ = torch.__version__[:5]

PRETRAINED_MODEL_URI = "https://github.com/kislerdm/pos_tagger/raw/master/flair-pos-pretrained/train_2/best-model.pt.gz"

# import model abstract class
module_name = "model_template"
file_path = f"{pathlib.Path(__file__).absolute().parents[1]}/{module_name}.py"
spec = importlib.util.spec_from_file_location(module_name, file_path)
model_template = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_template)


class Dataset(UniversalDependenciesDataset):
    def __init__(self, path: str):
        """Dataset class.
        
        Args:
          path: Path to conull dataset.
        """
        self.in_memory: bool = True
        self.path_to_conll_file: pathlib.Path = pathlib.Path(
            path
        ) if path else pathlib.Path(".")
        self.total_sentence_count: int = 0
        self.sentences: List[Sentence] = []
        if path:
            super(Dataset, self).__init__(
                path_to_conll_file=self.path_to_conll_file, in_memory=self.in_memory
            )


class Corpus(Corpus_flair):
    def __init__(self, path_train: str, path_dev: str = None, path_test: str = None):
        """Corpus class.
        
        Args:
          path_train: Path to conull train dataset.
          path_dev: Path to conull dev dataset.
          path_test: Path to conull test dataset.
        """
        super(Corpus, self).__init__(
            train=Dataset(path_train),
            dev=Dataset(path_dev),
            test=Dataset(path_test),
            name=pathlib.Path(path_train).name,
        )


def tokenization(document: str) -> List[Sentence]:
    """Tokenizer function.
    
    Args:
      document: String with lines of tokens separated by single 'space'.
    
    Return:
      List of tokenized sentences.  
    """
    return [
        Sentence(sentence_str)
        for sentence_str in tokenizer(document, pattern="\n+", gaps=True)
    ]
    

