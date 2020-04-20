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
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from nltk.tokenize import regexp_tokenize as tokenizer
import warnings
from tagger_framework.utils.logger import getLogger
import fastjsonschema
warnings.filterwarnings("ignore")

# fix required for flair
# see the issue https://github.com/flairNLP/flair/issues/1528

torch.__version__ = torch.__version__[:5]

# set seed for reproducibility
torch.manual_seed(2020)

# import model abstract class
module_name = "model_template"
file_path = f"{pathlib.Path(__file__).absolute().parents[1]}/{module_name}.py"
spec = importlib.util.spec_from_file_location(module_name, file_path)
model_template = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_template)


logs = getLogger("model_ops", kill=False)

MODEL_CONFIG_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "title": "Flair U-PoS tagger model config",
    "additionalProperties": false,
    "properties": {
        "learning_rate": {
            "type": "number",
            "description": "Initial learning rate",
            "default": 1.0,
            "examples": [
                0.05, 0.1, 0.5
            ]
        },
        "min_learning_rate": {
            "type": "number",
            "description": "Learning rate threshold to not go below when anneal.",
            "default": 0,
            "minimum": 0,
            "examples": [
                0.05, 0.1, 0.5
            ]
        },
        "anneal_factor": {
            "type": "number",
            "description": "Learning rate annealing factor.",
            "default": 0.5,
            "examples": [
                0.5, 0.2, 0.1
            ]
        },
        "batch_size": {
            "type": "integer",
            "default": 32,
            "minimum": 1,
            "examples": [
                8, 16, 32, 128
            ]
        },
        "epochs_max": {
            "type": "integer",
            "description": "Max number of epochs to train.",
            "default": 1,
            "examples": [
                10.0
            ]
        },
        "patience": {
            "type": "integer",
            "description": "How many epoch to wait before learning rate annealing.",
            "default": 1,
            "minimum": 0,
            "examples": [
                1, 2, 3
            ]
        }
    }
}

model_config_validator = fastjsonschema.compile(MODEL_CONFIG_SCHEMA)


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
                path_to_conll_file=self.path_to_conll_file, 
                in_memory=self.in_memory
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
    

class Model(model_template.Model):
    def _model_definition(self) -> SequenceTagger:
        """Function to define and compile the model.
        
        It uses pre-trained model 'pos-fast' optimized for CPU 
        (GPU is not supported as of 2020-04).
        
        See list of available pretrained models here:  
        
        Returns:
          Model object.
        """
        return SequenceTagger.load('pos-fast')

    def train(self, 
              corpus: Corpus,
              evaluate: bool = True,
              config: dict = None) -> Union[None,
                                           List[NamedTuple("model_eval", 
                                                           dataset=str,
                                                           accuracy=float)]]:
        """Train method.

        Args:
          corpus: Corpus to train model.
          evaluate: Flag to return evaluation of the model.
          config: Training config dict.

        Returns: 
          namedtuple with metrics values:
              "accuracy": float
        """
        if self.model is None:
            self._model_definition()
        
        # prepare tags dictionary
        tag_dictionary = corpus.make_tag_dictionary('upos')
        tag_dictionary_size = len(tag_dictionary)
        
        # change the dictionary tags
        self.model.tag_dictionary = tag_dictionary
        # set tagger type to universal PoS
        self.model.tag_type = "upos"
        # size of the tagset (required to )
        self.model.tagset_size = tag_dictionary_size
        # freeze all layers
        for param in self.model.parameters():
            param.requires_grad = False
        # replace and (by default) unfreeze the output layer
        self.model.linear = torch.nn.Linear(512, tag_dictionary_size)
        # reset transitions so they aligned in size with new tagset
        self.model.transitions = torch.nn.Parameter(
            torch.randn(tag_dictionary_size, tag_dictionary_size)
        )
        
        # define trainer
        trainer = ModelTrainer(self.model,
                               corpus)
        
        # launch training
        trainer.train("/tmp/",
                      learning_rate=1,
                      mini_batch_size=32,
                      max_epochs=1,
                      anneal_factor=0.5,
                      embeddings_storage_mode='cpu',
                      patience=2,
                      shuffle=True,
                      monitor_train=False, 
                      checkpoint=False, save_final_model=False)
        
        if evaluate:
            return self.evaluate(corpus)
        return None
      
    def load(self, path: str) -> None:
        """Model loader method.

        Args:
          path: Path to load model from.
        
        Rises:
          IOError, Error: Occurred when loading/deserializing the obj.
        """
        self.model = SequenceTagger.load(pathlib.Path(path))
        return