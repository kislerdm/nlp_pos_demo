# Dmitry Kisler © 2020-present
# www.dkisler.com

import os
import pathlib
from typing import Tuple, List, Dict, Union
import torch
from flair import device as torch_device
from flair.data import Sentence
from flair.data import Corpus as Corpus_flair
from flair.datasets import UniversalDependenciesDataset
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from nltk.tokenize import regexp_tokenize as tokenizer
from tagger_framework.utils.logger import getLogger
from tagger_framework.tagger.pos.evaluation import model_performance
from tagger_framework.tagger.pos.model_template import Model as Template
import fastjsonschema
import re
import warnings


warnings.filterwarnings("ignore")

# fix required for flair
# see the issue https://github.com/flairNLP/flair/issues/1528

torch.__version__ = torch.__version__[:5]

# set seed for reproducibility
torch.manual_seed(2020)

logs = getLogger("model_ops", kill=False)

TAG_REGEX = re.compile(r"<(\w+)>", re.I)

IN_DOCKER = os.getenv("RUN_IN_DOCKER", False)


class Dataset(UniversalDependenciesDataset):
    def __init__(self, path: str = None):
        """Dataset class.
        
        Args:
          path: Path to conllu dataset.
        """
        self.in_memory: bool = True
        self.path_to_conll_file: pathlib.Path = pathlib.Path(path) if path else pathlib.Path(".")
        self.total_sentence_count: int = 0
        self.sentences: List[Sentence] = []
        if path:
            super(Dataset, self).__init__(
                path_to_conll_file=self.path_to_conll_file, 
                in_memory=self.in_memory
            )

    def get_tags(self) -> List[List[str]]:
        """Extractor of tags from sentences."""
        if self.total_sentence_count == 0:
            return []

        return [
            TAG_REGEX.findall(sentence.to_tagged_string("upos"))
            for sentence in self.sentences
        ]

    def get_tokens(self) -> List[List[str]]:
        """Extractor of tokens from sentences."""
        if self.total_sentence_count == 0:
            return []

        output = []
        for sentence in self.sentences:
            output.append([token.text for token in sentence.tokens])
        return output


class Corpus(Corpus_flair):
    def __init__(self, 
                 path_train: str, 
                 path_dev: str = None, 
                 path_test: str = None):
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


class Model(Template):
    MODEL_CONFIG_SCHEMA = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "title": "Flair U-PoS tagger model config",
        "additionalProperties": False,
        "properties": {
            "learning_rate": {
                "type": "number",
                "description": "Initial learning rate",
                "default": 1.0,
                "examples": [0.05, 0.1, 0.5],
            },
            "min_learning_rate": {
                "type": "number",
                "description": "Learning rate threshold to not go below when anneal.",
                "default": 0.00001,
                "minimum": 0,
                "examples": [0.05, 0.1, 0.5],
            },
            "anneal_factor": {
                "type": "number",
                "description": "Learning rate annealing factor.",
                "default": 0.5,
                "examples": [0.5, 0.2, 0.1],
            },
            "mini_batch_size": {
                "type": "integer",
                "default": 32,
                "minimum": 1,
                "examples": [8, 16, 32, 128],
            },
            "max_epochs": {
                "type": "integer",
                "description": "Max number of epochs to train.",
                "default": 1,
                "examples": [10.0],
            },
            "patience": {
                "type": "integer",
                "description": "How many epoch to wait before learning rate annealing.",
                "default": 1,
                "minimum": 0,
                "examples": [1, 2, 3],
            },
        },
    }

    def _model_definition(self) -> SequenceTagger:
        """Function to define and compile the model.
        
        It uses pre-trained model 'pos-fast' optimized for CPU 
        (GPU is not supported as of 2020-04).
        
        See list of available pretrained models here:  
        
        Returns:
          Model object.
        """
        MODEL_URI = "pos-fast"
        if IN_DOCKER:
            MODEL_URI = pathlib.Path("/app/en-pos-ontonotes-fast-v0.4.pt")
        return SequenceTagger.load(MODEL_URI)

    def train(self, 
              corpus: Corpus, 
              evaluate: bool = True, 
              config: dict = None) -> Union[None, 
                                            Dict[str, 
                                                 Dict[str, float]]]:
        """Train method.

        Args:
          corpus: Corpus to train model.
          evaluate: Flag to return evaluation of the model.
          config: Training config dict.

        Returns: 
          Model evaluation metrics.
        """
        if self.model is None:
            self._model_definition()

        # prepare tags dictionary
        tag_dictionary = corpus.make_tag_dictionary("upos")
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

        # flexibility to select optimizer is omitted due to time constraints
        # SGD is used
        trainer = ModelTrainer(self.model, corpus)

        # get training config
        train_config = Model.get_default_train_config()

        # update configs
        if config:
            # validate input config
            try:
                fastjsonschema.validate(Model.MODEL_CONFIG_SCHEMA, config)
                for k, v in config.items():
                    train_config[k] = v
            except fastjsonschema.JsonSchemaException as ex:
                logs.send(f"Provided training configs not valid: {ex}", kill=False)
                logs.send(
                    f"Default training configs are being used.",
                    is_error=False,
                    kill=False,
                )

        # launch training
        trainer.train(
            "/tmp",
            embeddings_storage_mode="cpu",
            shuffle=True,
            monitor_train=True,
            monitor_test=True,
            checkpoint=False,
            save_final_model=False,
            param_selection_mode=False,
            **train_config,
        )

        if evaluate:
            return self.evaluate(corpus)
        return None

    def evaluate(self, 
                 corpus: Union[Corpus, Dataset]) -> Dict[str,
                                                         Dict[str, float]]:
        """Model metrics evaluation.

        Args:
          corpus: Corpus/Dataset to evaluate model.
            
        Returns:
          Model evaluation metrics.
        """

        def _eval(y_true: Dataset, 
                  y_pred: List[List[Tuple[str]]]) -> Dict[str, float]:
            """Function to evaluate model performance using prediction accuracy.
            
            Args:
              y_true: Real tags.
              y_pred: Tags prediction.
            
            Returns:
              Accuracy in the range between 0.0 and 1.0, 
              or None in case of an empty input.
                
            Raises:
              ValueError: Exception occurred when input lists' length don't match.
            """
            y_pred_converted = []
            for sentence in y_pred:
                y_pred_converted.append(
                    TAG_REGEX.findall(sentence.to_tagged_string("upos"))
                )
            del y_pred
            
            return model_performance(y_true.get_tags(), 
                                     y_pred_converted)
        
        if not isinstance(corpus, Dataset):
            prediction = self.model.predict(corpus.train)
            output = {"train": _eval(corpus.train, prediction)}
            
            if corpus.dev:
                prediction = self.model.predict(corpus.dev)
                output['dev'] = _eval(corpus.dev, prediction)

            if corpus.test:
                prediction = self.model.predict(corpus.test)
                output['test'] = _eval(corpus.test, prediction)
        else: 
            prediction = self.model.predict(corpus)
            output = _eval(corpus, prediction)
            
        return output

    def predict(self, sentences: List[Sentence]) -> Union[List[List[Tuple[str]]], None]:
        """Method to tag tokens from the list of sentences.

        Args:
          sentences: Sentences.
        
        Returns:
          List of lists with tuples of (form, tag)
        """

        def _predicted_sentence_converter(sentence: Sentence) -> List[Tuple[str]]:
            """Converter to match with output convention.
            
            Args:
              sentense: Tokenized sentence.
            
            Returns:
              List of lists with tuples of (form, tag)
            """
            tokens_list = [token.text for token in sentence.tokens]
            tags_list = TAG_REGEX.findall(sentence.to_tagged_string("upos"))
            return [(token, tag) for token, tag in zip(tokens_list, tags_list)]

        if self.model is None:
            return None

        return [
            _predicted_sentence_converter(sentence)
            for sentence in self.model.predict(sentences)
        ]

    def save(self, path: str) -> None:
        """Model saver method.

        Args:
          path: Path to save model into.
        
        Raises:
          Except: Occurred when saving error happened.
        """
        self.model.save(pathlib.Path(path))
        return

    def load(self, path: str) -> SequenceTagger:
        """Model loader method.

        Args:
          path: Path to load model from.
        
        Rises:
          IOError, Error: Occurred when loading/deserializing the obj.
        """
        return SequenceTagger.load(pathlib.Path(path))

    @staticmethod
    def get_default_train_config() -> dict:
        """Default model config extractor."""
        return {
            k: v["default"] for k, v in Model.MODEL_CONFIG_SCHEMA["properties"].items()
        }
