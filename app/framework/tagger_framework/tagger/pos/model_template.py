# Dmitry Kisler © 2020-present
# www.dkisler.com

from typing import List, Tuple, Any, NamedTuple, Union
from abc import ABC, abstractmethod
from collections import namedtuple


class Corpus(ABC):
    def __init__(self, 
                 train: str,
                 dev: str = None):
        """Corpus class."""
        self.train = Corpus._build_corpus(train)
        self.dev = Corpus._build_corpus(dev)
    
    @staticmethod
    @abstractmethod
    def _build_corpus(document: str) -> Any:
        """Function to define corpus
        
        Args:
          document: String document.
        """
        return


def tokenze(document: str) -> Any:
    """Tokenizer function.
    
    Args:
      document: String with lines of tokens separated by single 'space'.
    
    Return:
      List of tokenized sentences.
    """
    pass


class Model(ABC):
    model_eval = namedtuple("model_eval", ["dataset", 
                                           "accuracy"])

    def __init__(self, 
                 path: str = None):
        """"Model definition class
        
        Args:
          path: Path to pre-trained model.
        
        Raises:
          IOError if model file doesn't exist.
        """
        if path:
            try:
                self.model = self.load(path)
            except IOError as ex:
                raise ex
        else:
            self.model = self._model_definition()

    @abstractmethod
    def _model_definition(self) -> Any:
        """Function to define and compile the model.
        
        Returns:
          Model object.
        """
        pass

    @abstractmethod
    def train(self, 
              corpus: Corpus,
              evaluate: bool = True) -> Union[None,
                                             List[NamedTuple("model_eval", 
                                                             dataset=str,
                                                             accuracy=float)]]:
        """Train method.

        Args:
          corpus: Corpus to train model.
          evaluate: Flag to return evaluation of the model.

        Returns: 
          namedtuple with metrics values: 
              "accuracy": float
        """
        if self.model is None:
            self._model_definition()

        # train model
        if evaluate:
            return self.evaluate(corpus)
        return None

    @abstractmethod
    def evaluate(self, 
                 corpus: Corpus = None) -> List[NamedTuple("model_eval", 
                                                           dataset=str,
                                                           accuracy=float)]:
        """Model metrics evaluation.

        Args:
          corpus: Corpus to evaluate model.

        Returns:
          namedtuple with metrics values: 
              "accuracy": float
        """
        return None
      
    @abstractmethod
    def predict(self, 
                sentences: List[str]) -> dict:
        """Method to tag tokens from the list of sentences.

        Args:
          sentences: Sentences.
        """
        if self.model is None:
            return None
        pass

    @abstractmethod
    def save(self, path: str):
        """Model saver method.

        Args:
          path: Path to save model into.
        """
        pass

    @abstractmethod
    def load(self, path: str):
        """Model loader method.

        Args:
          path: Path to load model from.
        """
        pass

