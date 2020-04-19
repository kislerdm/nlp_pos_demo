# Dmitry Kisler © 2020-present
# www.dkisler.com

from typing import List, Tuple, Any, NamedTuple, Union
from abc import ABC, abstractmethod
from collections import namedtuple


class Corpus():
    def __init__(self, 
                 train: str,
                 dev: str = None):
        """Corpus class."""
        self.train = train
        self.dev = dev
    
    @property
    def train(self):
        return self.train
    
    @property
    def dev(self):
        return self.dev    


class Model(ABC):
    model_eval = namedtuple("model_eval", ["accuracy"])

    def __init__(self, config: dict = None, path: str = None):
        """"Model definition class
        
        Args:
          config: Model config dictionary.
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
              evalute: bool = True) -> Union[None,
                                             NamedTuple("model_eval", 
                                                      accuracy=float)]:
        """Train method.

        Args:
          corpus: Corpus to train model.

        Returns: 
          namedtuple with metrics values: 
              "accuracy": float
        """
        if self.model is None:
            self._model_definition()

        # train model

        return None

    @abstractmethod
    def evaluate(self, 
                 corpus: Corpus = None) -> NamedTuple("model_eval", 
                                                      accuracy=float):
        """Model metrics evaluation.

        Args:
          corpus: Corpus to evaluate model.

        Returns:
          namedtuple with metrics values: 
              "accuracy": float
        """
        return None

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

    @abstractmethod
    def predict(self, sentenses: List[List[str]]) -> List[List[Tuple[str]]]:
        """Method to tag tokens from the list of sentences.

        Args:
          X: Features values.
        """
        if self.model is None:
            return None
        pass
