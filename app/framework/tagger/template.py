# Dmitry Kisler © 2020-present
# www.dkisler.com

from typing import List, Tuple
from abc import ABC, abstractmethod
from collections import namedtuple
import numpy


class Model(ABC):
    """"Model definition class"""
    model_eval = namedtuple('model_eval', ['accuracy'])

    def __init__(self,
                 model=None):
        self.model = model

    @abstractmethod
    def _model_definition(self,
                          config: None) -> Any:
        """Function to define and compile the model.
        
        Args:
          config: Model's hyperparameters.
        
        Returns:
          Model object.
        """
        pass

    @abstractmethod
    def train(self,
              X: pd.DataFrame,
              y: pd.Series) -> NamedTuple('model_eval',
                                          accuracy=float):
        """Train method.

        Args:
          X: Features values.
          y: Target column values.

        Returns: 
          namedtuple with metrics values: 
              "accuracy": float
        """
        if self.model is None:
            self._model_definition()

        # train step

        # eval step
        y_pred = None
        model_eval = self.score(y_true=y, y_pred=y_pred)
        return model_eval

    @classmethod
    def score(cls,
              y_true: numpy.array,
              y_pred: numpy.array) -> NamedTuple('model_eval',
                                                 accuracy=float):
        """Model metrics evaluation.

        Args:
          y_true: True tag values vector.
          y_pred: Predicted tag values vector.

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
    def tag(self, X: List[List[str]]) -> List[List[Tuple[str]]]:
        """Method to tag tokens from the list of sentences.

        Args:
          X: Features values.
        """
        if self.model is None:
            return None
        pass
