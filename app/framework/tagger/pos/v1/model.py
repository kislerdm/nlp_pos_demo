# Dmitry Kisler © 2020-present
# www.dkisler.com

import pathlib
import importlib.util
import nltk
import warnings
warnings.simplefilter(action='ignore', 
                      category=FutureWarning)


# import model abstract class
module_name = "template"
file_path = f"{pathlib.Path(__file__).parent}/{module_name}.py"
spec = importlib.util.spec_from_file_location(module_name, file_path)
model_template = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_template)


RULES = [
    (r"^(an|a|the)$", "DET"),
    (r"^(of|in|to|for|on|with|at|from|by|inside|outside)$", "ADP"),
    (r"^(also|so|then|just|more|as|very|well|even|most)$", "ADV"),
    (r"^(and|or|but|\&|both|either|nor|so|though|although|however)$", "CCONJ"),
    (f"^(yes|jup|yeah|yey|well|no|neh|meh|oh|yeah|hey|okay|yep|OK)$", "INTJ"),
    (f"^(that|if|when|as|how|where|because|while|after)$", "SCONJ"),
    (r"^(\.|\;|\:|\,|\'|\"|\"\"|\''|\]|\[|\(|\)|\?|\!)$", "PUNCT"),
    (r"^(\\|``|`|#|@|%|\$)$", "SYM"),
    (r"^-?[0-9]+(\.[0-9]+)?$", "NUM"),
    (r"^[a-zA-Z0-9\.\-]+@[a-zA-Z0-9\.\-]+\.[a-zA-Z]+$", "PRON"),
    (r"(.*ing|.*ish)$", "ADJ"),
    (r"^(.*es|.*ed)$", "VERB"),
]


class Model(template.Model):
    def __init__(self,
                 model=None):
      """Rule-based PoS tagger."""
      self.model = model

    def _model_definition(self):
        """Function to define and compile the model.

        Returns:
          model object
        """
        if self.model is None:
          self.model = nltk.RegexpTagger(RULES, 
                                         backoff=nltk.DefaultTagger('NOUN'))

    def train(self,
              X: pd.DataFrame,
              y: pd.Series) -> NamedTuple('model_eval',
                                          mse=float):
        """Train method

        Args:
            X: pd.DataFrame with features values
            y: target column values

        Returns: 
            namedtuple with metrics values: 
                "mse": float
        """
        if self.model is None:
          self._model_definition()

        # evalute on train set
        y_pred = self.predict(X)
        model_eval = self.score(y_true=y, y_pred=y_pred)
        return model_eval

    def save(self, path: str):
        """Model saver method.

        Args:
          path: Path to save model into.

        Raises:
          IOError, save error
        """
        pass

    def load(self, path: str):
        """Model loader method.

        Args:
          path: Path to load model from.

        Raises:
          IOError, load error.
        """
        pass

    def tag(self, X: List[List[str]]) -> List[List[Tuple[str]]]:
        """Method to tag tokens from the list of sentences.

        Args:
          X: Features values.
        """
        if self.model is None:
            return None
        pass
