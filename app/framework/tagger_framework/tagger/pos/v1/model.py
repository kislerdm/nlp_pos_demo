# Dmitry Kisler © 2020-present
# www.dkisler.com

import pathlib
import importlib.util
from typing import Tuple, List, Union, NamedTuple
from nltk import DefaultTagger, RegexpTagger
from nltk.tokenize import regexp_tokenize as tokenizer
from pyconll import iter_from_string as conllu_iterator
from tagger_framework.utils.io_fs import save_obj_pkl, load_obj_pkl, corpus_reader
import warnings


warnings.simplefilter(action='ignore', 
                      category=FutureWarning)


# import model abstract class
module_name = "model_template"
file_path = f"{pathlib.Path(__file__).absolute().parents[1]}/{module_name}.py"
spec = importlib.util.spec_from_file_location(module_name, file_path)
model_template = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_template)


class Corpus(model_template.Corpus):
    def _build_dataset(self, path: str) -> List[List[Tuple[str]]]:
        """Function to define dataset.
        
        Args:
          path: Path to corpus file.
        
        Raises:
          IOError: Occurred on reading/unpacking error.
        """
        if not path:
            return []
                  
        document, err = corpus_reader(path)
        if err:
            raise IOError(err)
          
        if document is None:
          return []
        sentences = []
        for sentence in conllu_iterator(document):
            sentences.append(
                [(token.form, token.upos) 
                 for token in sentence]
            )
        return sentences


def tokenization(document: str) -> List[List[str]]:
    """Tokenizer function.
    
    Args:
      document: String with lines of tokens separated by single 'space'.
    
    Return:
      List of tokenized sentences.  
    """
    return [
        tokenizer(sentence, pattern='\s+', gaps=True)
        for sentence in tokenizer(document, pattern='\n+', gaps=True)
    ]


class Model(model_template.Model):
    RULES = [
        (r"^(an|a|the)$", "DET"),
        (r"^(of|in|to|for|on|with|at|from|by|inside|outside)$", "ADP"),
        (r"^(also|so|then|just|more|as|very|well|even|most)$", "ADV"),
        (r"^(and|or|but|\&|both|either|nor|so|though|although|however)$", "CCONJ"),
        (r"^(yes|jup|yeah|yey|well|no|neh|meh|oh|yeah|hey|okay|yep|OK)$", "INTJ"),
        (r"^(that|if|when|as|how|where|because|while|after)$", "SCONJ"),
        (r"^(\.|\;|\:|\,|\'|\"|\"\"|\''|\]|\[|\(|\)|\?|\!)$", "PUNCT"),
        (r"^(\\|``|`|#|@|%|\$)$", "SYM"),
        (r"^-?[0-9]+(\.[0-9]+)?$", "NUM"),
        (r"^[a-zA-Z0-9\.\-]+@[a-zA-Z0-9\.\-]+\.[a-zA-Z]+$", "PRON"),
        (r"(.*ing|.*ish)$", "ADJ"),
        (r"^(.*es|.*ed)$", "VERB"),
    ]
    
    def _model_definition(self) -> RegexpTagger:
        """Function to define and compile the model.
        
        Returns:
          Model object.
        """
        t0 = DefaultTagger('NOUN')
        return RegexpTagger(Model.RULES, backoff=t0)
            
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
          config: Training config dict (not used for this model).

        Returns: 
          namedtuple with metrics values: 
              "accuracy": float
        """
        if self.model is None:
            self._model_definition()
        
        if evaluate:
            return self.evaluate(corpus)
        return None
    
    def evaluate(self, 
                 corpus: Corpus) -> List[NamedTuple("model_eval", 
                                                    dataset=str,
                                                    accuracy=float)]:
        """Model metrics evaluation.

        Args:
          corpus: Corpus to evaluate model.

        Returns:
          namedtuple with metrics values: 
              "accuracy": float
        """
        output = [model_template.Model.model_eval(dataset="train",
                                                  accuracy=self.model.evaluate(corpus.train))]
        if corpus.dev:
            output.append(model_template.Model.model_eval(dataset="dev",
                                                          accuracy=self.model.evaluate(corpus.dev)))
        if corpus.test:
            output.append(model_template.Model.model_eval(dataset="test",
                                                          accuracy=self.model.evaluate(corpus.test)))
        
        return output
      
    def predict(self, 
                sentences: List[List[str]]) -> Union[List[List[Tuple[str]]], 
                                                     None]:
        """Method to tag tokens from the list of sentences.

        Args:
          sentences: Sentences.
        
        Returns:
          List of lists with tuples of (form, tag)
        """
        if self.model is None:
            return None
        return self.model.tag_sents(sentences)

    def save(self, path: str) -> None:
        """Model saver method.

        Args:
          path: Path to save model into.
        
        Raises:
          IOError, pickle.PicklingError: Occurred on writing/pickling error.
        """
        save_obj_pkl(self.model, path)
        return
    
    def load(self, path: str) -> RegexpTagger:
        """Model loader method.

        Args:
          path: Path to load model from.
        
        Rises:
          IOError, pickle.UnpicklingError: Occurred when loading/deserializing the obj.
        """
        return load_obj_pkl(path)
