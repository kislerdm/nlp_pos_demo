# Dmitry Kisler © 2020-present
# www.dkisler.com

import pathlib
import importlib.util
from typing import Tuple, List, Union, NamedTuple
from nltk import DefaultTagger, UnigramTagger
from nltk.tokenize import regexp_tokenize as tokenizer
from pyconll import iter_from_string as conllu_iterator
from tagger_framework.utils.io_fs import save_obj_pkl, load_obj_pkl
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
    def __init__(self, 
                 train: str,
                 dev: str = None):
        """Corpus class."""
        self.train = Corpus._build_corpus(train)
        self.dev = Corpus._build_corpus(dev)

    @staticmethod
    def _build_corpus(document: str) -> List[List[Tuple[str]]]:
        """Function to define corpus
        
        Args:
          document: String document.
        """
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
    def _model_definition(self) -> UnigramTagger:
        """Function to define and compile the model.
        
        Returns:
          Model object.
        """
        t0 = DefaultTagger('NOUN')
        return UnigramTagger([[(".", "PUNCT")]], backoff=t0)
            
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
        
        self.model = UnigramTagger(corpus.train, 
                                   backoff=DefaultTagger('NOUN'))
        
        if evaluate:
            return self.evaluate(corpus)
        return None
    
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
        output = [model_template.Model.model_eval(dataset="train",
                                                  accuracy=self.model.evaluate(corpus.train))]
        if corpus.dev:
            output.append(model_template.Model.model_eval(dataset="dev",
                                                          accuracy=self.model.evaluate(corpus.dev)))
        
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

    def save(self, path: str):
        """Model saver method.

        Args:
          path: Path to save model into.
        
        Raises:
          IOError: Occurred when saving error happed.
        """
        raise NotImplementedError("The model cannot be pickled")
        # err = save_obj_pkl(self.model, path)
        # if err:
        #     raise IOError(err)
    
    def load(self, path: str):
        """Model loader method.

        Args:
          path: Path to load model from.
        
        Rises:
          IOError, Error: Occurred when loading/deserializing the obj.
        """
        raise NotImplementedError("The model cannot be pickled")
        # obj, err = load_obj_pkl(path)
        # if err:
        #     raise Exception(err)
        # self.model = obj
        # del obj
