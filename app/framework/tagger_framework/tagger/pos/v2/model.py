# Dmitry Kisler © 2020-present
# www.dkisler.com

import pathlib
from typing import Tuple, List, Dict, Union
from nltk import DefaultTagger, UnigramTagger
from nltk.tokenize import regexp_tokenize as tokenizer
from pyconll import iter_from_string as conllu_iterator
from tagger_framework.utils.io_fs import save_obj_pkl, load_obj_pkl, corpus_reader
from tagger_framework.tagger.pos.evaluation import model_performance
from tagger_framework.tagger.pos.model_template import Model as Template
from tagger_framework.tagger.pos.model_template import Dataset, Corpus
import warnings


warnings.simplefilter(action='ignore', 
                      category=FutureWarning)


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


class Model(Template):
    def _model_definition(self) -> UnigramTagger:
        """Function to define and compile the model.
        
        Returns:
          Model object.
        """
        t0 = DefaultTagger('NOUN')
        return UnigramTagger([[(".", "PUNCT")]], backoff=t0)
            
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
          config: Training config dict (not used for this model).

        Returns: 
          Model evaluation metrics.
        """
        if self.model is None:
            self._model_definition()
        
        self.model = UnigramTagger(corpus.train.sentences,
                                   backoff=DefaultTagger('NOUN'))
        
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
        def _tag_extractor(sentence: List[Tuple[str]]) -> List[str]:
            return [token[1] for token in sentence]

        if not isinstance(corpus, Dataset):
            prediction_tags = [_tag_extractor(sentence)
                               for sentence in self.predict(corpus.train.get_tokens())]
            output = {"train": model_performance(corpus.train.get_tags(),
                                                 prediction_tags)}

            if corpus.dev:
                prediction_tags = [_tag_extractor(sentence)
                                   for sentence in self.predict(corpus.dev.get_tokens())]
                output['dev'] = model_performance(corpus.dev.get_tags(),
                                                  prediction_tags)

            if corpus.test:
                prediction_tags = [_tag_extractor(sentence)
                                   for sentence in self.predict(corpus.test.get_tokens())]
                output['test'] = model_performance(corpus.test.get_tags(),
                                                   prediction_tags)
        else:
            prediction_tags = [_tag_extractor(sentence)
                               for sentence in self.predict(corpus.get_tokens())]
            output = model_performance(corpus.get_tags(),
                                       prediction_tags)

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
    
    def load(self, path: str) -> UnigramTagger:
        """Model loader method.

        Args:
          path: Path to load model from.
        
        Rises:
          IOError, pickle.UnpicklingError: Occurred when loading/deserializing the obj.
        """
        return load_obj_pkl(path)
