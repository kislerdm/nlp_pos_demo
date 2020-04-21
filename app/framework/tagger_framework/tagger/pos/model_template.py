# Dmitry Kisler © 2020-present
# www.dkisler.com

from typing import List, Tuple, Dict, Any, Union
from abc import ABC, abstractmethod
from tagger_framework.utils.io_fs import corpus_reader
from pyconll import iter_from_string as conllu_iterator


class Dataset():
    def __init__(self, path: str = None):
        """Dataset class.
        
        Args:
          path: Path to conllu dataset.
        
        Raises:
          IOError: Occurred on reading/unpacking error.
        """
        self.total_sentence_count = 0
        self.sentences = []
        
        document, err = corpus_reader(path)
        if err:
            raise IOError(err)
        
        for sentence in conllu_iterator(document):
            self.sentences.append(
                [(token.form, token.upos)
                 for token in sentence]
            )
            self.total_sentence_count += 1

        def __len__(self):
            return self.total_sentence_count
          
        def __getitem__(self, index: int = 0) -> List[Tuple[str]]:
            return self.sentences[index]      
    
    def get_tags(self) -> List[List[str]]:
        """Extractor of tags from sentences."""
        if self.total_sentence_count == 0:
            return []
        
        def _get_tags_from_sentence(sentence: List[Tuple[str]]) -> List[str]:
            return [token[1] for token in sentence]
            
        return [
            _get_tags_from_sentence(sentence)
            for sentence in self.sentences
        ]

    def get_tokens(self) -> List[List[str]]:
        """Extractor of tokens from sentences."""
        if self.total_sentence_count == 0:
            return []

        def _get_tokens_from_sentence(sentence: List[Tuple[str]]) -> List[str]:
            return [token[0] for token in sentence]

        return [
            _get_tokens_from_sentence(sentence)
            for sentence in self.sentences
        ]
      

class Corpus():
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
        self.train = Dataset(path_train)
        self.dev = Dataset(path_dev)
        self.test = Dataset(path_test)


def tokenization(document: str) -> Any:
    """Tokenizer function.
    
    Args:
      document: String with lines of tokens separated by single 'space'.
    
    Return:
      List of tokenized sentences.
    """
    pass


class Model(ABC):
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

        # train model
        if evaluate:
            return self.evaluate(corpus)
        return None

    @abstractmethod
    def evaluate(self, 
                 corpus: Corpus = None) -> Dict[str,
                                                Dict[str, float]]:
        """Model metrics evaluation.

        Args:
          corpus: Corpus to evaluate model.

        Returns:
          Model evaluation metrics.
        """
        return None
      
    @abstractmethod
    def predict(self, sentences: List[Any]) -> Any:
        """Method to tag tokens from the list of sentences.

        Args:
          sentences: Tokenize sentences.
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
