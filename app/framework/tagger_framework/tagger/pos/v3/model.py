# Dmitry Kisler © 2020-present
# www.dkisler.com

import torch
from flair import device as torch_device
from flair.data import Sentence
from flair.datasets import UniversalDependenciesCorpus
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
import warnings
warnings.filterwarnings("ignore")


# fix required for flair
torch.__version__ = torch.__version__[:5]

PRETRAINED_MODEL_URI =\
  "https://github.com/kislerdm/pos_tagger/raw/master/flair-pos-pretrained/train_2/best-model.pt.gz"




