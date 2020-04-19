import pathlib
import torch
from flair import device as torch_device
from flair.data import Sentence
from flair.datasets import UniversalDependenciesCorpus
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
import time
from logger import getLogger
import sys
import warnings
warnings.filterwarnings("ignore")


# fix required for flair
torch.__version__ = '1.4.0'

DIR = pathlib.Path(__file__).absolute().parent

BUCKET_DATA = f"{DIR}/../../data/UD_English-GUM"
FILE_TRAIN = f"{BUCKET_DATA}/en_gum-ud-train.conllu"
FILE_TEST = f"{BUCKET_DATA}/en_gum-ud-test.conllu"
FILE_DEV = f"{BUCKET_DATA}/en_gum-ud-test.conllu"


def build_corpus():
    return UniversalDependenciesCorpus(data_folder=BUCKET_DATA,
                                       train_file=pathlib.Path(FILE_TRAIN),
                                       test_file=pathlib.Path(FILE_TEST),
                                       dev_file=pathlib.Path(FILE_DEV))
    

if __name__ == "__main__":
    logs = getLogger("flair-trainer", kill=True)
     
    t00 = time.time()
    
    logs.send("Read corpus.", is_error=False)
    t0 = time.time()
    try:
        georgetown_corpus = build_corpus()
    except Exception as ex:
        logs.send(f"Reading error. {ex}")
    logs.send(f"Done. Elapsed time: {round(time.time() - t0, 2)} sec.",
              is_error=False)
    
    logs.send("Prepare tags dictionary.", is_error=False)
    tag_dictionary = georgetown_corpus.make_tag_dictionary('upos')
    tag_dictionary_size = len(tag_dictionary)
    logs.send("Done.", is_error=False)
    
    logs.send("Read pretrained model.", is_error=False)
    t0 = time.time()
    try:
        tagger = SequenceTagger.load('pos-fast')
    except Exception as ex:
        logs.send(f"Tagger loading error. {ex}")
    logs.send(f"Done. Elapsed time: {round(time.time() - t0, 2)} sec.", 
              is_error=False)
    
    logs.send("Adjust model params.", is_error=False)
    # change the dictionary tags
    tagger.tag_dictionary = tag_dictionary
    # minor adjustmend of the PoS -> Universal PoS
    tagger.tag_type = "upos"
    # size of the tagset (required to )
    tagger.tagset_size = tag_dictionary_size
    # freeze all layers
    for param in tagger.parameters():
        param.requires_grad = False
    # replace and (by default) unfreeze the output layer
    tagger.linear = torch.nn.Linear(512, tag_dictionary_size)
    # reset transitions so they aligned in size with new tagset
    tagger.transitions = torch.nn.Parameter(
        torch.randn(tag_dictionary_size, tag_dictionary_size)
    )
    logs.send("Done.", is_error=False)    

    logs.send("Set trainer.", is_error=False)
    trainer = ModelTrainer(tagger,
                           georgetown_corpus)
    logs.send("Done.", is_error=False)

    logs.send("Train.", is_error=False)
    t0 = time.time()
    trainer.train("/tmp/",
                  learning_rate=0.1,
                  mini_batch_size=32,
                  max_epochs=2,
                  embeddings_storage_mode='cpu',
                  patience=2,
                  shuffle=True,
                  monitor_train=True,
                  monitor_test=True, 
                  checkpoint=True)
    logs.send(f"Done. Elapsed time: {round(time.time() - t0, 2)} sec.",
              is_error=False)

    logs.send("Save model.", is_error=False)
    t0 = time.time()
    tagger.save(f"{DIR}/model.pt")
    
    logs.send(f"Completed. Elapsed time: {round(time.time() - t00, 2)} sec.",
              is_error=False)
