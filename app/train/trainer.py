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


# fix required for flair
torch.__version__ = '1.4.0'

DIR = pathlib.Path(__file__).parent

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
    
    logs.send("Read corpus.")
    t0 = time.time()
    georgetown_corpus = build_corpus()
    logs.send(f"Done. Elapsed time: {round(time.time() - t0, 2)} sec.")
    
    logs.send("Prepare tags dictionary.")
    tag_dictionary = georgetown_corpus.make_tag_dictionary('upos')
    logs.send("Done.")
    
    logs.send("Read pretrained model.")
    t0 = time.time()
    tagger = SequenceTagger.load('pos-fast')
    logs.send(f"Done. Elapsed time: {round(time.time() - t0, 2)} sec.")
    
    logs.send("Adjust model params.")
    tagger.tag_dictionary = tag_dictionary
    tagger.tag_type = "upos"
    logs.send("Done.")    

    logs.send("Set trainer.")
    trainer = ModelTrainer(tagger,
                           georgetown_corpus)
    logs.send("Done.")

    logs.send("Train.")
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
    logs.send(f"Done. Elapsed time: {round(time.time() - t0, 2)} sec.")

    logs.send("Save model.")
    t0 = time.time()
    tagger.save(f"{DIR}/model.pt")
    
    logs.send(f"Completed. Elapsed time: {round(time.time() - t00, 2)} sec.")
