# NLP Classificaiton Framework

[![license](https://img.shields.io/pypi/l/ansicolortags.svg)](./LICENSE)
[![pyversion](https://img.shields.io/static/v1?label=python&color=blue&message=3.7.7%20|%203.8.2)](./)
[![coverage](https://img.shields.io/static/v1?label=coverage&color=brightgreen&message=78%25)](./)
[![test](https://img.shields.io/static/v1?label=tests&color=success&message=100%25)](./)

[Google Python Style Guide](http://google.github.io/styleguide/pyguide.html) is being followed and is recommended to stick to.

*The framework's main **objective*** is, to provide a flexible extensible foundation for ML NLP classification (e.g. for the Part-of-Speach tagging) services development. The library uses OOP paradigm and relays on the python [abstract base classes](https://docs.python.org/3/library/abc.html).

## Package structure

```bash
.
├── __init__.py                         <- should contain __version__
├── tagger
│   └── pos                             <- PoS tagger (e.g. NER models can be added as well)
│       ├── evaluation.py               <- models evaluation function(s)
│       ├── model_template.py           <- abstract classes definition
│       ├── v1						    <- model version v1
│       │   └── model.py                <- model definition - classes: Model, Dataset, Corpus
│       ├── v2                          ...
│       │   └── model.py        
│       └── v3
│           ├── install_dependencies.sh <- model additional dependencies, e.g. OS, or C libraries
│           └── model.py
└── utils                               <- additional modules, e.g. for io operations
    ├── corpus_operations.py            <- conllu data mapping to json functions
    ├── io_fs.py                        <- file system io operations
    └── logger.py                       <- custom logs class
```

Every new model (model version/iteration) can be added as a module to the tagger_framework subdirectories. For example, PoS tagger *v2* to be added to [`tagger_framework/tagger/pos/v2`](./tagger_framework/tagger/pos/v2), NER tagger to be added to tagger_framework/tagger/ner/v1 etc. tagger_framework/tagger/pos/v2 must contain the module [model.py](./tagger_framework/tagger/pos/v2/model.py) which has to define the model class "Model", the data set class "Dataset" and the corpus class "Corpus" (the structure which includes between two and three Dataset objects, i.e. train/dev, or train/dev/test data sets objects).  

## tagger_framework/tagger/pos/vX break-down

### [model.py](./tagger_framework/tagger/pos/v3/model.py) break-down

#### Model

The class relies on the abstract class **Model** from [model_template.py](./tagger_framework/tagger/pos/model_template.py). The following methods are required to be defined:

- "_model_definition": defines model architecture/logic.
- "train": defines the model training (i.e. sets optimizer, hyper-parameters) and performs model training.
- "evaluate": evaluates the model in terms of prediction performance metrics (see the function [model_performance](./tagger_framework/tagger/pos/evaluation.py)).
- "predict": runs model prediction given a list of input data points (tokenized "sentences").
- "save": saves the model, e.g. as a pickle binary.
- "load": restores trained model, e.g. from pickle file.

#### Corpus

The class is required to have attributes *train*, *dev* and *test* with corresponding data set objects.

#### Dataset

The class is meant to read a data set given its path and tokenize its "sentences".

It should include the attributes:

- "sentences": list of tokenized sentences.
- "total_sentence_count": count of the sentences.

The following methods are required to be defined (*\_\_init\_\_* method sould cover file reading):

- "get_tags": extracts classes (i.e. PoS tags) from the input data set.
- "get_tokens": extracts tokens.

#### tokenization

The function is meant to parse and tokenize sentences provided for model prediction.

**!Note!** The sentences must consist of tokens separated by "space" ('\s+').

**!Note!** One sentence per document line is allowed. 

### Additional scripts/deps

Every sub-directory/sub-package/sub-module can include as many modules as necessarily.

#### OS dependencies installation

OS dependencies to be specified in `install_dependencies.sh` as `pkgs` argument. For example:

```bash
pkgs='gcc'
```

The script can also include model specific python dependencies for sake of framework flexibility.

## Misc

The package can be extended limitless with sub-modules. For example, it includes a [module for file system io operations](./tagger_framework/utils/io_fs.py) and can be further extended to have modules with functions to facilitate interactions with cloud providers (e.g. AWS, GCP). 
