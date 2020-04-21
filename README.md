# PoS Tagger for English Words based on the Georgetown University Multilayer Corpus

Assessment task for the position of ML Engineer @BackHQ. The original description can be found [here](./docu/Machine%20Learning%20Challenge%20-%20Back.pdf).

## Goal

To build a Part-of-Speech tagger using [Georgetown University Multilayer Corpus](https://github.com/UniversalDependencies/UD_English-GUM/tree/master).

### Problem

Despite the fact that the PoS problem may be not of a great usefulness per se, PoS tagging is of a certain importance as an input for text classification (e.g. for sentiment, or styling analysis and syntactic parsing), or as a benchmark for algorithms performance.

We all learned PoS tagging in school where we mostly used rule-based method to classify a given word. This method was quite accurate when applying to a stand-alone word, it was however often failing in case of a rarely spoken, or foreign language origin word. The rules weren't often working when a word was put into a sentence, the words sequence/context was becoming important in such case. In some rare cases, i.e. idiomatic expressions, even words sequence analysis didn't work.

The PoS tagging problem is known since decades in computational linguistics. It was approached using different methods:

- *Rule based*, based on the regex rules (it's also used for words normalization).
- *Lexical based*, based on PoS tag occurrence with a word in the text corpus.
- *Probabilistic/contextual based*, based on the probability of a PoS tags sequence occurrence.
- *ANN*, based on sub-words embeddings.

## Project flow

A data science project flow described in [the article](https://towardsdatascience.com/how-to-power-up-your-product-by-machine-learning-with-python-microservice-pt-1-de0f2b434bec) is being followed.

### Steps

1. Develop a framework to train and serve models
2. Develop a baseline model
3. Develop models iteratively to improve tagging accuracy
4. (*) Deploy model as a web server to be accessible over HTTP as Rest API
5. (*) Setup a train pipeline on GCP (**).

(*) - additional steps

(**) - the cloud provided is selected based on the free leftover credits I have :)

#### Model Development

Three logical steps of model development are being followed:

- Model **v1**, *rule-based tagger*. It must be better than random PoS class guessing and should presumably yield to the accuracy of 50-60%. Defult class labling (e.g. labling every token as 'NOUN') would give over 20% accuracy alone (see the corpus composition [here](./data/UD_English-GUM/stats.xml)).
- Model **v2**, *1-gram freq. tagger*, mix of lexical and contextual method. It can reach over 80% accuracy.
- Model **v3**, *ANN tagger*. At that point, one needs to seriously consider the trade-off between time investment and business requirements. State-of-the-art model performance (every additional 1% after 90% accuracy) can cost months of human hours and years of computation run time with no guarantee of model performance improvement. Luckily, deep learning abstraction creates wide prospect of opportunities for transfer learning. Thanks to NLP community and range of research groups, there is a handful of great frameworks and pre-trained [PoS and NER tagging models](http://nlpprogress.com/english/part-of-speech_tagging.html) with the tate-of-the-art parformance of above 95%. 

I am going to use [flair](https://github.com/flairNLP/flair) to develop the model **v3**. It is an open source framework with low entry barrier built on top of [pytorch](https://pytorch.org/). It is being constantly developed and maintained by the [ML and NLP group](https://www.informatik.hu-berlin.de/en/forschung-en/gebiete/ml-en/) at Humboldt-Universität zu Berlin. As the model base, a pre-trainned tagger [**'pos-fast'**](https://github.com/flairNLP/flair/blob/master/resources/docs/TUTORIAL_2_TAGGING.md) is being used.

## Accuracy comparison on *test* dataset

|Model version|Accuracy|
|-|-:|
|v1 (rule-based tagger)|0.540|
|v2 (1-gram tagger)|0.840|
|v3 (ANN tagger)|0.932*|

## How to run

### Requirements

The following programs are required:

```yaml
  git-lfs:
    ver: '>= 2.10.0'
  docker:
    server:
      ver: '>= 19.03.8'
    client:
      ver: '>= 19.03.8'
    api:
      ver: '>= 1.40'
  docker-compose:
    ver: '>= 1.25.4'
  ```

Port **9999** to be *open* and *not busy*.

**!NOTE!** you should have **sudo** access rights in the env where you plan to run the code.

#### Requirements installation

**Docker**:

- [Ubuntu](https://docs.docker.com/install/linux/docker-ce/ubuntu/)
- [MacOs](https://docs.docker.com/docker-for-mac/install/)
- [Windows](https://docs.docker.com/docker-for-windows/install/)

**Git LFS**: 

see [here](https://git-lfs.github.com/)

*Framework tested on the following OS*:

```bash
- MacOS Mojave
- Ubuntu 16.04
```