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

1. Develop a baseline model
2. Develop a framework to train and serve models
3. Train models iteratively to improve final model performance
4. (*) Deploy model as a web server to be accessible over HTTP as Rest API
5. (*) Setup a train pipeline on GCP (**).

(*) - additional steps

(**) - the cloud provided is selected based on the free leftover credits I have :)

## Solution

A rule-based model to be used as a baseline. It should easily beat random guessing and cover good 70-80% of the corpus cases.

## How to run

### Requirements

The following programs are required:

```yaml
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

Find details:

- [Ubuntu](https://docs.docker.com/install/linux/docker-ce/ubuntu/)
- [MacOs](https://docs.docker.com/docker-for-mac/install/)
- [Windows](https://docs.docker.com/docker-for-windows/install/)
