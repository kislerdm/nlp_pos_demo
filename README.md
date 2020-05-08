# PoS Tagger for English Words based on the Georgetown University Multilayer Corpus

## Goal

To build a Part-of-Speech tagger using [Georgetown University Multilayer Corpus](https://github.com/UniversalDependencies/UD_English-GUM/tree/master).

### Problem

Despite the fact that the PoS problem may be not of a great usefulness per se, PoS tagging is of a definite importance as an input for text classification (e.g. for sentiment, or styling analysis and syntactic parsing), or as a benchmark for algorithms performance.

We all learned PoS tagging in school where we mostly used rule-based method to classify a given word. This method was quite accurate when applying to a stand-alone word, it was however often failing in case of a rarely spoken, or foreign language origin word. The rules weren't often working when a word was put into a sentence, the words sequence/context was becoming important in such case. In some rare cases, i.e. idiomatic expressions, even words sequence analysis didn't work.

The PoS tagging problem is known since decades in computational linguistics. It was approached using different methods:

- *Rule based*, based on the regex rules (it's also used for words normalization).
- *Lexical based*, based on PoS tag occurrence with a word in the text corpus.
- *Probabilistic/contextual based*, based on the probability of a PoS tags sequence occurrence.
- *ANN*, based on sub-words embeddings.

This project illustrates a generic approach towards solving ML (or classical) software problem with business implications and focus on deliverability, maintainability and scalability of solution.

## Project flow

A data science project flow described in [the article](https://towardsdatascience.com/how-to-power-up-your-product-by-machine-learning-with-python-microservice-pt-1-de0f2b434bec) is being followed.

### Objectives

1. Develop a framework to train and serve models
2. Develop a baseline model
3. Develop models iteratively to improve PoS tagging accuracy

#### Model Development

The logical steps of model development:

- Model **v1**, *rule-based tagger*. It must be better than random PoS class guessing and should presumably yield to the accuracy of 50-60%. Default class labling (e.g. labling every token as 'NOUN') would give over 20% accuracy alone (see the corpus composition [here](./data/UD_English-GUM/stats.xml)).
- Model **v2**, *1-gram freq. tagger*, mix of lexical and contextual method. It can reach over 80% accuracy.
- Model **v3**, *ANN tagger*. At that point, one needs to seriously consider the trade-off between time investment and business requirements. State-of-the-art model performance (every additional 1% after 90% accuracy) can cost months of human hours and years of computation run time with no guarantee of model performance improvement. Luckily, deep learning abstraction creates wide prospect of opportunities for transfer learning. Thanks to NLP community and many research groups, there is a handful of great frameworks and several pre-trained [PoS and NER tagging models](http://nlpprogress.com/english/part-of-speech_tagging.html) with the state-of-the-art performance (accuracy, or F1 score) of above 0.95.

The opensource library [flair](https://github.com/flairNLP/flair) is being used to develop the model **v3**. It is build on top of [pytorch](https://pytorch.org/) and has "low entry barrier". It is being under active development and maintenance by community and the [ML and NLP group](https://www.informatik.hu-berlin.de/en/forschung-en/gebiete/ml-en/) at Humboldt-Universität zu Berlin. 

As the **v3** model base, a pre-trained tagger [**'pos-fast'**](https://github.com/flairNLP/flair/blob/master/resources/docs/TUTORIAL_2_TAGGING.md) is being used.

## Accuracy comparison

|Model                 |Accuracy|F1 score (weighted)|Training time [s]|Size [bytes]|Prediction time [s] (*)|
|:-                    |      -:|                 -:|               -:|          -:|                     -:|
|v1 (rule-based tagger)|    0.54|               0.49|            ~ 3.5|         835|                  < 0.7|
|v2 (1-gram tagger)    |    0.84|               0.83|            ~ 1.7|      221803|                  ~ 0.3|
|v3 (DNN tagger)       |   ~ 1.0|              ~ 1.0|           ~ 3670|    75183392|                   > 50|

**!Note!** Model predictive performance metrics were evaluated using the [sklearn.metrics](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics) based [function](./app/framework/tagger_framework/tagger/pos/evaluation.py) on the [train data set](./data/UD_English-GUM/en_gum-ud-train.conllu).

**!Note!** The **v3** model was trained for 41 epochs (with limit set to 100) with learning rate annealing by the factor of 0.5 after every 2nd epoch with no loss improvement.

**!Note!** (*) The time required to run prediction was measured during models evaluation, ergo the numbers cannot be taken as the proper benchmark, but rather as a qualitative comparison. The time measured on 5yo machine with the specs:

```yaml
OS: MacOS Mojave
CPU: Intel Core i5
RAM: 8GB 1867MHz (DDR3)
```

### Trade-off

Model v3 has noticeable accuracy improvement compared to the model v2 (I'd question the accuracy of 1.0, even though sklearn was used to compute the metrics). The improvements however come at the price of model complexity, hence its high size and computation power requirements as well as higher maintenance costs (in terms of human hours). These factors must be taken to account for projects risks assessment. One should carefully assess if higher model performance (in terms of accuracy, or other prediction quality metric) brings enough business value to be worth development time and extra operational costs.

## PoS Framework

One of the set project's objectives is, to illustrate real-life approach one can take to build a flexible and extendable base for NLP services. A python package/framework is proposed. See details in [the package documentation](./app/framework/README.md).

## Framework runner application

The framework is operated by the runner application which include three services:

- train: the service to train a model defined in the framework with the use of [conllu](https://universaldependencies.org/format.html) data sets. **!Note!** train and dev sets are required.
- evaluate: the service to evaluate the model using conllu data set, i.e. test data set.
- serve: the service to run prediction using (pre-)trained models (defined in the framework/package) given the file of tokenized sentences.  **!Note!** one sentence per line, tokens separated by "space".

*Every service is dockerized and can be deployed to k8s cluster, or as a AWS Fargate, or GCP Cloud Run service.*

The services are operated with a help of the bash routing/triggering scripts [./run.sh](./run.sh) and [runner.sh](./app/service/runner.sh).

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

**!NOTE!** you may need **sudo** access rights (depending on your docker settings) in the env where you plan to run the code.

#### Requirements installation

**Docker**:

- [Ubuntu](https://docs.docker.com/install/linux/docker-ce/ubuntu/)
- [MacOs](https://docs.docker.com/docker-for-mac/install/)
- [Windows](https://docs.docker.com/docker-for-windows/install/)

**Git LFS**: 

see [here](https://git-lfs.github.com/)

*Framework and app have been tested on the following OS*:

```bash
- MacOS Mojave
- Ubuntu 16.04
```

### Repo

Please clone the repo to run the app (**!Note!** [git-lfs](https://git-lfs.github.com/) must be installed):

```bash
git clone git@github.com:kislerdm/nlp_pos_demo.git . --recursive
```

or (over HTTP)

```bash
git clone https://github.com/kislerdm/nlp_pos_demo.git . --recursive
```

### Run instructions

The main trigger script is `./run.sh` can be executed to trigger test and serve/predict services.

You can get more information by calling helper function (or by simply executing the script)

```bash
./run.sh -h
```

### Tests

To trigger a set of e2e tests, run:

```bash
./run.sh test
```

**!Note!** The tests may take up to 5-10 minutes.

### Build the app

To build the app for a `MODEL_VERSION`, run:

```bash
./run.sh build ${MODEL_VERSION}
```

### Run training/prediction/evaluate services

To run train/predict/evaluate service, run the following command:

```bash
./run.sh ${TYPE} ${MODEL_VERSION} -h
```
(it outputs the [argparse](https://docs.python.org/3/library/argparse.html) helper for expected stdin parameters)

where

- TYPE is *train*, or *serve*, or *evaluate*
- MODEL_VERSION is the model [version](#Model%20Development)

#### Examples

***Example***: train and evaluate the [**v1** model](#Model%20Development) on the whole corpus.

```bash
./run.sh train v1 \
  --path-train en_gum-ud-train.conllu \
  --path-dev en_gum-ud-dev.conllu \
  --path-test en_gum-ud-test.conllu \
  --path-model-out v1/v1.pt
```

*Expected stdout logs*:

```bash
2020-04-22 20:15:42.967 [INFO ] [service/train/v1] Init.
2020-04-22 20:15:44.086 [INFO ] [service/train/v1] Reading data and building corpus.
2020-04-22 20:15:45.502 [INFO ] [service/train/v1] Defining the model.
2020-04-22 20:15:45.507 [INFO ] [service/train/v1] Start model training.
2020-04-22 20:15:49.019 [INFO ] [service/train/v1] Training completed. Elapsed time 3.51 sec.
2020-04-22 20:15:49.020 [INFO ] [service/train/v1] Model score:
{
  "train": {
    "accuracy": 0.5221330275229358,
    "f1_micro": 0.5221330275229358,
    "f1_macro": 0.38634727886202297,
    "f1_weighted": 0.4732954069400854
  },
  "dev": {
    "accuracy": 0.5388315269672289,
    "f1_micro": 0.5388315269672289,
    "f1_macro": 0.38474140644524296,
    "f1_weighted": 0.48794378870557026
  },
  "test": {
    "accuracy": 0.5403165033911078,
    "f1_micro": 0.5403165033911078,
    "f1_macro": 0.3882471826600896,
    "f1_weighted": 0.4867607976410461
  }
}
2020-04-22 20:15:49.023 [INFO ] [service/train/v1] Saving model to /model/v1/v1.pt
```

The resulting model is expected to be found in [`./model/v1.pt`](./model/v1.pt).

***Example***: run prediction with the model trained on the step above.

```bash
./run.sh serve v1 \
  --path-model v1/v1.pt \
  --path-input test.txt \
  --path-output test.json
```

*Expected stdout logs*:

```bash
2020-04-22 20:16:47.218 [INFO ] [service/serve/v1] Loading the model.
2020-04-22 20:16:47.233 [INFO ] [service/serve/v1] Read the input data.
2020-04-22 20:16:47.247 [INFO ] [service/serve/v1] Prepare/tokenize data.
2020-04-22 20:16:47.249 [INFO ] [service/serve/v1] Run prediction.
2020-04-22 20:16:47.250 [INFO ] [service/serve/v1] Convert prediction to output format.
2020-04-22 20:16:47.251 [INFO ] [service/serve/v1] Write prediction results to /data_prediction/test.json
2020-04-22 20:16:47.271 [INFO ] [service/serve/v1] Prediction completed. Elapsed time 0.05 sec.
```

The model prediction results file is expected to be found in [`./data/prediction/output/test.json`](./data/prediction/output/test.json).

The serve/prediction service output format:

```js
[
  {
    "form": List[String],
    "upos": List[String],
  },
  {
    "form": List[String],
    "upos": List[String],
  },...,
  {
    "form": List[String],
    "upos": List[String],
  },
]
```

***Example***: run model v1 evaluation using *test* data set.

```bash
./run.sh evaluate v1 \
--path-model v1/v1.pt \
--path-input UD_English-GUM/en_gum-ud-test.conllu \
--dir-output v1
```

*Expected stdout logs*:

```bash
2020-04-23 01:12:13.432 [INFO ] [service/evaluate/v1] Loading the model.
2020-04-23 01:12:13.449 [INFO ] [service/evaluate/v1] Reading data and preparing dataset.
2020-04-23 01:12:13.705 [INFO ] [service/evaluate/v1] Starting model performance evaluation.
2020-04-23 01:12:14.378 [INFO ] [service/evaluate/v1] Writing evaluation results to /model/v1.
2020-04-23 01:12:14.394 [INFO ] [service/evaluate/v1] Model evaluation completed. Elapsed time 0.96 sec.
```

The evaluation results file when running above command is expected to be found in [`./model/v1/`](./model/v1/metrics_v1_20200423T01121587604334Z.json).

The evaluation service output format:

```json
{
  "accuracy": 0.5403165033911078,
  "f1_micro": 0.5403165033911078,
  "f1_macro": 0.3882471826600896,
  "f1_weighted": 0.4867607976410461,
  "meta": {
    "model": {
      "version": "v1",
      "framework_version": "1.0.0"
    },
    "data": {
      "relative_path": "UD_English-GUM/en_gum-ud-test.conllu",
      "volume": {
        "sentences_count": 890,
        "tags_count": 15924
      }
    }
  }
}
```

**!Note!** 

```yaml
The path is relative:
  model: "to the ./model directory"
  data:
    train: "to the ./data/UD_English-GUM directory"
    serve: 
      input: "to the ./data/prediction/input directory"
      output: "to the ./data/prediction/output directory"
    evaluate: "to the ./data directory"
```
