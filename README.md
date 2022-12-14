# VoteBot

This repository contains data and code for the paper ___What are Pros and Cons? Stance Detection and Summarization on Feature Requests___ in ICSE'23.



## Overview

We propose VoteBot, which automatically detects stance on a feature request (i.e., stance detection) and summarizes the opinions (i.e., stance summarization), for facilitating the decision making of feature requests.



### Stance Detection

* Extract the reply-to relations among the comments.
* Incorporate these relations into a BERT-based classifier.



### Stance Summarization

* Acquire the semantic relevance and argumentative relations within a comment.
* Incorporate them with a graph-based ranking algorithm.



## Main Requirements

### Stance Detection

```
python 3.8

torch                   1.9.0+cu111
torchaudio              0.9.0
torchvision             0.10.0+cu111
```

### Stance Summarization

```
python 3.6

torch==1.6.0+cu101
torchvision==0.7.0+cu101
spacy==3.0.6
spacy-legacy==3.0.6
spacy-universal-sentence-encoder==0.4.3
```

### Pre-trained model and path

* The path of [bert-base-uncased](https://huggingface.co/bert-base-uncased): ``VoteBot_SD/bert_pretrain/``.
* The path of [pacsum_models](https://drive.google.com/file/d/1wbMlLmnbD_0j7Qs8YY8cSCh935WKKdsP/view?usp=sharing): ``VoteBot_SS/models/``.

## Data

* Data path for Stance Detection: ``VoteBot_SD/data/``.
* Data path for Stance Summarization: ``VoteBot_SS/data/``.

## Code

* VoteBot_SD is for Stance Detection.

* VoteBot_SS is for Stance Summarization.


## Running

```python
# stance detection
python run.py --model bert

# stance summarization
python run.py
```

## Example Results

![example](https://user-images.githubusercontent.com/112673904/189326644-6d1c5321-e6f9-4f98-bfce-38c2d0626f51.png)


