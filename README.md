# VoteBot

This repository contains data and code for the paper **What are Pros and Cons? Stance Detection and Summarization on Feature Request**.



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

#### Data Format

**The data formats of the two datasets mentioned above:**

* **Stance Detection**
  
  ```
  "tokens": comment sentences.
  "label": ground-truth stance polarity.
  "login": the user name of this commenter.
  "character": the role of this commenter.
  "reply": the commenter role of child comments.
  "beReplied": the commenter role of parent comments.
  ```
  
* **Stance Summarization**
  
  ```
  "doc": comment sentences.
  "target": the index of the ground-truth summary sentence.
  "issue_sim": semantic relevance between each comment sentence and the feature description.
  "prob": argumentative relations of each comment sentence (i.e., the probability where the argumentative relation is predicted as MajorClaim).
  ```



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

![image](https://raw.githubusercontent.com/KeyL99/VoteBot/main/images/example.png)

