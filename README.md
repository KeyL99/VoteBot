# VoteBot_Issue

This repository contains data and code for votebot stance detection and  summarization.



We propose votebot automatically detecting stance (in favor or against) on a feature request and summarizing the opinions, which can facilitate the decision making of feature requests.



### stance detection:

* extract the reply-to relations among the comments
* incorporate these relations into a BERT-based classifier



### stance summarization:

* acquire the semantic relevance and argumentative relations within a comment
* incorporate them with a graph-based ranking algorithm



To get our pre-trained model：

* [model1](https://huggingface.co/bert-base-uncased) for bert-base-uncased
* [model2](https://drive.google.com/file/d/1wbMlLmnbD_0j7Qs8YY8cSCh935WKKdsP/view?usp=sharing) for pacssum_models



### Code：

* VoteBot_SS is for stance detection

* VoteBot_SS is for stance summarization



### Running

```python
# stance detection
python run.py --model bert

# stance summarization
python run.py
```





### Questions

If you have a question please either:

- Open an issue on [github](https://github.com/KeyL99/VoteBot/issues).

