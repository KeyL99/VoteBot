from gensim_preprocess import preprocess_documents


def evaluate_f1(model_tags, references, sen_num_per_doc):
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    assert len(model_tags) == len(references)
    assert len(model_tags) == len(sen_num_per_doc)
    for i in range(len(sen_num_per_doc)):
        #print(i,model_tags[i],":<",references[i])
        temp = 0
        for model_tag in model_tags[i]:
            if model_tag in references[i]:
                true_positive += 1
                temp += 1
            else:
                false_positive += 1
        false_negative += len(references[i]) - temp
        true_negative += sen_num_per_doc[i] - len(model_tags[i]) - len(references[i]) + temp


    precision = true_positive * 1.0 / (true_positive + false_positive + 1e-6)
    recall = true_positive * 1.0 / (true_positive + false_negative + 1e-6)
    fmeasure = (2.0 * precision * recall) / (precision + recall + 1e-6)

    return {'p': precision, 'r': recall, 'f1': fmeasure}


def clean_text_by_sentences(text):
    """Tokenize a given text into sentences, applying filters and lemmatize them.

    Parameters
    ----------
    text : str
        Given text.

    Returns
    -------
    list of :class:`~gensim.summarization.syntactic_unit.SyntacticUnit`
        Sentences of the given text.

    """
    original_sentences = text
    filtered_sentences = [join_words(sentence) for sentence in preprocess_documents(original_sentences)]

    return filtered_sentences


def join_words(words, separator=" "):
    """Concatenates `words` with `separator` between elements.

    Parameters
    ----------
    words : list of str
        Given words.
    separator : str, optional
        The separator between elements.

    Returns
    -------
    str
        String of merged words with separator between elements.

    """
    return separator.join(words)
