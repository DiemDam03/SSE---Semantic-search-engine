import numpy as np
import data_handler

corpus = ['data science is one of the most important fields of science',
          'this is one of the best data science courses',
          'data scientists analyze data' ]

def build_vocab(corpus):
    vocab = set()
    for text in corpus:
        texts = data_handler.preprocessor(text)
        vocab.update(texts)
    return vocab

def index_corpus(corpus, vocabulary):
    word_to_index = {word: i for i, word in enumerate(vocabulary)}
    vectors = []
    for text in corpus:
        vector = np.zeros(len(vocabulary))
        tokens = data_handler.preprocessor(text)
        for token in tokens:
            if token in word_to_index:
                vector[word_to_index[token]] += 1
        vectors.append(vector)
    return vectors

