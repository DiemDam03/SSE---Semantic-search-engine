import re
import unicodedata

corpus = ['data science is one of the most important fields of science',
          'this is one of the best data science courses',
          'data scientists analyze data' ]

query = "What is data science?"

def preprocessor(text):
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split() 

def build_vocab(corpus):
    vocab = set()
    for text in corpus:
        texts = preprocessor(text)
        vocab.update(texts)
    return vocab

vocab = build_vocab([query])
# vocab = build_vocab(corpus)
print("Vocabulary:", vocab)

