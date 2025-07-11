import pandas as pd
import unicodedata
import re
import numpy as np

def movie_dataset_processing(source='ml-latest-small/movies.csv'):
    movie = pd.read_csv(source) # Load the movie dataset
    movie['text'] = movie['title'] + ' | ' + movie['genres'] # Combine title and genres into a single text column
    corpus = movie['text'].tolist() 
    return corpus

def preprocessing(text):
    text = unicodedata.normalize("NFC", text) # normalize unicode
    text = text.lower() # convert to lowercase
    text = re.sub(r"[^\w\s]", "", text) # remove punctuation
    text = re.sub(r"\s+", " ", text).strip() # remove extra spaces
    return text.split() 

def query_handling(query_tokens, database):

    all_tokens = set()
    for sentence, _ in database:
        if not isinstance(sentence, str):
            sentence = " | ".join(sentence) 
        tokens = preprocessing(sentence)
        all_tokens.update(tokens)
    
    vocab = list(all_tokens)
    vocab_dict = {token: i for i, token in enumerate(vocab)}
    
    # Tạo vector cho query
    query_vector = np.zeros(len(vocab))
    for token in query_tokens:
        if token in vocab_dict:
            query_vector[vocab_dict[token]] += 1

    # Normalize query vector
    if np.linalg.norm(query_vector) > 0:
        query_vector = query_vector / np.linalg.norm(query_vector)
    return query_vector