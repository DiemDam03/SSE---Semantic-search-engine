import pandas as pd
import preproc

def data_handler(source='ml-latest-small/movies.csv'):
    movie = pd.read_csv(source)

    movie['text'] = movie['title'] + ' | ' + movie['genres']
    
    corpus = movie['text'].tolist()

    corpus = [preproc.preprocessor(doc) for doc in corpus]
    
    return corpus
