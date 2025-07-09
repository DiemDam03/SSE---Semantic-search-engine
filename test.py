from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer("all-MiniLM-L6-v2") 
import pandas as pd
import re
import faiss
import unicodedata

movie = pd.read_csv('ml-latest-small/movies.csv')  # read the movie dataset

movie['text'] = movie['title'] + ' | ' + movie['genres']
corpus = movie['text'].tolist()

# id_map = movie['movieId'].tolist()

def preprocessor(text):
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split() 

# corpus = [
#     "The weather is lovely today.",
#     "It's so sunny outside!",
#     "He drove to the stadium.",
# ]
embeddings = model.encode(corpus, convert_to_numpy=True, device='cpu')

# query = "funny animated fantasy movie"
query = input("What movie are you looking for? ") 
query_embedding = model.encode(query)
query_vec = model.encode([query], convert_to_numpy=True, device='cpu')

# cosine_scores = util.cos_sim(query_embedding, corpus_embeddings)

# results = zip(corpus, cosine_scores[0])
# for text, score in sorted(results, key=lambda x: x[1], reverse=True):
#     print(f"{score:.4f} => {text}")

similarities = model.similarity(embeddings, query_vec)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

_, I = index.search(query_vec, k=5)  # k = num of nearest neighbors

for idx in I[0]:
    print(movie.iloc[idx]['title'], '|', movie.iloc[idx]['genres'])
