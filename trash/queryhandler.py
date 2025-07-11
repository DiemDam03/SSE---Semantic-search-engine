#query handler
from sentence_transformers import SentenceTransformer, util
import faiss
import numpy as np
import preproc
import pandas as pd

model = SentenceTransformer("all-MiniLM-L6-v2") 

# load saved data
emb = np.load('embeddings.npy')#?????
index = faiss.read_index("movie_index.index")

# load movie dataset
movie = pd.read_csv("ml-latest-small/movies.csv")

# take user input
query = input("What movie are you looking for? ")

# preprocess query
query = preproc.preprocessor(query)
query = ' '.join(query)  # join the list back to a string
# encode query
query_embedding = model.encode(query, 
                                convert_to_numpy=True, 
                                device='cpu')

# model.similarity(emb, query_embedding)

# cosine similarity search

top_k = 10
D, I = index.search(query_embedding.reshape(1, -1), k=top_k) 

for i, idx in enumerate(I[0]):
    print(f"{i+1}. {movie.iloc[idx]['title']} | {movie.iloc[idx]['genres']} (dist: {D[0][i]:.4f})")


# Re-ranking results based on cosine similarity
top_k_embeddings = emb[I[0]]

cos_scores = util.cos_sim(query_embedding, top_k_embeddings)[0].cpu().numpy()

reranked_indices = np.argsort(-cos_scores)

print("\nRe-ranked results:")
for rank, idx in enumerate(reranked_indices):
    movie_idx = I[0][idx]
    print(f"{rank+1}. {movie.iloc[movie_idx]['title']} | {movie.iloc[movie_idx]['genres']} (score: {cos_scores[idx]:.4f})")
