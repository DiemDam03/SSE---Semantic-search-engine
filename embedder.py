from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import faiss
import dataproc

model = SentenceTransformer("all-MiniLM-L6-v2") 
corpus = dataproc.data_handler()

# sbert embeddings
embeddings = model.encode(corpus, 
                          convert_to_numpy=True, 
                          show_progress_bar=True, 
                          device='cpu')

# faiss index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

#Saving the index and embeddings
faiss.write_index(index, "movie_index.index")
np.save('embeddings.npy', embeddings)
