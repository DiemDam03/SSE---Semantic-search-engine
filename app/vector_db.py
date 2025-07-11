# import os
# import data_handler  
# import ancient_emb
# import pickle

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# DATASET_PATH = os.path.join(BASE_DIR, "ml-latest-small", "movies.csv")
# VECTOR_FILE = os.path.join(BASE_DIR, "vector_store.pkl")

# def generate_vectors():
#     dataset = data_handler.movie_dataset_processing(DATASET_PATH)
#     vectorized_dataset = ancient_emb.vectorize_corpus(dataset)
#     return dataset, vectorized_dataset

# def storing_vectors(dataset, vectorized_dataset):
#     vector_store = {sentence: vector for sentence, vector in zip(dataset, vectorized_dataset)}
#     with open(VECTOR_FILE, "wb") as f:
#         pickle.dump(vector_store, f)
#     print(f"[INFO] Vector store saved to {VECTOR_FILE}")

# def loading_vectors():
#     if not os.path.exists(VECTOR_FILE):
#         raise FileNotFoundError(f"{VECTOR_FILE} does not exist.")
#     with open(VECTOR_FILE, "rb") as f:
#         vector_store = pickle.load(f)
#     return vector_store

# dataset, vectorized_data = generate_vectors()
# storing_vectors(dataset, vectorized_data)

# vector_db.py - Improved version
import os
import data_handler  
import ancient_emb
import pickle
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "ml-latest-small", "movies.csv")
VECTOR_FILE = os.path.join(BASE_DIR, "vector_store.pkl")

def generate_vectors():
    """Tạo vectors từ dataset"""
    print("Loading dataset...")
    dataset = data_handler.movie_dataset_processing(DATASET_PATH)
    print(f"Dataset loaded with {len(dataset)} movies")
    
    print("Generating vectors...")
    vectorized_dataset = ancient_emb.vectorize_corpus(dataset)
    print(f"Generated {len(vectorized_dataset)} vectors")
    
    return dataset, vectorized_dataset

def storing_vectors(dataset, vectorized_dataset):
    """Lưu vectors vào file"""
    print("Storing vectors...")
    
    # Đảm bảo vectors có định dạng nhất quán
    processed_vectors = []
    for i, vector in enumerate(vectorized_dataset):
        if isinstance(vector, (list, tuple)):
            vector = np.array(vector)
        
        # Đảm bảo vector không rỗng
        if len(vector) == 0:
            print(f"Warning: Empty vector at index {i}")
            vector = np.zeros(100)  # Default vector size
        
        processed_vectors.append(vector)
    
    # Tạo dictionary mapping từ sentence tới vector
    vector_store = {}
    for sentence, vector in zip(dataset, processed_vectors):
        vector_store[sentence] = vector
    
    # Lưu vào file
    with open(VECTOR_FILE, "wb") as f:
        pickle.dump(vector_store, f)
    
    print(f"[INFO] Vector store saved to {VECTOR_FILE}")
    print(f"[INFO] Total entries: {len(vector_store)}")

def loading_vectors():
    """Load vectors từ file"""
    if not os.path.exists(VECTOR_FILE):
        print(f"Vector file not found at {VECTOR_FILE}")
        raise FileNotFoundError(f"{VECTOR_FILE} does not exist.")
    
    print("Loading vectors from file...")
    with open(VECTOR_FILE, "rb") as f:
        vector_store = pickle.load(f)
    
    print(f"[INFO] Loaded {len(vector_store)} vectors from {VECTOR_FILE}")
    return vector_store

def initialize_database():
    """Khởi tạo database nếu chưa có"""
    if not os.path.exists(VECTOR_FILE):
        print("Vector database not found. Creating new one...")
        dataset, vectorized_data = generate_vectors()
        storing_vectors(dataset, vectorized_data)
        print("Vector database created successfully!")
    else:
        print("Vector database already exists.")

# Khởi tạo database khi import module
if __name__ == "__main__":
    initialize_database()