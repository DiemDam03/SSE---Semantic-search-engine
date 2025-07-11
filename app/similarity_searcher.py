# import numpy as np 
# from numpy.linalg import norm

# def cosine_similarity(vector1, vector2):
#     cosine = np.dot(vector1, vector2) / (norm(vector1) * norm(vector2))
#     return cosine

# def similarity_search_top_k(database_vectors, query_vectors, top_k):
#     similarities = []
#     for stored_vector in database_vectors:
#         sim = cosine_similarity(query_vectors, stored_vector)
#         similarities.append(sim)
#     ranked = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)
#     return ranked[:top_k]

# # new
# def similarity_search_top_k(database, query_vector, top_k):
#     similarities = []
#     for sentence, vector in database:
#         sim = cosine_similarity(query_vector, vector)
#         similarities.append((sentence, sim))
#     ranked = sorted(similarities, key=lambda x: x[1], reverse=True)
#     return ranked[:top_k]

# similarity_searcher.py - Fixed version
import numpy as np 
from numpy.linalg import norm

def cosine_similarity(vector1, vector2):
    # Đảm bảo vectors có cùng độ dài
    if len(vector1) != len(vector2):
        # Padding với zeros nếu cần
        max_len = max(len(vector1), len(vector2))
        vector1 = np.pad(vector1, (0, max_len - len(vector1)), 'constant')
        vector2 = np.pad(vector2, (0, max_len - len(vector2)), 'constant')
    
    # Tránh chia cho 0
    norm1 = norm(vector1)
    norm2 = norm(vector2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    cosine = np.dot(vector1, vector2) / (norm1 * norm2)
    return cosine

def similarity_search_top_k(database, query_vector, top_k):
    similarities = []
    
    for sentence, vector in database:
        # Đảm bảo vector có đúng định dạng
        if isinstance(vector, (list, tuple)):
            vector = np.array(vector)
        
        sim = cosine_similarity(query_vector, vector)
        similarities.append((sentence, sim))
    
    # Sắp xếp theo similarity score giảm dần
    ranked = sorted(similarities, key=lambda x: x[1], reverse=True)
    
    # Trả về top k kết quả
    return ranked[:top_k]