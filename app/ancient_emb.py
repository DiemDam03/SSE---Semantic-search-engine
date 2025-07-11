# import math
# from collections import Counter
# import numpy as np
# import data_handler

# def compute_tf(corpus):
#     tokens = [token for sentence in corpus for token in data_handler.preprocessing(sentence)]
#     total = len(tokens)
#     tf = Counter(tokens)
#     return {term: count / total for term, count in tf.items()}

# def compute_idf(corpus):
#     sentence_count = len(corpus)
#     df = Counter()
#     for sentence in corpus:
#         tokens = set(data_handler.preprocessing(sentence))
#         for token in tokens:
#             df[token] += 1
#     return {term: math.log(sentence_count / df[term]) for term in df}

# def compute_tfidf(tf, idf):
#     return {term: tf[term] * idf.get(term, 0.0) for term in tf}

# def vectorize_sentence(sentence, tfidf_vocab):
#     tokens = data_handler.preprocessing(sentence)
#     vector = np.array([tfidf_vocab.get(token, 0.0) for token in tokens])
#     return vector

# def vectorize_corpus(corpus):
#     tf = compute_tf(corpus)
#     idf = compute_idf(corpus)
#     tfidf = compute_tfidf(tf, idf)
#     vectors = [vectorize_sentence(sentence, tfidf) for sentence in corpus]
#     return vectors

# ancient_emb.py - Improved version
import math
from collections import Counter
import numpy as np
import data_handler

def compute_tf(corpus):
    """Tính Term Frequency cho toàn bộ corpus"""
    print("Computing TF...")
    all_tokens = []
    for sentence in corpus:
        tokens = data_handler.preprocessing(sentence)
        all_tokens.extend(tokens)
    
    total = len(all_tokens)
    tf = Counter(all_tokens)
    
    # Normalize TF
    tf_normalized = {term: count / total for term, count in tf.items()}
    print(f"Computed TF for {len(tf_normalized)} unique terms")
    return tf_normalized

def compute_idf(corpus):
    """Tính Inverse Document Frequency"""
    print("Computing IDF...")
    sentence_count = len(corpus)
    df = Counter()
    
    for sentence in corpus:
        tokens = set(data_handler.preprocessing(sentence))
        for token in tokens:
            df[token] += 1
    
    # Tính IDF với smoothing để tránh log(0)
    idf = {}
    for term in df:
        idf[term] = math.log(sentence_count / (df[term] + 1)) + 1
    
    print(f"Computed IDF for {len(idf)} unique terms")
    return idf

def compute_tfidf(tf, idf):
    """Tính TF-IDF scores"""
    print("Computing TF-IDF...")
    tfidf = {}
    for term in tf:
        tfidf[term] = tf[term] * idf.get(term, 0.0)
    
    print(f"Computed TF-IDF for {len(tfidf)} terms")
    return tfidf

def create_vocabulary(corpus):
    """Tạo vocabulary từ corpus"""
    print("Creating vocabulary...")
    all_tokens = set()
    for sentence in corpus:
        tokens = data_handler.preprocessing(sentence)
        all_tokens.update(tokens)
    
    vocab = list(all_tokens)
    vocab_dict = {token: i for i, token in enumerate(vocab)}
    print(f"Created vocabulary with {len(vocab)} unique terms")
    return vocab, vocab_dict

def vectorize_sentence(sentence, vocab_dict, tfidf):
    """Chuyển đổi sentence thành vector dựa trên vocabulary"""
    tokens = data_handler.preprocessing(sentence)
    vector = np.zeros(len(vocab_dict))
    
    # Đếm frequency của mỗi token trong sentence
    token_counts = Counter(tokens)
    
    for token, count in token_counts.items():
        if token in vocab_dict:
            idx = vocab_dict[token]
            # Sử dụng TF-IDF score
            vector[idx] = tfidf.get(token, 0.0) * count
    
    # Normalize vector
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    
    return vector

def vectorize_corpus(corpus):
    """Vectorize toàn bộ corpus"""
    print(f"Vectorizing corpus of {len(corpus)} sentences...")
    
    # Tạo vocabulary
    vocab, vocab_dict = create_vocabulary(corpus)
    
    # Tính TF-IDF
    tf = compute_tf(corpus)
    idf = compute_idf(corpus)
    tfidf = compute_tfidf(tf, idf)
    
    # Vectorize từng sentence
    vectors = []
    for i, sentence in enumerate(corpus):
        if i % 1000 == 0:
            print(f"Vectorized {i}/{len(corpus)} sentences")
        
        vector = vectorize_sentence(sentence, vocab_dict, tfidf)
        vectors.append(vector)
    
    print(f"Vectorization complete! Generated {len(vectors)} vectors")
    return vectors

