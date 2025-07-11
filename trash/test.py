import numpy as np

# Example sentences
sentences = [
    "I eat mango",
    "mango is my favorite fruit",
    "mango, apple, oranges are fruits",
    "fruits are good for health",
]

# Tokenization and vocabulary creation
vocabulary = set()
for sentence in sentences:
    tokens = sentence.lower().split()
    vocabulary.update(tokens)

word_to_index = {word: i for i, word in enumerate(vocabulary)}

# Vectorization
sentence_vectors = {}
for sentence in sentences:
    tokens = sentence.lower().split()
    vector = np.zeros(len(vocabulary))
    for token in tokens:
        vector[word_to_index[token]] += 1
    sentence_vectors[sentence] = vector

# VectorStore class (simplified)
class VectorStore:
    def __init__(self):
        self.vector_data = {}

    def add_vector(self, vector_id, vector):
        self.vector_data[vector_id] = vector

    def find_similar_vectors(self, query_vector, num_results=2):
        results = []
        for vector_id, vector in self.vector_data.items():
            similarity = np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
            results.append((vector_id, similarity))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:num_results]

# Store vectors
vector_store = VectorStore()
for sentence, vector in sentence_vectors.items():
    vector_store.add_vector(sentence, vector)

# Query
query_sentence = "Mango is the best fruit"
query_vector = np.zeros(len(vocabulary))
for token in query_sentence.lower().split():
    if token in word_to_index:
        query_vector[word_to_index[token]] += 1

similar_sentences = vector_store.find_similar_vectors(query_vector, num_results=2)

# Output
print("Query Sentence:", query_sentence)
print("Similar Sentences:")
for sentence, similarity in similar_sentences:
    print(f"{sentence}: Similarity = {similarity:.4f}")
