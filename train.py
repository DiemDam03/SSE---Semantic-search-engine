from sentence_transformers import models, SentenceTransformer
from datasets import load_dataset


transformer = models.Transformer("all-MiniLM-L6-v2")
pooling = models.Pooling(transformer.get_word_embedding_dimension(), pooling_mode="mean")

model = SentenceTransformer(modules=[transformer, pooling])

dataset = load_dataset("csv", data_files="..")
