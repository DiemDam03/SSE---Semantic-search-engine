from flask import Flask, request, jsonify
from vector_db import loading_vectors
from ancient_emb import compute_tf, compute_idf, compute_tfidf
import similarity_searcher 
import data_handler

app = Flask(__name__)

vector_store = loading_vectors()
corpus = list(vector_store.keys())
database_vectors = list(vector_store.values())

query = data_handler.query_taking()
query_vectors = data_handler.query_handling(query)


@app.route("/search", methods=["POST"])
def search():
    data = request.json
    query = data.get("query", "")
    top_k = similarity_searcher.similarity_search_top_k(query_vectors, 10)
    return jsonify(top_k)

if __name__ == "__main__":
    app.run(debug=True)
