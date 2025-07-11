from flask import Flask, jsonify, request, render_template
import vector_db
import similarity_searcher
import data_handler

app = Flask(__name__)

@app.route("/search", methods=["GET"])
def search():
    try:
        query = request.args.get("q", "").strip()
        if not query:
            return jsonify({"error": "Query is required"}), 400
        
        data = vector_db.loading_vectors()
        
        # Convert data to list of (sentence, vector) tuples
        database = [(sentence, vector) for sentence, vector in data.items()]

        query_tokens = data_handler.preprocessing(query)
        if not query_tokens:
            return jsonify({"error": "Invalid query"}), 400
        
        # Tạo vector cho query
        query_vector = data_handler.query_handling(query_tokens, database)
        
        # Tìm kiếm similarity - returns list of tuples (sentence, similarity_score)
        top_k_results = similarity_searcher.similarity_search_top_k(database, query_vector, 10)
        
        # Format results - extract sentence from tuple
        results = []
        for sentence_tuple in top_k_results:
            # sentence_tuple is (sentence, similarity_score)
            sentence = sentence_tuple[0]  # Get the sentence from the tuple
            similarity_score = sentence_tuple[1]  # Get the similarity score
            
            parts = sentence.split(' | ')
            title = parts[0] if len(parts) > 0 else sentence
            genres = parts[1] if len(parts) > 1 else "Unknown"
            
            results.append({
                "title": title,
                "genres": genres,
                "similarity_score": float(similarity_score)  # Add similarity score to results
                })

        return jsonify({"results": results})
        
    except Exception as e:
        print(f"Error in search: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == '__main__':
    try:
        # Khởi tạo vector database nếu chưa có
        print("Initializing vector database...")
        vector_db.loading_vectors()
        print("Vector database loaded successfully!")
    except FileNotFoundError:
        print("Vector database not found. Creating new one...")
        # Tạo vectors mới nếu chưa có
        dataset, vectorized_data = vector_db.generate_vectors()
        vector_db.storing_vectors(dataset, vectorized_data)
        print("Vector database created successfully!")
    
    print("Starting Flask app...")
    app.run(host='0.0.0.0', port=5000, debug=True)