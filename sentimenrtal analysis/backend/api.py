import joblib
import re
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

def clean_text_basic(t: str) -> str:
    t = str(t).lower()
    t = re.sub(r"<br\s*/?>", " ", t)       
    t = re.sub(r"http\S+|www\.\S+", " ", t) 
    t = re.sub(r"[^a-z0-9\s']", " ", t)     
    t = re.sub(r"\s+", " ", t).strip()      
    return t

# Load Model and Vectorizer from the parent directory
try:
    model = joblib.load("../optimized_imdb_model.pkl")
    vectorizer = joblib.load("../tfidf_vectorizer.pkl")
except Exception as e:
    print(f"Error loading model or vectorizer: {e}")
    model = None
    vectorizer = None

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or vectorizer is None:
        return jsonify({"error": "Model or vectorizer not loaded."}), 500

    data = request.json
    if not data or 'review' not in data:
        return jsonify({"error": "No review provided."}), 400

    review = data['review']
    
    if review.strip() == "":
        return jsonify({"error": "Review cannot be empty."}), 400

    try:
        cleaned_review = clean_text_basic(review)
        review_vector = vectorizer.transform([cleaned_review])
        prediction = int(model.predict(review_vector)[0])
        
        try:
            probability = model.predict_proba(review_vector)[0]
            confidence = float(max(probability))
        except:
            confidence = None

        return jsonify({
            "prediction": prediction,
            "confidence": confidence
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8502)
