import os
import numpy as np
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import io

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Model & class labels
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "mobilenetv2_model.keras")
IMG_SIZE = (128, 128)

# Class labels in alphabetical order — this matches the order Keras
# assigns when using flow_from_directory (sorted directory names).
CLASS_LABELS = [
    "Ants",
    "Bees",
    "Beetle",
    "Caterpillar",
    "Earthworms",
    "Earwig",
    "Grasshopper",
    "Moth",
    "Slug",
    "Snail",
    "Wasp",
    "Weevil",
]

# Descriptions for each pest class (short, informative)
CLASS_DESCRIPTIONS = {
    "Ants": "Small social insects that can damage crops by farming aphids and disrupting root systems.",
    "Bees": "Important pollinators, but some species can become agricultural pests in certain contexts.",
    "Beetle": "Beetles are among the most destructive pests, feeding on leaves, roots, and stored grain.",
    "Caterpillar": "Larval stage of butterflies/moths; causes heavy leaf damage by voracious feeding.",
    "Earthworms": "Generally beneficial for soil health, but can sometimes indicate over-moist conditions.",
    "Earwig": "Nocturnal insects that feed on soft fruits, seedlings, and plant shoots.",
    "Grasshopper": "Known for swarming and devastating crops; can consume their body weight in food daily.",
    "Moth": "Adult moths are mostly harmless, but their larvae (caterpillars) damage crops significantly.",
    "Slug": "Soft-bodied pests that feed on seedlings and leaves, especially in damp environments.",
    "Snail": "Similar to slugs, snails chew large holes in leaves and can decimate young plants.",
    "Wasp": "Predatory insects; some species help control pests, but others can damage ripe fruits.",
    "Weevil": "Small beetles with elongated snouts; notorious for destroying stored grains and seeds.",
}

print("Loading model …")
model = load_model(MODEL_PATH)
print("Model loaded successfully!")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        # Read and preprocess the image
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img = img.resize(IMG_SIZE)
        img_array = img_to_array(img) / 255.0  # rescale same as training
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        predictions = model.predict(img_array, verbose=0)[0]

        # Build results — top 5
        indexed = list(enumerate(predictions))
        indexed.sort(key=lambda x: x[1], reverse=True)
        top5 = indexed[:5]

        results = []
        for idx, confidence in top5:
            label = CLASS_LABELS[idx]
            results.append(
                {
                    "label": label,
                    "confidence": round(float(confidence) * 100, 2),
                    "description": CLASS_DESCRIPTIONS.get(label, ""),
                }
            )

        return jsonify({"predictions": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, port=5001)
