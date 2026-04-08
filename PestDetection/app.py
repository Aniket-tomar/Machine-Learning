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

# Detailed pest information: description, crop effects, harmfulness, and remedies
PEST_INFO = {
    "Ants": {
        "description": "Small social insects that can damage crops by farming aphids and disrupting root systems.",
        "harmful": True,
        "crop_effects": [
            "Protect and farm aphids/mealybugs, leading to increased sap-sucking pest populations",
            "Tunnel through soil around roots, disrupting water and nutrient uptake",
            "Spread sooty mold by promoting honeydew-producing insects",
            "Can damage seedlings and young transplants by undermining root zones",
        ],
        "remedies": [
            "Apply food-grade diatomaceous earth around plant bases",
            "Use borax-based bait stations placed near ant trails",
            "Flood ant mounds with soapy water (2 tbsp dish soap per gallon)",
            "Plant mint, tansy, or garlic as natural repellents around crop borders",
            "Introduce beneficial nematodes to target underground colonies",
        ],
    },
    "Bees": {
        "description": "Important pollinators essential for crop yields; generally beneficial to agriculture.",
        "harmful": False,
        "crop_effects": [
            "Pollinate flowers of fruits, vegetables, and oilseed crops — boosting yields by 30-80%",
            "Rarely cause crop damage; occasional minor flower damage during nectar collection",
            "Their presence indicates a healthy ecosystem around the farm",
        ],
        "remedies": [],  # Not harmful — no remedies needed
    },
    "Beetle": {
        "description": "Beetles are among the most destructive agricultural pests, feeding on leaves, roots, and stored grain.",
        "harmful": True,
        "crop_effects": [
            "Skeletonize leaves — reducing photosynthesis and weakening the plant",
            "Larvae (grubs) feed on roots, causing wilting and plant death",
            "Bore into stems and fruits, creating entry points for fungal diseases",
            "Destroy stored grains and seeds in post-harvest storage (up to 30% losses)",
            "Can transmit bacterial wilt and other crop diseases",
        ],
        "remedies": [
            "Handpick beetles early morning when they are sluggish",
            "Apply neem oil spray (2-3 ml/liter) at weekly intervals",
            "Use row covers / floating row covers during peak beetle season",
            "Introduce predatory insects: ladybugs, lacewings, and parasitic wasps",
            "Rotate crops annually to break beetle life cycles in the soil",
            "Apply Bacillus thuringiensis (Bt) for larval control",
        ],
    },
    "Caterpillar": {
        "description": "Larval stage of butterflies/moths; voracious leaf feeders that cause heavy crop damage.",
        "harmful": True,
        "crop_effects": [
            "Devour leaves rapidly — a single caterpillar can consume several leaves per day",
            "Bore into fruits (e.g., corn earworm, tomato hornworm), making them unmarketable",
            "Roll and web leaves together, reducing photosynthetic area",
            "Heavy infestations can completely defoliate young plants",
            "Frass (droppings) can promote fungal growth on foliage",
        ],
        "remedies": [
            "Apply Bt (Bacillus thuringiensis) spray — organic and highly effective",
            "Release Trichogramma parasitic wasps to destroy eggs before hatching",
            "Handpick caterpillars and destroy them (effective in small gardens)",
            "Spray neem oil (3 ml/liter) to deter feeding and disrupt growth",
            "Use pheromone traps to monitor and reduce adult moth populations",
            "Encourage birds (wrens, sparrows) with nesting boxes near fields",
        ],
    },
    "Earthworms": {
        "description": "Highly beneficial soil organisms that improve soil structure, drainage, and fertility.",
        "harmful": False,
        "crop_effects": [
            "Aerate soil through burrowing — improving root growth and water penetration",
            "Break down organic matter into nutrient-rich castings (vermicompost)",
            "Improve soil structure and reduce compaction",
            "Their presence is a strong indicator of healthy, fertile soil",
        ],
        "remedies": [],  # Beneficial — no remedies needed
    },
    "Earwig": {
        "description": "Nocturnal insects that feed on soft fruits, seedlings, and plant shoots.",
        "harmful": True,
        "crop_effects": [
            "Chew irregular holes in leaves, flowers, and soft fruits (strawberries, stone fruits)",
            "Damage seedlings by eating cotyledons and young shoots overnight",
            "Feed on corn silk, reducing pollination and kernel development",
            "Can damage ornamental flowers and greenhouse crops",
        ],
        "remedies": [
            "Set up oil-baited traps: shallow containers with vegetable oil and soy sauce",
            "Place rolled-up damp newspaper as traps — collect and discard each morning",
            "Remove garden debris, mulch piles, and hiding spots near crops",
            "Apply food-grade diatomaceous earth around plant bases",
            "Encourage natural predators: toads, ground beetles, and birds",
        ],
    },
    "Grasshopper": {
        "description": "Highly destructive swarming insects that can devastate entire fields of crops.",
        "harmful": True,
        "crop_effects": [
            "Consume entire leaves, stems, and even bark — can eat their body weight daily",
            "Swarms can strip a field bare in hours, causing total crop loss",
            "Damage cereal crops (wheat, rice, millet) during grain-filling stage",
            "Feed on a wide range of crops — vegetables, legumes, fruits, and grasses",
            "Leave behind frass that can contaminate harvested produce",
        ],
        "remedies": [
            "Apply Nosema locustae (biological bait) early in the season for nymphs",
            "Use neem-based sprays to deter feeding and disrupt molting",
            "Encourage natural predators: birds, chickens, and guinea fowl in fields",
            "Till soil in autumn to expose and destroy grasshopper egg pods",
            "Plant trap crops (e.g., tall grasses) at field borders to divert them",
            "Apply kaolin clay spray on valuable crops to create a physical barrier",
        ],
    },
    "Moth": {
        "description": "Adult moths are mostly harmless, but their larvae (caterpillars) cause significant crop damage.",
        "harmful": True,
        "crop_effects": [
            "Larvae bore into fruits, bolls, and pods — major pest of cotton, tomato, and corn",
            "Leaf-mining moth larvae create tunnels inside leaves, destroying tissues",
            "Stored grain moths (Indian meal moth) contaminate cereals and flour",
            "Codling moth larvae cause wormy apples and pears",
            "Diamondback moth is the most destructive pest of cruciferous vegetables worldwide",
        ],
        "remedies": [
            "Install pheromone traps to monitor and mass-trap adult moths",
            "Apply Bt (Bacillus thuringiensis) when larvae are young and small",
            "Use light traps at night to attract and kill adult moths",
            "Release egg parasitoids (Trichogramma spp.) for biological control",
            "Practice crop rotation and remove crop residues after harvest",
            "Store grains in airtight containers with bay leaves as natural repellent",
        ],
    },
    "Slug": {
        "description": "Soft-bodied pests that feed on seedlings and leaves, especially in damp environments.",
        "harmful": True,
        "crop_effects": [
            "Devour seedlings overnight — can kill an entire tray of transplants",
            "Create large irregular holes in leaves, reducing marketability of leafy vegetables",
            "Feed on strawberries, lettuce, and brassicas at ground level",
            "Leave slime trails that can promote fungal diseases on crops",
            "Damage below-ground tubers (potatoes) by creating cavities",
        ],
        "remedies": [
            "Set up beer traps (shallow dishes) to attract and drown slugs",
            "Apply iron phosphate-based slug pellets (safe for pets and wildlife)",
            "Create barriers with crushed eggshells, copper tape, or wood ash",
            "Water in the morning (not evening) to reduce nighttime moisture they need",
            "Encourage natural predators: hedgehogs, frogs, ground beetles, and ducks",
            "Hand-collect slugs at night with a flashlight (most effective method)",
        ],
    },
    "Snail": {
        "description": "Similar to slugs, snails chew large holes in leaves and can decimate young plants.",
        "harmful": True,
        "crop_effects": [
            "Chew large ragged holes in leaves, stems, and flowers",
            "Destroy young seedlings and transplants — especially lettuces and brassicas",
            "Feed on ripening fruits (strawberries, citrus) near the ground",
            "Can transmit plant pathogens through contaminated slime trails",
            "Giant African snails are a quarantine pest that can devastate tropical crops",
        ],
        "remedies": [
            "Hand-collect snails in the early morning or after rain",
            "Apply iron phosphate pellets around susceptible crops",
            "Use copper barriers (tape or mesh) around raised beds",
            "Remove hiding places: boards, stones, and dense ground cover near crops",
            "Introduce predatory snails or encourage ducks and chickens",
            "Spray diluted coffee (1 part coffee to 10 parts water) as a deterrent",
        ],
    },
    "Wasp": {
        "description": "Predatory insects; many species are beneficial pest controllers, but some damage ripe fruits.",
        "harmful": False,
        "crop_effects": [
            "Parasitic wasps are among the most effective biological control agents",
            "Predatory wasps feed on caterpillars, aphids, and other crop pests",
            "Some species (yellowjackets) may damage ripe grapes, figs, and stone fruits",
            "Overall, wasps provide far more benefit than harm to agriculture",
        ],
        "remedies": [],  # Generally beneficial
    },
    "Weevil": {
        "description": "Small beetles with elongated snouts; notorious for destroying stored grains and field crops.",
        "harmful": True,
        "crop_effects": [
            "Rice weevil and grain weevil bore into kernels, destroying stored grain from inside",
            "Boll weevil devastated cotton crops historically — can cause 60-90% yield loss",
            "Vine weevil larvae feed on roots of strawberries, ornamentals, and nursery crops",
            "Adults notch leaves creating characteristic crescent-shaped feeding marks",
            "Post-harvest grain losses can reach 30-40% in untreated storage",
        ],
        "remedies": [
            "Store grains at low moisture (<12%) in airtight containers",
            "Freeze infested grain at -18°C for 72 hours to kill all life stages",
            "Use pheromone traps to detect weevil presence early in storage",
            "Apply food-grade diatomaceous earth to stored grain as a preventive",
            "Introduce parasitic wasps (Anisopteromalus calandrae) in grain stores",
            "Practice field sanitation — remove crop debris and volunteer plants",
        ],
    },
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
            info = PEST_INFO.get(label, {})
            results.append(
                {
                    "label": label,
                    "confidence": round(float(confidence) * 100, 2),
                    "description": info.get("description", ""),
                    "harmful": info.get("harmful", False),
                    "crop_effects": info.get("crop_effects", []),
                    "remedies": info.get("remedies", []),
                }
            )

        return jsonify({"predictions": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, port=5001)
