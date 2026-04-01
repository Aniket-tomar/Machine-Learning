# 🐛 PestDetect — AI-Powered Agricultural Pest Identifier

A web application that uses deep learning to identify common agricultural pests from images. Upload a photo and get instant classification results powered by a fine-tuned **MobileNetV2** model.

---

## ✨ Features

- **12-Class Pest Classification** — Identifies Ants, Bees, Beetles, Caterpillars, Earthworms, Earwigs, Grasshoppers, Moths, Slugs, Snails, Wasps, and Weevils
- **Top-5 Predictions** — Returns the five most likely pest classes with confidence percentages
- **Pest Descriptions** — Each prediction includes a brief agricultural impact description
- **Drag & Drop Upload** — Modern, intuitive image upload with preview
- **Dark-Themed UI** — Premium glassmorphism design with smooth animations
- **RESTful API** — JSON-based `/predict` endpoint for easy integration

---

## 🛠 Tech Stack

| Layer       | Technology                   |
|-------------|------------------------------|
| **Model**   | MobileNetV2 (TensorFlow / Keras) |
| **Backend** | Flask (Python)               |
| **Frontend**| HTML, CSS, JavaScript        |
| **Dataset** | Agriculture-Pests (5,494 images, 12 classes) |

---

## 📁 Project Structure

```
PestDetection/
├── app.py                      # Flask application (routes & prediction logic)
├── mobilenetv2_model.keras     # Pre-trained model weights
├── requirements.txt            # Python dependencies
├── Untitled3 (1).ipynb         # Model training notebook (Google Colab)
├── templates/
│   └── index.html              # Main web page
├── static/
│   ├── css/
│   │   └── style.css           # Dark-themed styles
│   └── js/
│       └── app.js              # Frontend logic (drag & drop, API calls)
└── test_images/
    └── beetle/                 # Sample test images
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd PestDetection
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**

   ```bash
   python app.py
   ```

4. **Open in browser**

   Navigate to [http://localhost:5000](http://localhost:5000)

---

## 📡 API Usage

### `POST /predict`

Upload an image file and receive pest classification results.

**Request:**

```bash
curl -X POST -F "file=@pest_image.jpg" http://localhost:5000/predict
```

**Response:**

```json
{
  "predictions": [
    {
      "label": "Beetle",
      "confidence": 87.34,
      "description": "Beetles are among the most destructive pests, feeding on leaves, roots, and stored grain."
    },
    {
      "label": "Weevil",
      "confidence": 5.21,
      "description": "Small beetles with elongated snouts; notorious for destroying stored grains and seeds."
    }
  ]
}
```

---

## 🧠 Model Training

The model was trained in Google Colab using the **Agriculture-Pests** dataset. Three architectures were compared:

| Model         | Validation Accuracy |
|---------------|---------------------|
| Custom CNN    | ~42.4%              |
| ResNet50      | ~12.3%              |
| **MobileNetV2** | **~80.2%**       |

**MobileNetV2** was selected and further optimized through:

- **Transfer Learning** — ImageNet pre-trained weights with frozen base layers
- **Fine-Tuning** — Unfreezing the top 30 layers with a low learning rate (`1e-4`)
- **Data Augmentation** — Random flips, rotations, and zooms
- **Callbacks** — EarlyStopping and ReduceLROnPlateau for efficient training
- **Batch Normalization & Dropout** — Added for regularization

The training notebook is included as `Untitled3 (1).ipynb`.

---

## 🐞 Supported Pest Classes

| #  | Pest         | Description |
|----|-------------|-------------|
| 1  | Ants        | Small social insects that damage crops by farming aphids |
| 2  | Bees        | Important pollinators, but can be pests in certain contexts |
| 3  | Beetle      | Feed on leaves, roots, and stored grain |
| 4  | Caterpillar | Larval stage of butterflies/moths; heavy leaf damage |
| 5  | Earthworms  | Generally beneficial but may indicate over-moist conditions |
| 6  | Earwig      | Nocturnal feeders on soft fruits and seedlings |
| 7  | Grasshopper | Can consume their body weight in food daily |
| 8  | Moth        | Adult moths are mostly harmless; larvae damage crops |
| 9  | Slug        | Feed on seedlings and leaves in damp environments |
| 10 | Snail       | Chew large holes in leaves; decimate young plants |
| 11 | Wasp        | Some species help control pests; others damage fruits |
| 12 | Weevil      | Notorious for destroying stored grains and seeds |

---

## 📄 License

This project is for educational purposes.
