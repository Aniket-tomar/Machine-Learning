import numpy as np
import librosa
import joblib
import tensorflow as tf
import streamlit as st

# Page config
st.set_page_config(page_title="Language Detector", page_icon="🎧", layout="centered")

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
    }
    .result-box {
        padding: 20px;
        border-radius: 15px;
        background-color: #1E1E1E;
        text-align: center;
        font-size: 20px;
        font-weight: bold;
        color: #00FFAA;
    }
    .greeting-box {
        padding: 20px;
        border-radius: 15px;
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #00FFAA33;
        text-align: center;
        margin-top: 15px;
    }
    .greeting-text {
        font-size: 26px;
        font-weight: bold;
        color: #FFD700;
        margin-bottom: 6px;
    }
    .greeting-translation {
        font-size: 14px;
        color: #94a3b8;
        font-style: italic;
    }
    </style>
""", unsafe_allow_html=True)

# Load model
model = tf.keras.models.load_model("language_model.keras")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("label_encoder.pkl")

# "Hello, how are you doing?" in each supported Indian language
GREETINGS = {
    "Hindi":      "नमस्ते, आप कैसे हैं?",
    "Bengali":    "হ্যালো, আপনি কেমন আছেন?",
    "Telugu":     "హలో, మీరు ఎలా ఉన్నారు?",
    "Marathi":    "नमस्कार, तुम्ही कसे आहात?",
    "Tamil":      "வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்?",
    "Gujarati":   "હેલો, તમે કેમ છો?",
    "Kannada":    "ಹಲೋ, ನೀವು ಹೇಗಿದ್ದೀರಿ?",
    "Malayalam":  "ഹലോ, നിങ്ങൾ എങ്ങനെയിരിക്കുന്നു?",
    "Punjabi":    "ਸਤ ਸ੍ਰੀ ਅਕਾਲ, ਤੁਸੀਂ ਕਿਵੇਂ ਹੋ?",
    "Odia":       "ନମସ୍କାର, ଆପଣ କେମିତି ଅଛନ୍ତି?",
    "Urdu":       "ہیلو، آپ کیسے ہیں؟",
    "Assamese":   "নমস্কাৰ, আপুনি কেনে আছে?",
    "English":    "Hello, how are you doing?",
    "Sanskrit":   "नमस्ते, भवान् कथम् अस्ति?",
    "Nepali":     "नमस्ते, तपाईंलाई कस्तो छ?",
    "Sindhi":     "هيلو، توهان ڪيئن آهيو؟",
    "Konkani":    "नमस्कार, तूं कसो आसा?",
    "Manipuri":   "ꯍꯦꯜꯂꯣ, ꯅꯍꯥꯛ ꯀꯝꯗꯧꯔꯤ?",
    "Bodo":       "हेलो, नों मानो बे?",
    "Dogri":      "नमस्ते, तुस्सी किव्वें हो?",
    "Maithili":   "प्रणाम, अहाँ कोना छी?",
    "Santali":    "ᱡᱚᱦᱟᱨ, ᱟᱢ ᱚᱠᱟ ᱢᱮᱱᱟ?",
}

# Feature extraction
def extract_features(file):
    audio, sr = librosa.load(file, duration=3)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)

# Title
st.markdown("<h1 style='text-align: center;'>🎧 Audio Language Detection</h1>", unsafe_allow_html=True)
st.write("Upload an audio file and detect its language instantly 🚀")

# File uploader
uploaded_file = st.file_uploader("📂 Upload Audio", type=["mp3", "wav"])

if uploaded_file:
    st.audio(uploaded_file, format="audio/wav")

    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())

    if st.button("🔍 Detect Language"):
        with st.spinner("Analyzing audio... ⏳"):
            features = extract_features("temp.wav")
            features = features.reshape(1, -1)
            features = scaler.transform(features)

            prediction = model.predict(features)
            predicted_label = np.argmax(prediction)
            confidence = np.max(prediction)

            language = encoder.inverse_transform([predicted_label])[0]

        # Result box
        st.markdown(f"""
            <div class="result-box">
                🎯 Predicted Language: {language}
            </div>
        """, unsafe_allow_html=True)

        # Greeting in the detected language
        greeting = GREETINGS.get(language, GREETINGS.get("English"))
        st.markdown(f"""
            <div class="greeting-box">
                <div class="greeting-text">💬 {greeting}</div>
                <div class="greeting-translation">"Hello, how are you doing?" in {language}</div>
            </div>
        """, unsafe_allow_html=True)

        # Confidence bar
        st.write("Confidence Score:")
        st.progress(float(confidence))

        # Show raw probabilities (optional)
        with st.expander("📊 See detailed probabilities"):
            probs = {
                encoder.inverse_transform([i])[0]: float(prediction[0][i])
                for i in range(len(prediction[0]))
            }
            st.json(probs)