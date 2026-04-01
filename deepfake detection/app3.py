import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import time

# ==========================================
# 1. Page Configuration
# ==========================================
st.set_page_config(
    page_title="Deepfake Detector",
    page_icon="🛡️",
    layout="centered"
)

# ==========================================
# 2. Custom CSS for Premium Dark UI
# ==========================================
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #0a0a1a 0%, #0d1117 50%, #0a0a1a 100%);
        font-family: 'Inter', sans-serif;
    }

    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        max-width: 800px;
    }

    /* Header area */
    .hero-header {
        text-align: center;
        padding: 2rem 0 1rem 0;
    }

    .hero-header h1 {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00d2ff 0%, #7b2ff7 50%, #ff6b9d 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }

    .hero-subtitle {
        color: #8b949e;
        font-size: 1.05rem;
        font-weight: 300;
        line-height: 1.6;
        max-width: 600px;
        margin: 0 auto;
    }

    /* Upload card */
    .upload-card {
        background: linear-gradient(145deg, rgba(22,27,34,0.95), rgba(13,17,23,0.95));
        border: 1px solid rgba(48,54,61,0.6);
        border-radius: 16px;
        padding: 2.5rem 2rem;
        margin: 1.5rem 0;
        text-align: center;
        transition: all 0.3s ease;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }

    .upload-card:hover {
        border-color: rgba(123,47,247,0.4);
        box-shadow: 0 8px 32px rgba(123,47,247,0.15);
    }

    .upload-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        display: block;
    }

    .upload-title {
        color: #e6edf3;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }

    .upload-desc {
        color: #8b949e;
        font-size: 0.9rem;
    }

    /* Result cards */
    .result-card {
        border-radius: 16px;
        padding: 2rem;
        margin: 1.5rem 0;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        animation: fadeInUp 0.6s ease-out;
    }

    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .result-real {
        background: linear-gradient(145deg, rgba(16,185,129,0.15), rgba(6,78,59,0.2));
        border: 1px solid rgba(16,185,129,0.4);
    }

    .result-fake {
        background: linear-gradient(145deg, rgba(239,68,68,0.15), rgba(127,29,29,0.2));
        border: 1px solid rgba(239,68,68,0.4);
    }

    .result-verdict {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0.5rem 0;
        letter-spacing: -0.5px;
    }

    .result-real .result-verdict {
        color: #10b981;
    }

    .result-fake .result-verdict {
        color: #ef4444;
    }

    .result-emoji {
        font-size: 4rem;
        display: block;
        margin-bottom: 0.5rem;
    }

    .confidence-bar-container {
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
        height: 12px;
        margin: 1rem auto;
        max-width: 400px;
        overflow: hidden;
    }

    .confidence-bar {
        height: 100%;
        border-radius: 12px;
        transition: width 1.5s cubic-bezier(0.25, 0.46, 0.45, 0.94);
    }

    .confidence-bar-real {
        background: linear-gradient(90deg, #059669, #10b981, #34d399);
    }

    .confidence-bar-fake {
        background: linear-gradient(90deg, #b91c1c, #ef4444, #f87171);
    }

    .confidence-text {
        color: #e6edf3;
        font-size: 1.4rem;
        font-weight: 700;
        margin-top: 0.5rem;
    }

    .confidence-label {
        color: #8b949e;
        font-size: 0.85rem;
        font-weight: 400;
        text-transform: uppercase;
        letter-spacing: 2px;
    }

    /* Stats row */
    .stats-row {
        display: flex;
        gap: 1rem;
        margin: 1.5rem 0;
    }

    .stat-card {
        flex: 1;
        background: rgba(22,27,34,0.8);
        border: 1px solid rgba(48,54,61,0.6);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }

    .stat-value {
        color: #e6edf3;
        font-size: 1.5rem;
        font-weight: 700;
    }

    .stat-label {
        color: #8b949e;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-top: 0.3rem;
    }

    /* Frame gallery */
    .frame-section {
        background: rgba(22,27,34,0.6);
        border: 1px solid rgba(48,54,61,0.4);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1.5rem 0;
    }

    .frame-section-title {
        color: #e6edf3;
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }

    /* Processing status */
    .processing-step {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.6rem 0;
        color: #8b949e;
        font-size: 0.9rem;
    }

    .processing-step.active {
        color: #e6edf3;
    }

    .processing-step.done {
        color: #10b981;
    }

    .step-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #30363d;
    }

    .step-dot.active {
        background: #7b2ff7;
        box-shadow: 0 0 8px rgba(123,47,247,0.5);
        animation: pulse 1.5s infinite;
    }

    .step-dot.done {
        background: #10b981;
    }

    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.3); opacity: 0.7; }
    }

    /* Footer */
    .footer-note {
        text-align: center;
        color: #484f58;
        font-size: 0.75rem;
        margin-top: 2rem;
        padding: 1rem 0;
        border-top: 1px solid rgba(48,54,61,0.3);
    }

    /* Override Streamlit defaults */
    .stFileUploader label {
        color: #e6edf3 !important;
    }

    .stFileUploader > div {
        background: rgba(22,27,34,0.6) !important;
        border: 2px dashed rgba(123,47,247,0.3) !important;
        border-radius: 12px !important;
    }

    .stFileUploader > div:hover {
        border-color: rgba(123,47,247,0.6) !important;
    }

    /* Progress bar overrides */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #7b2ff7, #00d2ff) !important;
    }

    /* Streamlit button */
    .stButton > button {
        background: linear-gradient(135deg, #7b2ff7, #5b1fd7) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 16px rgba(123,47,247,0.3) !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 24px rgba(123,47,247,0.5) !important;
    }

    /* Spinner override */
    .stSpinner > div {
        border-top-color: #7b2ff7 !important;
    }

    div[data-testid="stExpander"] {
        background: rgba(22,27,34,0.6);
        border: 1px solid rgba(48,54,61,0.4);
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. Load Model (Cached)
# ==========================================
@st.cache_resource
def load_deepfake_model():
    """Load the deepfake detection model only once."""
    try:
        from tensorflow.keras.models import load_model
        model = load_model("deepfake_model.h5")
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {str(e)}")
        st.stop()

@st.cache_resource
def load_face_detector():
    """Load OpenCV face detector."""
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    return face_cascade

# ==========================================
# 4. Helper Functions
# ==========================================
def extract_frames(video_path, max_frames=20):
    """Extract evenly-spaced frames from a video."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0

    if total_frames <= 0:
        cap.release()
        return [], 0, 0, 0

    # Calculate which frames to sample
    step = max(1, total_frames // max_frames)
    frame_indices = list(range(0, total_frames, step))[:max_frames]

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    return frames, total_frames, fps, duration

def detect_and_crop_faces(frame, face_cascade, target_size=(128, 128)):
    """Detect faces in a frame and return cropped, resized face images."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    face_crops = []
    face_locations = []
    for (x, y, w, h) in faces:
        # Add padding around the face
        pad_w = int(w * 0.2)
        pad_h = int(h * 0.2)
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(frame.shape[1], x + w + pad_w)
        y2 = min(frame.shape[0], y + h + pad_h)

        face_crop = frame[y1:y2, x1:x2]
        face_resized = cv2.resize(face_crop, target_size)
        face_crops.append(face_resized)
        face_locations.append((x, y, w, h))

    return face_crops, face_locations

def preprocess_face(face_img):
    """Preprocess a face image for the model."""
    # Convert BGR to RGB
    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    # Normalize to [0, 1]
    face_normalized = face_rgb.astype(np.float32) / 255.0
    # Add batch dimension
    return np.expand_dims(face_normalized, axis=0)

def analyze_video(video_path, model, face_cascade, progress_callback=None):
    """Full pipeline: extract frames → detect faces → predict."""
    results = {
        "total_frames_analyzed": 0,
        "faces_detected": 0,
        "predictions": [],
        "frame_results": [],
        "annotated_frames": []
    }

    # Step 1: Extract frames
    if progress_callback:
        progress_callback("Extracting video frames...", 0.1)

    frames, total_frames, fps, duration = extract_frames(video_path, max_frames=20)

    if not frames:
        return None, "Could not extract frames from the video."

    results["total_frames"] = total_frames
    results["fps"] = fps
    results["duration"] = duration
    results["sampled_frames"] = len(frames)

    # Step 2: Detect faces and predict
    if progress_callback:
        progress_callback("Detecting faces and analyzing...", 0.3)

    all_predictions = []

    for i, frame in enumerate(frames):
        if progress_callback:
            prog = 0.3 + (0.6 * (i / len(frames)))
            progress_callback(f"Analyzing frame {i+1}/{len(frames)}...", prog)

        face_crops, face_locations = detect_and_crop_faces(frame, face_cascade)

        frame_annotated = frame.copy()

        if face_crops:
            for face_img, (x, y, w, h) in zip(face_crops, face_locations):
                preprocessed = preprocess_face(face_img)
                prediction = model.predict(preprocessed, verbose=0)[0][0]
                all_predictions.append(prediction)
                results["faces_detected"] += 1

                # Annotate frame
                is_fake = prediction > 0.5
                color = (0, 0, 255) if is_fake else (0, 255, 0)
                label = f"{'FAKE' if is_fake else 'REAL'} {prediction*100:.1f}%"
                cv2.rectangle(frame_annotated, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame_annotated, label, (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        results["annotated_frames"].append(cv2.cvtColor(frame_annotated, cv2.COLOR_BGR2RGB))
        results["total_frames_analyzed"] += 1

    if progress_callback:
        progress_callback("Compiling results...", 0.95)

    if not all_predictions:
        return None, "No faces detected in the video. Please upload a video with visible faces."

    results["predictions"] = all_predictions
    results["mean_prediction"] = np.mean(all_predictions)
    results["is_fake"] = results["mean_prediction"] > 0.5
    results["confidence"] = abs(results["mean_prediction"] - 0.5) * 2  # scale to 0-1

    return results, None

# ==========================================
# 5. UI Layout
# ==========================================

# Hero header
st.markdown("""
<div class="hero-header">
    <h1>🛡️ Deepfake Detector</h1>
    <p class="hero-subtitle">
        Powered by deep learning to detect manipulated videos. 
        Upload a video and our AI will analyze facial patterns to determine authenticity.
    </p>
</div>
""", unsafe_allow_html=True)

# Upload section
st.markdown("""
<div class="upload-card">
    <span class="upload-icon">🎬</span>
    <div class="upload-title">Upload Your Video</div>
    <div class="upload-desc">Supported: MP4, AVI, MOV, MKV • Max size: 200MB</div>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose a video file",
    type=["mp4", "avi", "mov", "mkv"],
    label_visibility="collapsed"
)

if uploaded_file is not None:
    # Show video preview
    st.video(uploaded_file)

    # Analyze button
    if st.button("🔍 Analyze Video", type="primary", use_container_width=True):
        # Load model and face detector
        model = load_deepfake_model()
        face_cascade = load_face_detector()

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()

        def update_progress(msg, val):
            status_text.markdown(f"⏳ **{msg}**")
            progress_bar.progress(val)

        try:
            results, error = analyze_video(tmp_path, model, face_cascade, update_progress)
        finally:
            # Clean up temp file
            os.unlink(tmp_path)

        progress_bar.progress(1.0)
        status_text.empty()
        progress_bar.empty()

        if error:
            st.error(f"⚠️ {error}")
        else:
            # ==========================================
            # 6. Display Results
            # ==========================================
            is_fake = results["is_fake"]
            mean_pred = results["mean_prediction"]
            confidence = results["confidence"] * 100

            # Determine display values  
            if is_fake:
                verdict = "FAKE"
                emoji = "🚨"
                css_class = "result-fake"
                bar_class = "confidence-bar-fake"
                verdict_detail = "This video appears to be digitally manipulated."
            else:
                verdict = "REAL"
                emoji = "✅"
                css_class = "result-real"
                bar_class = "confidence-bar-real"
                verdict_detail = "This video appears to be authentic."

            # Main result card
            st.markdown(f"""
            <div class="result-card {css_class}">
                <span class="result-emoji">{emoji}</span>
                <div class="result-verdict">{verdict}</div>
                <div class="confidence-label">Model Confidence</div>
                <div class="confidence-bar-container">
                    <div class="confidence-bar {bar_class}" style="width: {confidence:.1f}%;"></div>
                </div>
                <div class="confidence-text">{confidence:.1f}%</div>
                <p style="color: #8b949e; margin-top: 0.75rem; font-size: 0.9rem;">{verdict_detail}</p>
            </div>
            """, unsafe_allow_html=True)

            if is_fake:
                st.error(f"### 🚨 Deepfake Detected — {confidence:.1f}% Confidence")
            else:
                st.success(f"### ✅ Authentic Video — {confidence:.1f}% Confidence")
                st.balloons()

            # Stats row
            st.markdown(f"""
            <div class="stats-row">
                <div class="stat-card">
                    <div class="stat-value">{results['sampled_frames']}</div>
                    <div class="stat-label">Frames Analyzed</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{results['faces_detected']}</div>
                    <div class="stat-label">Faces Detected</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{results['duration']:.1f}s</div>
                    <div class="stat-label">Duration</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{results['fps']:.0f}</div>
                    <div class="stat-label">FPS</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Detailed analysis expander
            with st.expander("📊 Detailed Frame-by-Frame Analysis"):
                if results["predictions"]:
                    st.markdown("**Per-Face Prediction Scores:**")
                    st.markdown("*(Score > 0.5 = Fake, Score ≤ 0.5 = Real)*")

                    # Create a simple chart of predictions
                    import pandas as pd
                    pred_df = pd.DataFrame({
                        "Face #": [f"Face {i+1}" for i in range(len(results["predictions"]))],
                        "Fake Score": results["predictions"]
                    })
                    st.bar_chart(pred_df.set_index("Face #"))

                    # Stats
                    preds = np.array(results["predictions"])
                    fake_count = np.sum(preds > 0.5)
                    real_count = np.sum(preds <= 0.5)
                    st.markdown(f"""
                    | Metric | Value |
                    |--------|-------|
                    | Total faces analyzed | **{len(preds)}** |
                    | Classified as Fake | **{fake_count}** ({fake_count/len(preds)*100:.1f}%) |
                    | Classified as Real | **{real_count}** ({real_count/len(preds)*100:.1f}%) |
                    | Mean Fake Score | **{np.mean(preds):.4f}** |
                    | Min Score | **{np.min(preds):.4f}** |
                    | Max Score | **{np.max(preds):.4f}** |
                    """)

            # Annotated frames gallery
            with st.expander("🖼️ Analyzed Frames Gallery"):
                st.markdown("*Frames with detected faces highlighted (Green = Real, Red = Fake)*")
                cols_per_row = 3
                annotated = results["annotated_frames"]
                for row_start in range(0, len(annotated), cols_per_row):
                    cols = st.columns(cols_per_row)
                    for j, col in enumerate(cols):
                        idx = row_start + j
                        if idx < len(annotated):
                            col.image(annotated[idx], caption=f"Frame {idx+1}", use_container_width=True)

# Footer
st.markdown("""
<div class="footer-note">
    🛡️ Deepfake Detector • Powered by Deep Learning<br>
    Note: Results are probabilistic and should not be used as sole evidence.
</div>
""", unsafe_allow_html=True)
