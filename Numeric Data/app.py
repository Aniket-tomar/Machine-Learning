import pandas as pd
import numpy as np
import re
from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

app = Flask(__name__)

# ── Load and train the model on startup ──────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv("salary_data_cleaned.csv")

# Keep only relevant columns
df_clean = df[['Job Title', 'Job Description', 'Company Name', 'avg_salary']].copy()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

df_clean['text'] = (
    df_clean['Job Title'] + " " +
    df_clean['Company Name'] + " " +
    df_clean['Job Description']
)
df_clean['text'] = df_clean['text'].apply(clean_text)

X = df_clean['text']
y = df_clean['avg_salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tfidf = TfidfVectorizer(max_features=15000, ngram_range=(1, 2), sublinear_tf=True)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

model = LinearRegression()
model.fit(X_train_tfidf, y_train)

preds = model.predict(X_test_tfidf)
r2 = r2_score(y_test, preds)
mae = mean_absolute_error(y_test, preds)
print(f"Model ready — R²: {r2:.2f}, MAE: ${mae:.2f}K")

# Gather unique values for dropdown options
job_titles = sorted(df['Job Title'].dropna().unique().tolist())
company_names = sorted(df['Company Name'].dropna().unique().tolist())

# ── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html",
                           job_titles=job_titles,
                           company_names=company_names,
                           r2=round(r2, 4),
                           mae=round(mae, 2))

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    job_title = data.get("job_title", "")
    company_name = data.get("company_name", "")
    job_description = data.get("job_description", "")

    combined = f"{job_title} {company_name} {job_description}"
    cleaned = clean_text(combined)
    vectorized = tfidf.transform([cleaned])
    prediction = model.predict(vectorized)[0]

    return jsonify({
        "salary": round(float(prediction), 2),
        "salary_low": round(float(prediction - mae), 2),
        "salary_high": round(float(prediction + mae), 2),
    })

if __name__ == "__main__":
    app.run(debug=True, port=5002)
