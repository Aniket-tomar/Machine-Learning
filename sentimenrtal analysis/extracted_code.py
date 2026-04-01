# ============================================================
# IMDB Sentiment Analysis Model Training Pipeline
# ============================================================
!pip install -q joblib

import pandas as pd
import numpy as np
import re
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# 1. Load the dataset
# Ensure you have uploaded 'IMDB Dataset.csv' to your Colab session
file_path = "/kaggle/input/datasets/nujyadav/imdb-dataset/IMDB Dataset.csv"

if not os.path.exists(file_path):
    print(f"❌ File not found! Please upload '{file_path}' using the folder icon on the left.")
else:
    print(f"✅ Reading dataset: {file_path}")
    df = pd.read_csv(file_path)
    
    # 2. Basic Dataset Stats (as performed in the reference code)
    print("\n=== Shape ===")
    print(df.shape)
    print("\n=== Info ===")
    df.info()
    print("\n=== Missing Values ===")
    print(df.isnull().sum())
    print("\n=== Class Distribution ===")
    print(df['sentiment'].value_counts())
    
    # 3. Data Cleaning & Feature Engineering
    print("\n✅ Performing data cleaning and feature engineering...")
    
    def clean_text_basic(t: str) -> str:
        t = str(t).lower()
        t = re.sub(r"<br\s*/?>", " ", t)       # Remove IMDB HTML breaks
        t = re.sub(r"http\S+|www\.\S+", " ", t) # Remove URLs
        t = re.sub(r"[^a-z0-9\s']", " ", t)     # Remove special characters
        t = re.sub(r"\s+", " ", t).strip()      # Remove extra whitespaces
        return t

    # Clean the reviews
    df['clean_text'] = df['review'].apply(clean_text_basic)
    
    # Extract structural features (as per reference code)
    df["char_len"] = df["clean_text"].apply(len).astype(float)
    df["word_count"] = df["clean_text"].apply(lambda x: len(x.split())).astype(float)
    df["avg_word_len"] = df["clean_text"].apply(lambda x: np.mean([len(w) for w in x.split()]) if len(x.split()) else 0).astype(float)
    
    # Encode sentiment into 1 (Positive) and 0 (Negative)
    df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    
    # Drop empty rows just in case
    df = df.dropna(subset=['clean_text', 'label'])
    
    # 4. Train-Test Split (80% Train, 20% Test)
    print("\n✅ Splitting dataset into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
    )
    
    # 5. Preprocessing: TF-IDF Vectorization
    print("✅ Vectorizing text using TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # 6. Initialize Models to Compare
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Linear SVC": LinearSVC(max_iter=1000, dual=False),
        "Multinomial Naive Bayes": MultinomialNB()
    }
    
    # 7. Train and Evaluate
    print("\n✅ Training and evaluating models (this may take a minute)...")
    results = []
    trained_models = {}
    
    for name, model in models.items():
        print(f" -> Training {name}...")
        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)
        acc = accuracy_score(y_test, y_pred)
        
        results.append({"Model": name, "Accuracy": acc})
        trained_models[name] = model
        print(f"    {name} Accuracy: {acc:.4f}")
    
    # 8. Compare Performances
    results_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False).reset_index(drop=True)
    
    print("\n=== FINAL RESULTS (Sorted) ===")
    print(results_df)
    
    # Visual Comparison Plot
    plt.figure(figsize=(8, 5))
    bars = plt.bar(results_df['Model'], results_df['Accuracy'], color=['#4C72B0', '#55A868', '#C44E52'])
    plt.ylim(0.8, 1.0)
    plt.title("Algorithm Performance Comparison (IMDB Sentiment Analysis)")
    plt.ylabel("Accuracy Score")
    
    # Add accuracy labels on top of the bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.002, f"{yval:.4f}", va='bottom', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # 9. Dump the best model dynamically
    best_model_name = results_df.iloc[0]['Model']
    best_model = trained_models[best_model_name]
    best_acc = results_df.iloc[0]['Accuracy']
    
    print(f"\n🏆 Best Performing Model: {best_model_name} (Accuracy: {best_acc:.4f})")
    
    model_filename = "best_imdb_model.pkl"
    vectorizer_filename = "tfidf_vectorizer.pkl"
    
    # Save Model & Vectorizer
    joblib.dump(best_model, model_filename)
    joblib.dump(vectorizer, vectorizer_filename)
    
    print(f"✅ Saved the best model as: '{model_filename}'")
    print(f"✅ Saved the TF-IDF vectorizer as: '{vectorizer_filename}'")
    print("💡 You can download these files from the 'Files' tab (folder icon) on the left panel in Colab.")
# ============================================================
# Optimize the Best IMDB Sentiment Analysis Model (Kaggle Version)
# ============================================================
import pandas as pd
import numpy as np
import re
import os
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

# 1. Locate the dataset in Kaggle
file_path = None
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        if filename == 'IMDB Dataset.csv':
            file_path = os.path.join(dirname, filename)
            break

# Fallback in case it's in the working directory
if not file_path and os.path.exists("IMDB Dataset.csv"):
    file_path = "/kaggle/input/datasets/nujyadav/imdb-dataset/IMDB Dataset.csv"

# 2. Define Kaggle output paths where the previous models were saved
model_filename = "/kaggle/working/best_imdb_model.pkl"
vectorizer_filename = "/kaggle/working/tfidf_vectorizer.pkl"

if not file_path or not os.path.exists(model_filename):
    print("❌ Missing required files! Ensure the dataset is added and you've run the training script first.")
else:
    print(f"✅ Reading dataset from: {file_path}")
    df = pd.read_csv(file_path)
    
    # 3. Re-apply Data Cleaning
    def clean_text_basic(t: str) -> str:
        t = str(t).lower()
        t = re.sub(r"<br\s*/?>", " ", t)       
        t = re.sub(r"http\S+|www\.\S+", " ", t) 
        t = re.sub(r"[^a-z0-9\s']", " ", t)     
        t = re.sub(r"\s+", " ", t).strip()      
        return t

    print("✅ Cleaning data and preparing splits...")
    df['clean_text'] = df['review'].apply(clean_text_basic)
    df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    df = df.dropna(subset=['clean_text', 'label'])
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
    )
    
    # Load and apply the saved vectorizer
    print("✅ Loading TF-IDF Vectorizer...")
    vectorizer = joblib.load(vectorizer_filename)
    X_train_tfidf = vectorizer.transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # 4. Load the saved best model
    best_model = joblib.load(model_filename)
    model_type = type(best_model).__name__
    print(f"\n✅ Loaded Best Model Type: {model_type}")
    
    # 5. Define Hyperparameter Grids based on model type
    param_grid = {}
    if model_type == "LogisticRegression":
        param_grid = {
            'C': [0.1, 1.0, 10.0],           # Inverse of regularization strength
            'penalty': ['l2'],               # Regularization type
            'solver': ['lbfgs', 'liblinear'] # Algorithm to use
        }
    elif model_type == "LinearSVC":
        param_grid = {
            'C': [0.01, 0.1, 1.0, 10.0],
            'loss': ['hinge', 'squared_hinge'],
            'max_iter': [1000, 2000]
        }
    elif model_type == "MultinomialNB":
        param_grid = {
            'alpha': [0.1, 0.5, 1.0, 2.0]    # Additive smoothing parameter
        }
    else:
        print("⚠️ Model type not recognized for specific grid search. Using empty grid.")
        param_grid = {}

    # 6. Perform Grid Search for Optimization
    if param_grid:
        print(f"\n⚙️ Starting Hyperparameter Tuning for {model_type}...")
        print("   (This may take a few minutes as it trains multiple variations to find the absolute best...)")
        
        # cv=3 means 3-fold cross-validation. n_jobs=-1 uses all available CPU cores in Kaggle.
        grid_search = GridSearchCV(best_model, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
        grid_search.fit(X_train_tfidf, y_train)
        
        optimized_model = grid_search.best_estimator_
        
        print("\n🏆 Optimization Complete!")
        print(f"   Best Parameters Found: {grid_search.best_params_}")
        
        # Evaluate Optimized Model
        y_pred = optimized_model.predict(X_test_tfidf)
        opt_acc = accuracy_score(y_test, y_pred)
        print(f"   Optimized Accuracy on Test Set: {opt_acc:.4f}")
        
        # 7. Save the optimized model to Kaggle's working directory
        optimized_filename = "/kaggle/working/optimized_imdb_model.pkl"
        joblib.dump(optimized_model, optimized_filename)
        print(f"\n✅ Saved the highly optimized model as: '{optimized_filename}'")
        print("💡 You can find this file in the 'Output' section on the right sidebar in Kaggle to download.")
        
    else:
        print("\nℹ️ No optimization grid defined for this model type. Kept original model.")
