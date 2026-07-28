

import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

def train_and_evaluate(input_path="data/scaled_credit_data.csv", model_dir="models"):
    """
    Data load karta hai, Train-Test split karta hai, Logistic Regression aur Random Forest
    train karke metrics evaluate karta hai aur trained models save karta hai.
    """
    if not os.path.exists(input_path):
        print(f"❌ Error: File {input_path} nahi mili. Pehle preprocessing.py chalayein.")
        return

    print("⏳ Model Training Process Start Ho Raha Hai...\n")
    df = pd.read_csv(input_path)

    # 1. Separate Features (X) & Target (y)
    X = df.drop(columns=['Creditworthy'])
    y = df['Creditworthy']

    # 2. Train-Test Split (80% Train, 20% Test)
    # stratify=y preserves class proportions
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"📊 Dataset Split Complete:")
    print(f"   - Training Samples: {X_train.shape[0]}")
    print(f"   - Testing Samples : {X_test.shape[0]}\n")

    # Models Dictionary
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    }

    os.makedirs(model_dir, exist_ok=True)

    # Loop through models
    for name, model in models.items():
        print(f"==================================================")
        print(f"🚀 Training Model: {name}")
        print(f"==================================================")

        # Model Training (Pattern Fitting)
        model.fit(X_train, y_train)

        # Predictions (Class Predictions: 0 or 1)
        y_pred = model.predict(X_test)
        
        # Probabilities (Probability of being Class 1)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Metric Calculations
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_prob)

        print(f"✅ Performance Metrics for {name}:")
        print(f"   - Accuracy  : {acc * 100:.2f}%")
        print(f"   - Precision : {prec * 100:.2f}%")
        print(f"   - Recall    : {rec * 100:.2f}%")
        print(f"   - F1-Score  : {f1 * 100:.2f}%")
        print(f"   - ROC-AUC   : {roc:.4f}\n")

        # Save Model to Disk (Pickle/Joblib)
        model_filename = os.path.join(model_dir, f"{name.lower().replace(' ', '_')}.pkl")
        joblib.dump(model, model_filename)
        print(f"💾 Model Saved Successfully at: {model_filename}\n")

if __name__ == "__main__":
    train_and_evaluate(input_path="data/scaled_credit_data.csv", model_dir="models")