# ==============================================================================
# FILE NAME: src/preprocessing.py
# ==============================================================================

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler

def preprocess_data(input_path="data/engineered_credit_data.csv", 
                    output_path="data/scaled_credit_data.csv",
                    scaler_path="models/scaler.pkl"):
    """
    Features ko scale karta hai aur fitted scaler ko disk par save karta hai.
    """
    if not os.path.exists(input_path):
        print(f"❌ Error: File {input_path} nahi mili. Pehle feature_engineering.py chalayein.")
        return None

    print("⏳ Data Preprocessing & Feature Scaling Start Ho Raha Hai...")
    df = pd.read_csv(input_path)

    X = df.drop(columns=['Creditworthy'])
    y = df['Creditworthy']

    scaler = StandardScaler()
    X_scaled_array = scaler.fit_transform(X)

    # Save fitted scaler for inference
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)
    print(f"💾 Fitted Scaler successfully saved at: {scaler_path}")

    X_scaled_df = pd.DataFrame(X_scaled_array, columns=X.columns)
    processed_df = pd.concat([X_scaled_df, y.reset_index(drop=True)], axis=1)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    processed_df.to_csv(output_path, index=False)
    print(f"💾 Scaled Dataset saved at: {output_path}")

    return processed_df

if __name__ == "__main__":
    preprocess_data()