# ==============================================================================
# FILE NAME: src/inference.py
# ==============================================================================

import pandas as pd
import numpy as np
import joblib
import os

class CreditScoringInferenceEngine:
    def __init__(self, model_path="models/random_forest.pkl", scaler_path="models/scaler.pkl"):
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            raise FileNotFoundError("❌ Model ya Scaler file nahi mili. Pehle training flow run karein.")

        print("⏳ Memory mein Trained Model aur Scaler Load Ho Raha Hai...")
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        print("✅ Models Successfully Loaded & Ready for Real-Time Scoring!\n")

    def _engineer_applicant_features(self, raw_data_df):
        df = raw_data_df.copy()
        epsilon = 1e-6

        df['DTI_Ratio'] = df['Debt'] / (df['Income'] + epsilon)
        df['Savings_to_Income'] = df['Savings'] / (df['Income'] + epsilon)
        df['Expense_to_Income'] = (df['Monthly_Expenses'] * 12) / (df['Income'] + epsilon)
        df['Risk_Flag_Count'] = (df['Previous_Defaults'] > 0).astype(int) + \
                                (df['Credit_Utilization'] > 0.8).astype(int)

        return df

    def predict_credit_risk(self, applicant_dict):
        raw_df = pd.DataFrame([applicant_dict])
        engineered_df = self._engineer_applicant_features(raw_df)

        feature_columns = self.scaler.feature_names_in_
        engineered_df = engineered_df[feature_columns]

        scaled_features_array = self.scaler.transform(engineered_df)

        prediction = self.model.predict(scaled_features_array)[0]
        approval_probability = self.model.predict_proba(scaled_features_array)[0][1]

        risk_score_percentage = (1.0 - approval_probability) * 100
        decision_status = "APPROVED" if prediction == 1 else "REJECTED"

        return {
            "Applicant Name": applicant_dict.get("Name", "Unknown"),
            "Decision": decision_status,
            "Approval Confidence": f"{approval_probability * 100:.2f}%",
            "Default Risk Percentage": f"{risk_score_percentage:.2f}%",
            "Risk Category": "LOW RISK" if risk_score_percentage < 30 else ("MEDIUM RISK" if risk_score_percentage < 60 else "HIGH RISK")
        }