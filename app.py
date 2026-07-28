# ==============================================================================
# FILE NAME: app.py
# PURPOSE  : Flask Web Application & REST API for Credit & Risk Scoring UI
# ==============================================================================

from flask import Flask, render_template, request, jsonify
from src.inference import CreditScoringInferenceEngine
import os

app = Flask(__name__)

# Initialize ML Engine
try:
    engine = CreditScoringInferenceEngine(
        model_path="models/random_forest.pkl",
        scaler_path="models/scaler.pkl"
    )
except Exception as e:
    print(f"⚠️ Model load warn: {e}")
    engine = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/score', methods=['POST'])
def score_applicant():
    if not engine:
        return jsonify({"error": "ML Engine not loaded. Train models first!"}), 500

    try:
        data = request.get_json()
        
        # Format payload numeric fields
        applicant_data = {
            "Name": str(data.get("Name", "Applicant")),
            "Age": float(data.get("Age", 30)),
            "Income": float(data.get("Income", 50000)),
            "Debt": float(data.get("Debt", 10000)),
            "Credit_Utilization": float(data.get("Credit_Utilization", 0.3)),
            "Loan_Amount": float(data.get("Loan_Amount", 15000)),
            "Loan_Duration": float(data.get("Loan_Duration", 36)),
            "Previous_Defaults": int(data.get("Previous_Defaults", 0)),
            "Payment_History": float(data.get("Payment_History", 0.9)),
            "Employment_Length": float(data.get("Employment_Length", 5)),
            "Num_Credit_Cards": int(data.get("Num_Credit_Cards", 2)),
            "Monthly_Expenses": float(data.get("Monthly_Expenses", 1500)),
            "Existing_Loans": int(data.get("Existing_Loans", 1)),
            "Credit_History_Length": float(data.get("Credit_History_Length", 5)),
            "Savings": float(data.get("Savings", 10000))
        }

        result = engine.predict_credit_risk(applicant_data)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)