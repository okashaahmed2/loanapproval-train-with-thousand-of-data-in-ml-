
import pandas as pd
import numpy as np
import os

def engineer_features(input_path="data/clean_credit_data.csv", output_path="data/engineered_credit_data.csv"):
    """
    Clean dataset se financial indicators calculate karke naye columns add karta hai.
    """
    if not os.path.exists(input_path):
        print(f"❌ Error: File {input_path} nahi mili. Pehle data_cleaning.py chalayein.")
        return None

    print("⏳ Feature Engineering Process Start Ho Raha Hai...")
    df = pd.read_csv(input_path)

    # Small constant to prevent Division-by-Zero
    epsilon = 1e-6

    # 1. Debt-to-Income Ratio (DTI)
    df['DTI_Ratio'] = df['Debt'] / (df['Income'] + epsilon)

    # 2. Savings-to-Income Ratio
    df['Savings_to_Income'] = df['Savings'] / (df['Income'] + epsilon)

    # 3. Annual Expenses to Income Ratio
    df['Expense_to_Income'] = (df['Monthly_Expenses'] * 12) / (df['Income'] + epsilon)

    # 4. Total Risk Score Proxy (Combining past flags)
    df['Risk_Flag_Count'] = (df['Previous_Defaults'] > 0).astype(int) + \
                            (df['Credit_Utilization'] > 0.8).astype(int)

    print(f"✅ Created 4 New Engineered Features successfully!")
    print(f"📊 New Dataset Shape: {df.shape} (Added new columns)")

    # Save to disk
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"💾 Feature Engineered Dataset saved at: {output_path}")

    return df

if __name__ == "__main__":
    eng_df = engineer_features(input_path="data/clean_credit_data.csv", output_path="data/engineered_credit_data.csv")
    if eng_df is not None:
        print("\nNew Features Sample Preview:")
        print(eng_df[['DTI_Ratio', 'Savings_to_Income', 'Expense_to_Income', 'Risk_Flag_Count']].head(3))