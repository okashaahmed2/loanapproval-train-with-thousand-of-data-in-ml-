# ==============================================================================
# FILE NAME: src/data_cleaning.py
# PURPOSE  : Missing value imputation, outlier handling, and data saving.
# ==============================================================================

import pandas as pd
import numpy as np
import os

def clean_data(input_path="data/credit_data.csv", output_path="data/clean_credit_data.csv"):
    """
    Missing values fill karta hai aur Outliers ko IQR method se cap karta hai.
    """
    if not os.path.exists(input_path):
        print(f"❌ Error: File {input_path} nahi mili. Pehle dataset_generator.py chalayein.")
        return None

    print("⏳ Data Cleaning Process Start Ho Raha Hai...")
    df = pd.read_csv(input_path)

    # --------------------------------------------------------------------------
    # STEP 1: DUPLICATES REMOVAL
    # --------------------------------------------------------------------------
    initial_rows = len(df)
    df = df.drop_duplicates()
    print(f"✅ Duplicates Removed: {initial_rows - len(df)} rows dropped.")

    # --------------------------------------------------------------------------
    # STEP 2: MISSING VALUES IMPUTATION (MEDIAN METHOD)
    # --------------------------------------------------------------------------
    # Income ke NaN values ko Median se fill karna
    income_median = df['Income'].median()
    df['Income'] = df['Income'].fillna(income_median)

    # Employment Length ke NaN values ko Median se fill karna
    emp_median = df['Employment_Length'].median()
    df['Employment_Length'] = df['Employment_Length'].fillna(emp_median)

    print(f"✅ Missing Values Imputed! Total Nulls Remaining: {df.isnull().sum().sum()}")

    # --------------------------------------------------------------------------
    # STEP 3: OUTLIER HANDLING (IQR WINSORIZATION / CAPPING)
    # --------------------------------------------------------------------------
    columns_to_cap = ['Income', 'Debt', 'Loan_Amount', 'Monthly_Expenses', 'Savings']

    for col in columns_to_cap:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Capping values using np.where
        df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
        df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])

    print("✅ Outliers successfully capped using IQR Winsorization.")

    # --------------------------------------------------------------------------
    # STEP 4: SAVE CLEAN DATASET TO DISK
    # --------------------------------------------------------------------------
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"💾 Cleaned Data saved at: {output_path}")

    return df

if __name__ == "__main__":
    clean_df = clean_data(input_path="data/credit_data.csv", output_path="data/clean_credit_data.csv")
    if clean_df is not None:
        print(f"\nCleaned Data Preview (Rows: {clean_df.shape[0]}, Cols: {clean_df.shape[1]}):")
        print(clean_df.head(3))