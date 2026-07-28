

import pandas as pd
import numpy as np
import os

def generate_and_save_credit_data(file_path="../data/credit_data.csv", n_samples=10000):
    """
    10,000 customers ka synthetic banking data generate karke CSV file mein save karta hai.
    """
    # Step 1: Random Seed set karna taake har baar same numbers generate hon
    np.random.seed(42)

    print("⏳ Data Generation Process Start Ho Raha Hai...")

    # Step 2: Individual Financial Features Generate karna
    # --------------------------------------------------
    # Customer ki umar (18 se 70 saal ke beech)
    age = np.random.randint(18, 70, size=n_samples)
    
    # Salana Income ($12,000 se $200,000 ke beech, Normal Bell Curve distribution)
    income = np.random.normal(loc=55000, scale=20000, size=n_samples).clip(12000, 200000)
    
    # Existing Debt/Karza
    debt = np.random.normal(loc=15000, scale=8000, size=n_samples).clip(0, 80000)
    
    # Credit Limit Utilization Ratio (0.01 = 1% used, 0.99 = 99% used)
    credit_utilization = np.random.uniform(0.01, 0.99, size=n_samples)
    
    # Loan ki demand amount
    loan_amount = np.random.normal(loc=20000, scale=10000, size=n_samples).clip(2000, 100000)
    
    # Loan wapas karne ki muddat (Months mein)
    loan_duration = np.random.choice([12, 24, 36, 48, 60], size=n_samples)
    
    # Pehle kitni baar loan default kiya hai (0, 1, 2, 3...)
    previous_defaults = np.random.poisson(lam=0.3, size=n_samples).clip(0, 5)
    
    # On-time payment history percentage (60% se 100%)
    payment_history = np.random.uniform(60.0, 100.0, size=n_samples)
    
    # Job/Business experience (Saal mein)
    employment_length = np.random.exponential(scale=5, size=n_samples).clip(0, 40)
    
    # Kitne Active Credit Cards hain
    num_credit_cards = np.random.randint(1, 10, size=n_samples)
    
    # Har mahine ka kharcha (Monthly Expenses)
    monthly_expenses = (income / 12) * np.random.uniform(0.3, 0.7, size=n_samples)
    
    # Abhi chalne wale doosre loans
    existing_loans = np.random.randint(0, 5, size=n_samples)
    
    # Pehla credit card/loan kab liya tha
    credit_history_length = (age - 18) * np.random.uniform(0.1, 0.8, size=n_samples)
    
    # Bank mein pari hui Savings
    savings = np.random.exponential(scale=10000, size=n_samples)

    # Step 3: Realistic Target Column (Creditworthy) calculate karna
    # ---------------------------------------------------------------
    # Mathematical Formula based on Banking Rules
    risk_score = (
        0.00005 * income 
        - 0.00008 * debt 
        - 3.5 * credit_utilization 
        - 1.2 * previous_defaults 
        + 0.05 * payment_history 
        + 0.03 * employment_length
        - 0.00004 * loan_amount
    )

    # Sigmoid Math Formula: Numbers ko 0.0 aur 1.0 ke beech ki Probability me convert karna
    probabilities = 1 / (1 + np.exp(-risk_score))
    
    # Decision Threshold: Agar probability > 0.5 hai toh 1 (Good Customer), warna 0 (Risky)
    creditworthy = np.where(probabilities > 0.5, 1, 0)

    # Step 4: Python Dictionary banana
    # -------------------------------
    data_dict = {
        'Age': age,
        'Income': np.round(income, 2),
        'Debt': np.round(debt, 2),
        'Credit_Utilization': np.round(credit_utilization, 4),
        'Loan_Amount': np.round(loan_amount, 2),
        'Loan_Duration': loan_duration,
        'Previous_Defaults': previous_defaults,
        'Payment_History': np.round(payment_history, 2),
        'Employment_Length': np.round(employment_length, 1),
        'Num_Credit_Cards': num_credit_cards,
        'Monthly_Expenses': np.round(monthly_expenses, 2),
        'Existing_Loans': existing_loans,
        'Credit_History_Length': np.round(credit_history_length, 1),
        'Savings': np.round(savings, 2),
        'Creditworthy': creditworthy
    }

    # Step 5: Pandas DataFrame mein Convert karna
    # -------------------------------------------
    df = pd.DataFrame(data_dict)

    # Real-world Jaisa banane ke liye kuch Missing Values (NaN) Inject karna
    nan_indices_inc = np.random.choice(df.index, size=150, replace=False)
    df.loc[nan_indices_inc, 'Income'] = np.nan

    nan_indices_emp = np.random.choice(df.index, size=100, replace=False)
    df.loc[nan_indices_emp, 'Employment_Length'] = np.nan

    # Step 6: File save karne se pehle folder check karna
    # --------------------------------------------------
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Step 7: Disk par CSV File Save karna
    # ------------------------------------
    df.to_csv(file_path, index=False)
    print(f"✅ Successful! Dataset ban kar save ho gaya hai yahan: {file_path}")
    
    return df

# ❌ Pehle yeh tha:
# generate_and_save_credit_data(file_path="../data/credit_data.csv")

# ✅ Isay aese badal dein:
if __name__ == "__main__":
    df = generate_and_save_credit_data(file_path="data/credit_data.csv")
    
    print("\n--- DATA LOADING CHECK ---")
    loaded_df = pd.read_csv("data/credit_data.csv")
    print(f"Dataset Shape in RAM: {loaded_df.shape}")