import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

# Terminal warnings ko chupane ke liye
warnings.filterwarnings('ignore')

def perform_eda(file_path="../data/credit_data.csv"):
    """
    Dataset load karke uski statistical summary aur graphical visualisations karta hai.
    """
    # File ki maujoodgi check karna
    if not os.path.exists(file_path):
        print(f"❌ Error: File {file_path} nahi mili. Pehle dataset_generator.py chalayein.")
        return

    # 1. Dataset Load Karna
    # ---------------------
    print("⏳ Data Load Ho Raha Hai...\n")
    df = pd.read_csv(file_path)

    # 2. Structural Inspection
    # ------------------------
    print("--- 1. DATASET SHAPE (Rows, Columns) ---")
    print(df.shape)
    print("\n--- 2. DATASET INFO ---")
    print(df.info())
    print("\n--- 3. SUMMARY STATISTICS ---")
    print(df.describe().T) # .T matrix ko transpose karta hai taake padhna aasan ho
    print("\n--- 4. MISSING VALUES ---")
    print(df.isnull().sum())
    
    print("\n--- 5. TARGET CLASS DISTRIBUTION ---")
    # Dekhna ke kitne % log creditworthy hain vs defaulter
    print(df['Creditworthy'].value_counts(normalize=True))

    # 3. Visualizations (Graphs)
    # --------------------------
    print("\n📊 Graph Plotting Start Ho Rahi Hai...")
    
    # Ek figure mein 4 graphs banayenge
    plt.figure(figsize=(16, 12))

    # Graph 1: Target Variable (Imbalance Check)
    plt.subplot(2, 2, 1)
    sns.countplot(x='Creditworthy', data=df, palette='Set2')
    plt.title('Distribution of Creditworthy (1 = Good, 0 = Bad)')

    # Graph 2: Credit Utilization vs Risk
    plt.subplot(2, 2, 2)
    sns.boxplot(x='Creditworthy', y='Credit_Utilization', data=df, palette='Set2')
    plt.title('Credit Utilization Impact on Credit Score')

    # Graph 3: Previous Defaults Effect
    plt.subplot(2, 2, 3)
    sns.countplot(x='Previous_Defaults', hue='Creditworthy', data=df, palette='Set2')
    plt.title('Past Defaults vs Future Approval')

    # Graph 4: Income Distribution
    plt.subplot(2, 2, 4)
    # dropna() zaruri hai warna seaborn NaN values pe erorr dega
    sns.histplot(df['Income'].dropna(), kde=True, color='teal', bins=30)
    plt.title('Income Distribution (KDE Curve)')

    plt.tight_layout()
    plt.show()

    # 4. Correlation Heatmap
    # ----------------------
    plt.figure(figsize=(12, 10))
    # Pearson correlation matrix banata hai
    corr_matrix = df.corr()
    
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
    plt.title('Feature Correlation Heatmap')
    plt.show()

if __name__ == "__main__":
    perform_eda(file_path="../data/credit_data.csv")