import pandas as pd
from pathlib import Path

# 1. Define your local dataset path
file_path = Path(__file__).with_name("healthcare_dataset.csv")

print("--- Healthcare Dataset EDA Engine (Local) ---")

try:
    # 2. Load the raw data from your local system
    df = pd.read_csv(file_path)
    print("Dataset successfully loaded from local file!\n")
except FileNotFoundError:
    print(f"Error: Could not find {file_path}. Please check your folder structure.")
else:
    # 3. Analyze the Structure
    print(f"Total Patient Records (Rows): {df.shape[0]}")
    print(f"Total Features (Columns): {df.shape[1]}\n")

    # 4. Preview the first few rows to understand the format
    print("--- Data Preview (Top 3 Rows) ---")
    print(df.head(3))
    print("\n")

    # 5. Hunt for Missing Data (Null values)
    print("--- Missing Data Check ---")
    missing_data = df.isnull().sum()

    # Filter to only show columns that actually have missing data
    missing_data = missing_data[missing_data > 0]

    if missing_data.empty:
        print("Excellent! No missing data detected in any column.")
    else:
        print("Warning: Missing data found in the following columns:")
        print(missing_data)

print("\n-------------------------------------")
