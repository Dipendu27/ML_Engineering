import pandas as pd
from pathlib import Path

# 1. Define file paths
project_dir = Path(__file__).parent
input_file = project_dir / "healthcare_dataset.csv"
output_file = project_dir / "cleaned_healthcare_data.csv"

print("--- Healthcare Data Cleaning Pipeline ---")

# 2. Load the raw data
df = pd.read_csv(input_file)

# 3. Handle Missing Data (Imputation)
# Calculate the median BMI and fill the NaN values
missing_before = df['bmi'].isnull().sum()
bmi_median = df['bmi'].median()
df['bmi'] = df['bmi'].fillna(bmi_median)

print(f"Fixed {missing_before} missing values by imputing the median BMI: {bmi_median}")

# Verify our fix
missing_after = df['bmi'].isnull().sum()
print(f"Missing BMI values remaining: {missing_after}\n")

# 4. Text Serialization for RAG
# Convert each row into a natural language string for our embedding model
print("Converting patient records into text profiles...")

def create_patient_profile(row):
    return f"Patient is a {row['age']}-year-old {row['gender']}. Hypertension: {row['hypertension']}. Heart Disease: {row['heart_disease']}. BMI: {row['bmi']}. Smoking Status: {row['smoking_status']}."

# Apply the function to create a new column
df['patient_profile'] = df.apply(create_patient_profile, axis=1)

# Preview the text profiles
print("\n--- Profile Preview ---")
print(df['patient_profile'].head(2).values)
print("\n")

# 5. Save the cleaned dataset
df.to_csv(output_file, index=False)
print(f"Success! Cleaned data saved to: {output_file}")
print("-----------------------------------------")
