import numpy as np
import pandas as pd

print("--- Day 2: Clinical Data Engineering Engine --- \n")

# 1. NumPy: Generating Synthetic Patient Biomarkers
# Let's mock blood pressure readings for 5 patients over 3 hospital visits.
# This creates a 2D Array (Matrix) of shape (5, 3)
np.random.seed(42)  # Ensures reproducible results
biomarkers = np.random.randint(110, 150, size=(5, 3))
print("Step 1: Raw NumPy Biomarker Matrix (Shape: 5x3):")
print(biomarkers, "\n")

# Calculate the mean blood pressure per patient (axis=1 means across columns)
patient_means = np.mean(biomarkers, axis=1)
print(f"Calculated Patient Mean Blood Pressures: {patient_means}\n")

# 2. Pandas: Building a Clinical Study DataFrame
# Imagine we have raw clinical trial records with some missing data.
raw_data = {
    'Patient_ID': [101, 102, 103, 104, 105],
    'Age': [45, np.nan, 29, 61, 52],  # Note the missing value (NaN)
    'Condition': ['Hypertension', 'Diabetes', 'Healthy', 'Hypertension', 'Diabetes'],
    'Clinical_Notes': [
        "Patient exhibits elevated resting systolic pressure.",
        "Type 2 diabetes checks. Follow up required.",
        "Routine checkup. Vital signs normal.",
        "Severe chronic hypertension tracking.",
        "BGL unstable. Adjusting metformin dosage."
    ]
}

df = pd.DataFrame(raw_data)
print("Step 2: Initial Pandas Clinical DataFrame:")
print(df, "\n")

# 3. Data Cleaning: Handling Missing Values (Imputation)
# We can't feed NaN values into an ML model. Let's fill missing Age with the median age.
median_age = df['Age'].median()
df['Age'] = df['Age'].fillna(median_age)
print(f"Step 3: Imputed missing ages with median ({median_age}):")
print(df, "\n")

# 4. Data Slicing & Filtering
# Real-world demand: Filter out only the high-risk 'Hypertension' patients over 40.
high_risk_patients = df[(df['Condition'] == 'Hypertension') & (df['Age'] > 40)]
print("Step 4: Filtered High-Risk Hypertension Patients:")
print(high_risk_patients[['Patient_ID', 'Age', 'Clinical_Notes']])