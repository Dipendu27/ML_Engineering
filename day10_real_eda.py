import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

print("--- Day 10: Real-World Data Cleaning & EDA ---\n")

project_dir = Path(__file__).parent
mock_file = project_dir / 'kaggle_mock_dataset.csv'
chart_file = project_dir / 'Figure_2.png'

# 1. Simulate downloading a "Messy" CSV file from Kaggle
messy_data = {
    'Patient_ID': [1, 2, 3, 4, 5, 6, 7],
    'Age': [45, 150, 29, np.nan, 52, 38, -5],  # 150 and -5 are impossible outliers
    'Diagnosis': ['Diabetes', 'hypertension', 'DIABETES', 'Healthy', 'hypertension ', 'Unknown', 'Healthy'],
    'Blood_Pressure': [130, 180, 115, np.nan, 140, 125, 118]
}
messy_df = pd.DataFrame(messy_data)
messy_df.to_csv(mock_file, index=False)
print("📥 Mock messy dataset saved as 'kaggle_mock_dataset.csv'")

# ---------------------------------------------------------
# 2. The ML Engineer Pipeline Begins Here
# ---------------------------------------------------------
print("🚀 Loading and inspecting raw data...")
df = pd.read_csv(mock_file)

print("\n❌ RAW DATA (Notice the errors, NaNs, and case differences):")
print(df)

# 3. Data Cleaning Sequence
print("\n🧹 Starting Data Sanitization...")

# A. Standardize Text (Lowercasing and stripping accidental spaces)
df['Diagnosis'] = df['Diagnosis'].str.lower().str.strip()

# B. Handle Outliers (Filter out impossible ages)
# Keep only patients between 0 and 120 years old
df = df[(df['Age'] > 0) & (df['Age'] < 120) | df['Age'].isna()]

# C. Impute Missing Values (Fill NaNs)
# Fill missing age with the median, and missing BP with the mean
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Blood_Pressure'] = df['Blood_Pressure'].fillna(df['Blood_Pressure'].mean())

print("\n✅ CLEANED DATA:")
print(df)

# 4. Visualization (Proving our data is clean)
plt.figure(figsize=(10, 5))
sns.set_theme(style="whitegrid")

# Plot the cleaned diagnosis counts
sns.countplot(data=df, x='Diagnosis', hue='Diagnosis', palette='viridis', legend=False)
plt.title('Patient Diagnosis Distribution (Cleaned)')
plt.xlabel('Diagnosis Categorization')
plt.ylabel('Number of Patients')

plt.tight_layout()
plt.savefig(chart_file, dpi=150)
print(f"\n📊 Generating chart... Saved to: {chart_file.name}")
if plt.get_backend().lower() != 'agg':
    plt.show()

# Clean up the mock file after we're done
if mock_file.exists():
    mock_file.unlink()
