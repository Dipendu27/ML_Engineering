import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

print("--- Day 14: Random Forest Classifier (Patient Readmission) ---\n")

project_dir = Path(__file__).parent
importance_chart = project_dir / "Figure_4.png"

# 1. Generate Synthetic Clinical Data
np.random.seed(42)
n_patients = 1200

ages = np.random.randint(30, 90, n_patients)
bmis = np.random.normal(28, 6, n_patients)
blood_pressures = np.random.normal(130, 20, n_patients)
days_in_hospital = np.random.randint(1, 14, n_patients) # Initial length of stay

# Logic: Longer stays, older age, and higher BMI increase readmission risk
risk_score = (ages * 0.02) + (bmis * 0.05) + (days_in_hospital * 0.3) + np.random.normal(0, 1.5, n_patients)
# Top ~30% risk scores result in readmission (1)
readmitted = (risk_score > 6.0).astype(int)

df = pd.DataFrame({
    'Age': ages, 
    'BMI': bmis, 
    'Blood_Pressure': blood_pressures, 
    'Days_In_Hospital': days_in_hospital,
    'Readmitted': readmitted
})

print(f"📊 Dataset: {df['Readmitted'].sum()} patients were readmitted out of {n_patients}.\n")

# 2. Train-Test Split
# Like standard decision trees, Random Forests do NOT require data scaling!
X = df.drop('Readmitted', axis=1) # Features (Everything except the target)
y = df['Readmitted']              # Target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Model Training
print("🚀 Planting and Training the Random Forest (100 Trees)...")
# n_estimators=100 means we are building 100 individual decision trees
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# 4. Evaluation
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"\n✅ Forest Accuracy: {accuracy * 100:.2f}%")
print("\n🏥 Classification Report:")
print(classification_report(y_test, predictions, target_names=['Healthy', 'Readmitted']))

# 5. Extracting Feature Importance
# Let's find out what the 100 trees collectively decided was the most important factor
importances = model.feature_importances_
feature_names = X.columns

print("🌳 Generating Feature Importance Chart... Check your taskbar/dock!")
plt.figure(figsize=(10, 6))
sns.set_theme(style="whitegrid")

# Create a bar chart of the importances
sns.barplot(x=importances, y=feature_names, hue=feature_names, palette="mako", legend=False)
plt.title("Random Forest: Feature Importance for Hospital Readmission")
plt.xlabel("Importance Score (Adds up to 1.0)")
plt.ylabel("Clinical Feature")
plt.tight_layout()
plt.savefig(importance_chart, dpi=150)
print(f"Saved feature importance chart to: {importance_chart.name}")
if plt.get_backend().lower() != "agg":
    plt.show()
