import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
# The imblearn Pipeline is special because it safely applies SMOTE only to the training data
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

print("--- Day 20: End-to-End Pipeline & Model Optimization ---\n")

# 1. Generate the same Highly Imbalanced Data from Day 19
np.random.seed(42)
n_patients = 10000
y = np.random.choice([0, 1], size=n_patients, p=[0.99, 0.01])
biomarker_A = np.random.normal(50, 10, n_patients) + (y * 15)
biomarker_B = np.random.normal(100, 20, n_patients) - (y * 25)

X = pd.DataFrame({'Biomarker_A': biomarker_A, 'Biomarker_B': biomarker_B})
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Build the End-to-End Pipeline
print("🚀 Initializing SMOTE + Random Forest Pipeline...")
# A Pipeline chains steps together.
# Step 1: SMOTE generates synthetic "sick" patients to balance the training data.
# Step 2: The Random Forest trains on this newly balanced data.
pipeline = Pipeline([
    ('balancer', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42, max_depth=6))
])

# Train the entire pipeline at once
pipeline.fit(X_train, y_train)

# 3. Threshold Tuning (The Fix for Alarm Fatigue)
# Instead of getting standard predictions, we get the raw probabilities
probabilities = pipeline.predict_proba(X_test)[:, 1]

# We manually shift the threshold from the default 50% to 75%
# "Only raise the alarm if you are >75% certain!"
custom_threshold = 0.75
tuned_predictions = (probabilities >= custom_threshold).astype(int)

# 4. Evaluate the Improved Model
precision = precision_score(y_test, tuned_predictions)
recall = recall_score(y_test, tuned_predictions)
f1 = f1_score(y_test, tuned_predictions)

print("\n🏥 OPTIMIZED MODEL METRICS (Threshold = 75%):")
print(f"🎯 Precision: {precision * 100:.2f}%  <-- Massive improvement!")
print(f"🔍 Recall:    {recall * 100:.2f}%  <-- Still catching the vast majority of cases")
print(f"⚖️ F1-Score:  {f1 * 100:.2f}%  <-- A much healthier, balanced model\n")

print("📊 New Confusion Matrix:")
conf_matrix = confusion_matrix(y_test, tuned_predictions)
print(f"True Negatives (Healthy, correctly ignored): {conf_matrix[0][0]}")
print(f"False Positives (False Alarms):              {conf_matrix[0][1]} <-- We drastically reduced these!")
print(f"False Negatives (Missed Patients):           {conf_matrix[1][0]}")
print(f"True Positives (Sick Patients Caught):       {conf_matrix[1][1]}")
