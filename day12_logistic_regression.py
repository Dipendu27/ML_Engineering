import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("--- Day 12: Logistic Regression Classifier ---\n")

# 1. Generate Synthetic Classification Data
np.random.seed(42)
n_patients = 1000

ages = np.random.randint(20, 80, n_patients)
bmis = np.random.normal(25, 6, n_patients)
heart_rates = np.random.normal(70, 15, n_patients)

# We create a hidden "Risk Score". If it's high enough, they have hypertension.
risk_scores = (ages * 0.05) + (bmis * 0.2) + (heart_rates * 0.05) + np.random.normal(0, 1, n_patients)
# Convert the continuous risk score into a binary Yes (1) / No (0)
hypertension_diagnosis = (risk_scores > 10).astype(int)

df = pd.DataFrame({
    'Age': ages, 'BMI': bmis, 'Heart_Rate': heart_rates, 'Hypertension': hypertension_diagnosis
})

print(f"📊 Dataset Balance: {df['Hypertension'].sum()} patients with Hypertension, {n_patients - df['Hypertension'].sum()} Healthy\n")

# 2. Train-Test Split
X = df[['Age', 'BMI', 'Heart_Rate']]
y = df['Hypertension']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Data Scaling (CRUCIAL for Logistic Regression)
scaler = StandardScaler()
# We 'fit' the scaler only on the training data, then transform both train and test
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Model Training
print("🚀 Training the Logistic Regression Model...")
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# 5. Evaluation & The Confusion Matrix
predictions = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, predictions)

print(f"\n✅ Overall Model Accuracy: {accuracy * 100:.2f}%")

print("\n🏥 Confusion Matrix (How did the model make mistakes?):")
conf_matrix = confusion_matrix(y_test, predictions)
print(f"True Negatives (Correctly predicted Healthy):      {conf_matrix[0][0]}")
print(f"False Positives (Falsely predicted Hypertension):  {conf_matrix[0][1]}  <-- AI False Alarm")
print(f"False Negatives (Missed the Hypertension!):        {conf_matrix[1][0]}  <-- DANGEROUS")
print(f"True Positives (Correctly predicted Hypertension): {conf_matrix[1][1]}")

# 6. Real-World Inference (Probabilities)
# Let's test a my health
new_patient = pd.DataFrame({'Age': [23], 'BMI': [22.5], 'Heart_Rate': [72]})
# Don't forget to scale the new patient!
new_patient_scaled = scaler.transform(new_patient)

# predict_proba returns the exact % probability from the Sigmoid curve
probability = model.predict_proba(new_patient_scaled)[0][1] 
prediction = model.predict(new_patient_scaled)[0]

print(f"\n🔮 Inference: 23-year-old, BMI 22.5, HR 72")
print(f"👉 Prediction: {'Hypertension' if prediction == 1 else 'Healthy'}")
print(f"👉 AI Confidence (Probability): {probability * 100:.1f}%")