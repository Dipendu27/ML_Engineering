import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("--- Day 15: XGBoost Classifier (Diabetes Risk Prediction) ---\n")

# 1. Generate Synthetic Clinical Data
np.random.seed(42)
n_patients = 1500

ages = np.random.randint(25, 75, n_patients)
bmis = np.random.normal(29, 6, n_patients)
fasting_blood_sugar = np.random.normal(100, 25, n_patients)
insulin_levels = np.random.normal(85, 30, n_patients)

# Logic: High risk requires a combination of factors, not just one.
# XGBoost is excellent at finding these multi-variable "if-this-then-that" conditions.
risk_score = (ages * 0.01) + (bmis * 0.1) + (fasting_blood_sugar * 0.05) + (insulin_levels * 0.02) + np.random.normal(0, 2, n_patients)

# Top ~25% risk scores result in a Diabetes Diagnosis (1)
diabetes_diagnosis = (risk_score > 12.5).astype(int)

df = pd.DataFrame({
    'Age': ages, 
    'BMI': bmis, 
    'Fasting_Blood_Sugar': fasting_blood_sugar, 
    'Insulin_Level': insulin_levels,
    'Diabetes': diabetes_diagnosis
})

print(f"📊 Dataset: {df['Diabetes'].sum()} patients diagnosed out of {n_patients}.\n")

# 2. Train-Test Split
X = df.drop('Diabetes', axis=1)
y = df['Diabetes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Model Training (The XGBoost Engine)
print("🚀 Training the XGBoost Model (Sequential Tree Boosting)...")

# XGBoost has specific hyperparameters. 
# learning_rate controls how aggressively each tree fixes the previous tree's mistakes.
model = xgb.XGBClassifier(
    n_estimators=150,       # Number of sequential trees to build
    learning_rate=0.1,      # Step size for error correction
    max_depth=4,            # Keep individual trees relatively shallow
    random_state=42,
    eval_metric='logloss'   # The calculus loss function used to grade the mistakes
)

# Train the model
model.fit(X_train, y_train)

# 4. Evaluation
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"\n✅ XGBoost Accuracy: {accuracy * 100:.2f}%")

print("\n🏥 Classification Report:")
print(classification_report(y_test, predictions, target_names=['Healthy', 'Diabetic']))

# 5. Let's test it on a highly specific edge-case patient
print("🔮 Inference Edge Case: Young patient (28) but High Blood Sugar (140) & High BMI (34)")
edge_case = pd.DataFrame({'Age': [28], 'BMI': [34], 'Fasting_Blood_Sugar': [140], 'Insulin_Level': [110]})

probability = model.predict_proba(edge_case)[0][1]
diagnosis = model.predict(edge_case)[0]

print(f"👉 Prediction: {'Diabetic Risk' if diagnosis == 1 else 'Healthy'}")
print(f"👉 AI Confidence (Probability): {probability * 100:.1f}%")
