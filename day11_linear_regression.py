import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

print("--- Day 11: Linear Regression Prediction Model ---\n")

# 1. Generate our Tabular Dataset (similar to Day 9)
np.random.seed(42)
n_patients = 1000

ages = np.random.randint(20, 80, n_patients)
bmis = np.random.normal(28, 5, n_patients)
# Formula: Base BP (80) + Age factor + BMI factor + Random Noise
blood_pressures = 80 + (ages * 0.4) + (bmis * 1.1) + np.random.normal(0, 5, n_patients)

df = pd.DataFrame({'Age': ages, 'BMI': bmis, 'Target_BP': blood_pressures})
print("📊 First 3 rows of patient data:")
print(df.head(3), "\n")

# 2. Train-Test Split (CRUCIAL ML CONCEPT)
# We must split our data. The model learns on the "Training" set (80%),
# and we test it on the "Testing" set (20%) to ensure it hasn't just memorized the answers.
X = df[['Age', 'BMI']] # Features (Inputs)
y = df['Target_BP']    # Target (Output)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"✂️ Data Split: {len(X_train)} Training Patients | {len(X_test)} Testing Patients\n")

# 3. Model Initialization & Training
print("🚀 Training the Linear Regression Model...")
model = LinearRegression()
# .fit() is where the math happens. The model finds the optimal weights.
model.fit(X_train, y_train)

# Let's peek under the hood at what the model learned!
print(f"🧠 Learned Weights -> Age Multiplier: {model.coef_[0]:.2f}, BMI Multiplier: {model.coef_[1]:.2f}")
print(f"🧠 Base Intercept -> {model.intercept_:.2f}\n")

# 4. Evaluation & Prediction
# Now we ask the model to predict the BP for the 200 patients in the test set
predictions = model.predict(X_test)

# How wrong was our model on average?
mae = mean_absolute_error(y_test, predictions)
print(f"📈 Evaluation: On average, the model's predictions are off by {mae:.2f} mmHg.")

# 5. Real-World Inference on a New Patient
new_patient = pd.DataFrame({'Age': [50], 'BMI': [30.0]})
predicted_bp = model.predict(new_patient)

print(f"\n🔮 Inference: Predicting BP for a 50-year-old patient with a BMI of 30...")
print(f"👉 Predicted Systolic Blood Pressure: {predicted_bp[0]:.1f}")