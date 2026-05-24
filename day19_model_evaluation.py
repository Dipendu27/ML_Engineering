import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

print("--- Day 19: Advanced Model Evaluation (The Accuracy Trap) ---\n")

project_dir = Path(__file__).parent
roc_chart = project_dir / "Figure_7.png"

# 1. Generate Highly Imbalanced Data (Rare Disease)
np.random.seed(42)
n_patients = 10000

# 99% Healthy (0), 1% Rare Disease (1)
# This perfectly mimics real-world rare anomaly detection
y = np.random.choice([0, 1], size=n_patients, p=[0.99, 0.01])

# Create synthetic biomarkers
# Healthy patients have normal levels, sick patients have slightly elevated levels
biomarker_A = np.random.normal(50, 10, n_patients) + (y * 15)
biomarker_B = np.random.normal(100, 20, n_patients) - (y * 25)

X = pd.DataFrame({'Biomarker_A': biomarker_A, 'Biomarker_B': biomarker_B})

print(f"📊 Dataset: {len(X)} Total Patients.")
print(f"   ⚠️ WARNING: Only {sum(y)} patients have the rare disease ({sum(y)/len(y)*100:.1f}%).\n")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Train the Model
print("🚀 Training Random Forest Classifier...")
# 'class_weight=balanced' forces the AI to pay extreme attention to the rare 1%
model = RandomForestClassifier(random_state=42, class_weight='balanced', max_depth=4)
model.fit(X_train, y_train)

# 3. Predict & Evaluate
predictions = model.predict(X_test)
# probabilities gives us the exact % confidence, not just a 1 or 0
probabilities = model.predict_proba(X_test)[:, 1] 

# Calculate the Big 4 Metrics
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print("🏥 Model Evaluation Metrics:")
print(f"✅ Accuracy:  {accuracy * 100:.2f}%  <-- Looks amazing, right?")
print(f"🎯 Precision: {precision * 100:.2f}%  <-- When AI says 'Sick', it's only right this % of the time (False Alarms)")
print(f"🔍 Recall:    {recall * 100:.2f}%  <-- Out of all TRULY sick patients, the AI found this %")
print(f"⚖️ F1-Score:  {f1 * 100:.2f}%  <-- The true grade of the model's performance\n")

# 4. Plot ROC Curve
# Calculate False Positive Rate (fpr) and True Positive Rate (tpr)
fpr, tpr, thresholds = roc_curve(y_test, probabilities)
roc_auc = auc(fpr, tpr)

print(f"📈 ROC-AUC Score: {roc_auc:.3f} (1.00 is perfect, 0.50 is random guessing)")
print("🎨 Generating ROC Curve plot... Check your taskbar/dock!")

plt.figure(figsize=(8, 6))
# Plot the AI's performance line
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
# Plot a baseline showing what random guessing looks like
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guessing')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (False Alarms)')
plt.ylabel('True Positive Rate (Recall / Catching the Disease)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(roc_chart, dpi=150)
print(f"Saved ROC curve to: {roc_chart.name}")
if plt.get_backend().lower() != "agg":
    plt.show()
