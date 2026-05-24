import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

print("--- Day 13: Decision Tree Classifier ---\n")

project_dir = Path(__file__).parent
tree_jpeg = project_dir / "Decision_Tree.jpeg"
tree_png = project_dir / "Figure_3.png"

# 1. Generate Synthetic Clinical Data
np.random.seed(42)
n_patients = 800

ages = np.random.randint(20, 85, n_patients)
bmis = np.random.normal(26, 5, n_patients)
bps = np.random.normal(120, 15, n_patients)

# Logic: High Priority Follow-up if (Age > 60 AND BP > 140) OR (BMI > 35)
follow_up = np.where(((ages > 60) & (bps > 140)) | (bmis > 35), 1, 0)

df = pd.DataFrame({'Age': ages, 'BMI': bmis, 'Blood_Pressure': bps, 'Priority_FollowUp': follow_up})

print(f"📊 Dataset: {df['Priority_FollowUp'].sum()} patients flagged for priority follow-up.\n")

# 2. Train-Test Split
# Notice: We DO NOT need to use StandardScaler today! 
# Trees don't care about the scale of the data; they only care about making splits.
X = df[['Age', 'BMI', 'Blood_Pressure']]
y = df['Priority_FollowUp']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Model Training
print("🚀 Training the Decision Tree...")
# We set max_depth=3 so the tree doesn't overfit and remains easy to read visually
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

# 4. Evaluation
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"✅ Model Accuracy: {accuracy * 100:.2f}%\n")

# 5. Visualizing the "Machine's Brain"
print("🌳 Generating Tree Visualization... Check your taskbar/dock!")
plt.figure(figsize=(14, 8))
plot_tree(
    model, 
    feature_names=['Age', 'BMI', 'BP'], 
    class_names=['Routine', 'High Priority'], 
    filled=True, 
    rounded=True,
    fontsize=10
)
plt.title("Decision Tree Logic for Patient Follow-Ups")
plt.tight_layout()
plt.savefig(tree_jpeg, dpi=200)
plt.savefig(tree_png, dpi=150)
print(f"Saved tree visualizations to: {tree_jpeg.name} and {tree_png.name}")
if plt.get_backend().lower() != "agg":
    plt.show()
