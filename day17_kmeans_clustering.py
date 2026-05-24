import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

print("--- Day 17: Unsupervised Learning (K-Means Clustering) ---\n")

project_dir = Path(__file__).parent
cluster_chart = project_dir / "Figure_5.png"

# 1. Generate Synthetic UNLABELED Clinical Data
# We intentionally create 3 distinct "hidden" groups of patients
np.random.seed(42)

# Group A: Young & Healthy (Low BMI, Normal BP)
group_a = np.random.normal(loc=[25, 22, 110], scale=[5, 2, 10], size=(150, 3))
# Group B: Middle-aged & High Risk (High BMI, High BP)
group_b = np.random.normal(loc=[50, 32, 145], scale=[8, 3, 12], size=(150, 3))
# Group C: Elderly & Moderate Risk (Normal BMI, Elevated BP)
group_c = np.random.normal(loc=[70, 25, 135], scale=[6, 2, 10], size=(150, 3))

# Combine them all and shuffle so the AI doesn't know the groups exist
all_patients = np.vstack([group_a, group_b, group_c])
np.random.shuffle(all_patients)

df = pd.DataFrame(all_patients, columns=['Age', 'BMI', 'Blood_Pressure'])
print(f"📊 Received raw data for {len(df)} patients. No diagnoses provided!\n")

# 2. Data Scaling (CRITICAL FOR K-MEANS)
# K-Means calculates physical distance. If Age is in the 70s and BMI is in the 20s, 
# it will unfairly prioritize Age. We must scale them to be equal!
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# 3. Model Training (The Clustering Engine)
print("🚀 Initializing K-Means to find 3 hidden patient segments...")
# We explicitly ask the AI to find 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)

# Notice we use .fit_predict(X). There is NO y_train! 
# The AI generates the labels itself.
df['AI_Assigned_Cluster'] = kmeans.fit_predict(scaled_data)

# 4. Analysis
print("\n✅ AI successfully segmented the patients!")
# Let's see the average stats for the groups the AI discovered
cluster_summary = df.groupby('AI_Assigned_Cluster').mean().round(1)
print("\n🏥 Average Vitals per AI-Discovered Cluster:")
print(cluster_summary)

# 5. Visualizing the Clusters
print("\n🎨 Generating 2D Cluster Visualization... Check your taskbar/dock!")
plt.figure(figsize=(10, 6))
sns.set_theme(style="whitegrid")

# We plot Age vs Blood Pressure, and color them by the AI's discovered clusters
sns.scatterplot(
    data=df, 
    x='Age', 
    y='Blood_Pressure', 
    hue='AI_Assigned_Cluster', 
    palette=['#1f77b4', '#ff7f0e', '#2ca02c'], # Blue, Orange, Green
    s=70, 
    alpha=0.8
)

plt.title('K-Means Patient Segmentation (Age vs. Blood Pressure)')
plt.xlabel('Patient Age')
plt.ylabel('Systolic Blood Pressure')
# Move the legend outside the plot
plt.legend(title='AI Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(cluster_chart, dpi=150)
print(f"Saved cluster visualization to: {cluster_chart.name}")
if plt.get_backend().lower() != "agg":
    plt.show()
