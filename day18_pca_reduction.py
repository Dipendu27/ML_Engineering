import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer

print("--- Day 18: Dimensionality Reduction with PCA ---\n")

project_dir = Path(__file__).parent
pca_chart = project_dir / "Figure_6.png"

# 1. Load a High-Dimensional Medical Dataset
# This dataset has 30 distinct measurements for every tumor.
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
# 0 = Malignant (Dangerous), 1 = Benign (Safe)
df['Diagnosis'] = data.target 

print(f"📊 Loaded Dataset Shape: {len(df)} patients with {df.shape[1]-1} distinct dimensions.")
print("   (Trying to plot this normally would require a 30-dimensional graph...)\n")

# 2. Data Scaling (ABSOLUTELY CRITICAL FOR PCA)
# If we don't scale the data, PCA will think features with large numbers are more important.
features = df.drop('Diagnosis', axis=1)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# 3. Apply PCA (Squishing 30 dimensions down to 2)
print("🚀 Running Principal Component Analysis (Compressing 30D -> 2D)...")
pca = PCA(n_components=2) # We explicitly ask for 2 dimensions for a 2D graph
principal_components = pca.fit_transform(scaled_features)

# Create a new DataFrame with our compressed math components
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
# Map the 0/1 back to words for the chart legend
pca_df['Diagnosis'] = df['Diagnosis'].replace({0: 'Malignant', 1: 'Benign'})

# 4. Analyze Information Retention (Explained Variance)
# How much of the original 30D truth survived the compression to 2D?
variance_kept = sum(pca.explained_variance_ratio_) * 100
print(f"✅ PCA successfully reduced 30 dimensions to 2!")
print(f"✅ The new 2D graph retains {variance_kept:.2f}% of the original complex information.\n")

# 5. Visualizing the Compressed Data
print("🎨 Generating 2D PCA Visualization... Check your taskbar/dock!")
plt.figure(figsize=(10, 6))
sns.set_theme(style="whitegrid")

# We plot our two math-generated components against each other
sns.scatterplot(
    data=pca_df, 
    x='PC1', 
    y='PC2', 
    hue='Diagnosis', 
    palette=['#d62728', '#2ca02c'], # Red, Green
    alpha=0.7,
    s=60
)

plt.title('30-Dimensional Breast Cancer Data Reduced to 2D (PCA)')
plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% Variance)')
plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% Variance)')
plt.legend(title='True Diagnosis')
plt.tight_layout()
plt.savefig(pca_chart, dpi=150)
print(f"Saved PCA visualization to: {pca_chart.name}")
if plt.get_backend().lower() != "agg":
    plt.show()
