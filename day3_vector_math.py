import mlx.core as mx

print("--- Day 3: Clinical Semantics & Matrix Engines --- \n")

# 1. Simulate embedding vectors for 3 clinical terms
# Imagine these arrays represent semantic concepts extracted from text
hyper_tension_vector = mx.array([0.9, 0.1, 0.0])  # Strong focus on cardiovascular
diabetes_vector      = mx.array([0.1, 0.8, 0.1])  # Strong focus on endocrine/blood sugar
blood_press_vector   = mx.array([0.8, 0.2, 0.0])  # High correlation to hypertension

# 2. Compute similarity manually using the Dot Product
# High dot product = High contextual similarity
similarity_1 = mx.sum(hyper_tension_vector * blood_press_vector)
similarity_2 = mx.sum(hyper_tension_vector * diabetes_vector)

print(f"Dot Product (Hypertension vs Blood Pressure): {similarity_1.item():.2f}")
print(f"Dot Product (Hypertension vs Diabetes):       {similarity_2.item():.2f}")
print("-> Notice how the related medical concepts yield a much higher score.\n")

# 3. Scale up to Matrix Multiplication
# Let's stack patient clinical summaries into a Matrix (2 patients, 3 medical traits)
# Patient 0: Mostly high blood pressure symptoms
# Patient 1: Deep diabetic/blood sugar issues
patient_matrix = mx.array([
    [0.85, 0.15, 0.00],  
    [0.05, 0.90, 0.05]
])

# Stack our medical category targets into a comparison matrix (Shape: 3x2)
category_matrix = mx.array([
    [0.9, 0.1],  # Hypertension profile
    [0.1, 0.8],  # Diabetes profile
    [0.0, 0.1]
])

# Perform a native Matrix Multiplication using MLX (matmul)
# This evaluates both patients against both diseases instantly on your M5 chip
analysis_results = mx.matmul(patient_matrix, category_matrix)

print("Step 3: MLX Accelerated Patient Diagnosis Matrix (Patients x Diseases):")
print(analysis_results)
print("\nRow 0 represents Patient 0 (High score in Column 0 indicates Hypertension risk)")
print("Row 1 represents Patient 1 (High score in Column 1 indicates Diabetes risk)")