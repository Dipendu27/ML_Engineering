import torch
import numpy as np

print("--- Day 21: PyTorch Tensor Fundamentals ---\n")

# 1. Creating Tensors
print("🚀 1. Tensor Creation")
# From a standard Python list
tensor_1d = torch.tensor([1.0, 2.0, 3.0, 4.0])
print(f"1D Tensor: {tensor_1d} | Shape: {tensor_1d.shape}")

# From a NumPy array (The bridge between Classical ML and Deep Learning)
numpy_array = np.array([[1, 2], [3, 4]])
tensor_2d = torch.from_numpy(numpy_array)
print(f"2D Tensor from NumPy:\n{tensor_2d} | Shape: {tensor_2d.shape}\n")

# Generating random tensors (How neural networks initialize their 'brains')
# Creates a 3x3 matrix of random numbers between 0 and 1
random_tensor = torch.rand(3, 3)
print(f"Random 3x3 Tensor:\n{random_tensor}\n")


# 2. Tensor Math (Matrix Multiplication)
print("🧮 2. Tensor Operations")
# Let's simulate a miniature neural network layer
# Inputs: 1 patient, 3 features [Age, BMI, BP]
patient_input = torch.tensor([[55.0, 28.0, 130.0]])

# Weights: The AI's learned parameters connecting 3 inputs to 2 hidden neurons
weights = torch.tensor([
    [0.1, 0.2],  # Weights for neuron 1
    [-0.1, 0.5], # Weights for neuron 2
    [0.3, -0.2]  # Weights for neuron 3
])

# Matrix Multiplication (@ is the python shortcut for torch.matmul)
# This multiplies the patient data by the AI weights
hidden_layer_output = patient_input @ weights
print(f"Patient Input Shape: {patient_input.shape}")
print(f"Weights Shape: {weights.shape}")
print(f"Neural Network Layer Output:\n{hidden_layer_output} | Shape: {hidden_layer_output.shape}\n")


# 3. Reshaping (The most common task for ML Engineers)
print("📐 3. Tensor Reshaping")
# Imagine we have a flat list of 16 pixel values from a medical scan
flat_image = torch.arange(1, 17) # Creates tensor [1, 2, ..., 16]
print(f"Flat Data: {flat_image} | Shape: {flat_image.shape}")

# A Neural Network needs this as a 4x4 2D grid!
# .view() instantly reshapes the tensor without copying data
grid_image = flat_image.view(4, 4)
print(f"Reshaped 4x4 Grid:\n{grid_image} | Shape: {grid_image.shape}")

# -1 is a PyTorch magic trick. It means "calculate this dimension for me"
# Let's reshape it into batches of 2
batched_data = flat_image.view(2, -1)
print(f"\nReshaped into 2 Batches:\n{batched_data} | Shape: {batched_data.shape}")
