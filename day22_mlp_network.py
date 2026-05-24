import torch
import torch.nn as nn

print("--- Day 22: Multi-Layer Perceptron (PyTorch nn.Module) ---\n")

# 1. Define the Neural Network Architecture
# We inherit from nn.Module, which gives our class PyTorch superpowers
class PatientRiskMLP(nn.Module):
    def __init__(self):
        super(PatientRiskMLP, self).__init__()

        # Define the layers (The Brain's Structure)
        # Layer 1: Takes 3 patient inputs (Age, BMI, BP) and expands to 16 hidden neurons
        self.hidden_layer_1 = nn.Linear(in_features=3, out_features=16)

        # Layer 2: Takes the 16 neurons and narrows them down to 8 neurons
        self.hidden_layer_2 = nn.Linear(in_features=16, out_features=8)

        # Layer 3 (Output): Takes the 8 neurons and outputs 1 final prediction (Risk Score)
        self.output_layer = nn.Linear(in_features=8, out_features=1)

        # Define the Activation Functions (The "Sparks" between neurons)
        self.relu = nn.ReLU()       # Used for hidden layers
        self.sigmoid = nn.Sigmoid() # Used for the final output to squash it between 0% and 100%

    def forward(self, x):
        # This function defines exactly how the data flows through the brain
        # Step 1: Pass input through Layer 1, then apply ReLU activation
        x = self.hidden_layer_1(x)
        x = self.relu(x)

        # Step 2: Pass through Layer 2, then apply ReLU activation
        x = self.hidden_layer_2(x)
        x = self.relu(x)

        # Step 3: Pass through Output Layer, then apply Sigmoid to get a probability
        x = self.output_layer(x)
        prediction = self.sigmoid(x)

        return prediction

# 2. Instantiate the Model
# This physically creates the initialized weights in your computer's memory
model = PatientRiskMLP()
print("🧠 Neural Network Architecture Built Successfully:")
print(model)
print("-" * 50)

# 3. Simulate a Forward Pass (No training yet, just testing the plumbing)
print("\n🚀 Simulating a Forward Pass...")

# Create a mock patient tensor [Age, BMI, BP]
# Note: In PyTorch, we almost always use float32 for neural networks
patient_data = torch.tensor([[55.0, 28.0, 130.0]], dtype=torch.float32)
print(f"📥 Input Tensor Shape: {patient_data.shape}")

# To pass data through the model, you simply call the model like a function!
# This automatically triggers the forward() method.
with torch.no_grad(): # We tell PyTorch not to calculate gradients yet to save memory
    risk_probability = model(patient_data)

print(f"\n📤 Output Tensor Shape: {risk_probability.shape}")
print(f"👉 Raw Untrained AI Prediction: {risk_probability.item() * 100:.2f}% Risk")
print("\n(Note: Because the model is untrained, the weights are random, so this prediction is currently just a random guess!)")
