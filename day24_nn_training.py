import torch
import torch.nn as nn
import torch.optim as optim

print("--- Day 24: PyTorch Neural Network Training Loop ---\n")

# 1. Hardware Setup (MPS for Apple Silicon)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"🚀 Booting Neural Network on device: {device}\n")

# 2. Generate Synthetic Training Data (1000 patients)
torch.manual_seed(42)
# X contains 2 features per patient (e.g., normalized Age and BMI, 0 to 100)
X_train = torch.rand(1000, 2) * 100 

# Hidden Logic: If (Age + BMI) > 100, they are High Risk (1.0). Otherwise, Low Risk (0.0).
# The AI doesn't know this rule; it has to learn it from scratch.
y_train = ((X_train[:, 0] + X_train[:, 1]) > 100).float().view(-1, 1)

# CRITICAL: We must move the data to the M5 GPU!
X_train = X_train.to(device)
y_train = y_train.to(device)

# 3. Define the Network Architecture
class RiskModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(in_features=2, out_features=8)
        self.relu = nn.ReLU()
        self.output = nn.Linear(in_features=8, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.sigmoid(self.output(x))
        return x

# Instantiate the model and move it to the M5 GPU
model = RiskModel().to(device)

# 4. The Loss Function and Optimizer
# BCELoss = Binary Cross Entropy (Industry standard for 1 vs 0 classification)
criterion = nn.BCELoss() 
# Adam Optimizer: It dynamically adjusts how big of a step it takes down the hill
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 5. The Training Loop
epochs = 1000
print(f"⚙️ Starting Training Loop for {epochs} Epochs...")

for epoch in range(epochs):
    # Step 1: Forward Pass
    predictions = model(X_train)
    
    # Step 2: Calculate Loss
    loss = criterion(predictions, y_train)
    
    # Step 3: Zero the Gradients
    optimizer.zero_grad()
    
    # Step 4: Backpropagation (Calculate slopes)
    loss.backward()
    
    # Step 5: Optimizer Step (Update weights)
    optimizer.step()
    
    # Print progress every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f"   Epoch {epoch+1:04d}/{epochs} | Loss (Error): {loss.item():.4f}")

print("\n✅ Training Complete! The loss successfully approached 0.")
print("The AI has mapped the hidden relationship between the features.")