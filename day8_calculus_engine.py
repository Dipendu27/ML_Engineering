import mlx.core as mx

print("--- Day 8: MLX Autograd & Backpropagation Engine ---\n")

# 1. The Setup
# Let's say we have a patient's resting heart rate feature (x) = 2.0
# And we know their actual risk score (y_true) = 10.0
x = mx.array([2.0])
y_true = mx.array([10.0])

# Our AI model starts with a completely random, bad guess for its "weight"
weight = mx.array([1.0]) 

# 2. Define the Loss Function (Mean Squared Error)
def calculate_loss(w, input_val, target_val):
    prediction = w * input_val
    # Error squared so it's always positive and punishes large mistakes heavily
    loss = mx.mean((prediction - target_val) ** 2)
    return loss

# 3. The Calculus Engine
# This MLX function magically computes both the Loss (altitude) and the Gradient (slope)
# using automatic differentiation on your M5 chip.
loss_and_grad_fn = mx.value_and_grad(calculate_loss)

learning_rate = 0.05 # How big of a step we take down the hill

print("🚀 Starting Training Loop...\n")

# 4. Gradient Descent Loop (The AI "Learning")
for epoch in range(1, 11):
    # Compute the current loss and the gradient
    loss_val, gradient = loss_and_grad_fn(weight, x, y_true)
    
    print(f"Epoch {epoch:02d} | Weight: {weight.item():.4f} | Loss: {loss_val.item():.4f} | Gradient (Slope): {gradient.item():.4f}")
    
    # Backpropagation: Update the weight by stepping in the OPPOSITE direction of the gradient
    weight = weight - (learning_rate * gradient)

print(f"\n✅ Training Complete. The model optimized the weight to: {weight.item():.4f} (Ideal is 5.000)")
