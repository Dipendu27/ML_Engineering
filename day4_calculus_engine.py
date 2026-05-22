import mlx.core as mx

# 1. Define a simple mathematical function
# Let's pretend this represents our network's "Loss" or "Error"
# Function: f(x) = 3x^2 + 2x + 1
def loss_function(x):
    return 3 * (x ** 2) + 2 * x + 1

# 2. Use MLX to automatically calculate the derivative (gradient)
# This is the exact underlying mechanism of Backpropagation
grad_fn = mx.value_and_grad(loss_function)

# 3. Test the gradient at a specific point (e.g., x = 2.0)
x_val = mx.array(2.0)
loss_val, gradient = grad_fn(x_val)

print("--- MLX Calculus Engine ---")
print(f"Input value (x): {x_val.item()}")
print(f"Calculated Loss: {loss_val.item()}")

# The analytical derivative of 3x^2 + 2x + 1 is 6x + 2.
# At x = 2.0, it should perfectly calculate: 6(2) + 2 = 14.0
print(f"Calculated Gradient (Rate of Change): {gradient.item()}")
print("---------------------------")
print("If your gradient is 14.0, your M5 Autograd engine is fully operational!")
