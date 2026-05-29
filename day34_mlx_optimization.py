import torch
import mlx.core as mx
import time

print("--- Day 34: Apple MLX vs PyTorch Overhead & Lazy Evaluation ---\n")

matrix_size = 10000
torch_device = "mps" if torch.backends.mps.is_available() else "cpu"


def synchronize_torch_if_needed():
    if torch_device == "mps":
        torch.mps.synchronize()


print(f"📐 Target: Multiplying three {matrix_size} x {matrix_size} matrices (100 Million numbers each).")
print("This usually causes massive memory spikes in standard frameworks.\n")

# ---------------------------------------------------------
# Part 1: PyTorch (Eager Evaluation)
# ---------------------------------------------------------
print("🔴 Running PyTorch (Eager Evaluation)...")
synchronize_torch_if_needed()
start_time = time.time()

# PyTorch allocates this directly into memory the exact millisecond this line runs
pt_A = torch.rand(matrix_size, matrix_size, device=torch_device)
pt_B = torch.rand(matrix_size, matrix_size, device=torch_device)
pt_C = torch.rand(matrix_size, matrix_size, device=torch_device)

# PyTorch immediately calculates step 1, stores it in RAM, then calculates step 2
pt_result = torch.matmul(torch.matmul(pt_A, pt_B), pt_C)
synchronize_torch_if_needed()

pytorch_time = time.time() - start_time
print(f"⏱️ PyTorch blocked your code for: {pytorch_time:.4f} seconds while it calculated.")
print("   (It spiked your RAM during this entire time!)\n")


# ---------------------------------------------------------
# Part 2: Apple MLX (Lazy Evaluation)
# ---------------------------------------------------------
print("🟢 Running Apple MLX (Lazy Evaluation)...")
start_time = time.time()

# MLX arrays. Notice we don't need .to("device"). It natively uses the unified GPU.
mx_A = mx.random.uniform(shape=(matrix_size, matrix_size))
mx_B = mx.random.uniform(shape=(matrix_size, matrix_size))
mx_C = mx.random.uniform(shape=(matrix_size, matrix_size))

# We chain the matrix multiplications (@ is the python operator for matmul)
mx_result = (mx_A @ mx_B) @ mx_C

mlx_execution_time = time.time() - start_time
print(f"⏱️ MLX 'executed' the code in: {mlx_execution_time:.6f} seconds!")
print("   Wait, how is it nearly instant? Because it didn't actually do the math yet!")
print("   It just built a lightweight 'Computation Graph' in memory.\n")

# ---------------------------------------------------------
# Part 3: Forcing the MLX Evaluation
# ---------------------------------------------------------
print("🟢 Forcing MLX to actually compute the result (mx.eval)...")
start_time = time.time()

# mx.eval() forces the GPU to run the graph it built.
# Because it knows the entire graph in advance, it optimizes the memory perfectly,
# skipping unnecessary temporary storage that PyTorch would have created.
mx.eval(mx_result)

mlx_compute_time = time.time() - start_time
print(f"⏱️ MLX actual computation time: {mlx_compute_time:.4f} seconds.")

print("-" * 50)
print("💡 THE ML ENGINEER TAKEAWAY:")
print("By using Lazy Evaluation, MLX allows you to queue up massive Neural Network layers")
print("without immediately crashing your computer's RAM. It optimizes the whole pipeline")
print("before allocating a single byte of memory. This is how we run LLMs locally on Macs!")
