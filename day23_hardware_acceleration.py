import torch
import time

print("--- Day 23: Apple Silicon (MPS) Hardware Acceleration ---\n")

# 1. Verify MPS Availability
# We did a quick check on Day 1, but this is the standard PyTorch syntax you will use in every project.
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"✅ Hardware Accelerator Found: {device}")
else:
    print("❌ MPS not found. Defaulting to CPU.")
    device = torch.device("cpu")

print("-" * 50)

def synchronize_if_needed():
    if device.type == "mps":
        torch.mps.synchronize()


# 2. Define the massive matrix size
# 10,000 x 10,000 = 100 Million numbers per tensor!
matrix_size = 10000

print(f"Generating two {matrix_size}x{matrix_size} matrices... (This takes a few seconds)")
# Generate two massive tensors. By default, these are created on the CPU.
tensor_A_cpu = torch.rand(matrix_size, matrix_size)
tensor_B_cpu = torch.rand(matrix_size, matrix_size)

# 3. CPU Benchmark (The Slow Way)
print("\n🐌 Running Matrix Multiplication on the CPU...")
start_time = time.time()

# Perform the math
result_cpu = torch.matmul(tensor_A_cpu, tensor_B_cpu)

cpu_time = time.time() - start_time
print(f"⏱️ CPU Time: {cpu_time:.4f} seconds")

# 4. Moving Tensors to the GPU
# The .to(device) command is how we push data to the M5 chip
print("\n🚀 Pushing 200 Million numbers to the M5 GPU (MPS)...")
tensor_A_mps = tensor_A_cpu.to(device)
tensor_B_mps = tensor_B_cpu.to(device)

# 5. GPU Benchmark (The Fast Way)
print("⚡️ Running Matrix Multiplication on the MPS Engine...")

# Warm-up run (PyTorch often takes a millisecond to initialize the GPU connection on the very first run)
_ = torch.matmul(tensor_A_mps, tensor_B_mps)
synchronize_if_needed()

# The actual timed run
synchronize_if_needed()
start_time = time.time()

result_mps = torch.matmul(tensor_A_mps, tensor_B_mps)
synchronize_if_needed()

mps_time = time.time() - start_time
print(f"⏱️ MPS Time: {mps_time:.4f} seconds")

# 6. The Verdict
speedup = cpu_time / mps_time
print("-" * 50)
print(f"🏆 The Apple M5 GPU was {speedup:.2f}x faster than the CPU!")
