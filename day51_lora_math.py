print("--- Day 51: The Math Behind LoRA (Low-Rank Adaptation) ---\n")

# 1. The Standard Neural Network Layer (Dense Matrix)
# Imagine a single self-attention layer inside Llama-3
input_dim = 4096
output_dim = 4096

# Calculate standard parameters (Weights + Biases)
standard_params = (input_dim * output_dim) + output_dim
print(f"🧠 Standard Layer Parameters (Full Fine-Tuning): {standard_params:,}")
print("   (Updating this requires massive amounts of VRAM for gradients and optimizer states.)\n")

# 2. The LoRA Injection (The Adapter)
# We freeze the 16.7 million weights above.
# We then inject two TINY matrices: Matrix A and Matrix B.
# The "Rank" (r) determines how wide these matrices are. r=8 is standard for language.
lora_rank = 8

# Matrix A compresses the input (4096 dimensions -> down to 8)
lora_matrix_a_params = input_dim * lora_rank

# Matrix B decompresses the signal (8 dimensions -> back to 4096)
lora_matrix_b_params = lora_rank * output_dim

total_lora_params = lora_matrix_a_params + lora_matrix_b_params
print(f"🧩 LoRA Adapter Parameters (r={lora_rank}): {total_lora_params:,}")

# 3. The Efficiency Calculation
reduction_percentage = (total_lora_params / standard_params) * 100

print("\n" + "=" * 50)
print(f"📉 PARAMETER REDUCTION: We are only training {reduction_percentage:.2f}% of the weights!")
print("=" * 50)
print("\n💡 THE ML ENGINEER TAKEAWAY:")
print("By freezing the base model and only training the LoRA adapters, we reduced the")
print("computational workload by 99.6%. This is the exact mathematical breakthrough")
print("that allows you to fine-tune an 8-Billion parameter LLM natively on a 16GB Mac!")