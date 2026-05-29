import subprocess
import sys
import time

print("--- Day 36: Local MLX Model Quantization (16-bit to 4-bit) ---\n")

# We use the original, un-quantized PyTorch model from the creators of TinyLlama
hf_repo = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print(f"📦 Source Model: {hf_repo} (Standard 16-bit PyTorch weights)")
print("🗜️ Target: Compressing to 4-bit MLX natively...\n")

start_time = time.time()

# The mlx_lm.convert CLI tool automatically:
# 1. Downloads the 16-bit PyTorch model
# 2. Converts it into Apple's native safetensors format
# 3. Quantizes the weights to 4-bit (reducing size by 75%)
# 4. Saves it locally to a folder named 'mlx_model'
print("⚙️ Running MLX Conversion & Quantization Engine...")
try:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "mlx_lm",
            "convert",
            "--hf-path",
            hf_repo,
            "--mlx-path",
            "mlx_model",
            "-q",  # This flag triggers 4-bit quantization
        ],
        check=True,
    )

    print(f"\n✅ Quantization Complete in {time.time() - start_time:.1f} seconds!")
    print("-" * 50)
    print("💡 THE ML ENGINEER TAKEAWAY:")
    print("Look at your VS Code file explorer. You now have a new folder named 'mlx_model'.")
    print("Inside that folder are your customized, 4-bit compressed Apple Silicon weights.")
    print("You just compressed a neural network to use 75% less RAM without losing its intelligence!")

except subprocess.CalledProcessError:
    print("\n❌ Error: Quantization failed. Ensure mlx-lm is installed and you have internet access.")
