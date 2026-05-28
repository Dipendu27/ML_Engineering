from huggingface_hub import hf_hub_download
from llama_cpp import Llama
import time

print("--- Day 32: 4-Bit Quantization (GGUF & Llama.cpp) ---\n")

# 1. Download the Quantized Model
# We are downloading TinyLlama, but specifically the "Q4_K_M" version (4-bit Quantized)
# The standard model is ~2.2 GB. This quantized version is only ~600 MB!
repo_id = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
filename = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

print(f"📥 Downloading/Verifying 4-bit GGUF model: {filename}")
print("   (This is only ~600MB instead of 2.2GB!)\n")

model_path = hf_hub_download(repo_id=repo_id, filename=filename)

# 2. Load the Model into the Apple M5 Metal Engine
print("🚀 Loading Quantized Model into RAM...")
# n_gpu_layers=-1 tells it to offload 100% of the math to your Apple Silicon GPU
llm = Llama(
    model_path=model_path,
    n_gpu_layers=-1,
    n_ctx=2048, # Context window size (how many words it can remember)
    verbose=False # Hides the messy C++ backend logs
)

# 3. The Prompt Template (Same as yesterday)
prompt = (
    "<|system|>\n"
    "You are a concise AI engineering tutor. Answer in one sentence under 20 words.</s>\n"
    "<|user|>\n"
    "Why does quantizing model weights to 4 bits help run language models on a laptop?</s>\n"
    "<|assistant|>\n"
)
print("🗣️ User Prompt: Why does quantizing model weights to 4 bits help run language models on a laptop?\n")

# 4. Generate the Text
print("⚙️ Generating Response on the M5 GPU (4-Bit INT4)...")
start_time = time.time()

output = llm(
    prompt,
    max_tokens=40,
    temperature=0.0,
    stop=["</s>"],
    echo=False # Don't print the prompt back to us
)

generation_time = time.time() - start_time
response_text = output['choices'][0]['text'].strip()
if "." in response_text:
    response_text = response_text.split(".", 1)[0].strip() + "."

print("\n🤖 AI Response:")
print(response_text)
print("-" * 50)
print(f"⏱️ Generation Speed: {generation_time:.2f} seconds")
print("💡 Notice: The 4-bit model can still produce useful local text while using much less memory.")
