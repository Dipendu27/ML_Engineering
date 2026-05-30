import mlx.core as mx
from mlx_lm import load, generate
import time

print("--- Day 39: M5 Neural Engine & Hardware Profiling ---\n")

# 1. Reset the Apple Metal memory trackers so we get a clean slate
mx.metal.reset_peak_memory()

# 2. Load the Model
model_id = "mlx-community/Meta-Llama-3-8B-Instruct-4bit"
print(f"🚀 Loading {model_id}...")
model, tokenizer = load(model_id)

# 3. Check memory JUST from loading the model (Idle State)
# get_active_memory() returns bytes, so we divide by (1024^3) to get Gigabytes
idle_mem_gb = mx.metal.get_active_memory() / (1024 ** 3)
print(f"📊 Metal VRAM Usage (Idle): {idle_mem_gb:.2f} GB")

# 4. Formulate the Prompt
prompt = "Explain the exact biological mechanics of mRNA vaccines in 3 sentences."
messages = [{"role": "user", "content": prompt}]
formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# 5. Stress the Neural Engine
print("\n⚙️ Generating text and stressing the M5 GPU...")
start_time = time.time()

# We hide verbose output so we can focus strictly on the hardware stats
response = generate(model, tokenizer, prompt=formatted_prompt, max_tokens=150, verbose=False)

gen_time = time.time() - start_time

# 6. Profile the Peak Hardware Usage during Generation
# This tells us how high the RAM spiked while doing the math
peak_mem_gb = mx.metal.get_peak_memory() / (1024 ** 3)

# This tells us how much memory the KV Cache (Context Window) took up
cache_mem_gb = mx.metal.get_cache_memory() / (1024 ** 3)

print(f"\n🤖 AI Output:\n{response}")
print("\n" + "=" * 50)
print("📈 APPLE SILICON (METAL) PROFILING REPORT:")
print("=" * 50)
print(f"⏱️ Total Generation Time: {gen_time:.2f} seconds")
print(f"🪫 Idle Model Footprint:  {idle_mem_gb:.2f} GB")
print(f"🗄️ KV Cache Allocation:   {cache_mem_gb:.2f} GB")
print(f"🔥 Peak VRAM Spiked To:  {peak_mem_gb:.2f} GB (Max memory used during calculation)")
print("-" * 50)
print("💡 THE ML ENGINEER TAKEAWAY:")
print("Notice that your Peak VRAM stayed well below your Mac's 16GB limit.")
print("If you had used the 16-bit PyTorch model, that Peak VRAM would have hit ~18GB,")
print("causing an immediate Out-Of-Memory (OOM) crash!")