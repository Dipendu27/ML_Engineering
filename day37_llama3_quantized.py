from mlx_lm import load, generate
import time

print("--- Day 37: Pre-Quantized Llama-3 Inference via Apple MLX ---\n")

# 1. Target the Community-Quantized Model
# This is Meta's 8-Billion parameter model, compressed to 4-bit.
# It requires about 4.5GB of RAM to load, fitting easily into a 16GB Mac.
model_id = "mlx-community/Meta-Llama-3-8B-Instruct-4bit"

# 🛑 ML ENGINEER FALLBACK 🛑
# Note: Meta requires you to digitally "sign" their terms on Hugging Face to access Llama-3.
# If you get a "401 Unauthorized" error, uncomment the Mistral model below to bypass it!
# model_id = "mlx-community/Mistral-7B-Instruct-v0.2-4bit"

print(f"🚀 Downloading & Loading {model_id}...")
print("   (This is ~4.5GB. It may take a few minutes on the first run.)\n")

model, tokenizer = load(model_id)

# 2. Formulate the Medical Prompt
# Notice we are asking a highly complex reasoning question, not just a fact retrieval.
conversation = [
    {"role": "system", "content": "You are an expert clinical diagnostician. Provide concise, highly technical answers."},
    {"role": "user", "content": "A 45-year-old patient presents with sudden onset sharp chest pain that worsens with deep inspiration and improves when leaning forward. ECG shows diffuse ST elevations. What is the most likely diagnosis?"}
]

print("🗣️ Doctor's Prompt: Analyzing chest pain symptoms...")

prompt = tokenizer.apply_chat_template(
    conversation,
    tokenize=False,
    add_generation_prompt=True
)

# 3. Native MLX Generation
print("\n⚙️ Generating Diagnostic Report (M-Series GPU)...\n")
start_time = time.time()

# We pass 'verbose=True' so we can watch the AI stream the text in real-time
response = generate(
    model,
    tokenizer,
    prompt=prompt,
    max_tokens=250,
    verbose=True
)

generation_time = time.time() - start_time
print("\n" + "-" * 50)
print(f"⏱️ Generation Speed: {generation_time:.2f} seconds")
print("💡 Notice: An 8-Billion parameter model just ran flawlessly on your laptop because of 4-bit quantization!")
