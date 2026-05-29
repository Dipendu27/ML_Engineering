from mlx_lm import load, generate
import time

print("--- Day 35: Native Apple MLX Text Generation ---\n")

# 1. Load the Model and Tokenizer natively into MLX
# We use the 'mlx-community' repository on Hugging Face.
# These are identical models, but their weights have been pre-converted from PyTorch to MLX format.
model_id = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
print(f"🚀 Loading MLX Model: {model_id}...")
print("   (Because this uses unified memory, loading is incredibly fast!)\n")

# The load() function handles both the neural network and the tokenizer simultaneously
model, tokenizer = load(model_id)

# 2. Formulate the Prompt (Using the Day 33 Chat Template technique)
conversation = [
    {"role": "system", "content": "You are a concise AI engineering tutor. Answer in one sentence under 25 words."},
    {"role": "user", "content": "Explain why Apple MLX is useful for running local AI models on a Mac."}
]

print("🗣️ User Prompt: Explain why Apple MLX is useful for running local AI models on a Mac.")

# Apply the exact chat template Qwen was trained on
prompt = tokenizer.apply_chat_template(
    conversation,
    tokenize=False,
    add_generation_prompt=True
)

# 3. Native MLX Generation
print("\n⚙️ Generating Response via Apple MLX Engine...")
start_time = time.time()

# The mlx_lm.generate function is purpose-built for M-series chips
response = generate(
    model,
    tokenizer,
    prompt=prompt,
    max_tokens=60,
    verbose=False # We handle printing manually
)

generation_time = time.time() - start_time
# Rough calculation of tokens per second (assuming ~1 word = ~1.3 tokens)
estimated_tokens = len(response.split()) * 1.3
tps = estimated_tokens / generation_time

print("\n🤖 AI Response:")
response_text = response.strip()
if "." in response_text:
    response_text = response_text.split(".", 1)[0].strip() + "."
print(response_text)
print("-" * 50)
print(f"⏱️ Generation Speed: {generation_time:.2f} seconds")
print(f"⚡️ Speed Metric: ~{tps:.1f} Tokens Per Second (Extremely fast for local hardware!)")
print("💡 Notice: Your Mac didn't sound like a jet engine, because MLX didn't spike the memory overhead!")
