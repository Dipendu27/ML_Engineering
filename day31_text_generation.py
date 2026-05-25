import torch
from transformers import GenerationConfig, pipeline
import warnings

warnings.filterwarnings("ignore")

print("--- Day 31: Local LLM Text Generation ---\n")

# 1. Load the Model
# TinyLlama is small enough to run entirely in your Mac's memory.
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
print(f"🚀 Loading Generative Model: {model_id}")
print("   (This will download ~2GB of weights the first time you run it...)")

# We use the pipeline, but this time we explicitly route it to your M5 GPU (MPS)
device = "mps" if torch.backends.mps.is_available() else "cpu"

generator = pipeline(
    "text-generation",
    model=model_id,
    device=device
)

# 2. The Prompt Template
# Chat models require special "tags" so they know who is talking.
# Without these, the AI gets confused about whether it is the user or the assistant.
prompt = "<|system|>\nYou are a helpful medical AI assistant. Answer with exactly three short numbered items.</s>\n<|user|>\nCan you list three common symptoms of the seasonal flu?</s>\n<|assistant|>\n"

print("\n🗣️ User Prompt:")
print("Can you list three common symptoms of the seasonal flu?")

# 3. Generation (The Next-Word Engine)
print("\n⚙️ Generating Response on the M5 Chip (Deterministic Decoding)...")
prompt_token_count = len(generator.tokenizer(prompt)["input_ids"])
generation_config = GenerationConfig(
    max_length=prompt_token_count + 80,
    do_sample=False,
    pad_token_id=generator.tokenizer.eos_token_id,
    eos_token_id=generator.tokenizer.eos_token_id,
)

output = generator(
    prompt,
    generation_config=generation_config,
    return_full_text=False,             # Only return the AI's answer, hide our prompt
    clean_up_tokenization_spaces=False  # Keep BPE-tokenizer spacing behavior explicit
)

print("\n🤖 AI Response:")
raw_response = output[0]["generated_text"].strip()
numbered_items = []

for line in raw_response.splitlines():
    stripped_line = line.strip()
    if stripped_line.startswith(("1.", "2.", "3.")):
        numbered_items.append(stripped_line)
    if len(numbered_items) == 3:
        break

if len(numbered_items) == 3:
    response = "Here are three common symptoms of the seasonal flu:\n\n" + "\n".join(numbered_items)
else:
    response = raw_response

print(response)
