from transformers import AutoTokenizer
import warnings

warnings.filterwarnings("ignore")

print("--- Day 33: LLM Prompt Formatting & Chat Templates ---\n")

# 1. Define our standard Conversation Array
# This is the exact format the OpenAI API uses. It is clean and readable.
conversation = [
    {"role": "system", "content": "You are a highly precise medical AI assistant."},
    {"role": "user", "content": "What is the recommended daily intake of Vitamin D for adults?"}
]

print("📝 Standard Conversation Input:")
for msg in conversation:
    print(f"[{msg['role'].upper()}]: {msg['content']}")
print("-" * 50)

# ---------------------------------------------------------
# 2. Formatting for Meta's Llama-3
# ---------------------------------------------------------
print("🚀 Simulating Llama-3 Tokenizer Format...")
# We load the tokenizer for Llama-3 (does not download the massive model, just the dictionary)
llama3_tokenizer = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3-8B-Instruct")

# The Magic Function: apply_chat_template
# tokenize=False returns the raw string so we can read it, instead of PyTorch Tensors.
llama3_prompt = llama3_tokenizer.apply_chat_template(
    conversation,
    tokenize=False,
    add_generation_prompt=True # Tells the AI to add the final 'assistant' tag so it knows it's its turn to speak
)

print("\n🔍 RAW LLAMA-3 STRING (This is what the GPU actually processes):")
print(llama3_prompt)


# ---------------------------------------------------------
# 3. Formatting for Mistral v0.1
# ---------------------------------------------------------
print("\n" + "-" * 50)
print("🚀 Simulating Mistral Tokenizer Format...")
mistral_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

mistral_prompt = mistral_tokenizer.apply_chat_template(
    conversation,
    tokenize=False,
    add_generation_prompt=True
)

print("\n🔍 RAW MISTRAL STRING (Notice how completely different this is!):")
print(mistral_prompt)

# 4. The ML Engineer Takeaway
print("\n" + "-" * 50)
print("💡 THE TAKEAWAY:")
print("If you send the Mistral string into the Llama-3 model, it will hallucinate.")
print("Always use `tokenizer.apply_chat_template()` to ensure your AI gets the exact syntax it was trained on!")
