from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors

print("--- Day 40: Advanced Inference Optimization (Final) ---\n")

# 1. Load the Quantized Model
model_id = "mlx-community/Meta-Llama-3-8B-Instruct-4bit"
print(f"🚀 Loading {model_id} into Unified Memory...")
model, tokenizer = load(model_id)

# 2. Initialize Conversation History
conversation_history = [
    {"role": "system", "content": "You are a clinical AI assistant. Answer in 2-3 sentences max."}
]

# 3. Configure optimized generation controls once, then reuse them every turn
sampler = make_sampler(temp=0.7, top_p=0.9)
logits_processors = make_logits_processors(repetition_penalty=1.15)

print("\n✅ Highly-Tuned AI Assistant Ready. (Type 'quit' or 'exit' to end)")
print("-" * 50)

# 4. The Continuous Inference Loop
while True:
    user_input = input("\n🧑‍⚕️ You: ")
    if user_input.lower() in ['quit', 'exit']:
        print("Ending session. Goodbye!")
        break
    if not user_input.strip():
        continue

    conversation_history.append({"role": "user", "content": user_input})
    prompt = tokenizer.apply_chat_template(
        conversation_history,
        tokenize=False,
        add_generation_prompt=True
    )

    print("🤖 AI: ", end="", flush=True)

    full_response = ""

    # Pass kwargs directly into the MLX stream generator
    generator = stream_generate(
        model,
        tokenizer,
        prompt,
        max_tokens=250,
        sampler=sampler,
        logits_processors=logits_processors
    )

    for response_piece in generator:
        chunk = response_piece.text

        # The Bug Fix! Break the loop if the AI tries to print its stop token
        if "<|eot_id|>" in chunk or "<|end_of_text|>" in chunk:
            break

        print(chunk, end="", flush=True)
        full_response += chunk

    print()  # New line after the stream finishes

    conversation_history.append({"role": "assistant", "content": full_response})

    # Context Window Management
    if len(conversation_history) > 11:
        print("\n[⚠️ System: Approaching Context Limit. Dropping oldest chat turn...]")
        del conversation_history[1:3]
