from mlx_lm import load, stream_generate

print("--- Day 38: Robust Local Inference & Chat Streaming (Final) ---\n")

# 1. Load the Quantized Model (from Day 37)
model_id = "mlx-community/Meta-Llama-3-8B-Instruct-4bit"
print(f"🚀 Loading {model_id} into Unified Memory...")
model, tokenizer = load(model_id)

# 2. Initialize the Conversation History
# We start the array with the System Prompt. This will persist for the entire session.
conversation_history = [
    {"role": "system", "content": "You are a concise, highly intelligent medical assistant. Answer in 2-3 sentences max."}
]

print("\n✅ AI Assistant Ready. (Type 'quit' or 'exit' to end the session)")
print("-" * 50)

# 3. The Continuous Inference Loop
while True:
    # A. Get User Input
    user_input = input("\n🧑‍⚕️ You: ")

    if user_input.lower() in ['quit', 'exit']:
        print("Ending session. Goodbye!")
        break

    if not user_input.strip():
        continue

    # B. Append User Message to History
    conversation_history.append({"role": "user", "content": user_input})

    # C. Apply the Chat Template to the ENTIRE history
    prompt = tokenizer.apply_chat_template(
        conversation_history,
        tokenize=False,
        add_generation_prompt=True
    )

    print("🤖 AI: ", end="", flush=True)

    # D. Stream the Output
    full_response = ""
    for response_piece in stream_generate(model, tokenizer, prompt, max_tokens=250):
        # FIX 1: Extract the text string from the MLX GenerationResponse object
        chunk = response_piece.text

        # FIX 2: Catch Llama-3's hidden stop tokens to prevent runaway generation!
        if "<|eot_id|>" in chunk or "<|end_of_text|>" in chunk:
            break

        print(chunk, end="", flush=True)
        full_response += chunk

    print()  # New line after the stream finishes

    # E. Append the AI's Response to the History
    # If we don't do this, the AI won't remember what it just said!
    conversation_history.append({"role": "assistant", "content": full_response})

    # F. Context Window Management (Basic Guardrail)
    # If the history gets too long, we drop the oldest messages (keeping the system prompt)
    if len(conversation_history) > 11:  # System + 5 turns (user/ai pairs)
        print("\n[⚠️ System: Approaching Context Limit. Dropping oldest chat turn...]")
        # Keep index 0 (System), delete index 1 and 2 (Oldest User/AI pair)
        del conversation_history[1:3]
