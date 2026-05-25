import torch
import torch.nn.functional as F

print("--- Day 25: Transformers & Self-Attention (From Scratch) ---\n")

# 1. Simulate an Embedded Sentence
# Imagine a 3-word sentence: "Apple is sweet"
# In NLP, words are converted to arrays of numbers (Embeddings)
word_apple = [1.0, 0.0, 0.5, 0.2]
word_is    = [0.0, 1.0, 0.0, 0.0]
word_sweet = [0.5, 0.0, 1.0, 0.2]

# Stack them into a sequence matrix (3 words, 4 dimensions each)
sequence = torch.tensor([word_apple, word_is, word_sweet])
print("📥 Input Sentence Matrix (3 words):")
print(sequence)
print(f"Shape: {sequence.shape}\n")

# 2. Queries, Keys, and Values (Q, K, V)
# In a real Transformer, the AI learns weights to create Q, K, and V.
# For basic intuition, we'll use the raw sequence for all three.
Q = sequence # Query: What is this word looking for?
K = sequence # Key: What does this word offer?
V = sequence # Value: What is the actual mathematical payload of this word?

# 3. Step A: Calculate Raw Attention (The Dot Product)
# We multiply the Queries by the Keys to see how "similar" every word is to every other word.
# .transpose(0, 1) flips the matrix so the math aligns.
raw_attention = torch.matmul(Q, K.transpose(0, 1))

print("🧮 Raw Attention Scores (Matrix Multiplication):")
print(raw_attention)
print("-> Notice how 'Apple' (Row 0) and 'sweet' (Row 2) have a high mathematical crossover!\n")

# 4. Step B: Softmax (The Attention Weights)
# We squish the raw scores into percentages that add up to exactly 1.0 (100%)
attention_weights = F.softmax(raw_attention, dim=-1)

print("⚖️ Attention Weights (Softmax Percentages):")
print(attention_weights.round(decimals=2))
print("\n🔍 Look closely at Row 0 ('Apple').")
print("It pays 49% attention to itself, but ALSO pays 38% attention to 'sweet'.")
print("This is exactly how Transformers understand CONTEXT.\n")

# 5. Step C: Apply Attention to the Output
# We multiply our percentage weights by the original Values.
context_aware_output = torch.matmul(attention_weights, V)

print("📤 Final Context-Aware Output:")
print(context_aware_output.round(decimals=2))
print(f"Shape: {context_aware_output.shape}")
print("\n✅ The final output is still 3 words, but now the math inside 'Apple' has been permanently altered by the math inside 'sweet'!")
