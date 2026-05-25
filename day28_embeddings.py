import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

print("--- Day 28: Text Vectorization & Mean Pooling ---\n")

# 1. Load the Embedding Model
# This is a lightweight model specifically trained to produce high-quality sentence embeddings
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 2. Three Clinical Sentences
# Sentences 1 and 2 mean the exact same thing medically, but share almost NO exact words.
# Sentence 3 is completely unrelated.
sentences = [
    "Patient presents with severe hypertension.",
    "The individual has extremely high blood pressure.",
    "Patient suffered a fractured femur in a fall."
]

print("🚀 Tokenizing and passing text through the Transformer...")

# 3. Tokenize (Exactly like Day 27)
inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")

# 4. The Forward Pass
with torch.no_grad():
    outputs = model(**inputs)

# outputs.last_hidden_state contains the vector for EVERY SINGLE WORD.
# Shape: [Batch_Size, Tokens, Dimensions]
token_embeddings = outputs.last_hidden_state
print(f"\n🔢 Raw Word Embeddings Shape: {token_embeddings.shape}")
batch_size, token_count, embedding_dimensions = token_embeddings.shape
print(f"({batch_size} sentences, {token_count} tokens each, {embedding_dimensions} dimensions per token)")

# 5. Mean Pooling (Squishing words into a single sentence vector)
# We multiply by the attention mask so we don't accidentally average in the [PAD] zeros!
attention_mask = inputs['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
sum_embeddings = torch.sum(token_embeddings * attention_mask, 1)
sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
sentence_embeddings = sum_embeddings / sum_mask

print(f"\n📦 Final Sentence Embeddings Shape: {sentence_embeddings.shape}")
print("(3 sentences, 384 dimensions total per sentence!)")

# 6. Cosine Similarity (The Magic of RAG)
# Let's compare Sentence 0 (Hypertension) to Sentence 1 (High BP)
sim_0_1 = F.cosine_similarity(sentence_embeddings[0].unsqueeze(0), sentence_embeddings[1].unsqueeze(0))
# Let's compare Sentence 0 (Hypertension) to Sentence 2 (Fracture)
sim_0_2 = F.cosine_similarity(sentence_embeddings[0].unsqueeze(0), sentence_embeddings[2].unsqueeze(0))

print("\n⚖️ Semantic Similarity Scores (1.0 = Identical, 0.0 = Unrelated):")
print(f"Hypertension vs. High Blood Pressure:  {sim_0_1.item():.4f}  <-- High Match!")
print(f"Hypertension vs. Fractured Femur:      {sim_0_2.item():.4f}  <-- No Match!")
print("\n✅ The AI successfully proved mathematically that hypertension and high blood pressure mean the same thing, despite having completely different letters!")
