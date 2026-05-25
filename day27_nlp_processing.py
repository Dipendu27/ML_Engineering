import torch
from transformers import AutoTokenizer
import warnings

warnings.filterwarnings("ignore")

print("--- Day 27: Advanced NLP Processing & Tokenization ---\n")

# 1. Load the Tokenizer
# BERT is the foundational model for text embeddings and processing
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
print(f"✅ Loaded Tokenizer: {model_name}")

# 2. Messy, Real-World Medical Notes (Varying Lengths)
medical_notes = [
    "Patient has a mild headache.", # Short
    "Patient admitted to ER with severe chest pain, shortness of breath, and elevated heart rate. Administered aspirin and ordered an immediate ECG.", # Long
    "Follow-up scheduled." # Very short
]

print("\n🚀 Processing a batch of raw medical notes...")

# 3. The ML Engineer Tokenization Pipeline
# padding=True: Adds [PAD] tokens to the short sentences so they match the longest one in the batch.
# truncation=True: Cuts off text if it exceeds the model's max limit.
# return_tensors='pt': Outputs PyTorch tensors ready for the M5 GPU.
tokens = tokenizer(
    medical_notes,
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt"
)

# 4. Analyzing the Tensors
print("\n🔢 Input IDs (The padded numbers fed to the AI):")
print(tokens["input_ids"])
sequence_length = tokens["input_ids"].shape[1]
print(f"Shape: {tokens['input_ids'].shape} -> Notice how all 3 sentences are now exactly {sequence_length} tokens long!")

print("\n🎭 Attention Mask (Telling the AI what to ignore):")
print(tokens["attention_mask"])
print("-> 1 = Real Word | 0 = Ignore this (Padding)")

# 5. Decoding (Reverse Engineering)
# Let's translate the first (short) sentence back into English to see what the Tokenizer actually did.
print("\n🔍 Decoding Note 1 to see the hidden Special Tokens:")
decoded_note_1 = tokenizer.decode(tokens["input_ids"][0])
print(decoded_note_1)

print("\n🔍 Decoding Note 3 to see the hidden Special Tokens:")
decoded_note_3 = tokenizer.decode(tokens["input_ids"][2])
print(decoded_note_3)

print("\n-> Notice the [CLS] (Start of sentence), [SEP] (End of sentence), and [PAD] (Filler) tokens!")
