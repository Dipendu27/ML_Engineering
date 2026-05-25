import torch
# AutoClasses automatically detect the architecture of the model you are downloading
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import warnings

# Suppress minor warnings for clean terminal output
warnings.filterwarnings("ignore")

print("--- Day 26: Hugging Face & Pre-Trained Transformers ---\n")

# ---------------------------------------------------------
# Part 1: The Easy Way (The Pipeline API)
# ---------------------------------------------------------
print("🚀 Part 1: The High-Level Pipeline API")
# This automatically downloads a lightweight classification model (~250MB)
# and sets up all the PyTorch infrastructure in the background.
analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

patient_note_1 = "The patient is responding wonderfully to the new hypertension medication and feels great."
patient_note_2 = "Patient complains of severe chest pain and dizziness. Immediate intervention required."

print(f"Note 1: '{patient_note_1}'")
print(f"AI Analysis: {analyzer(patient_note_1)}\n")

print(f"Note 2: '{patient_note_2}'")
print(f"AI Analysis: {analyzer(patient_note_2)}\n")


# ---------------------------------------------------------
# Part 2: The ML Engineer Way (Under the Hood)
# ---------------------------------------------------------
print("⚙️ Part 2: Under the Hood (Manual Tokenization & Inference)")

model_name = "distilbert-base-uncased-finetuned-sst-2-english"

# A. The Tokenizer: A Transformer cannot read letters. It only reads ID numbers.
# The tokenizer chops words into pieces and assigns them their specific ID from the model's dictionary.
tokenizer = AutoTokenizer.from_pretrained(model_name)

# B. The Model: The actual PyTorch Neural Network architecture
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# 1. Tokenize the text (return_tensors='pt' tells it to output PyTorch Tensors!)
raw_text = "Patient feels highly fatigued today."
inputs = tokenizer(raw_text, return_tensors="pt")

print(f"Original Text: {raw_text}")
print("🔢 Tokenized PyTorch Tensor (This is what the AI actually 'sees'):")
print(inputs['input_ids'], "\n")

# 2. Run the Forward Pass
with torch.no_grad(): # We are just predicting, not training
    outputs = model(**inputs) # The ** unpacks the dictionary directly into the model

# 3. View the Raw Output (Logits)
# Logits are the raw, un-squished mathematical output of the neural network's final layer.
print("🧠 Raw Model Output (Logits):")
print(outputs.logits)

# 4. Convert Logits to Probabilities (using Softmax, just like yesterday!)
probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(f"\n📊 Final Probabilities [Negative, Positive]:")
print(probabilities.round(decimals=4))
