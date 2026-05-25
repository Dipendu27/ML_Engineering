import time
from sentence_transformers import SentenceTransformer

print("--- Day 29: Embedding Model Benchmarking ---\n")

# A batch of medical documents to process
medical_corpus = [
    "Patient presents with severe hypertension and acute chest pain.",
    "The individual has extremely high blood pressure and a history of cardiac events.",
    "Patient suffered a fractured femur in a fall from a ladder.",
    "Routine checkup shows normal vitals and healthy BMI.",
    "Follow-up scheduled for next Tuesday regarding the MRI results."
] * 100 # We multiply by 100 to simulate a batch of 500 documents

print(f"📊 Benchmarking payload: {len(medical_corpus)} medical documents.\n")

# ---------------------------------------------------------
# Model 1: all-MiniLM-L6-v2 (The Speed Demon)
# ---------------------------------------------------------
print("🚀 Loading Model 1: all-MiniLM-L6-v2...")
# SentenceTransformer automatically handles tokenization, PyTorch routing, and Mean Pooling!
model_minilm = SentenceTransformer('all-MiniLM-L6-v2')

print("⏱️  Generating Embeddings...")
start_time = time.time()
# Just one line of code to embed 500 documents!
embeddings_minilm = model_minilm.encode(medical_corpus)
time_minilm = time.time() - start_time

print(f"✅ MiniLM Time: {time_minilm:.4f} seconds")
print(f"📐 Vector Dimensions: {embeddings_minilm.shape[1]} numbers per document\n")


# ---------------------------------------------------------
# Model 2: BAAI/bge-small-en-v1.5 (The Accuracy Champion)
# ---------------------------------------------------------
print("🚀 Loading Model 2: BAAI/bge-small-en-v1.5...")
model_bge = SentenceTransformer('BAAI/bge-small-en-v1.5')

print("⏱️  Generating Embeddings...")
start_time = time.time()
embeddings_bge = model_bge.encode(medical_corpus)
time_bge = time.time() - start_time

print(f"✅ BGE Time: {time_bge:.4f} seconds")
print(f"📐 Vector Dimensions: {embeddings_bge.shape[1]} numbers per document\n")


# ---------------------------------------------------------
# The Verdict
# ---------------------------------------------------------
print("-" * 50)
print("🏆 BENCHMARK RESULTS:")
if time_minilm < time_bge:
    print(f"Fastest Model: all-MiniLM-L6-v2 ({(time_bge / time_minilm):.2f}x faster)")
else:
    print(f"Fastest Model: BGE-Small ({(time_minilm / time_bge):.2f}x faster)")

print("\n💡 ML Engineer Note:")
print("Both models output 384 dimensions. MiniLM is compact and widely used for fast baselines.")
print("BGE-Small is newer and currently performs very strongly on embedding benchmarks for its size.")
print("For our final RAG system, BGE-Small is the highly recommended choice!")
