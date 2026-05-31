from pathlib import Path
import warnings

import torch
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

warnings.filterwarnings("ignore")

print("--- Day 47: Semantic Search & Vector Retrieval ---\n")

db_dir = Path("clinical_rag_db")

if not db_dir.exists():
    print(f"❌ Error: Could not find '{db_dir}'. Please run Day 46 first!")
    raise SystemExit(1)

# 1. Initialize the Embedding Engine
# CRITICAL: You MUST use the exact same embedding model you used for ingestion.
# If you embed the query with a different model, it will plot the coordinates on a completely different map!
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"🧠 Loading BGE-Small Embedding Engine ({device})...")
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": device},
    encode_kwargs={"normalize_embeddings": True}
)
print(f"✅ Embedding device: {device}")

# 2. Connect to the Populated Database
print("📂 Connecting to ChromaDB Storage...")
vectorstore = Chroma(
    collection_name="patient_records",
    embedding_function=embeddings,
    persist_directory=str(db_dir)
)

# 3. The Natural Language Query
# Notice we are asking about 'treatment', but the text might say 'administered' or 'prescribed'.
doctor_query = "What is the recommended treatment for a patient with acute pericarditis?"
print(f"\n🗣️ Doctor's Query: '{doctor_query}'")

# 4. Execute Semantic Search (K-Nearest Neighbors)
print("🔍 Searching for the top 3 most relevant clinical chunks...")
# k=3 tells the database to return the 3 closest mathematical neighbors.
# with_score returns the actual distance metric so we can see how confident the AI is.
results = vectorstore.similarity_search_with_score(doctor_query, k=3)

# 5. Parse and Display the Results
print("\n" + "=" * 50)
print("🏆 RETRIEVAL RESULTS:")
print("=" * 50)

for i, (doc, score) in enumerate(results):
    # Because we normalized our embeddings, this score represents Cosine Distance.
    # Lower distance = closer match! (e.g., 0.2 is an incredible match, 0.9 is unrelated)
    print(f"\n🥇 MATCH #{i+1} (Distance Score: {score:.4f})")
    print(f"📄 Citation: {doc.metadata.get('source')} | Page: {doc.metadata.get('page')}")
    print(f"📝 Content: '{doc.page_content}'")

print("\n" + "-" * 50)
print("💡 THE ML ENGINEER TAKEAWAY:")
print("You just bypassed traditional keyword search! The database successfully retrieved")
print("clinically relevant chunks, including the paragraph about NSAIDs and Colchicine,")
print("because the vectors were close to each other in mathematical space. It also")
print("preserved the page metadata for citations.")
