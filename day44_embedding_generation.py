import time
import warnings
from pathlib import Path

import torch
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

warnings.filterwarnings("ignore")

print("--- Day 44: LangChain Batch Embedding Generation ---\n")

pdf_path = Path("patient_discharge.pdf")
if not pdf_path.exists():
    print(f"❌ Error: Could not find '{pdf_path}'. Please run Day 42's script first!")
    raise SystemExit(1)

# ---------------------------------------------------------
# Step 1 & 2: Ingest and Chunk (Recap of Days 42 & 43)
# ---------------------------------------------------------
print("📥 Ingesting and Chunking PDF...")
loader = PyPDFLoader(str(pdf_path))
raw_documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
chunks = text_splitter.split_documents(raw_documents)
print(f"✅ Created {len(chunks)} semantic chunks.\n")

# ---------------------------------------------------------
# Step 3: Initialize the LangChain Embedding Model
# ---------------------------------------------------------
# We use BAAI/bge-small-en-v1.5 (The champion from our Day 29 benchmark!)
model_name = "BAAI/bge-small-en-v1.5"
print(f"🧠 Loading Embedding Model: '{model_name}'...")

# Route math to Apple Silicon GPU when available, otherwise fall back to CPU.
device = "mps" if torch.backends.mps.is_available() else "cpu"
model_kwargs = {"device": device}
# BGE-Small performs best when vectors are mathematically normalized (scaled to 1)
encode_kwargs = {"normalize_embeddings": True}

embeddings_engine = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
print(f"✅ Embedding device: {device}")

# ---------------------------------------------------------
# Step 4: Batch Vectorization
# ---------------------------------------------------------
print("\n⚡️ Firing M-Series GPU for Batch Vectorization...")
start_time = time.time()

# Extract just the raw text strings from our LangChain Document objects
text_contents = [chunk.page_content for chunk in chunks]

# embed_documents is LangChain's standard method for batch processing arrays of text
vector_embeddings = embeddings_engine.embed_documents(text_contents)

compute_time = time.time() - start_time

print("-" * 50)
print(f"✅ Successfully converted {len(vector_embeddings)} chunks into math in {compute_time:.3f} seconds!")
print("-" * 50)

# 5. Inspect the Mathematical Output
print("\n🔬 Inspecting Chunk 1's Embedding Vector:")
first_vector = vector_embeddings[0]

print(f"Vector Dimensions: {len(first_vector)} numbers (This maps to the 384D mathematical space)")
print(f"Vector Preview:   [{first_vector[0]:.4f}, {first_vector[1]:.4f}, {first_vector[2]:.4f}, ..., {first_vector[-1]:.4f}]")

print("\n💡 THE ML ENGINEER TAKEAWAY:")
print("Your documents are no longer just English text. They are now highly structured")
print("mathematical coordinates. Because we used 'normalize_embeddings=True', the")
print("distance between any two vectors can be perfectly calculated using Cosine Similarity!")
