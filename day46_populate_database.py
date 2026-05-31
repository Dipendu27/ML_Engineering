from pathlib import Path
import warnings

import torch
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

warnings.filterwarnings("ignore")

print("--- Day 46: Populating ChromaDB (RAG Ingestion Pipeline) ---\n")

pdf_path = Path("patient_discharge.pdf")
db_dir = Path("clinical_rag_db")

if not pdf_path.exists():
    print(f"❌ Error: Could not find '{pdf_path}'. Please run Day 42 first!")
    raise SystemExit(1)

# ---------------------------------------------------------
# Step 1: Ingest & Chunk (From Days 42 & 43)
# ---------------------------------------------------------
print("📥 Loading and chunking clinical PDF...")
loader = PyPDFLoader(str(pdf_path))
raw_docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=30)
chunks = text_splitter.split_documents(raw_docs)
print(f"✅ Generated {len(chunks)} semantic chunks.")

# ---------------------------------------------------------
# Step 2: Connect to the Database (From Day 45)
# ---------------------------------------------------------
print("\n🧠 Initializing BGE-Small Embedding Engine...")
device = "mps" if torch.backends.mps.is_available() else "cpu"
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": device},
    encode_kwargs={"normalize_embeddings": True}
)
print(f"✅ Embedding device: {device}")

print("📂 Connecting to ChromaDB Storage...")
vectorstore = Chroma(
    collection_name="patient_records",
    embedding_function=embeddings,
    persist_directory=str(db_dir)
)

# ---------------------------------------------------------
# Step 3: Secure Insertion (Deduplication via Chunk IDs)
# ---------------------------------------------------------
print("\n🔐 Generating unique IDs to prevent duplicates...")
chunk_ids = []

for idx, chunk in enumerate(chunks):
    # Extract metadata safely
    source = chunk.metadata.get("source", "unknown_file")
    page = chunk.metadata.get("page", 0)

    # Create a deterministic ID: "filename.pdf:page:chunk_index"
    unique_id = f"{source}:page_{page}:chunk_{idx}"
    chunk_ids.append(unique_id)

print("⚡️ Vectorizing and writing to disk (M-Series GPU)...")

# add_documents handles the batch embedding AND the database insertion in one move!
vectorstore.add_documents(documents=chunks, ids=chunk_ids)

# ---------------------------------------------------------
# Step 4: Verification
# ---------------------------------------------------------
# .get() pulls the underlying data so we can verify the DB size
current_db_size = len(vectorstore.get()["ids"])

print("-" * 50)
print(f"🎉 SUCCESS! Total chunks currently in ChromaDB: {current_db_size}")
print("-" * 50)

print("\n💡 THE ML ENGINEER TAKEAWAY:")
print("Run this script a second time. You will notice the DB size does NOT increase.")
print("Because we generated explicit IDs, Chroma recognized the data was already there")
print("and updated it instead of duplicating it. Your RAG pipeline is now robust!")
