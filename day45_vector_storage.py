from pathlib import Path

import torch
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import warnings

warnings.filterwarnings("ignore")

print("--- Day 45: Initializing the Persistent Vector Database ---\n")

# 1. Initialize the Embedding Engine
# The database needs this engine so it knows HOW to convert future text into math
model_name = "BAAI/bge-small-en-v1.5"
print(f"🧠 Loading Embedding Engine: '{model_name}'...")
device = "mps" if torch.backends.mps.is_available() else "cpu"
embeddings_engine = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs={"device": device},
    encode_kwargs={"normalize_embeddings": True}
)
print(f"✅ Embedding device: {device}")

# 2. Define the Persistent Storage Path
# This is the physical folder on your Mac where the database files will live
persist_directory = Path("clinical_rag_db")
print(f"\n📂 Storage Target: {persist_directory}")

# 3. Initialize the LangChain Vector Store (Infrastructure Setup)
print("⚙️ Initializing ChromaDB Storage Backend...")

# If the folder doesn't exist, Chroma creates it.
# If it does exist, Chroma connects to it.
vectorstore = Chroma(
    collection_name="patient_records",
    embedding_function=embeddings_engine,
    persist_directory=str(persist_directory)
)

# 4. Verify Database Setup
# .get() retrieves the underlying data structures. We check how many IDs exist.
existing_records = len(vectorstore.get()['ids'])

print("\n✅ Database Setup Complete!")
print(f"📊 Total documents currently in storage: {existing_records}")

print("\n" + "=" * 50)
print("💡 THE ML ENGINEER TAKEAWAY:")
print("Look at your VS Code File Explorer. A new folder named 'clinical_rag_db' just appeared!")
print("Inside, Chroma built a highly optimized SQLite database. Because we bound the")
print("HuggingFaceEmbeddings directly to the Chroma object, this database is now 'smart'.")
print("Tomorrow, we will stream our PDF chunks directly into it!")
