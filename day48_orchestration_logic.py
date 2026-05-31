from pathlib import Path
import warnings

import torch
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

warnings.filterwarnings("ignore")

print("--- Day 48: RAG Orchestration & Prompt Injection ---\n")

db_dir = Path("clinical_rag_db")
if not db_dir.exists():
    print(f"❌ Error: Could not find '{db_dir}'. Please run Day 46 first!")
    raise SystemExit(1)

# 1. Connect to the Database
print("🧠 Loading Embedding Engine & Database...")
device = "mps" if torch.backends.mps.is_available() else "cpu"
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": device},
    encode_kwargs={"normalize_embeddings": True}
)
print(f"✅ Embedding device: {device}")

vectorstore = Chroma(
    collection_name="patient_records",
    embedding_function=embeddings,
    persist_directory=str(db_dir)
)

# 2. Initialize the LangChain Retriever
# The retriever abstracts the raw math search into a clean tool that simply returns documents
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # Get top 3 docs


# 3. The Orchestration Logic (Context Formatting)
def format_retrieved_documents(docs):
    """Takes raw Document objects and formats them into a clean string with citations."""
    formatted_text = ""
    for i, doc in enumerate(docs):
        # We manually inject the metadata so the LLM can read where the text came from
        formatted_text += f"[Document {i+1} | Source: {doc.metadata.get('source')} | Page: {doc.metadata.get('page')}]\n"
        formatted_text += f"{doc.page_content}\n\n"
    return formatted_text.strip()

# 4. The Master RAG Prompt Template
# This is the "System Instructions" that prevents hallucinations.
rag_prompt_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a highly precise clinical AI assistant.
Your only job is to answer the Doctor's Question using ONLY the pieces of clinical context provided below.
If you do not know the answer based strictly on the context, you must reply: "I do not have enough information."
Do not hallucinate or use outside knowledge. Always cite your source document.

--- RETRIEVED CLINICAL CONTEXT ---
{injected_context}
----------------------------------
<|eot_id|><|start_header_id|>user<|end_header_id|>
Doctor's Question: {user_question}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

# 5. Execute the Orchestration Pipeline
doctor_query = "What is the recommended treatment for a patient with acute pericarditis?"
print(f"🗣️ Doctor's Query: '{doctor_query}'\n")

print("⚙️ Step A: Retrieving documents from ChromaDB...")
retrieved_docs = retriever.invoke(doctor_query)

print("⚙️ Step B: Formatting raw chunks into a single context string...")
formatted_context = format_retrieved_documents(retrieved_docs)

print("⚙️ Step C: Injecting Context and Query into the Master Prompt...\n")
final_payload = rag_prompt_template.format(
    injected_context=formatted_context,
    user_question=doctor_query
)

# 6. View the Final Payload
print("=" * 50)
print("📦 FINAL PAYLOAD (The text we will send to Llama-3 tomorrow):")
print("=" * 50)
print(final_payload)
print("-" * 50)
print("💡 THE ML ENGINEER TAKEAWAY:")
print("Notice how the final payload contains the Llama-3 special tokens (<|start_header_id|>)!")
print("We just took chaotic vector math, raw PDF strings, and natural language, and orchestrated")
print("them into one structured, citation-aware, hallucination-resistant string.")
