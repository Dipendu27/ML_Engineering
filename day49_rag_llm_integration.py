from pathlib import Path
import warnings

import torch
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

warnings.filterwarnings("ignore")

print("--- Day 49: End-to-End RAG (Retrieval-Augmented Generation) ---\n")

db_dir = Path("clinical_rag_db")
if not db_dir.exists():
    print(f"❌ Error: Database not found. Please run Day 46 first!")
    raise SystemExit(1)

# ---------------------------------------------------------
# Step 1: Load the Retrieval Infrastructure (Phase 3 & 5)
# ---------------------------------------------------------
print("🧠 Loading Embedding Engine & Vector Database...")
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
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ---------------------------------------------------------
# Step 2: Load the Generative AI (Phase 4)
# ---------------------------------------------------------
model_id = "mlx-community/Meta-Llama-3-8B-Instruct-4bit"
print(f"🚀 Loading Generative AI: {model_id} into Unified Memory...")
llm_model, llm_tokenizer = load(model_id)

# ---------------------------------------------------------
# Step 3: Prompt Orchestration Functions (From Day 48)
# ---------------------------------------------------------
def format_docs(docs):
    return "\n\n".join(
        f"[Source: {doc.metadata.get('source')} | Page: {doc.metadata.get('page')}]\n{doc.page_content}"
        for doc in docs
    )


rag_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a precise clinical AI. Answer the Doctor's question using ONLY the provided clinical context.
If the context does not contain the answer, say "I do not have enough information."
Always cite the source document and page number in your response.

--- RETRIEVED CLINICAL CONTEXT ---
{context}
----------------------------------<|eot_id|><|start_header_id|>user<|end_header_id|>
Doctor's Question: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

# ---------------------------------------------------------
# Step 4: The RAG Execution Pipeline
# ---------------------------------------------------------
doctor_query = "What is the recommended treatment for a patient with acute pericarditis?"
print(f"\n🗣️ Doctor's Query: '{doctor_query}'\n")

print("🔍 Step A: Retrieving semantic chunks from ChromaDB...")
retrieved_chunks = retriever.invoke(doctor_query)
formatted_context = format_docs(retrieved_chunks)

print("🧩 Step B: Injecting context into Llama-3 Prompt Template...")
final_prompt = rag_template.format(context=formatted_context, question=doctor_query)

print("⚡️ Step C: Generating Answer (M5 Neural Engine)...\n")
print("🤖 AI: ", end="", flush=True)

# We use a very low temperature (0.1) so the AI acts like a strict robot, not a creative writer
sampler = make_sampler(temp=0.1)

generator = stream_generate(
    llm_model,
    llm_tokenizer,
    final_prompt,
    max_tokens=250,
    sampler=sampler
)

for response_piece in generator:
    chunk = response_piece.text
    if "<|eot_id|>" in chunk or "<|end_of_text|>" in chunk:
        break
    print(chunk, end="", flush=True)

print("\n\n" + "=" * 50)
print("💡 THE ML ENGINEER TAKEAWAY:")
print("You just built a fully functional Local RAG Pipeline!")
print("The AI didn't guess the answer from the internet. It read the specific PDF")
print("chunks retrieved by the Vector Database and summarized them natively on your Mac.")
