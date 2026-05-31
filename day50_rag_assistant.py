from pathlib import Path
import warnings

import torch
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

warnings.filterwarnings("ignore")


class LocalRAGAssistant:
    """
    A production-grade, object-oriented Local RAG Assistant optimized
    for Apple Silicon utilizing LangChain, ChromaDB, and MLX.
    """
    def __init__(self, db_directory: str, model_id: str):
        self.db_directory = Path(db_directory)
        self.model_id = model_id

        # Verify infrastructure paths
        if not self.db_directory.exists():
            raise FileNotFoundError(f"Database directory '{self.db_directory}' not found. Please run Day 46 first.")

        self._initialize_infrastructure()

    def _initialize_infrastructure(self):
        """Loads models and establishes database connections once during initialization."""
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"🧠 Loading BGE-Small Embedding Engine ({device})...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True}
        )

        print("📂 Connecting to Persistent ChromaDB Storage...")
        self.vectorstore = Chroma(
            collection_name="patient_records",
            embedding_function=self.embeddings,
            persist_directory=str(self.db_directory)
        )
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})

        print(f"🚀 Loading Generative AI [{self.model_id}] into Unified Memory...")
        self.model, self.tokenizer = load(self.model_id)
        print("✨ All systems initialized and ready for execution.\n")

    def _format_context(self, retrieved_docs) -> str:
        """Extracts text and constructs markdown-style citations from retrieved chunks."""
        formatted_chunks = []
        for doc in retrieved_docs:
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", 0)
            formatted_chunks.append(f"[Source: {source} | Page: {page}]\n{doc.page_content}")
        return "\n\n".join(formatted_chunks)

    def ask(self, question: str, max_tokens: int = 250) -> str:
        """Executes the complete RAG loop: Retrieval -> Prompt Formatting -> Streaming Generation."""
        # A. Semantic Search Retrieval
        retrieved_docs = self.retriever.invoke(question)
        context = self._format_context(retrieved_docs)

        # 🛠️ ML ENGINEER DEBUGGER: Let's see what ChromaDB actually found!
        print(f"\n[DEBUG - WHAT THE AI SEES]:\n{context}\n")

        # B. Construct Hallucination-Proof Prompt
        rag_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a precise clinical AI. Answer the Doctor's question using ONLY the provided clinical context.
If the context does not contain the answer, say "I do not have enough information."
Always cite the source document and page number in your response.

--- RETRIEVED CLINICAL CONTEXT ---
{context}
----------------------------------<|eot_id|><|start_header_id|>user<|end_header_id|>
Doctor's Question: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
        final_prompt = rag_template.format(context=context, question=question)

        # C. Optimized Streaming Inference Execution
        sampler = make_sampler(temp=0.1)  # Deterministic, factual decoding
        generator = stream_generate(
            self.model,
            self.tokenizer,
            final_prompt,
            max_tokens=max_tokens,
            sampler=sampler
        )

        full_response = ""
        print("🤖 AI: ", end="", flush=True)
        for response_piece in generator:
            chunk = response_piece.text
            # Stop string guardrails
            if "<|eot_id|>" in chunk or "<|end_of_text|>" in chunk:
                break
            print(chunk, end="", flush=True)
            full_response += chunk
        print()  # Clean newline string terminal tracking
        return full_response

# ---------------------------------------------------------
# 4. Continuous Interactive Execution Loop
# ---------------------------------------------------------
if __name__ == "__main__":
    print("--- Day 50: Production-Ready Local RAG Framework ---\n")

    DB_PATH = "clinical_rag_db"
    MODEL = "mlx-community/Meta-Llama-3-8B-Instruct-4bit"

    try:
        # Initialize the production class instance (Heavy lifting occurs here ONCE)
        assistant = LocalRAGAssistant(db_directory=DB_PATH, model_id=MODEL)

        print("🏥 Local Clinical Assistant Active. Ask your questions below.")
        print("(Type 'exit' or 'quit' to terminate session)")
        print("-" * 60)

        while True:
            query = input("\n🧑‍⚕️ Enter Question: ")
            if query.lower() in ["exit", "quit"]:
                print("Terminating clinical AI instance. Goodbye!")
                break
            if not query.strip():
                continue

            # Execute instant search and generation via memory-pinned models
            assistant.ask(question=query)

    except Exception as e:
        print(f"\n❌ Initialization Error: {e}")
