import warnings
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

warnings.filterwarnings("ignore")

print("--- Day 43: Semantic Document Chunking & Overlap ---\n")

pdf_path = Path("patient_discharge.pdf")

if not pdf_path.exists():
    print(f"❌ Error: Could not find '{pdf_path}'. Please run Day 42's script first!")
    raise SystemExit(1)

# 1. Ingest the PDF (From Day 42)
print("📥 Ingesting PDF Document...")
loader = PyPDFLoader(str(pdf_path))
raw_documents = loader.load()

print(f"📄 Loaded {len(raw_documents)} raw pages.")

# 2. Configure the Splitter
# In production, chunk_size is usually ~500-1000.
# We use 100 here to aggressively force splitting so we can observe the overlap.
chunk_size = 100
chunk_overlap = 20

print(f"✂️ Initializing Recursive Splitter (Size: {chunk_size}, Overlap: {chunk_overlap})...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    length_function=len,
    separators=["\n\n", "\n", " ", ""],  # Paragraph -> line -> word -> char
)

# 3. Execute the Splitting Pipeline
print("⚙️ Splitting documents into semantic chunks...\n")
# Notice we pass the full array of Document objects. It automatically handles the iteration.
chunked_documents = text_splitter.split_documents(raw_documents)

print("-" * 50)
print(f"✅ Splitting Complete! The {len(raw_documents)} pages became {len(chunked_documents)} chunks.")
print("-" * 50)

# 4. Inspect the Resulting Chunks
for i in range(min(4, len(chunked_documents))):  # Print first 4 chunks
    chunk = chunked_documents[i]
    print(f"\n🧩 CHUNK {i+1} | Source: {chunk.metadata['source']} (Page {chunk.metadata['page']})")
    print(f"Length: {len(chunk.page_content)} chars")
    print(f"Content: '{chunk.page_content}'")

print("\n" + "=" * 50)
print("💡 THE ML ENGINEER TAKEAWAY:")
print("1. Chunk overlap protects local context when a longer section has to be split across boundaries.")
print("2. Notice the metadata. LangChain perfectly copied the 'page' metadata from the parent document")
print("   into every single child chunk. You never lose your citation tracking!")
