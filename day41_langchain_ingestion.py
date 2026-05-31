from pathlib import Path

from langchain_community.document_loaders import TextLoader

print("--- Day 41: LangChain Document Ingestion Pipeline ---\n")

# 1. Define the path to our raw data file
file_path = Path("clinical_guidelines.txt")

if not file_path.exists():
    print(f"❌ Error: Could not find '{file_path}'. Please create it first!")
    raise SystemExit(1)

# 2. Initialize the LangChain Loader
# In the real world, you might use S3DirectoryLoader or GoogleDriveLoader here.
# The beauty of LangChain is that the pipeline stays the same regardless of the source!
print("📥 Initializing LangChain TextLoader...")
loader = TextLoader(str(file_path), encoding="utf-8")

# 3. Execute the Ingestion
# .load() reads the file and converts it into a standardized LangChain 'Document' array
print("⚙️ Ingesting data into memory...\n")
documents = loader.load()

# 4. Inspect the Resulting Objects
print("-" * 50)
print(f"✅ Successfully ingested {len(documents)} document(s).")
print("-" * 50)

# Extract the first (and only) document we loaded
doc = documents[0]

# LangChain Documents have two critical attributes: page_content and metadata
print("📄 METADATA (Crucial for filtering databases later):")
print(doc.metadata)

print("\n📝 RAW PAGE CONTENT:")
# We print just the first 150 characters to verify ingestion
print(f"{doc.page_content[:150]}...\n")

print("-" * 50)
print("💡 THE ML ENGINEER TAKEAWAY:")
print("Notice how LangChain automatically created the 'metadata' dictionary containing the file source.")
print("When we have 10,000 hospital documents, this metadata is what allows the Vector DB to cite its sources!")
