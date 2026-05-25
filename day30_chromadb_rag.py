import chromadb
from sentence_transformers import SentenceTransformer
import os

print("--- Day 30: ChromaDB Vector Retrieval Engine ---\n")

# 1. Load our Champion Embedding Model (from Day 29)
print("🚀 Loading BGE-Small Embedding Model...")
model = SentenceTransformer('BAAI/bge-small-en-v1.5')

# 2. Initialize ChromaDB (Persistent Storage)
# Using PersistentClient means the database actually saves to a folder on your Mac.
# If you run this script tomorrow, the data will still be there!
db_path = "./clinical_vector_db"
client = chromadb.PersistentClient(path=db_path)

# Create a "Collection" (ChromaDB's word for a table/folder)
# We use get_or_create so it doesn't crash if we run the script twice
collection = client.get_or_create_collection(name="patient_records")

# 3. Our Medical Corpus & Metadata
documents = [
    "Patient presents with severe hypertension and acute chest pain.",
    "Patient suffered a fractured femur in a fall from a ladder.",
    "Routine checkup shows normal vitals and healthy BMI.",
    "Follow-up scheduled for next Tuesday regarding the MRI results."
]

# Every document in a Vector DB needs a unique ID and optional Metadata (for filtering)
document_ids = ["doc_001", "doc_002", "doc_003", "doc_004"]
metadata = [
    {"category": "ER_Admission", "priority": "High"},
    {"category": "Trauma", "priority": "High"},
    {"category": "Routine", "priority": "Low"},
    {"category": "Admin", "priority": "Medium"}
]

# 4. Generate the Embeddings (Converting text to math)
print(f"🧠 Embedding {len(documents)} medical records into 384-dimensional vectors...")
# ChromaDB requires standard Python lists, so we use .tolist() to convert from numpy/pytorch
embeddings = model.encode(documents).tolist()

# 5. Load the Database
# We push the raw text, the metadata, the IDs, and the mathematical vectors all at once.
collection.add(
    ids=document_ids,
    embeddings=embeddings,
    documents=documents,
    metadatas=metadata
)
print(f"✅ Successfully saved {collection.count()} records to '{db_path}'!\n")

# ---------------------------------------------------------
# 6. The RAG Query (Semantic Search)
# ---------------------------------------------------------
# Notice: Our query does NOT contain the words "hypertension" or "chest pain"
user_query = "Did anyone come into the hospital today with high blood pressure?"
print(f"🗣️ Doctor's Query: '{user_query}'")

# Step A: Convert the query into a vector using the EXACT SAME model
query_embedding = model.encode([user_query]).tolist()

# Step B: Search the Database for the Top 1 closest match (Nearest Neighbor)
print("🔍 Searching Vector Database for semantic matches...")
results = collection.query(
    query_embeddings=query_embedding,
    n_results=1 # Return the top 1 most relevant document
)

# 7. Print the Results
print("\n🏆 TOP RESULT FOUND:")
print(f"Document ID: {results['ids'][0][0]}")
print(f"Metadata:    {results['metadatas'][0][0]}")
print(f"Content:     {results['documents'][0][0]}")
print(f"Distance:    {results['distances'][0][0]:.4f} (Lower is closer/better)")

print("\n💡 SUCCESS: The Vector DB perfectly retrieved the hypertension document even though you searched for 'high blood pressure'!")