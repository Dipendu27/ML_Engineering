import chromadb
import warnings

# Suppress minor huggingface warnings for a clean terminal output
warnings.filterwarnings("ignore")

print("--- Day 7: ChromaDB Vector Engine Initialization ---\n")

# 1. Initialize a Local ChromaDB Client
# This runs entirely in memory on your Mac. No cloud APIs required.
chroma_client = chromadb.Client()

# Create a collection (similar to a table in SQL)
# Chroma will automatically download a lightweight, open-source embedding model 
# (all-MiniLM-L6-v2) in the background to convert our text to vectors.
collection = chroma_client.create_collection(name="patient_profiles")

# 2. Mocking Serialized Data from Day 6
# These represent the cleaned patient rows converted into natural language.
patient_documents = [
    "Patient 101: 45 year old male. Exhibits elevated resting systolic pressure. Diagnosed with Hypertension. BMI is 28.5.",
    "Patient 102: 61 year old female. Severe chronic hypertension tracking. Prescribed Lisinopril. BMI is 31.2.",
    "Patient 103: 29 year old male. Routine checkup. Vital signs normal. Healthy profile. BMI is 22.1.",
    "Patient 104: 52 year old female. Type 2 diabetes checks. BGL unstable. Adjusting metformin dosage. BMI is 29.8."
]

patient_ids = ["id_101", "id_102", "id_103", "id_104"]

print("📥 Embedding and loading patient records into Vector Database...")
# 3. Ingestion & Embedding
# Chroma automatically calculates the vectors and stores them.
collection.add(
    documents=patient_documents,
    ids=patient_ids
)
print("✅ Records successfully stored in ChromaDB.\n")

# 4. Semantic Search (The Retrieval in RAG)
# We ask a natural language question. Notice how we don't use exact matching words.
query_text = "elderly patients struggling with high blood pressure"

print(f"🔍 Searching Vector DB for: '{query_text}'")
results = collection.query(
    query_texts=[query_text],
    n_results=2 # Retrieve the top 2 closest mathematical matches
)

# 5. Display Results
print("\n🎯 Top 2 Semantic Matches Found:")
for i, doc in enumerate(results['documents'][0]):
    # The distance score tells us how close the vectors are (lower is closer)
    distance = results['distances'][0][i]
    print(f"Match {i+1} (Distance: {distance:.4f}): {doc}")