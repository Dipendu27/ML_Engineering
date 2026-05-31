import warnings
from pathlib import Path

from fpdf import FPDF
from langchain_community.document_loaders import PyPDFLoader

# Suppress minor font warnings from FPDF
warnings.filterwarnings("ignore")

print("--- Day 42: Parsing Clinical PDFs with LangChain ---\n")

# 1. Generate a Dummy Clinical PDF for our pipeline
pdf_path = Path("patient_discharge.pdf")

if not pdf_path.exists():
    print("📄 Generating a 2-page dummy clinical PDF...")
    pdf = FPDF()

    # Page 1: Diagnosis
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="PATIENT DISCHARGE SUMMARY", ln=1, align="C")
    pdf.cell(200, 10, txt="Patient: John Doe | Age: 45 | Ward: Cardiology", ln=2)
    pdf.multi_cell(0, 10, txt="Diagnosis: Acute Pericarditis.\nTreatment: Administered NSAIDs and Colchicine. Patient responded well. Instructed to rest and avoid strenuous activity for 2 weeks.")

    # Page 2: Follow-up
    pdf.add_page()
    pdf.cell(200, 10, txt="FOLLOW-UP INSTRUCTIONS", ln=1)
    pdf.multi_cell(0, 10, txt="Patient must schedule an echocardiogram in exactly 14 days. Monitor for fever or worsening chest pain.")

    pdf.output(str(pdf_path))
    print(f"✅ Created '{pdf_path}' in your project folder!\n")

# 2. The LangChain PyPDFLoader
print(f"🚀 Loading '{pdf_path}' into the LangChain Pipeline...")
loader = PyPDFLoader(str(pdf_path))

# .load() reads the binary PDF, extracts the text, and splits it automatically by PAGE
documents = loader.load()

print("-" * 50)
print(f"✅ Successfully extracted {len(documents)} pages from the PDF.")
print("-" * 50)

# 3. Inspect the Parsed Data
for i, doc in enumerate(documents):
    print(f"\n📑 PAGE {i+1} METADATA:")
    print(doc.metadata)

    print(f"📝 PAGE {i+1} CONTENT (Preview):")
    # We replace newline characters with spaces so it prints cleanly in the terminal
    preview = doc.page_content[:150].replace('\n', ' ')
    print(f"{preview}...")

print("\n" + "=" * 50)
print("💡 THE ML ENGINEER TAKEAWAY:")
print("Unlike a basic .txt file, PyPDFLoader automatically chunked the document by PAGE.")
print("Look at the metadata dictionary: it now contains 'page': 0 and 'page': 1.")
print("When our Vector Database retrieves this data later, it can pass this exact")
print("page number to the LLM so it can properly cite its sources to the doctor!")
