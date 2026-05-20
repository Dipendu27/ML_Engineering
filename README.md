# 60-Day AIML Engineering Journey 🚀

This repository tracks my transition into an AIML Engineer.  
Built from scratch — focusing on a privacy-first, 100% local Retrieval-Augmented Generation (RAG) system running natively on Apple Silicon.

---

## 🛠 Tech Stack

| Category | Tools |
|---|---|
| **Hardware** | MacBook Pro M5 (16GB RAM, 512GB SSD) |
| **Shell** | Bash / Zsh |
| **Version Control** | Git, GitHub, `.gitignore` configuration |
| **Core ML Engines** | Apple MLX, PyTorch (Metal Performance Shaders) |
| **Data Engineering** | Python 3.11, NumPy, Pandas |
| **Environment Management** | Python `venv`, pip requirements |
| **Editor** | VS Code (with Python & Jupyter extensions) |

---

## 📈 Learning Progress

## 🏗️ Phase 1: Environment & Mathematical Foundations (Days 1–10)

- [x] **Day 1:** Environment & Hardware Verification - Configure Python/VS Code and verify Apple MLX plus PyTorch MPS on Apple Silicon.
- [x] **Day 2:** Data Engineering Foundation - Generate synthetic clinical biomarkers with NumPy and clean/filter patient records with Pandas.
- [ ] **Day 3:** Multi-Dimensional Matrices & Dot Products - Simulate semantic matching with Apple MLX.
- [ ] **Day 4:** Advanced Data Engineering - Slicing, filtering, and manipulating tabular clinical data.
- [ ] **Day 5:** Data Engineering Practice - Synthesizing and cleaning mock patient datasets.
- [ ] **Day 6:** Math Basics - Linear algebra fundamentals and matrices.
- [ ] **Day 7:** Math Basics - Vector dot products and semantic math.
- [ ] **Day 8:** Math Basics - Calculus intuition and derivatives for backpropagation.
- [ ] **Day 9:** Exploratory Data Analysis (EDA) - Introduction to Matplotlib and Seaborn.
- [ ] **Day 10:** EDA Practice - Cleaning and visualizing messy medical datasets from Kaggle.

## 📊 Phase 2: Classical Machine Learning (Days 11–20)

- [ ] **Day 11:** Linear Regression - Building a continuous predictor model.
- [ ] **Day 12:** Logistic Regression - Building a classifier model.
- [ ] **Day 13:** Decision Trees - Learning how tree-based models make splits.
- [ ] **Day 14:** Random Forests - Predicting categorical outcomes, such as patient readmission.
- [ ] **Day 15:** Gradient Boosting - Introduction to XGBoost.
- [ ] **Day 16:** Advanced XGBoost - Training models on tabular data.
- [ ] **Day 17:** Unsupervised Learning - Implementing K-Means clustering.
- [ ] **Day 18:** Unsupervised Learning - Principal Component Analysis (PCA).
- [ ] **Day 19:** Model Evaluation - Mastering Precision, Recall, F1-score, and ROC-AUC.
- [ ] **Day 20:** ML Pipelines - Building an end-to-end scaling and training pipeline.

## 🧠 Phase 3: Deep Learning & Embeddings (Days 21–30)

- [ ] **Day 21:** PyTorch Fundamentals - Tensors and basic operations.
- [ ] **Day 22:** Multi-Layer Perceptron (MLP) - Building your first neural network.
- [ ] **Day 23:** Hardware Acceleration - Pushing tensors to the M5 GPU (`device = torch.device("mps")`).
- [ ] **Day 24:** Neural Network Training - Forward passes, loss functions, and backpropagation.
- [ ] **Day 25:** Transformers - Understanding self-attention architectures.
- [ ] **Day 26:** Hugging Face - Loading pre-trained models via the `transformers` library.
- [ ] **Day 27:** NLP Processing - Tokenization and text preparation.
- [ ] **Day 28:** Vectorization - Converting medical text into dense numerical vectors (embeddings).
- [ ] **Day 29:** Lightweight Models - Generating embeddings using small, efficient models.
- [ ] **Day 30:** Vector Databases - Spinning up a local ChromaDB instance for storage.

## 👁️ Phase 4: Vision & The MLX Engine (Days 31–40)

- [ ] **Day 31:** Native Computer Vision - Introduction to YOLO for object detection.
- [ ] **Day 32:** YOLO26-MLX - Implementing real-time vision natively on the Mac.
- [ ] **Day 33:** Medical Imaging - Applying vision to identify anomalies in scans.
- [ ] **Day 34:** Memory Optimization - Bypassing PyTorch overheads for Apple MLX efficiency.
- [ ] **Day 35:** The Apple MLX LLM - Introduction to local text generation.
- [ ] **Day 36:** Model Quantization - Understanding 16-bit to 4-bit compression for 16GB RAM limits.
- [ ] **Day 37:** Downloading Quantized Models - Using `mlx-lm`, such as Llama-3-8B-4bit.
- [ ] **Day 38:** Local Inference - Writing scripts to generate text natively.
- [ ] **Day 39:** Hardware Acceleration - Taking full advantage of the M5 neural engine.
- [ ] **Day 40:** Inference Optimization - Tuning generation parameters, including temperature and max tokens.

## ⚙️ Phase 5: Building Agentic RAG (Days 41–50)

- [ ] **Day 41:** Document Ingestion - Setting up a LangChain pipeline.
- [ ] **Day 42:** PDF Parsing - Reading and extracting text from clinical PDFs.
- [ ] **Day 43:** Chunking - Splitting documents into optimized semantic chunks.
- [ ] **Day 44:** Embedding Generation - Batch processing chunks into vectors.
- [ ] **Day 45:** Vector Storage Setup - Initializing the RAG database.
- [ ] **Day 46:** Populating ChromaDB - Inserting chunked embeddings securely.
- [ ] **Day 47:** Semantic Search - Querying the database for nearest neighbors.
- [ ] **Day 48:** Orchestration Logic - Programming the agentic retrieval step.
- [ ] **Day 49:** LLM Integration - Instructing the MLX LLM to read chunks and formulate answers.
- [ ] **Day 50:** RAG Fusion - End-to-end testing of the question-answering loop.

## 🚀 Phase 6: Fine-Tuning, UI & Deployment (Days 51–60)

- [ ] **Day 51:** PEFT & LoRA - Introduction to Parameter-Efficient Fine-Tuning.
- [ ] **Day 52:** Corporate Jargon - Preparing a dataset of specific medical terminology.
- [ ] **Day 53:** MLX Fine-Tuning - Running LoRA on the quantized model within 16GB limits.
- [ ] **Day 54:** Merging Weights - Applying the LoRA adapters to the base model.
- [ ] **Day 55:** User Interface - Introduction to Streamlit.
- [ ] **Day 56:** UI Development - Building a modern browser-based chat interface.
- [ ] **Day 57:** UI Integration - Connecting the Streamlit frontend to the RAG backend.
- [ ] **Day 58:** Profiling - Reviewing memory usage across the complete pipeline.
- [ ] **Day 59:** Optimization - Tweaking chunk sizes and prompts to maximize M5 generation speed.
- [ ] **Day 60:** Shipping - Finalizing `README.md` documentation and pushing to GitHub.

---

## ✅ Current Status

Day 2 is complete. The project now has:

- `test_env.py` for validating Apple Silicon ML acceleration with PyTorch MPS and Apple MLX.
- `day2_data_engine.py` for generating synthetic patient biomarker data, imputing missing clinical fields, and filtering high-risk hypertension records.
- `requirements.txt` with the Day 1 and Day 2 Python dependencies.

## 📂 Project Highlights

### ⚙️ Hardware Verification (`test_env.py`)

A sanity check script ensuring PyTorch communicates with the M5 GPU (MPS) and Apple MLX is successfully initialized.

```bash
python test_env.py
# --- Apple Silicon Hardware Test ---
# ✅ PyTorch is successfully utilizing the M5 GPU (MPS).
# ✅ Apple MLX is installed and functioning.
```

---

### 🧬 Clinical Data Engine (`day2_data_engine.py`)

Simulates clinical dataset manipulation for future RAG ingestion. Demonstrates fast NumPy array processing for patient biomarkers and Pandas DataFrame imputation for missing values in clinical notes.

```bash
python day2_data_engine.py
# --- Day 2: Clinical Data Engineering Engine ---
#
# Step 1: Raw NumPy Biomarker Matrix (Shape: 5x3):
# [[148 138 124] ... ]
#
# Step 3: Imputed missing ages with median (48.5):
#     Patient_ID   Age     Condition                                     Clinical_Notes
# 0          101  45.0  Hypertension  Patient exhibits elevated resting systolic pressure.
#
# Step 4: Filtered High-Risk Hypertension Patients:
#     Patient_ID   Age                                     Clinical_Notes
# 0          101  45.0  Patient exhibits elevated resting systolic pressure.
# 3          104  61.0              Severe chronic hypertension tracking.
```

---

## 💻 Local AI Execution & Validation

```bash
# 1. Create and activate isolated Python environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install Day 1 and Day 2 dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 3. Run hardware verification
python test_env.py

# 4. Run data engineering simulation
python day2_data_engine.py

# Verify clean git tracking (ignoring .venv)
git status
```

---

## 🎯 Target Roles

AIML Engineer · ML Ops Engineer · GenAI Developer

---

*Built with 💻 + 🧠 by Dipendu Mukherjee — one day at a time.*
