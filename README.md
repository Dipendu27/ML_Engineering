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

- [x] **Day 1:** Environment & Hardware Verification — isolated Python 3.11 setup, IDE configuration, and verifying MLX/PyTorch MPS on the M5 chip.
- [x] **Day 2:** Data Engineering Foundation — processing clinical biomarkers with NumPy arrays and cleaning/filtering tabular patient records with Pandas DataFrames.
- [ ] **Day 3:** Multi-Dimensional Matrices & Dot Products — simulating a semantic matching engine using Apple MLX.
- [ ] **Day 4-10:** Advanced Data Engineering & Math Basics — calculus intuition, EDA, Matplotlib, and Seaborn.
- [ ] **Days 11-20:** Classical Machine Learning — Scikit-learn, XGBoost, and Pipeline Evaluation.
- [ ] **Days 21-60:** Deep Learning, Vision, Agentic RAG, and Streamlit UI Deployment.

---

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
# 1. Activate isolated Python environment
source .venv/bin/activate

# 2. Run hardware verification
python test_env.py

# 3. Run data engineering simulation
python day2_data_engine.py

# Verify clean git tracking (ignoring .venv)
git status
```

---

## 🎯 Target Roles

AIML Engineer · ML Ops Engineer · GenAI Developer

---

*Built with 💻 + 🧠 by Dipendu Mukherjee — one day at a time.*
