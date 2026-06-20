# Setup / Transfer Guide — running the project on another laptop

This explains how to move the welding chatbot from one machine to another
(e.g. your 3050 laptop → a 5050 laptop for the demo).

> **Important:** You do **NOT** need a GPU to *run or demo* this project. It runs
> fine on CPU (RandomForest, SHAP, LIME, and even the DistilBERT attention all
> work on CPU in a few seconds). A GPU is only needed if you *fine-tune* a model.

---

## What must move with the project

These are **git-ignored**, so `git clone` alone will NOT bring them — copy or
regenerate them:

| Item | Why it matters | How to get it on the new machine |
|---|---|---|
| `.env` | holds your `GROQ_API_KEY` | **copy manually** (or recreate from `.env.example`) |
| `data/raw/*.csv` | the synthetic dataset | copy, **or** regenerate (deterministic, seed=42) |
| `data/processed/*.parquet` | fused features | copy, **or** regenerate |
| `models/*.joblib` | the trained detector | copy, **or** retrain |

Everything else (source code, `config/`, `requirements.txt`, the `.md` docs) is
in git.

**Never copy `.venv/`** — it contains machine-specific binaries and absolute
paths. Always recreate it on the new machine.

---

## Method A — Simple copy (recommended for a quick demo)

1. On the **old** laptop, copy the whole `manufacturing-chatbot` folder to a USB
   drive **but delete/exclude the `.venv` folder** (it's huge and won't work
   anyway).
2. On the **new** laptop, paste the folder (e.g. to `C:\manufacturing-chatbot`).
3. Open PowerShell in that folder and recreate the environment:
   ```powershell
   python -m venv .venv
   .venv\Scripts\python.exe -m pip install -r requirements.txt
   ```
4. Run it (see "Run the app" below).

This brings `data/`, `models/`, and `.env` along, so nothing needs regenerating.

---

## Method B — Git + regenerate (clean, no big files)

1. On the new laptop:
   ```powershell
   git clone <your-repo-url> manufacturing-chatbot
   cd manufacturing-chatbot
   python -m venv .venv
   .venv\Scripts\python.exe -m pip install -r requirements.txt
   ```
2. Recreate the git-ignored data + model (deterministic — produces the *same*
   data because the seed is fixed):
   ```powershell
   .venv\Scripts\python.exe src\data\generate_synthetic.py
   .venv\Scripts\python.exe src\model\train_anomaly.py
   ```
3. Create `.env` from the template and paste your Groq key:
   ```powershell
   copy .env.example .env
   # then edit .env and set GROQ_API_KEY=gsk_...
   ```

---

## Run the app

```powershell
.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
```
→ opens at http://localhost:8501

Sanity-check everything works:
```powershell
.venv\Scripts\python.exe -m pytest tests/ -q     # expect: all passing
```

---

## Requirements on the new laptop

- **Python 3.11** (this venv was built with 3.11.0 — match the 3.x minor version).
- **Internet on first run** — needed for two things:
  1. The **Groq API** (the LLM that phrases answers) uses your key over the internet.
  2. The **DistilBERT** attention model (~265 MB) downloads once from HuggingFace,
     then caches. *(To avoid this download, copy the folder
     `C:\Users\<you>\.cache\huggingface` from the old machine too.)*

---

## GPU notes (only if you will FINE-TUNE, not just demo)

`pip install -r requirements.txt` installs the **CPU** build of PyTorch, which is
all you need to demo. If you later want GPU training:

- **RTX 3050 / 40-series:** `pip install torch --index-url https://download.pytorch.org/whl/cu121`
- **RTX 5050 (Blackwell / 50-series):** use the newer build —
  `pip install torch --index-url https://download.pytorch.org/whl/cu128`
  (older `cu121` wheels do **not** support 50-series cards → "no kernel image" error).

Verify the GPU is seen:
```powershell
.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## Quick troubleshooting

| Symptom | Fix |
|---|---|
| `streamlit: command not found` | use `.venv\Scripts\python.exe -m streamlit run ...` |
| Answers look like raw text, not phrased | `.env` missing or `GROQ_API_KEY` not set / no internet |
| `Model or data not found` | run `generate_synthetic.py` then `train_anomaly.py` (Method B step 2) |
| First note query is slow | DistilBERT downloading once — wait, then it's cached |
| `torch.cuda.is_available()` is False | expected with the CPU build; only an issue for fine-tuning |
