# app_flask/config.py
import os

# ─────── Project root (one level up from this file) ────────────────
BASE_DIR = os.getenv("BASE_RAG_DATA_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ─────── Where your rag_chunks folder lives ────────────────────────
CHUNK_DIR = os.path.join(BASE_DIR, "rag_chunks")

# ─────── Where your FAISS index + mapping live ─────────────────────
MODEL_RAG_DIR = os.path.join(BASE_DIR, "model_rag")

INDEX_PATH   = os.path.join(MODEL_RAG_DIR, "legal-facts.index")
MAPPING_PATH = os.path.join(MODEL_RAG_DIR, "index_to_doc.pkl")

# ─────── Embedding + API settings ──────────────────────────────────
EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
API_URL        = os.getenv("API_URL", "")