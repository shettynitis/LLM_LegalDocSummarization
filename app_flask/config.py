import os

# e.g. http://fastapi_server:8000/summarize
API_URL: str = os.getenv("API_URL", "")

BASE_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR  = os.path.join(BASE_DIR, "2_vector_db", "model_rag")

INDEX_PATH     = os.path.join(MODEL_DIR, "legal-facts.index")
MAPPING_PATH   = os.path.join(MODEL_DIR, "index_to_doc.pkl")

# Embedding model name
EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"