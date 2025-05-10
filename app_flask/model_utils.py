import pickle
import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from .config import INDEX_PATH, MAPPING_PATH, EMB_MODEL_NAME, API_URL

# ─── Load FAISS index & metadata once ───────────────────────────────
_index = faiss.read_index(INDEX_PATH)

with open(MAPPING_PATH, "rb") as f:
    # metadata is expected to be a list of dicts, each with a "file_path" key
    _metadata = pickle.load(f)

_embedder = SentenceTransformer(EMB_MODEL_NAME)


def _retrieve(query: str, top_k: int = 5):
    """
    Returns a list of (score, snippet_text) tuples
    by searching the FAISS index.
    """
    q_emb = _embedder.encode([query])
    distances, indices = _index.search(np.array(q_emb), top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(_metadata):
            continue
        meta = _metadata[idx]
        snippet = ""
        try:
            with open(meta["file_path"], encoding="utf-8") as f:
                snippet = f.read()
        except Exception:
            pass
        results.append((float(dist), snippet))
    return results


def summarize_text(user_text: str) -> str:
    """
    1) Retrieve top‐k chunks for user_text
    2) Build a combined prompt
    3) POST to API_URL (if set) or fall back to a mock summary
    """
    # 1) RAG retrieval
    retrieved = _retrieve(user_text)

    # 2) Build the prompt
    prompt_parts = []
    for i, (score, snippet) in enumerate(retrieved, start=1):
        prompt_parts.append(f"### Context {i} (score={score:.3f}):\n{snippet}\n")
    prompt_parts.append(f"### Query:\n{user_text}\n")
    full_prompt = "\n".join(prompt_parts)

    # 3) Call real API or mock
    if API_URL:
        try:
            resp = requests.post(API_URL, json={"text": full_prompt})
            resp.raise_for_status()
            return resp.json().get("summary", "")
        except Exception as e:
            print(f"[summarize_text] API call failed: {e}")

    # —————————————————— MOCK ——————————————————
    snippet = user_text.strip().replace("\n", " ")
    return f"📄 MOCK SUMMARY (first 100 chars): {snippet[:100]}"