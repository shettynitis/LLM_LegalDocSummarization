import os
import pickle
import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from .config import INDEX_PATH, MAPPING_PATH, CHUNK_DIR, EMB_MODEL_NAME, API_URL

# ─── Load FAISS index & metadata once ───────────────────────────────
_index = faiss.read_index(INDEX_PATH)

with open(MAPPING_PATH, "rb") as f:
    _metadata = pickle.load(f)  # could be list of dicts or list of strings

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

        entry = _metadata[idx]
        # Determine how to get the path
        if isinstance(entry, dict) and "file_path" in entry:
            path = entry["file_path"]
        elif isinstance(entry, str):
            path = os.path.join(CHUNK_DIR, entry)
        else:
            # unexpected format—log and skip
            print(f"[retrieve] Unrecognized metadata entry #{idx!r}: {entry!r}")
            continue

        # Debug: show us the path
        print(f"[retrieve] Loading snippet from: {path}")

        snippet = ""
        if os.path.isfile(path):
            try:
                with open(path, encoding="utf-8") as f:
                    snippet = f.read().strip()
            except Exception as e:
                print(f"[retrieve] Error reading {path!r}: {e}")
        else:
            print(f"[retrieve] File not found: {path!r}")

        results.append((float(dist), snippet))
    return results


def summarize_text(user_text: str, max_new_tokens: int = 100) -> str:
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

    # Debug: print to console
    print("full_prompt:\n", full_prompt)

    # 3) Call real API or mock
    if API_URL:
        try:
            resp = requests.post(
                API_URL,
                json={
                    "prompt": full_prompt,
                    "max_new_tokens": max_new_tokens
                }
            )
            resp.raise_for_status()
            data = resp.json()
            # adjust the key below to match your API's response field
            return data.get("summary", data.get("generated_text", ""))
        except Exception as e:
            print(f"[summarize_text] API call failed: {e}")

    # —————————————————— MOCK ——————————————————
    flat = full_prompt.replace("\n", " ")
    return f"📄 MOCK SUMMARY (first 100 chars): {flat[:100]}"


if __name__ == "__main__":
    # simple manual test
    text = "Summarize clause 7.2"
    print(summarize_text(text, max_new_tokens=10))