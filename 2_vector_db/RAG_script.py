"""
RAG Pipeline: Supreme Court Judgment Summarization
--------------------------------------------------
[Raw facts files]
    ↓ clean & chunk
[Chunked text files]
    ↓ embed (Sentence-Transformer)
[FAISS vector DB]
    ↓ (at query time) embed(query) → FAISS.search → top-K chunks
    ↓ build prompt (snippets + instruction)
[Prompt]
    ↓ Llama-2 (fine-tuned on Zenodo)
[Generated summary]
"""

# ---- Step 1: Download Dataset from Kaggle ----
import os
from kaggle.api.kaggle_api_extended import KaggleApi

# Path where dataset will be saved
download_path = os.path.expanduser("/mnt/block/rag_data")
os.makedirs(download_path, exist_ok=True)

# Authenticate and download
api = KaggleApi()
api.authenticate()

api.dataset_download_files(
    dataset="deepcontractor/supreme-court-judgment-prediction",
    path=download_path,
    unzip=True
)

print(f"Dataset downloaded and extracted to: {download_path}")
# ---- Step 2: Load and Inspect CSV ----
import pandas as pd

csv_path = os.path.join(download_path, 'justice.csv')
assert os.path.isfile(csv_path), f"File not found: {csv_path}"
df = pd.read_csv(csv_path)

print(f"Loaded: {csv_path}")
print(f"Total rows: {len(df)}")
print("Columns:", df.columns.tolist())
print(df[['name', 'facts']].head(3))

# ---- Step 3: Clean and Save Each Document ----
import re
from tqdm import tqdm

BASE_DIR = os.path.expanduser('/mnt/block/rag_data')
OUT_DIR = os.path.join(download_path, 'rag_txt')
os.makedirs(OUT_DIR, exist_ok=True)

def clean_text(text):
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'\n{2,}', '\n\n', text)
    return "\n".join([line.strip() for line in text.split('\n') if line.strip()])

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Writing RAG texts"):
    name, facts = str(row.get("name", "")).strip(), str(row.get("facts", "")).strip()
    if not facts:
        continue
    safe_name = re.sub(r'[\\/:"*?<>|]+', '_', name) or f"case_{idx}"
    cleaned = clean_text(facts)
    fname = f"{idx:05d}_{safe_name}.txt"
    with open(os.path.join(OUT_DIR, fname), "w", encoding="utf-8") as f:
        f.write(cleaned)

# ---- Step 4: Chunk Documents with Overlap ----
from transformers import AutoTokenizer

CHUNK_DIR = os.path.join(download_path, 'rag_chunks')
os.makedirs(CHUNK_DIR, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
MAX_TOKENS, OVERLAP = 512, 64

for fn in os.listdir(OUT_DIR):
    if not fn.endswith(".txt"):
        continue
    with open(os.path.join(OUT_DIR, fn), encoding="utf-8") as f:
        text = f.read()
    toks, start, cid = tokenizer.encode(text), 0, 0
    while start < len(toks):
        chunk_toks = toks[start : start + MAX_TOKENS]
        chunk_text = tokenizer.decode(chunk_toks, skip_special_tokens=True)
        base = os.path.splitext(fn)[0]
        chunk_name = f"{base}_chunk{cid:03d}.txt"
        with open(os.path.join(CHUNK_DIR, chunk_name), "w", encoding="utf-8") as out:
            out.write(chunk_text)
        start += MAX_TOKENS - OVERLAP
        cid += 1
print(f"Chunks saved to: {CHUNK_DIR}")

# ---- Step 5: Embed and Build FAISS Index ----
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

MODEL_DIR = os.path.join(BASE_DIR, 'model_rag')
os.makedirs(MODEL_DIR, exist_ok=True)

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
dim = embedder.get_sentence_embedding_dimension()
index = faiss.IndexFlatL2(dim)

filenames = []
for fname in sorted(os.listdir(CHUNK_DIR)):
    if not fname.endswith(".txt"):
        continue
    with open(os.path.join(CHUNK_DIR, fname), encoding="utf-8") as f:
        vec = embedder.encode(f.read())
        index.add(np.array([vec]))
        filenames.append(fname)

faiss.write_index(index, os.path.join(MODEL_DIR, "legal-facts.index"))
with open(os.path.join(MODEL_DIR, "index_to_doc.pkl"), "wb") as f:
    pickle.dump(filenames, f)

# ---- Step 6: Retrieval Function ----
def retrieve(query: str, top_k: int = 5):
    index = faiss.read_index(os.path.join(MODEL_DIR, "legal-facts.index"))
    with open(os.path.join(MODEL_DIR, "index_to_doc.pkl"), "rb") as f:
        metadata = pickle.load(f)
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    q_emb = embedder.encode([query])
    distances, indices = index.search(np.array(q_emb), top_k)

    results = []
    for score, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        fname = metadata[idx]
        file_path = os.path.join(CHUNK_DIR, fname)
        chunk_id = fname.split("_chunk")[0]
        try:
            with open(file_path, encoding="utf-8") as f:
                snippet = f.read()
        except FileNotFoundError:
            snippet = ""
        results.append({
            "idx": idx,
            "chunk_id": chunk_id,
            "file_path": file_path,
            "score": float(score),
            "text": snippet
        })
    return results

if __name__ == "__main__":
    query = "Fourth Amendment search warrant scope"
    for res in retrieve(query, top_k=5):
        print(f"ID {res['idx']} (chunk {res['chunk_id']}, score {res['score']:.3f}):")
        print(res["text"][:300].replace("\n", " "), "...\n")