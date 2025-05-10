# test_retrieval.py
import os
import faiss
import pickle
from app_flask.model_utils import _retrieve
from app_flask.config      import INDEX_PATH, MAPPING_PATH

def main():
    # 1) Sanity‐check your artifacts
    print(f"INDEX_PATH    = {INDEX_PATH!r}")
    print(f"MAPPING_PATH  = {MAPPING_PATH!r}")

    if not os.path.isfile(INDEX_PATH):
        print("⚠️  FAISS index not found!")
        return
    if not os.path.isfile(MAPPING_PATH):
        print("⚠️  Metadata mapping not found!")
        return

    # 2) Inspect index & metadata sizes
    index = faiss.read_index(INDEX_PATH)
    with open(MAPPING_PATH, "rb") as f:
        metadata = pickle.load(f)

    print(f"FAISS index vectors: {index.ntotal}")
    print(f"Metadata entries    : {len(metadata)}")

    # 3) Run a sample retrieval
    query = "Fourth Amendment search warrant scope"
    top_k = 3
    print(f"\nRetrieving top {top_k} for query: {query!r}\n")
    results = _retrieve(query, top_k=top_k)

    if not results:
        print("⚠️  No results retrieved. Check that your index and metadata align!")
        return

    # 4) Print out what came back
    for i, (score, snippet) in enumerate(results, start=1):
        snippet_preview = snippet.replace("\n"," ")[:200]
        print(f"Result {i}: score={score:.3f}")
        print(f"{snippet_preview}...\n")

if __name__ == "__main__":
    main()