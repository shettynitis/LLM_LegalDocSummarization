# llm_tests/conftest.py  (with MLflow fallback)
import json, os, tempfile, shutil, numpy as np, pytest, onnxruntime as ort
from pathlib import Path
from transformers import AutoTokenizer

# ─── config via env vars ─────────────────────────────────────────────
MODEL_DIR   = Path(os.getenv("LLM_MODEL_DIR",  "../llama-legal-onnx"))
MODEL_URI   = os.getenv("LLM_MODEL_URI")          # ← if set, pull from MLflow
TEST_JSONL  = Path(os.getenv("LLM_TEST_FILE",  "data/test.jsonl"))
MAX_NEW     = int(os.getenv("LLM_MAX_NEW",     120))
PROVIDERS   = ["CUDAExecutionProvider", "CPUExecutionProvider"]
# ────────────────────────────────────────────────────────────────────

# --- download from MLflow if requested --------------------------------
if MODEL_URI:
    import mlflow
    _tmp = Path(tempfile.mkdtemp(prefix="onnx_dl_"))
    print(f"Downloading model from MLflow → {_tmp}")
    MODEL_DIR = Path(mlflow.artifacts.download_artifacts(
        artifact_uri=MODEL_URI, dst_path=_tmp))

@pytest.fixture(scope="session")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_DIR)

@pytest.fixture(scope="session")
def ort_session():
    return ort.InferenceSession((MODEL_DIR / "model.onnx").as_posix(),
                                providers=PROVIDERS)

@pytest.fixture(scope="session")
def test_data():
    rows = [json.loads(l) for l in open(TEST_JSONL, encoding="utf‑8")]
    sys_prompt = "Summarize the following legal text."
    inputs  = [f"### Instruction: {sys_prompt}\n\n### Input:\n{r['judgement']}\n\n### Response:\n"
               for r in rows]
    refs    = [r["summary"] for r in rows]
    return inputs, refs

# ‑‑‑‑ greedy+top‑k sampler used by all tests ‑‑‑‑
def _sample_top_k(logits, k=40, temp=0.8):
    logits = logits.astype(np.float32) / temp
    top = logits.argsort()[-k:]; mask = np.ones_like(logits, bool); mask[top]=False
    logits[mask] = -np.inf
    p = np.exp(logits-logits.max()); p /= p.sum()
    return int(np.random.choice(len(logits), p=p))

def _generate(sess, tok, prompt, max_new=MAX_NEW):
    enc = tok(prompt, return_tensors="np")
    ids, att = enc["input_ids"], enc["attention_mask"]
    for _ in range(max_new):
        pos = np.arange(ids.shape[1], dtype=np.int64)[None]
        logits = sess.run(None, {
            "input_ids":ids, "attention_mask":att, "position_ids":pos
        })[0]
        nxt = _sample_top_k(logits[0,-1])
        if nxt in {0,2,50256}: break
        step = np.array([[nxt]], np.int64)
        ids  = np.concatenate([ids, step],1)
        att  = np.concatenate([att, np.ones_like(step)],1)
    return tok.decode(ids[0], skip_special_tokens=True)

@pytest.fixture(scope="session")
def predict(ort_session, tokenizer):
    return lambda txt: _generate(ort_session, tokenizer, txt)

@pytest.fixture(scope="session")
def predictions(predict, test_data):
    inputs, _ = test_data
    return [predict(p) for p in inputs]
