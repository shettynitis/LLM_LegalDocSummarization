#!/usr/bin/env python3
"""
FastAPI + ONNX Runtime GPU inference
• Downloads the registered model bundle from MLflow once at startup
• Top‑k / temperature sampling with repetition penalty
"""

import os
import time
from pathlib import Path
from typing import Optional,List

import mlflow
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer
from huggingface_hub import login

login("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")
# ─────────── Sampling defaults ────────────
TEMPERATURE    = float(os.getenv("TEMPERATURE", 0.8))
TOP_K          = int(os.getenv("TOP_K", 40))
REPETITION_PEN = float(os.getenv("REPETITION_PENALTY", 1.15))
END_TOKENS     = {0, 2, 50256}           # pad, eos for Llama‑like, GPT‑2 eos
# ──────────────────────────────────────────


class InferenceRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 100


app = FastAPI(title="LLM ONNX Inference (via MLflow)")

# global handles filled on startup
#tokenizer: AutoTokenizer | None = None
#ort_session: ort.InferenceSession | None = None

tokenizer: Optional[AutoTokenizer] = None
ort_session: Optional[ort.InferenceSession] = None

# ────────────────────────────────
# Utilities
# ────────────────────────────────
def sample_top_k(logits: np.ndarray, top_k: int, temperature: float) -> int:
    """Sample one token id from the top‑k logits array."""
    logits = logits.astype(np.float32) / max(temperature, 1e-8)
    if top_k and top_k < logits.size:
        top_ids = logits.argsort()[-top_k:]          # biggest k
        mask = np.ones_like(logits, dtype=bool)
        mask[top_ids] = False
        logits[mask] = -np.inf                      # drop rest
    probs = np.exp(logits - np.max(logits))
    probs /= probs.sum()
    return int(np.random.choice(len(logits), p=probs))


# ────────────────────────────────
# Startup: pull model from MLflow
# ────────────────────────────────
@app.on_event("startup")
def load_from_mlflow() -> None:      # runs once per container
    global tokenizer, ort_session

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    model_uri    = os.getenv("MODEL_URI")
    if not model_uri:
        raise RuntimeError("MODEL_URI environment variable is not set")

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    # Download artifact folder (tokenizer + model.onnx) to a tmp dir
    local_dir = Path(
        mlflow.artifacts.download_artifacts(artifact_uri=model_uri)  # :contentReference[oaicite:0]{index=0}
    )
    tokenizer = AutoTokenizer.from_pretrained(local_dir)
    onnx_path = local_dir / "model.onnx"
    if not onnx_path.exists():
        raise RuntimeError(f"model.onnx not found in {local_dir}")

    ort_session = ort.InferenceSession(
        onnx_path.as_posix(),
        providers=["CPUExecutionProvider"]#"CUDAExecutionProvider", "CPUExecutionProvider"],  # :contentReference[oaicite:1]{index=1}
    )
    print("ONNX providers →", ort_session.get_providers())


# ────────────────────────────────
# Inference route
# ────────────────────────────────
@app.post("/generate")
def generate(req: InferenceRequest):
    try:
        if tokenizer is None or ort_session is None:
            raise RuntimeError("Model not loaded")

        enc = tokenizer(req.prompt, return_tensors="np")
        input_ids      = enc["input_ids"].astype(np.int64)
        attention_mask = enc["attention_mask"].astype(np.int64)

        generated = input_ids.copy()
        times: List[float] = []

        for _ in range(req.max_new_tokens):
            position_ids = np.arange(generated.shape[1], dtype=np.int64)[None, :]

            t0 = time.time()
            logits = ort_session.run(
                None,
                {
                    "input_ids":      generated,
                    "attention_mask": attention_mask,
                    "position_ids":   position_ids,
                },
            )[0]
            #times.append(time.time() - t0)

            # repetition penalty
            logits[0, -1, np.unique(generated)] /= REPETITION_PEN

            next_id = sample_top_k(logits[0, -1], TOP_K, TEMPERATURE)
            if next_id in END_TOKENS:
                break

            next_token        = np.array([[next_id]], dtype=np.int64)
            generated         = np.concatenate([generated, next_token], axis=1)
            attention_mask    = np.concatenate([attention_mask, np.ones_like(next_token)], axis=1)

        text = tokenizer.decode(generated[0], skip_special_tokens=True)
       # throughput = len(times) / max(sum(times), 1e-6)

        return {
            "generated_text": text#,
        #    "perf": {
         #       "new_tokens": len(times),
          #      "mean_ms":    round(float(np.mean(times) * 1000), 2),
           #     "throughput": round(float(throughput), 2)
           # }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
