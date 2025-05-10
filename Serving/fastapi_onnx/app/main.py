from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer

class InferenceRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 100

app = FastAPI(title="LLM ONNX Inference")

# Load tokenizer and ONNX session once at startup
tok = AutoTokenizer.from_pretrained("../model/")  # adjust path if needed
# For GPU inference, make sure onnxruntime-gpu is installed and NVIDIA drivers available
ort_session = ort.InferenceSession(
    "../model/model.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)

@app.post("/generate")
def generate(req: InferenceRequest):
    try:
        # Tokenize to numpy arrays
        enc = tok(req.prompt, return_tensors="np")
        input_ids = enc["input_ids"].astype(np.int64)
        attention_mask = enc["attention_mask"].astype(np.int64)
        # For causal-LM with past, you may need to construct position_ids
        position_ids = np.arange(input_ids.shape[1], dtype=np.int64)[None, :]

        outputs = ort_session.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
        )
        # Assume logits in outputs[0]; implement your own decoding loop
        # Here’s a simple greedy decode for demonstration:
        logits = outputs[0]
        next_token_id = int(np.argmax(logits[:, -1, :], axis=-1)[0])
        generated = tok.decode(next_token_id)
        return {"generated_token": generated}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
