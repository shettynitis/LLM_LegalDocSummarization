from fastapi import FastAPI
from pydantic import BaseModel
import onnxruntime as ort
from transformers import AutoTokenizer

MODEL_PATH = "/model/model_decoder_int4.onnx"
tokenizer  = AutoTokenizer.from_pretrained("/model")
session    = ort.InferenceSession(
                  MODEL_PATH,
                  providers=["TensorrtExecutionProvider",
                             "CUDAExecutionProvider",
                             "CPUExecutionProvider"])

class Req(BaseModel):
    prompt: str
    max_new_tokens: int = 120

app = FastAPI(title="Llama‑2 7B Legal Summariser")

@app.post("/generate")
def generate(r: Req):
    ins  = tokenizer(r.prompt, return_tensors="np")
    outs = session.run(None, {"input_ids":ins["input_ids"],
                              "attention_mask":ins["attention_mask"]})[0]
    # greedy next‑token demo
    next_id = outs.argmax(-1)[0, -1]
    return {"completion": tokenizer.decode(next_id, skip_special_tokens=True)}
