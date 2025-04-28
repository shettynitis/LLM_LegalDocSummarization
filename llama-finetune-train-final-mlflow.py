

import os
import json

#!pip uninstall -y bitsandbytes
#!pip install --no-cache-dir --upgrade bitsandbytes==0.45.5

#!python -m bitsand

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset, concatenate_datasets
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model

# Added MLflow imports (nothing else changed)
import mlflow
import mlflow.pytorch

DATA_DIR   = "../dataset/processed-IN-Ext"
RESULT_DIR = {"full": "../results_full", "lora": "../results_lora"}
SEED       = 42

# ----------------------------------------------------------------------
# MLflow global setup — exactly like the Food‑11 example
# ----------------------------------------------------------------------
mlflow.set_experiment("llama-legal-summariser")
# Make sure any stray run is closed
try:
    mlflow.end_run()
except Exception:
    pass
mlflow.start_run(log_system_metrics=True)

# Log GPU info upfront
import subprocess
_gpu_info = next(
    (subprocess.run(cmd, capture_output=True, text=True).stdout for cmd in ["nvidia-smi", "rocm-smi"]
     if subprocess.run(f"command -v {cmd}", shell=True, capture_output=True).returncode == 0),
    "No GPU found."
)
mlflow.log_text(_gpu_info, "gpu-info.txt")

# ================================
# The rest of your original code — UNCHANGED
# ================================

#!pip install dataset trl peft

from huggingface_hub import login
login("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")

preprocessed_data_dir = "../dataset/processed-IN-Ext/"

"""import torch, subprocess, re, sys
print("torch sees CUDA", torch.version.cuda)
!nvcc --version | head -n 1"""


"""# 1⃣  Point the dynamic loader to the CUDA 12.4 libs
!export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-12.4/targets/x86_64-linux/lib
# 2⃣  Make it permanent for the current Colab session
!echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-12.4/targets/x86_64-linux/lib' >> ~/.bashrc

import ctypes, bitsandbytes as bnb
print("bitsandbytes version:", bnb.__version__)
ctypes.cdll.LoadLibrary("libcudart.so")
print("✓  libcudart loaded — CUDA runtime visible")"""

#from transformers.quantizers.quantizer_bnb_8bit import is_bitsandbytes_available
#print("transformers sees bitsandbytes:", is_bitsandbytes_available())


def load_dataset(jsonl_file):
    """Load pre‑processed data and format it into a single text field."""
    with open(jsonl_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    system_prompt = "Summarize the following legal text."
    texts = [
        f"""### Instruction: {system_prompt}\n\n### Input:\n{item['judgement'].strip()[:10000]}\n\n### Response:\n{item['summary'].strip()}""".strip()
        for item in data
    ]
    return Dataset.from_dict({"text": texts})

# Load datasets (original names preserved)
train_file_A1 = os.path.join(preprocessed_data_dir, "full_summaries_A1.jsonl")
train_file_A2 = os.path.join(preprocessed_data_dir, "full_summaries_A2.jsonl")
train_dataset_A1 = load_dataset(train_file_A1)
train_dataset_A2 = load_dataset(train_file_A2)
train_data = concatenate_datasets([train_dataset_A1, train_dataset_A2])

from transformers import AutoModelForCausalLM, BitsAndBytesConfig

model_name = "meta-llama/Llama-2-7b-hf"

bnb_cfg = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)

tok = AutoTokenizer.from_pretrained(model_name)
tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_cfg,
    device_map="auto",
)

lora_config = LoraConfig(
    lora_alpha=8,
    lora_dropout=0.1,
    r=8,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

#!pip install evaluate rouge_score

import evaluate
rouge = evaluate.load("rouge")

def compute_metrics(eval_preds):
    gen_ids, labels = eval_preds
    preds = tok.batch_decode(gen_ids, skip_special_tokens=True)
    refs  = tok.batch_decode(labels,   skip_special_tokens=True)
    res   = rouge.compute(predictions=preds, references=refs)
    return {
        "rouge1": res["rouge1"].mid.fmeasure * 100,
        "rouge2": res["rouge2"].mid.fmeasure * 100,
        "rougeL": res["rougeL"].mid.fmeasure * 100,
    }

train_params = SFTConfig(
    output_dir="../results_lora",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=50,
    logging_steps=50,
    evaluation_strategy="steps",
    eval_steps=50,
    predict_with_generate=True,
    generation_max_length=256,
    learning_rate=5e-3,
    weight_decay=0.001,
    fp16=True,
    bf16=False,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="tensorboard",
    dataset_text_field="text",
    max_seq_length=4096,
)

# NOTE: eval_data was referenced in the original but not defined; leaving as‑is
fine_tuning = SFTTrainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=eval_data,  # <-- unchanged placeholder
    peft_config=lora_config,
    tokenizer=tok,
    args=train_params,
    compute_metrics=compute_metrics,
)

print("Starting fine‑tuning…")
fine_tuning.train()

print("Saving the fine‑tuned model…")
model.save_pretrained("../fine_tuned_lora_model")
tok.save_pretrained("../fine_tuned_lora_model")
print("Fine‑tuned model saved at '../fine_tuned_lora_model'")

# ----------------------------------------------------------------------
# MLflow *post‑training* logging — mirrors Food‑11 example
# ----------------------------------------------------------------------
try:
    eval_metrics = fine_tuning.evaluate()
    # The rouge compute already returns percentage, so we log as‑is
    mlflow.log_metrics(eval_metrics)
except Exception as e:
    # If evaluate fails (e.g. eval_data undefined), still close the run
    mlflow.log_text(str(e), "eval_error.txt")

# Save the LoRA adapter & tokenizer to MLflow artefacts
mlflow.pytorch.log_model(model, "lora_adapter")
mlflow.end_run()
