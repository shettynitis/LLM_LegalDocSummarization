# train_sft_with_ray.py
import os
import os, sys, subprocess, json

subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade',
                'bitsandbytes', 'transformers', 'accelerate',
                'datasets', 'trl', 'peft', 'evaluate',
                'lightning', 'torch', 'torchvision', 'mlflow','huggingface-hub'], check=True)

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from torch.optim import AdamW
from trl import SFTConfig
from peft import LoraConfig, get_peft_model
import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import mlflow, mlflow.pytorch
from huggingface_hub import login

# Ray imports
import ray
from ray.train import RunConfig, ScalingConfig
from ray.train.lightning import RayTrainReportCallback, RayDDPStrategy, prepare_trainer
from ray.train.torch import TorchTrainer

# ---------- ensure HF login & env setup ----------

from huggingface_hub import login

login("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")

# ---------- data loading helper ----------
def load_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        data = [json.loads(l) for l in f]
    sys_prompt = "Summarize the following legal text."
    texts = [
        f"### Instruction: {sys_prompt}\n\n### Input:\n{item['judgement'][:10000]}\n\n### Response:\n{item['summary']}"
        for item in data
    ]
    return Dataset.from_dict({"text": texts})

# ---------- the Ray “train_func” ----------
def train_func(config):
    # only rank0 does MLflow
    rank = int(os.environ.get("RANK",0))
    if rank==0:
        mlflow.set_experiment("lora_sft_ray")
        mlflow.pytorch.autolog()
        mlflow.start_run(log_system_metrics=True)
    
    # load data
    ds_dir = os.getenv("MERGED_DATASET_DIR","merged_dataset")
    train_ds = load_dataset(os.path.join(ds_dir,"train.jsonl"))
    val_ds   = load_dataset(os.path.join(ds_dir,"test.jsonl"))
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    
    def collate_fn(batch):
        toks = tokenizer([x["text"] for x in batch],
                         return_tensors="pt", padding=True,
                         truncation=True, max_length=4096)
        toks["labels"] = toks.input_ids.clone()
        return toks

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=config["batch_size"], shuffle=False,collate_fn=collate_fn)

    # build model + LoRA
    bnb = BitsAndBytesConfig(load_in_8bit=True)
    base = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        quantization_config=bnb,
        device_map="auto"
    )
    lora = LoraConfig(r=8, lora_alpha=8, lora_dropout=0.1, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(base, lora)

    # LightningModule wrapper
    class SFTModule(pl.LightningModule):
        def __init__(self, model, lr, wd):
            super().__init__()
            self.model = model
            self.lr, self.wd = lr, wd

        def training_step(self, batch, bi):
            out = self.model(**batch)
            self.log("train_loss", out.loss, prog_bar=True)
            return out.loss

        def validation_step(self, batch, bi):
            out = self.model(**batch)
            self.log("val_loss", out.loss, prog_bar=True)

        def configure_optimizers(self):
            params = [p for p in self.model.parameters() if p.requires_grad]
            return AdamW(params, lr=self.lr, weight_decay=self.wd)

    lit_mod = SFTModule(model, lr=config["lr"], wd=config["weight_decay"])

    # Lightning Trainer with RayDDPStrategy + Ray callback
    trainer = Trainer(
        max_epochs=config["num_epochs"],
        accelerator="gpu",
        devices="auto",
        strategy=RayDDPStrategy(),
        callbacks=[
            ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1),
            EarlyStopping(monitor="val_loss", patience=3),
            RayTrainReportCallback()        # ← report metrics back to Ray
        ],
        log_every_n_steps=50
    )

    # prepare for Ray
    trainer = prepare_trainer(trainer)

    # fit & validate
    trainer.fit(lit_mod, train_loader, val_loader)

    # end MLflow run on rank0
    if rank==0:
        mlflow.end_run()

# ---------- Ray Trainer entrypoint ----------
if __name__=="__main__":
    ray.init()  # or ray.init(address="auto") in a cluster

    scaling = ScalingConfig(
        num_workers=1,            # you can increase to number of nodes
        use_gpu=True,
        resources_per_worker={"GPU": 1, "CPU": 8}
    )
    runcfg = RunConfig(storage_path="s3://ray")

    torch_trainer = TorchTrainer(
        train_func=train_func,
        scaling_config=scaling,
        run_config=runcfg,
        train_loop_config={
            "model_name":"meta-llama/Llama-2-7b-hf",
            "batch_size":1,
            "lr":5e-3,
            "weight_decay":0.001,
            "num_epochs":3
        }
    )
    result = torch_trainer.fit()
    print("Finished with result:", result.metrics)
