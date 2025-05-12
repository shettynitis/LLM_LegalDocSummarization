# train_sft_with_lightning_ray_mlflow.py

import os, json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim import AdamW
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar
from lightning.pytorch.loggers import MLFlowLogger

import ray.train.lightning
import mlflow.pytorch
from ray.train.lightning import RayTrainReportCallback, RayDDPStrategy, RayLightningEnvironment
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig, RunConfig, FailureConfig

from peft import LoraConfig, get_peft_model
import torch
import mlflow

# ─── MLflow / Experiment setup ───────────────────────────────────
MLFLOW_URI = "http://129.114.25.240:8000"
EXPERIMENT  = "LLama-ray"

mlf_logger = MLFlowLogger(
    experiment_name=EXPERIMENT,
    tracking_uri=MLFLOW_URI,
)

# ─── Data loader helper ──────────────────────────────────────────
def load_dataset(jsonl_file):
    with open(jsonl_file, "r", encoding="utf-8") as f:
        data = [json.loads(l) for l in f]
    system_prompt = "Summarize the following legal text."
    texts = [
        f"### Instruction: {system_prompt}\n\n### Input:\n{item['judgement'][:10000]}\n\n### Response:\n{item['summary']}"
        for item in data
    ]
    return Dataset.from_dict({"text": texts})

def make_dataloaders(batch_size=1, max_len=1012):
    ds = load_dataset("/mnt/LLMData/train.jsonl")
    split = ds.train_test_split(test_size=0.1, seed=42)
    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", padding_side="left")
    tok.pad_token = tok.eos_token

    def collate_fn(batch):
        texts = [x["text"] for x in batch]
        out = tok(
            texts, return_tensors="pt",
            padding="max_length", truncation=True, max_length=max_len
        )
        out["labels"] = out.input_ids.clone()
        return out

    train_dl = DataLoader(split["train"], batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_dl   = DataLoader(split["test"],  batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return train_dl, val_dl

# ─── LightningModule definition ─────────────────────────────────
class SFTModule(pl.LightningModule):
    def __init__(self, lr=5e-3, wd=0.001):
        super().__init__()
        base = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True
        )
        base.gradient_checkpointing_enable()
        lora_cfg = LoraConfig(r=8, lora_alpha=8, lora_dropout=0.1, bias="none", task_type="CAUSAL_LM")
        self.model = get_peft_model(base, lora_cfg).to("cuda")

        self.save_hyperparameters({"lr": lr, "weight_decay": wd})

    def forward(self, **batch):
        return self.model(**batch)

    def training_step(self, batch, _):
        loss = self.model(**batch).loss
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, _):
        loss = self.model(**batch).loss
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        return AdamW(params, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

# ─── Ray Train loop ──────────────────────────────────────────────
def train_loop(_config):
    train_dl, val_dl = make_dataloaders(batch_size=1)
    model = SFTModule()

    # callbacks: save best by val_loss + report to Ray Tune
    ckpt_cb = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1,save_last=True, dirpath="checkpoints",filename="last")
    es_cb   = EarlyStopping(monitor="val_loss", patience=3, mode="min")
    tqdm_cb = TQDMProgressBar(refresh_rate=10)

    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="gpu",
        devices=1,
        strategy=RayDDPStrategy(),
        plugins=[RayLightningEnvironment()],
        logger=mlf_logger,
        callbacks=[ckpt_cb, es_cb, tqdm_cb, RayTrainReportCallback()],
        accumulate_grad_batches=4,
        gradient_clip_val=0.3,
        log_every_n_steps=50,
    )
    trainer = ray.train.lightning.prepare_trainer(trainer)

    checkpoint = train.get_checkpoint()
    if checkpoint:
        # Ray hands us a little directory with “last.ckpt” in it
        with checkpoint.as_directory() as ckpt_dir:
            ckpt_path = os.path.join(ckpt_dir, "last.ckpt")
        trainer.fit(
            model,
            train_dataloaders=train_dl,
            val_dataloaders=val_dl,
            ckpt_path=ckpt_path,
        )
    else:
        trainer.fit(
            model,
            train_dataloaders=train_dl,
            val_dataloaders=val_dl,
        )


    
    # at this point MLFlowLogger has already logged metrics & checkpoint
    # we can also log the final model one more time:
    #mlflow.pytorch.log_model(model, artifact_path="final_model")
    adapter_state = model.model.get_adapter_state_dict() #add
    torch.save(adapter_state, "adapter_weights.pt")  #add

# ...then log that file as a generic artifact:
    mlflow.log_artifact("adapter_weights.pt", artifact_path="ray_llama") #add

# ─── Entrypoint ─────────────────────────────────────────────────
if __name__ == "__main__":
    scale_cfg = ScalingConfig(
        num_workers=2, use_gpu=True,
        resources_per_worker={"GPU": 1, "CPU": 8},
    )
    run_cfg = RunConfig(
        storage_path="s3://ray",
        failure_config=FailureConfig(max_failures=-1),
    )
    trainer = TorchTrainer(
        train_loop,
        scaling_config=scale_cfg,
        run_config=run_cfg,
        train_loop_config={},  # no extra hyperparams
    )
    result = trainer.fit()
    print("Done, metrics:", result.metrics)
