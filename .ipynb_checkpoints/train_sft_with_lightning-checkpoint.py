# train_sft_with_lightning_ray_mlflow.py
# Ensure all necessary installs when run as a script
import subprocess
import sys


'''subprocess.run([
    sys.executable,
    "-m",
    "pip",
    "install",
    "--upgrade",
    "bitsandbytes",
    "transformers",
    "accelerate",
    "datasets",
    "trl",
    "peft",
    "evaluate",
    "lightning",
    "torch",
    "torchvision",
    "mlflow",
    "huggingface-hub",
    "ray[air]",
], check=True)'''

# -------------------------------------------------------------
# Imports
# -------------------------------------------------------------
import os
import json
from transformers import BitsAndBytesConfig
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model
import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar

import mlflow
import mlflow.pytorch

import ray.train.torch
import ray.train.lightning
from ray.train import ScalingConfig, RunConfig
from ray.train.torch import TorchTrainer
from lightning.pytorch.strategies import FSDPStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from ray.train.lightning import RayTrainReportCallback
from ray.train.lightning import RayDDPStrategy, RayLightningEnvironment

#from huggingface_hub import login

# -------------------------------------------------------------
# Login to Hugging Face (token should have access permissions)
# -------------------------------------------------------------
#login("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")

# -------------------------------------------------------------
# Ray Train loop function
# -------------------------------------------------------------
mlflow.set_tracking_uri("http://129.114.25.240:8000/") 
mlflow.set_experiment("LLama-ray")


def train_func(train_loop_config):
    """Distributed training loop executed by Ray workers."""
    import os
    import mlflow
    import mlflow.pytorch
    import ray.train as train
    import os
    #from huggingface_hub import login

    #login(os.getenv("HUGGINGFACE_HUB_TOKEN"))
    HF_TOKEN = os.environ["HF_TOKEN"] 

    # ------------------ Data Preparation ------------------
    preprocessed_data_dir = "/mnt/LLMData" #os.getenv("MERGED_DATA_DIR")

    def load_dataset(jsonl_file):
        with open(jsonl_file, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
        system_prompt = "Summarize the following legal text."
        texts = [
            f"""### Instruction: {system_prompt}\n\n### Input:\n{item['judgement'].strip()[:10000]}\n\n### Response:\n{item['summary'].strip()}"""
            for item in data
        ]
        return Dataset.from_dict({"text": texts})

    train_ds = load_dataset(os.path.join(preprocessed_data_dir, "train.jsonl"))
    test_ds = load_dataset(os.path.join(preprocessed_data_dir, "test.jsonl"))

    # Tokenizer and collate function
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-2-7b-hf", padding_side="left"
    )
    tokenizer.pad_token = tokenizer.eos_token
    MAX_LEN = 1012

    def collate_fn(batch):
        texts = [ex["text"] for ex in batch]
        tok = tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
        )
        tok["labels"] = tok.input_ids.clone()
        return tok

    train_loader = DataLoader(
        train_ds, batch_size=1, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False, collate_fn=collate_fn
    )

    # ------------------ Model Setup ------------------
    '''base_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf", 
         torch_dtype=torch.bfloat16, 
        low_cpu_mem_usage=True,
    )'''
    base_model = AutoModelForCausalLM.from_pretrained(
     "meta-llama/Llama-2-7b-hf",
      torch_dtype=torch.float16,
     low_cpu_mem_usage=True,
      )

    base_model.gradient_checkpointing_enable()  # ← drop activations

    lora_cfg = LoraConfig(r=8, lora_alpha=8, lora_dropout=0.1,
                      bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(base_model, lora_cfg).to("cuda")

    # ------------------ LightningModule ------------------
    class SFTLightningModule(pl.LightningModule):
        def __init__(self, model, lr, weight_decay):
            super().__init__()
            self.model = model
            self.lr = lr
            self.weight_decay = weight_decay

        def training_step(self, batch, batch_idx):
            outputs = self.model(**batch)
            loss = outputs.loss
            self.log("train_loss", loss, prog_bar=True, sync_dist=True)
            return loss

        def validation_step(self, batch, batch_idx):
            outputs = self.model(**batch)
            loss = outputs.loss
            self.log("val_loss", loss, prog_bar=True, sync_dist=True)

        def configure_optimizers(self):
            peft_params = [p for p in self.model.parameters() if p.requires_grad]
            return AdamW(peft_params, lr=self.lr, weight_decay=self.weight_decay)
        
        
    #precision fp16
    

    lit_model = SFTLightningModule(model, lr=5e-3, weight_decay=0.001)

    # ------------------ Callbacks ------------------
    checkpoint_cb = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1)
    earlystop_cb = EarlyStopping(monitor="val_loss", patience=3, mode="min")
    tqdm_cb = TQDMProgressBar(refresh_rate=10)

    # ------------------ Trainer ------------------
    '''trainer = pl.Trainer(
        max_epochs=1,
        accelerator="gpu",
        devices=1,
        strategy=ray.train.lightning.RayDDPStrategy(),
        plugins=[ray.train.lightning.RayLightningEnvironment()],
        gradient_clip_val=0.3,
        callbacks=[
            checkpoint_cb,
            earlystop_cb,
            RayTrainReportCallback(),
        ],
        log_every_n_steps=50,
    )'''
    from functools import partial
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer
    from ray.train.lightning import (
    RayLightningEnvironment,
    RayFSDPStrategy,          # ← Ray’s wrapper
)
    

    wrap_policy = partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={LlamaDecoderLayer},
    # keyword arg!
)

    # 2. build the Trainer
    
    trainer = pl.Trainer(
      max_epochs=3,
      accelerator="gpu",
      devices=1,                      # one GPU per Ray worker
      strategy=RayDDPStrategy(),      # ← switch to DDP
      plugins=[RayLightningEnvironment()],
      accumulate_grad_batches=4,
      gradient_clip_val=0.3,
      callbacks=[checkpoint_cb, earlystop_cb, RayTrainReportCallback()],
      log_every_n_steps=50,
 )

    


    # Prepare trainer for Ray distributed execution
    trainer = ray.train.lightning.prepare_trainer(trainer)

    # ------------------ MLflow Autologging ------------------
    # Enable MLflow only on the primary (rank 0) worker to avoid duplicate logs
    import ray.train as train

    try: 
        mlflow.end_run() # end pre-existing run, if there was one
    except:
        pass
    finally:
        mlflow.start_run(log_system_metrics=True) # Start MLFlow run

    # ------------------ Training ------------------
    trainer.fit(lit_model, train_loader, val_loader)

    # ------------------ Save model & tokenizer ------------------
    '''output_dir = "/mnt/LLMData/lightning_lora_model"
    os.makedirs(output_dir, exist_ok=True)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)'''

    # Log the saved model as an MLflow artifact (rank 0 only)
    

    # Return the validation loss for Ray Tune / Ray Train aggregates
    metrics = {"val_loss": trainer.callback_metrics.get("val_loss").item()}
    return metrics


# -------------------------------------------------------------
# Ray Trainer Configuration & Execution
# -------------------------------------------------------------

scaling_config = ScalingConfig(
    num_workers=2,
    use_gpu=True,
    resources_per_worker={"GPU": 1,"CPU":8},
)

run_config = RunConfig(storage_path="s3://ray")

trainer = TorchTrainer(
    train_func,
    scaling_config=scaling_config,
    run_config=run_config,
    train_loop_config={},  # no additional hyperparams passed for now
)
# Log metrics - the things we *measure* - to MLFlow
   

result = trainer.fit()

print("Training completed. Aggregated metrics:", result)
mlflow.pytorch.log_model(lit_model.model, "ray_llama")

mlflow.log_metrics(
    {"test_loss": metrics['val_loss']
    })

mlflow.end_run()

