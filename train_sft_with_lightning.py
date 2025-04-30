# train_sft_with_lightning.py
# Ensure all necessary installs when run as a script
import subprocess
import sys

# Install dependencies if not already present
subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade',
                'bitsandbytes', 'transformers', 'accelerate',
                'datasets', 'trl', 'peft', 'evaluate',
                'lightning', 'torch', 'torchvision', 'mlflow','huggingface-hub'], check=True)

import os
import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AdamW
from trl import SFTConfig
from peft import LoraConfig, get_peft_model
import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import mlflow
import mlflow.pytorch

# MLflow: configure autolog only on primary process
# (Lightning will spawn multiple processes for DDP; we only want rank 0 to log)
# We'll start the MLflow run after creating the Trainer, once we know our global_rank.

# placeholder; actual autologging setup moved below after Trainer instantiation

from huggingface_hub import login

login("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")

# ------------------ Data Preparation ------------------
preprocessed_data_dir = os.getenv('MERGED_DATASET_DIR', 'merged_dataset')

def load_dataset(jsonl_file):
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    system_prompt = "Summarize the following legal text."
    texts = []
    for item in data:
        text = f"""### Instruction: {system_prompt}\n\n### Input:\n{item['judgement'].strip()[:10000]}\n\n### Response:\n{item['summary'].strip()}"""
        texts.append(text)
    return Dataset.from_dict({'text': texts})

train_ds = load_dataset(os.path.join(preprocessed_data_dir, 'train.jsonl'))
test_ds  = load_dataset(os.path.join(preprocessed_data_dir, 'test.jsonl'))

# Tokenizer and collate function
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', padding_side='left')
tokenizer.pad_token = tokenizer.eos_token



def collate_fn(batch):
    texts = [ex['text'] for ex in batch]
    tok = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=4096)
    tok['labels'] = tok.input_ids.clone()
    return tok

train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate_fn)
val_loader   = DataLoader(test_ds,  batch_size=1, shuffle=False, collate_fn=collate_fn)

# ------------------ Model Setup ------------------
bnb_config = BitsAndBytesConfig(load_in_8bit=True)
base_model = AutoModelForCausalLM.from_pretrained(
    'meta-llama/Llama-2-7b-hf', quantization_config=bnb_config, device_map='auto'
)
lora_cfg = LoraConfig(r=8, lora_alpha=8, lora_dropout=0.1, bias='none', task_type='CAUSAL_LM')
model = get_peft_model(base_model, lora_cfg)
model.print_trainable_parameters()

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
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss
        self.log('val_loss', loss, prog_bar=True)

    def configure_optimizers(self):
        peft_params = [p for p in self.model.parameters() if p.requires_grad]
        return AdamW(peft_params, lr=self.lr, weight_decay=self.weight_decay)

# ------------------ Training ------------------
lit_model = SFTLightningModule(model, lr=5e-3, weight_decay=0.001)

checkpoint_cb = ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=1)
earlystop_cb   = EarlyStopping(monitor='val_loss', patience=3, mode='min')

trainer = Trainer(
    max_epochs=3,
    accelerator='gpu', devices='auto',
    gradient_clip_val=0.3,
    callbacks=[checkpoint_cb, earlystop_cb],
    log_every_n_steps=50
)

trainer.fit(lit_model, train_loader, val_loader)

# ------------------ Save ------------------
output_dir = 'lightning_lora_model'
os.makedirs(output_dir, exist_ok=True)
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Model and tokenizer saved to {output_dir}")
