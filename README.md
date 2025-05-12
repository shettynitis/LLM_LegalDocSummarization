# LLM Legal Document Summarization

This repository contains the code and configuration for the LLM Legal Document Summarization project, deployed on Chameleon Cloud. Follow the sections below to understand the system lifecycle from data ingestion to production serving, and see links to the specific implementation files.

---

## 1. Value Proposition

**Target Customer:** Legal analysts at corporate law firms who need fast, accurate summaries of incoming legal documents to accelerate review.

- **Customer Details:**
  - Receives >100 documents/day (.pdf, .docx)
  - Needs to look up previous judgements and their summaries by searching key words
  - Requires summary within minutes of upload
  - Ground-truth labels (expert summaries) available after review

**Design Influences:** Data size, latency requirements, retraining frequency.

## 2. Scale

- **Offline Data:** 10 GB raw documents (~20 K files)
- **Model Size:** Fine-tuned Llama-2-7B; training takes uses 2×A100 GPUs
- **Deployment Throughput:** ~500 inference requests/day (~1 req/min)

---

## 3. Architecture Diagram

![Untitled-2025-05-11-1923](https://github.com/user-attachments/assets/1c2e2095-0107-4f7b-ab7e-435c30a7360b)

## 4. Infrastructure & IaC

Provisioning and configuration via Terraform and Ansible:

- **Terraform:** [`Terraform configurations, variables, setting - DAY 0`](https://github.com/shettynitis/LLM_LegalDocSummarization/tree/main/ci-cd/tf/kvm)
- **Ansible Playbooks:** [`Ansible notebooks`](https://github.com/shettynitis/LLM_LegalDocSummarization/tree/main/ci-cd/ansible)
- **Argo CD**: [`Argo CD notebooks for 3 environments`](https://github.com/shettynitis/LLM_LegalDocSummarization/tree/main/ci-cd/ansible/argocd)

---

## 5. Persistent Storage

On Chameleon:

- **Object Store:** 

[Notebook with instructions to create and access object store, code to download dataset, preprocess, partition and store in the object store created](https://github.com/shettynitis/LLM_LegalDocSummarization/blob/main/1_data_pipeline/2_create_and_connect_to_object_store.ipynb)

Structure and contents:

[object-persist-project33](https://chi.tacc.chameleoncloud.org/project/containers/container/object-persist-project33/)
```
├── production.jsonl
├── test.jsonl
└── train.jsonl
```

- **Block Volume:**
 [Notebook with instructions to create, partition, add file system, access, run containers on block volume](https://github.com/shettynitis/LLM_LegalDocSummarization/blob/main/1_data_pipeline/3_create_block_storage.ipynb)

We created a block volume of 50 GiB initially, but extended to 100 GiB to store our ONNX model, RAG data, etc.

Structure and contents:

[block-persist-project33]([https://chi.tacc.chameleoncloud.org/project/containers/container/object-persist-project33/](https://kvm.tacc.chameleoncloud.org/project/volumes/f82ca11f-e219-4f3d-ba3e-5ba645f324d6/))
```
├── minio_data
  ├── mlflow-artifacts
  └── ray
├── postgres_data
└── rag_data
  ├── model_rag
    ├── index_to_doc.pkl
    └── legal-facts.index
  └── rag_chunks  
```
Mlflow artifacts folder contains all the artifacts generated during serving and training.
Ray folder contains as ray train related checkpoints
Postgress folder contains
Rag Data folder contains all data related to our RAG model `sentence-transformers/all-MiniLM-L6-v2` which includes the data chunks, vector db, mapping info.

---

## 6. Offline Data

### Training Dataset & Data Lineage

We use the **Zenodo Indian & UK Legal Judgments Dataset** containing ~20K court cases and corresponding human-written summaries.

- **Sources:** `IN-Abs`, `UK-Abs`, and `IN-Ext`
- **Data Size:** ~10 GB total, over 20,000 legal documents and associated summaries.
- **Format:** Paired `.txt` files for full judgments and summaries

#### Example Sample (`train.jsonl`)
```json
{
  "filename": "UKCiv2012.txt",
  "judgement": "The claimant seeks damages following breach of contract. The court heard evidence from both parties. After reviewing the statutory framework and case law precedent, the court finds that the defendant did not fulfill their obligations...",
  "summary": "The defendant breached the contract. The court awarded damages to the claimant.",
  "meta": {
    "doc_words": 2176,
    "sum_words": 132,
    "ratio": 0.06
  }
}
```
#### Relation to Customer Use Case

Our target user (a legal analyst at a law firm) regularly deals with such long-form judicial decisions. The Zenodo dataset closely mirrors their real-world workflow:
	•	They review lengthy judgment documents daily.
	•	They generate or consume summaries internally for client reporting.
	•	Our model mimics this process by learning from historic summaries.

#### About Production Samples

Production samples (the 10% test set):
	•	Contain no ground-truth summaries at inference time.
	•	In a deployed setting, these would represent new unseen judgments uploaded by users.
	•	Once reviewed by a human expert, feedback summaries could be used to retrain the model thus closing the feedback loop.

---

## 7. Data Pipeline

#### Processing Pipeline

Steps handled in [`data_preprocessing.py`](https://github.com/shettynitis/LLM_LegalDocSummarization/blob/main/1_data_pipeline/data_preprocessing.py):

1. **Ingestion:** Load documents from raw folders.
2. **Merging:** Combine segment-wise summaries if full summary not available.
3. **Cleaning:** Normalize unicode, remove extra whitespace, lowercase.
4. **Sanity checks:** Remove empty/duplicate/missing files.
5. **Filtering:** Retain samples with 50–1500 summary words and acceptable doc:summary ratios.
6. **Split:** 70% train, 20% test, 10% production — written to `*.jsonl`.

### Data Pipeline Overview

```
         ┌──────────────────────────────┐
         │    Raw Zenodo Dataset        │
         │  (/data/raw/* subfolders)    │
         └────────────┬─────────────────┘
                      │
                      ▼
         ┌──────────────────────────────┐
         │   Ingestion & File Loading   │
         │ - Load judgment + summary    │
         │ - Handle IN-Abs, UK-Abs,     │
         │   IN-Ext variants            │
         └────────────┬─────────────────┘
                      │
                      ▼
         ┌──────────────────────────────┐
         │     Merging Segment-wise     │
         │ - Combine partial summaries  │
         │   (facts, statute, etc.)     │
         └────────────┬─────────────────┘
                      │
                      ▼
         ┌──────────────────────────────┐
         │         Cleaning Text        │
         │ - Unicode normalization      │
         │ - Lowercasing                │
         │ - Remove extra whitespace    │
         └────────────┬─────────────────┘
                      │
                      ▼
         ┌──────────────────────────────┐
         │       Sanity Checks          │
         │ - Remove empty/missing files │
         │ - Check for duplicates       │
         └────────────┬─────────────────┘
                      │
                      ▼
         ┌──────────────────────────────┐
         │       Statistical Filter     │
         │ - 50–1500 summary words      │
         │ - Ratio: 1–50% of doc length │
         └────────────┬─────────────────┘
                      │
                      ▼
         ┌──────────────────────────────┐
         │        Split & Dump          │
         │ - 70% train                  │
         │ - 20% test                   │
         │ - 10% production             │
         │ → Output as `.jsonl` files   │
         └──────────────────────────────┘
```

---

## 8. Model Training

### 8.1 Provisioning our resources and Jupyter container
- We spin up our Ray head and worker nodes (each with 1×A100 GPU) using a small Jupyter notebook: [`Ray-Train/start_ray`](Ray-Train/start_ray.ipynb)

- We use the following notebook to submit our Ray job: [`Ray-Train/submit_ray`](Ray-Train/submit_ray.ipynb)

### 8.2 Fine-tuning with LoRA + Ray Train + Lightning + MLflow

- **Training script:** [`Ray-Train/sft_train_llama`](Ray-Train/sft_train_llama.py)
- **Frameworks:** PyTorch Lightning, Ray Train (DDP + fault‐tolerance), PEFT (LoRA), MLflow for experiment tracking  
- **Checkpointing:**  
  - We save both the best `val_loss` and the last epoch into `./checkpoints/` via Lightning’s `ModelCheckpoint(save_top_k=1, save_last=True)` callback.  
  - On worker restarts, Ray will supply the last checkpoint directory and Lightning will resume from `checkpoints/last.ckpt`.  

- **Logging:**  
  - Metrics (train/val loss, epochs) and checkpoint paths are automatically logged to MLflow via the `MLFlowLogger`.  
 

### 8.3 Experiment Tracking

- Compare runs in [`mlruns/`](http://129.114.25.240:8000/#/experiments/2?searchFilter=&orderByKey=attributes.start_time&orderByAsc=false&startTime=ALL&lifecycleFilter=Active&modelVersionFilter=All+Runs&datasetsFilter=W10%3D)

### 8.4 Retrain Code

- Retrain Yaml: [`train.yml`](ci-cd/workflows/train-model.yaml)
- Retrain-code: [`train.py`](ci-cd/flow(1).py)

---

## 9. Model Serving & Evaluation

### 9.1 API Endpoint

- FastAPI app: [`Code where we are creating the Fast API`](https://github.com/shettynitis/LLM_LegalDocSummarization/blob/main/Serving/fastapi_onnx/app/main.py)
- Dockerfile: [`Dockerfile`](https://github.com/shettynitis/LLM_LegalDocSummarization/blob/main/Serving/fastapi_onnx/Dockerfile)
- Input: User Prompt appended with RAG output
- Output: summary text

### 9.2 Offline Evaluation

- PyTest suite: [`tests/test_offline_eval.py`](tests/test_offline_eval.py)
- ROUGE scores logged to MLflow

### 9.3 Load Testing

- Locust script: [`tests/load_test.py`](tests/load_test.py)
- Results: [`experiments/load_test_report.html`](experiments/load_test_report.html)

### 9.4 Business-Specific Evaluation

- Evaluation plan: [`docs/business_eval.md`](docs/business_eval.md)

### 9.5 Staged Deployment

- Staging deployment: [`infrastructure/ansible/deploy_staging.yml`](infrastructure/ansible/deploy_staging.yml)

---

## 10. Online Data & Monitoring

- **Online Ingestion:** New documents POSTed to `/ingest` endpoint
- **Monitoring Dashboards:** Grafana dashboards in [`monitoring/grafana/`](monitoring/grafana/)
- **Data Drift:** [`src/monitor/drift.py`](src/monitor/drift.py)

---

## 11. CI/CD & Continuous Training

- GitHub Actions workflow: [`CI git merge test`]([.github/workflows/ci_cd.yml](https://github.com/shettynitis/LLM_LegalDocSummarization/blob/main/.github/workflows/ci.yml))
- Triggers: push to `main` → tests → build Docker images → deploy to staging
- Flask App: We have a flask app, which takes input from user, looks up on RAG, appends it to the user promt, sends the request with the new promt to our ONNX model through FastAPI, which then returns the summary. The summary is then appended to the UI, and user has the option to download the summary text. [Code](https://github.com/shettynitis/LLM_LegalDocSummarization/tree/main/app_flask)

---

## 12. Getting Started on Chameleon
