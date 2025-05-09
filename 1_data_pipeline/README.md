# Data Pipeline

This folder contains everything you need to spin up and manage our end-to-end data pipeline for legal-text model training. You’ll find notebooks and scripts to:

- **Create and tear down** a VM on KVM@TACC  
- **Download, preprocess, and store** raw Zenodo data in an object store  
- **Provision block storage** and run core services (MinIO, PostgreSQL, MLflow, Jupyter)  
- **Clean up** once you’re done  

---

## 1. Create the Server  
**Notebook:** `1_create_server.ipynb`  
Launches a new VM on KVM@TACC, installs prerequisites, and prepares the environment for the rest of the pipeline.

---

## 2. Object Store & Data Preprocessing  
**Notebook:** `2_create_and_connect_to_object_store.ipynb`  
- Creates an S3-compatible object store  
- Mounts it locally using `rclone`  
- Runs `docker-compose-data.yaml` with two services:  
  - **process-data** (calls `data_preprocessing.py`)  
  - **load-data** (pushes processed JSONL files into the object store)  

### `data_preprocessing.py`  
Ingests the raw Zenodo legal-text dump, then:  
1. Finds and (if needed) merges judgment–summary pairs  
2. Drops empty or duplicate files  
3. Cleans, normalizes, and lowercases the text  
4. Computes word/sentence counts and filters out outliers  
5. Shuffles and splits the data into 70% train, 20% test, 10% production  
6. Writes each split to a JSONL file (with metadata) for downstream use  

---

## 3. Block Storage & Core Services  
**Notebook:** `3_create_block_storage.ipynb`  
Shows how to compose and run `docker-compose-block.yaml`, which brings up:  
- **MinIO** (S3-compatible store, data in `/mnt/block/minio_data`, console on 9001)  
- **Bucket helper** (waits for MinIO, then ensures an `mlflow-artifacts` bucket exists)  
- **PostgreSQL** (MLflow metadata, stored at `/mnt/block/postgres_data`)  
- **MLflow server** (installs `psycopg2-binary` & `boto3` on startup, uses PostgreSQL + MinIO)  
- **Jupyter PyTorch notebook** (mounts your workspace and a read-only legal dataset, points `MLFLOW_TRACKING_URI` at the MLflow server so you can log experiments)

---

## 4. Delete the Server  
**Notebook:** `4_delete_vm.ipynb`  
Cleanly shuts down and removes the VM instance created in step 1.

---