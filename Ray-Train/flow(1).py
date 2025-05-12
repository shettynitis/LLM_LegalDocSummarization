# flow_retrain_prod.py
import os
import subprocess
import sys
from prefect import flow, task, get_run_logger
from mlflow.tracking import MlflowClient

# ─── CONFIG ─────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = "http://129.114.25.240:8000"
MODEL_REGISTRY_NAME = "LLamaRayModel"         # your registered model name
BASE_MODEL_ALIAS   = "development"            # or whatever alias your “base” model has
TRAIN_SCRIPT       = "sft_train_llama.py"
# ─────────────────────────────────────────────────────────────────────

@task
def fetch_base_model() -> str:
    """
    Download the latest 'development' version of your LoRA base model
    from the registry and return a local path to its artifacts.
    """
    logger = get_run_logger()
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    # get latest version under alias
    versions = client.get_latest_versions(MODEL_REGISTRY_NAME, stages=[BASE_MODEL_ALIAS])
    if not versions:
        raise RuntimeError(f"No model found under alias '{BASE_MODEL_ALIAS}'")
    version = versions[0].version
    model_uri = f"models:/{MODEL_REGISTRY_NAME}/{version}"
    local_dir = os.path.abspath(f"base_model_v{version}")
    logger.info(f"Downloading {model_uri} → {local_dir}")
    mlflow.pyfunc.load_model(model_uri, dst_path=local_dir)
    return local_dir

@task
def run_ray_retrain(base_model_path: str, prod_data_dir: str) -> str:
    """
    Invoke your Ray‐Trainer script in a subprocess, pointing it at prod_data_dir
    and seeding from base_model_path.
    Returns the new MLflow run ID.
    """
    logger = get_run_logger()
    env = os.environ.copy()
    env.update({
        "MLFLOW_TRACKING_URI":   MLFLOW_TRACKING_URI,
        "BASE_MODEL_DIR":        base_model_path,
        "TRAIN_DATA_DIR":        prod_data_dir,
        # you can also pass hyperparams here if you like…
    })
    cmd = [sys.executable, TRAIN_SCRIPT]
    logger.info(f"Launching retrain: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if res.returncode != 0:
        logger.error(res.stderr)
        raise RuntimeError("Ray retrain failed")
    # now pick up latest run
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    runs = client.search_runs(
        experiment_ids=["0"],
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    run_id = runs[0].info.run_id
    logger.info(f"Retrain completed in run {run_id}")
    return run_id

@task
def register_production_model(run_id: str) -> int:
    """
    Register the newly‐trained run as the 'production' version of your model.
    """
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    model_uri = f"runs:/{run_id}/model"
    mv = client.create_model_version(MODEL_REGISTRY_NAME, model_uri, run_id)
    client.transition_model_version_stage(
        name=MODEL_REGISTRY_NAME,
        version=mv.version,
        stage="Production",
        archive_existing_versions=True,
    )
    return mv.version

@flow
def retrain_on_production(prod_data_dir: str):
    """
    1. Download base model
    2. Fine‐tune it on prod_data_dir via Ray
    3. Register the new version in Production stage
    """
    base_model_dir = fetch_base_model()
    run_id         = run_ray_retrain(base_model_dir, prod_data_dir)
    prod_version   = register_production_model(run_id)
    return prod_version

if __name__ == "__main__":
    # e.g. python flow_retrain_prod.py /mnt/production/judgements
    import sys
    prod_dir = sys.argv[1] if len(sys.argv) > 1 else "/mnt/production/judgements"
    version = retrain_on_production(prod_dir)
    print(f"New production model version: {version}")
