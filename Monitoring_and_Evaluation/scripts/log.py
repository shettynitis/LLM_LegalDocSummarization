import subprocess, sys, mlflow, os, datetime, pathlib, json, numpy as np
from evaluate import load

RESULT = subprocess.run(["pytest", "-q", "llm_tests/tests"],
                        capture_output=True, text=True)
print(RESULT.stdout)
if RESULT.returncode:
    print("Tests failed – model NOT registered"); sys.exit(1)

# If we’re here everything passed
run_id   = os.getenv("MLFLOW_RUN_ID")           # parent training run
model_dir= pathlib.Path(os.getenv("LLM_MODEL_DIR","../llama-legal-onnx"))
client   = mlflow.tracking.MlflowClient()

with mlflow.start_run(run_id=run_id):
    rouge = load("rouge")
    rows  = json.load(open("latest_eval.json"))   # if you stored details
    mlflow.log_metrics(rows["aggregate"])
    mlflow.set_tag("phase","evaluation")

    mv = client.create_model_version(
        name="llama2_legal_summarizer",
        source=model_dir.as_posix(),
        run_id=run_id,
        description=f"Auto-registered on {datetime.datetime.utcnow()}",
    )
    print("All tests green – registered as version", mv.version)
