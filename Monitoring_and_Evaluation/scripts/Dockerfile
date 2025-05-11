# ---------- base: CUDA 12.4 runtime w/ Python 3.10 ----------
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# basic tools
RUN apt-get update && apt-get install -y git curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY llm_tests/requirements-test.txt /app/

# install deps: ORT GPU wheel + pytest stack
RUN pip install --no-cache-dir --extra-index-url \
      https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-nvidia-12-4/pypi/simple/ \
      "onnxruntime-gpu==1.18.0" \
      -r requirements-test.txt

# copy code, model, data
COPY scripts /app/llm_tests

# default entry: run tests, then register if green
CMD ["python", "-m", "llm_tests.register_if_green"]



