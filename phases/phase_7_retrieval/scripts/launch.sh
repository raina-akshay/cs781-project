#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES="${CS781_CUDA_VISIBLE_DEVICES:-0,1,2}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export FLA_DISABLE_BACKEND_DISPATCH=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$PWD/artifacts/triton-cache}"
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${PYTHON:-python}"
# End torchrun option parsing before our script's --run argument, which would
# otherwise be mistaken for an abbreviation of torchrun's --run-path option.
exec "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=3 -- scripts/run.py "$@"
