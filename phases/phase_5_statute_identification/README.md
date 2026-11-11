# Phase 5: Statute identification

The retained model combines the Phase 1 Gated DeltaNet and attention hybrid with EVA-initialized LoRA, token-level label attention, and ordered chunk context. It predicts any subset of 100 statute labels. Documents are processed in complete nonoverlapping chunks of 2,046 content tokens with BOS/EOS, without truncation.

The selected checkpoint is **epoch 10**. Selection-panel Macro-F1 was **0.4343785391**; full-development Macro-F1 was **0.4450731348**, Micro-F1 **0.5356969804**, and subset accuracy **0.07582752185**. Per-label thresholds were fitted on the separate calibration panel. These are local development results.

## Predict

Run from this directory with the repository dependencies installed:

```bash
python predict.py --input test.json --output prediction.json --archive submission.zip
```

Input is a JSON list of objects with string `id` and `text` fields. Output contains `id` and `pred`, where `pred` is a list of integer labels. `--batch-size` controls complete documents per inference batch and defaults to 1. CUDA and the pinned FLA implementation are used for scored inference. `--device cpu --reference` explicitly selects the slow reference implementation for small diagnostics.

Required assets are included: `weights/model.pt` contains the selected trained adapters, task head and calibrated thresholds; `weights/metadata.json` records their configuration and checksum. The shared foundation and tokenizer are in `../../assets/foundation/`. The prediction loader checks the foundation identity, exact learned-tensor keys, shapes and dtypes, and frozen-weight digest before inference. No optimizer state, training data, or EVA fitting-time initialization artifact is needed for prediction.

## Main source

- `lsi/model.py`, `adaptation.py`, and `eva.py`: task head, native adapters, and TRAIN-only EVA fitting.
- `lsi/data.py`, `train.py`, `runtime.py`, and `metrics.py`: full-document batching, training, checkpoint checks, threshold fitting, and evaluation.
- `scripts/prepare_data.py` and `initialize_eva.py`: data preparation and fitting-time initialization.
- `configs/`: retained training recipe and EVA fitting settings.
- `evaluation/evaluate.py`: official metric implementation.

For new training, obtain the official Phase 5 dataset, place the files at the paths in `metadata/raw-data.json`, prepare the token cache, and fit EVA using the included script. Training uses four ranks, global batch 24, and the original 25-epoch maximum with patience 5. From this directory, prepare the cache and fit the basis with:

```bash
bash scripts/launch.sh scripts/prepare_data.py
CUDA_VISIBLE_DEVICES=0 bash scripts/launch.sh scripts/initialize_eva.py \
  --config configs/baseline.json --fit-config configs/eva-fit.json \
  --output artifacts/eva/initialization.pt
```

Set `adaptation.initialization_sha256` and `adaptation.rank_pattern` in the training configuration to the actual fitted artifact's checksum and ranks. Then launch the retained training implementation with `bash scripts/launch.sh -m torch.distributed.run --standalone --nproc_per_node=4 -m lsi.train --config configs/baseline.json`. The compact inference export is a separate format and cannot be used as a training-resume checkpoint; resume uses the full training checkpoint containing optimizer, per-rank random state and the next batch cursor.

EVA's incremental PCA source and Apache-2.0 license are retained under `lsi/vendor/eva/`.
