# Phase 6: Legal summarization

The retained model applies LoRMA to the shared Phase 1 Gated DeltaNet and attention hybrid. A TRAIN-fitted TF-IDF selector chooses source passages, and the model generates a summary in rolling 512-token blocks with 384 tokens of recent summary context.

The selected checkpoint is **epoch 16**. On the 700-document local validation split, BERTScore F1 was **0.8441489023**, ROUGE-1 F1 **0.4753168842**, and ROUGE-L F1 **0.2620746538**. The recorded portal scores were **0.8406609297**, **0.4807160438**, and **0.2634446621**, respectively.

## Predict

Run from this directory with the repository dependencies installed:

```bash
python predict.py --input test.json --output prediction.json --archive submission.zip
```

Input is a JSON list of objects with string `id` and `document` fields. Output contains `id` and `summary`. `--batch-size` defaults to 1. Generation retains the selected model's limit of 6,144 new tokens and no-repeat 8-gram rule. CUDA and the pinned FLA implementation are used for scored inference. `--device cpu --reference` explicitly selects the slow CPU reference implementation for small diagnostics.

Required assets are included: `weights/model.pt` contains the selected LoRMA factors; `weights/selector.joblib` is the original fitted TF-IDF selector; `weights/metadata.json` records their checksums and configuration. The shared foundation and tokenizer are in `../../assets/foundation/`. Exact learned-tensor keys, shapes and dtypes, and the frozen-weight digest are checked when loading. Only load the supplied selector file, whose checksum is verified before deserialization. Inference does not require BERTScore weights or training examples.

## Main source

- `summ/model.py`, `adaptation.py`, and `foundation.py`: original hybrid model reconstruction and LoRMA.
- `summ/selection.py`, `data.py`, and `generation.py`: source selection, prompt formatting, complete-target training windows, and decoding.
- `summ/train.py`, `runtime.py`, `evaluation.py`, and `metrics.py`: training, checkpoint validation, and metric computation.
- `scripts/prepare_data.py`, `configs/baseline.json`, and `metadata/split.json`: retained preparation recipe and fixed TRAIN/validation IDs.
- `evaluation/evaluate.py`: official metric implementation.

For training, obtain the official Phase 6 release and the evaluation-only RoBERTa-large snapshot identified in `metadata/scorer-provenance.json`. Prepare the fixed train/validation split and token caches, then run:

```bash
bash scripts/launch.sh scripts/prepare_data.py --config configs/baseline.json
bash scripts/launch.sh -m torch.distributed.run --standalone --nproc_per_node=3 \
  -m summ.train --config configs/baseline.json
```

Training uses global batch 24, at most 20 epochs, at least 6 epochs, and patience 4. `--resume artifacts/run/latest.pt` restores the model, optimizer, per-rank random state and next batch cursor. The compact inference weights contain no optimizer state and cannot serve as a training-resume checkpoint.

The LoRMA attribution and CC-BY-NC-4.0 declaration are retained under `summ/vendor/lorma/`.
