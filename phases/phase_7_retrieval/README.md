# Phase 7: Prior-case retrieval

Selected method: partial tuning of the Phase 1 Gated DeltaNet and attention hybrid. The model trains the last four blocks and final normalization, represents each case using up to 16 chunks of 512 tokens, and combines global cosine similarity with directed chunk MaxSim. Training uses scheduled positive examples and periodically refreshed hard negatives. The selected development checkpoint was epoch 8; its development Micro-F1@7 was 0.384658143, and the submitted result scored 0.3608978145.

## Files

- `pcr/`: data preparation, model, objective, retrieval scoring, distributed training and inference.
- `configs/baseline.json`: selected experiment configuration.
- `predict.py`: inference with the selected epoch-8 checkpoint.
- `weights/model.pt`: the 51 tuned parameter tensors, without optimizer or training state.
- `weights/metadata.json`: checkpoint identity and configuration.
- `scripts/prepare.py`: token and lexical cache preparation.
- `scripts/launch.sh`: training and final prediction export.
- `scripts/validate_submission.py` and `evaluation/evaluate_phase7.py`: format validation and official development scoring.
- `data/raw/source-manifest.json`: dataset revision and expected file hashes.

## External inputs

Use the shared environment described in the repository README, including PyTorch, NumPy, SciPy and tokenizers. The native Phase 1 model also requires its specified flash-linear-attention environment. Inference uses one CUDA GPU. The existing training protocol requires three CUDA GPUs and retains its original batch size and distributed checks.

Place the six dataset files listed in `data/raw/source-manifest.json` under `data/raw/Train`, `data/raw/Dev` and `data/raw/Test`, preserving their filenames. Dataset files are not included.

The default foundation manifest is `../../assets/foundation/manifest.json`. It provides the original Phase 1 checkpoint, tokenizer and model source directory. Relative paths inside this manifest are resolved from its directory. The tokenizer is expected at `../../assets/foundation/tokenizer.json`; the model source is in `../phase_1_pretraining/legal_lm`. The repository includes the shared foundation weights and the selected retrieval weights. The tokenizer and model-weight compatibility checks are retained.

## Predict with the selected model

Run from this phase directory. Supply the official query file and the complete candidate pool; preserve case IDs and the original candidate order.

```bash
python predict.py --queries data/raw/Test/test.json \
  --candidates data/raw/Test/test_candidates.json --output prediction.json
```

The output maps each original query ID to seven candidate IDs. Inference regenerates chunk representations and embeddings as temporary files; it does not require saved candidate embeddings or lexical data because the selected epoch uses the joint neural score alone. `--work-dir` chooses where these temporary files are stored. Existing output files are not overwritten.

## Train

Run from this phase directory after supplying the six dataset files:

```bash
python scripts/prepare.py
bash scripts/launch.sh
```

The training pipeline selects the development checkpoint, replays its development predictions, creates test rankings, and writes `submission.zip`. A completed run can repeat its final export using:

```bash
bash scripts/launch.sh --export-best
python scripts/validate_submission.py submission.zip --split test
```

`PYTHON` can select the Python executable, and `CS781_CUDA_VISIBLE_DEVICES` can select the three GPU IDs. Checkpoint compatibility checks are retained. Training resume uses the saved configuration, dataset identities and optimizer state. Runs created in the original workspace use different paths and should remain there. The packaged weights were checked tensor by tensor against the selected checkpoint, and the shared foundation and task state were loaded successfully. Applying the packaged ranking code to the original saved scores reproduced all 237 test predictions. A complete GPU inference or training run has not been repeated from this package.

The official evaluator is retained byte-for-byte from the project's Phase 7 evaluation files. The dataset manifest records the source repository and revision.
