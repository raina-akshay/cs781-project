# Phase 3: rhetorical role prediction

The retained model uses the Phase 1 Gated DeltaNet/attention foundation, rank-stabilized LoRA (rank 16), and a hierarchical contextual classifier. It predicts 13 rhetorical roles using windows with 32 supervised core sentences and 16 context sentences on either side. The selected checkpoint was epoch 17; development Macro-F1 was 0.478214 and the recorded portal Macro-F1 was 0.477776.

## Inference

The selected task tensors are included in `weights/task.pt`. The shared foundation model, tokenizer and manifest are already included under the repository's `assets/foundation/` directory. The task checkpoint has no optimizer or RNG state.

From this directory, in the repository's CUDA environment:

```bash
python predict.py --input /path/to/CL_test.json /path/to/IT_test.json --output prediction.json
```

Each input uses the official test format: a JSON object mapping each document ID to its ordered list of sentence strings. One or both files can be supplied. For a renamed file, specify `--domain CL` or `--domain IT`; the domain is metadata and does not change the model. The output maps original document IDs to equally long lists of literal role labels, in sentence order. Inference needs only these test files and the included assets. No training or development data is loaded. Existing output files are not overwritten.

## Training

Training data is not included. Obtain the seven official files listed with pinned URLs and SHA-256 hashes in `configs/data-sources.json`. Keep each file's `Phase_3_RR/...` path beneath this phase's `data/raw/` directory. Authentication may be needed for the course dataset. Preprocessing checks the tokenizer and preserves all sentence tokens; the model applies its configured head/tail input limit.

```bash
python scripts/prepare_data.py
FLA_DISABLE_BACKEND_DISPATCH=1 python -m rr_context.train --config configs/train.json
```

To resume a newly trained packaged run, append `--resume` to its training command. The retained configuration uses one GPU, at most 60 epochs and development-based early stopping. It runs only the selected context-16 configuration. The compact included checkpoint is for inference; training begins a new run from the shared foundation.

`scripts/evaluate.py` is the course evaluator. `rr_context/` contains the selected model, adapter, training and inference implementation; `rr/` contains its shared data, metric and target-selection helpers. The included task weights reproduce the selected model.
