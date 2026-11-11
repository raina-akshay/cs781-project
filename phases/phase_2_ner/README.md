# Phase 2: legal named entity recognition

The selected model uses the shared Gated DeltaNet–attention foundation, rank-16 LoRMA+ adapters, a character-level bidirectional LSTM, and direct span supervision with coefficient 0.1. The holdout selected 26 epochs. The submitted model was then trained afresh on all 70 released training documents for that duration, retaining the 40-epoch learning-rate schedule. Its portal macro F1 was **0.5307413631**.

## Predict

Install the repository dependencies and run from this directory:

```bash
python predict.py --input /path/to/fold3-text.json --output prediction.json
```

Input is a JSON object mapping document IDs to unchanged text strings. Output maps each ID to `[start, end, label]` entity spans; offsets are Python Unicode character positions with an exclusive end. Inference requires a CUDA GPU with BF16 support and the pinned FLA implementation. The original 1,024-token windows, 256-token overlap, character BIO decoding, and leading-whitespace policy are preserved.

`weights/model.pt` contains only the selected model's 171 learned task tensors, character vocabulary, ordered labels, and configuration. The 213M foundation and tokenizer are loaded from `../../assets/foundation/`; no training data or cache is needed for prediction. Full reconstructed model tensors were checked for exact equality with the original checkpoint. Optimizer and random-state dumps are excluded.

## Prepare data and train

Obtain the official Phase 2 data described in the project guidelines. Keep its `Train/` and `Test/` structure, then run:

```bash
python prepare_data.py --raw-dir /path/to/Phase_2_LNER
python -m shared.s7_train --config configs/baseline.json
```

`configs/split.json` preserves the original fixed 56-document training and 14-document validation split. The trainer selects a duration on the holdout and then performs the full-data refit; a new training run may select a different duration. Training uses BIO cross-entropy, boundary BCE, entity-role cross-entropy, boundary ranking and direct span ranking. Their character, entity and comparison denominators are kept separate. `evaluation/evaluate_phase2.py` is the official scorer.
