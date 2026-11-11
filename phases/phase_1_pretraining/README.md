# Phase 1: Legal language-model pretraining

Selected model: a 213,084,768-parameter Gated DeltaNet and attention hybrid trained from scratch for 1.05 billion supervised token targets. The model has 12 blocks, width 1,024, a 32,768-token vocabulary and a 2,560-token context. Its submitted BERTScore F1 was 0.8325744867.

`legal_lm/` contains the model, tokenizer, data processing, training and generation code. `configs/hybrid.json` contains the selected training configuration. `configs/generation.json` contains the exact submitted decoding settings. The shared trained weights and tokenizer are stored once under `../../assets/foundation/` and reused by the later phases.

## Predict

Install the environment described in the root README and `../../requirements.txt`. The native model requires a CUDA GPU and the specified flash-linear-attention installation.

Run from this directory:

```bash
python predict.py --input /path/to/test.json --output prediction.json
```

The input is the official JSON mapping each example ID to a `text` field containing its 512-word prefix. Output preserves every ID and contains exactly 128 newly generated words in each `text` field. It does not repeat the input prefix. Existing output files are not overwritten.

Generation uses BF16 autocasting, temperature 0.85, top-p 0.95, seed 42 with an independent stable seed for each ID, repetition penalty 1.2, and a four-token repetition constraint. Both repetition controls apply to the generated suffix. The original generator checks complete word boundaries and raises an error if it cannot produce a valid continuation within the context and token limit.

## Training source

The retained command interface exposes the original download and preprocessing stages:

```bash
python -m legal_lm download --output data/raw --revision main
python -m legal_lm prepare --input-dir data/raw --output data/prepared
python -m legal_lm tokenizer --corpus data/prepared --output data/tokenizer
python -m legal_lm tokenize --corpus data/prepared --tokenizer data/tokenizer/tokenizer.json --output data/tokens --context-length 2560
```

The trainer uses the deduplicated ordinary-token view produced by `prepare_stage3`. Build the view after tokenization, then start training:

```bash
python -c "from legal_lm.stage3_data import prepare_stage3; prepare_stage3('data/tokens', 'data/training-view')"
export FLA_DISABLE_BACKEND_DISPATCH=1
export TRITON_CACHE_DIR="$PWD/artifacts/triton-cache"
torchrun --standalone --nproc_per_node=2 -m legal_lm train \
  --config configs/hybrid.json --view data/training-view \
  --run-dir artifacts/training
```

The data split, vocabulary, window order and source sampling affect the result. Reproducing the recorded run requires the same dataset revision and prepared data. New training checkpoints save optimizer, sampler and per-worker random states for resume with `--resume /path/to/latest.pt`.

The supplied compact foundation contains model weights and model metadata, without optimizer or random-state data. It is suitable for inference and downstream initialization, not resuming the original pretraining run. Checkpoint and tokenizer compatibility are checked before inference.
