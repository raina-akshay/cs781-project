# CS781: Legal language modelling and applications

This repository contains the selected implementation for each of the project's seven phases, with the weights and small supporting assets needed for inference. All downstream models use the same 213M-parameter Gated DeltaNet and attention hybrid trained in Phase 1.

| Phase | Task | Selected solution |
| --- | --- | --- |
| [1](phases/phase_1_pretraining/) | Legal language-model pretraining | Gated DeltaNet and attention hybrid |
| [2](phases/phase_2_ner/) | Named entity recognition | LoRMA+ with character modelling and direct span supervision; 26-epoch full-data refit |
| [3](phases/phase_3_rhetorical_roles/) | Rhetorical role labelling | Contextual sentence model with rank-stabilized LoRA; epoch 17 |
| [4](phases/phase_4_judgment_prediction/) | Court judgment prediction | Verified LoRMA solution 1; epoch 4 |
| [5](phases/phase_5_statute_identification/) | Statute identification | EVA-LoRA with token-level label attention; validation-selected epoch 10 |
| [6](phases/phase_6_summarization/) | Judgment summarization | LoRMA with TF-IDF source selection; epoch 16 |
| [7](phases/phase_7_retrieval/) | Prior-case retrieval | Partial tuning of the final four blocks, with global and chunk-level retrieval; epoch 8 |

Each phase directory has its own README, selected configuration, training and inference source, evaluation utilities, and task weights. Phase 4 uses the verified LoRMA submission selected for this release. Phase 5 uses the final validation-selected checkpoint. Results and input formats are documented in the individual READMEs.

## Layout

```text
assets/foundation/                 Shared model, tokenizer and manifest
phases/phase_1_pretraining/         Foundation source and generation
phases/phase_2_ner/                 Named entity recognition
phases/phase_3_rhetorical_roles/    Rhetorical role labelling
phases/phase_4_judgment_prediction/ Court judgment prediction
phases/phase_5_statute_identification/ Statute identification
phases/phase_6_summarization/       Judgment summarization
phases/phase_7_retrieval/           Prior-case retrieval
requirements.txt                  Runtime, training and evaluation dependencies
THIRD_PARTY_NOTICES.md             Upstream attribution and licenses
```

The foundation is stored once. Task checkpoints contain the learned adapters, heads, or tuned layers required by their selected models; they reuse unchanged foundation tensors at load time. The tokenizer, NER character vocabulary, statute thresholds, and summarization selector are included. No tensor precision was reduced during export. Datasets, training caches, optimizer states, random-state dumps, repeated foundation copies, and alternative experiment checkpoints are excluded.

## Environment

Use Python 3.11 on Linux. The production inference paths use CUDA and the pinned Flash Linear Attention implementation. Create an environment and install from the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
```

Some phase scripts expose an explicit CPU reference option for short diagnostics. Native CUDA inference is the intended execution mode. Training device counts and data preparation differ by phase; follow that phase's README. The official datasets can be obtained from the sources specified in the project guidelines and the included data-source metadata.

## Run a selected model

Enter its phase directory and use `predict.py` as documented there. The defaults find the included shared foundation and task weights automatically. For example:

```bash
cd phases/phase_2_ner
python predict.py --input /path/to/fold3-text.json --output prediction.json
```

Input datasets are supplied by the caller. Prediction does not require optimizer states or earlier experiment outputs. The compact checkpoints are inference exports; new training uses the included source and configurations.

## Upload this folder to GitHub

The model files are tracked with Git LFS. Install Git LFS before adding files, then run the following from this directory after creating the empty private repository `raina-akshay/cs781-project`:

```bash
git init
git checkout -b main
git lfs install --local
git add .
git commit -m "Add selected solutions and inference weights for all seven phases"
git remote add origin https://github.com/raina-akshay/cs781-project.git
git push -u origin main
```

The included `.gitattributes` stores `*.pt` and `*.joblib` files through LFS. For an existing repository with commits, copy this folder's contents into a clone of that repository and commit there instead. When cloning the published repository, run `git lfs install` and `git lfs pull` to obtain the weights.

## Validation

The retained weights, tokenizer and summarization selector match the selected exports byte for byte. Before-and-after CPU reference checks matched model outputs and the relevant losses, gradients, preprocessing and decoding; short training and resume checks also passed. Full CUDA training was not repeated during cleanup.

See [third-party notices](THIRD_PARTY_NOTICES.md) for upstream code, methods and licenses.
