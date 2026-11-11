# Phase 4: court judgment prediction

The retained model is the verified LoRMA solution. It adapts the original Phase 1 Gated DeltaNet/attention foundation with output-side LoRMA+, using rank 16 and alpha 16. Complete cases are processed in ordered 2,048-token chunks, followed by a bidirectional LSTM and learned attention pooling for binary classification.

The selected checkpoint is epoch 4. Development Macro-F1 was **0.819613**; the recorded portal Macro-F1 was **0.7927783153**, precision **0.8181818182**, and recall **0.7559055118**.

## Inference

`weights/task.pt` contains the exact selected task tensors and buffers, without optimizer or RNG state. The shared foundation model, tokenizer and manifest are included under the repository's `assets/foundation/` directory. Labels retain the course's original integer IDs 0 and 1.

From this directory, in the repository's CUDA environment:

```bash
python predict.py --input /path/to/test.json --output prediction.json
```

The input is a JSON list with string `id` and `text` fields. Released test labels, if present, are discarded. All case content is retained. Output is an ordered list of `{id,pred}` records, with strict integer 0/1 predictions. Existing outputs are not overwritten. Only test documents and the included assets are needed for inference. CUDA/FLA is the original inference backend; `--device cpu --reference` enables the slow CPU reference implementation for small diagnostics.

## Training

Training data is not included. The downloader uses the original pinned course dataset revision. Authenticate with Hugging Face if access is required.

```bash
python scripts/download_data.py
python scripts/prepare_data.py
FLA_DISABLE_BACKEND_DISPATCH=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 torchrun --standalone --nproc_per_node=3 -m cjpe.train --config configs/train.json
```

The retained configuration uses three GPU ranks, 12 cases per global update, six encoder chunks per microbatch, and the original development-based early stopping schedule. Resume a newly trained packaged run with `--resume`; training creates its own complete recovery checkpoints. The compact included checkpoint is for inference and does not contain optimizer or RNG state.

`scripts/evaluate.py` is the course evaluator. The LoRMA attribution and license declaration are preserved under `cjpe/vendor/lorma/`. The included task weights reproduce the selected model.
