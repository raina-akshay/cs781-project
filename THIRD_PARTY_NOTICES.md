# Third-party attribution

This file records the upstream material used by the retained implementations. It does not assign one license to the entire project or relicense third-party code, model assets, or course data. The notices and terms accompanying each component remain applicable.

## Flash Linear Attention and Gated DeltaNet

The Gated DeltaNet implementation follows the projections, convolution, decay and output-gating design in [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention/tree/3e52d5abb6ad85312846f39c3e946c2e9881a823), commit `3e52d5abb6ad85312846f39c3e946c2e9881a823`. CUDA recurrence uses that pinned implementation. Attribution: Songlin Yang, Yu Zhang, Zhiyuan Li and contributors. The project uses its own explicit cache handling and retains a PyTorch reference recurrence for diagnostics. The upstream MIT license is preserved in `phases/phase_1_pretraining/third_party/FLA-LICENSE`.

## LoRMA

The native adapters in Phases 2, 4 and 6 follow [LoRMA](https://github.com/Exploration-Lab/LoRMA/tree/ffb3c1de29a7f2ab880dc504a2e473b5a4b63292), commit `ffb3c1de29a7f2ab880dc504a2e473b5a4b63292`, by Harsh Bihany, Shubham Patel and Ashutosh Modi. The project evaluates the same identity-inflation formula through factorized activations instead of constructing a dense updated weight on every forward pass. The upstream README declares **CC-BY-NC-4.0**. The upstream README declaration and source provenance are retained under the Phase 4, Phase 5 and Phase 6 `vendor/lorma/` directories.

## EVA incremental PCA

Phase 5 uses an unchanged `incremental_pca.py` from [Hugging Face PEFT](https://github.com/huggingface/peft/tree/53c25fe4fdd7c6aa4b40db0560815ea570d32303), version 0.17.1, commit `53c25fe4fdd7c6aa4b40db0560815ea570d32303`. The file is distributed under **Apache-2.0**; its license and provenance are retained in `phases/phase_5_statute_identification/lsi/vendor/eva/`. Its own header credits adaptation from scikit-learn's incremental PCA. The surrounding EVA fitting, masking, rank allocation and native adapter integration are implemented in the project source.

## Retrieval methods

Phase 7 uses independently implemented legal-document chunking, late interaction and lexical fusion. The MaxSim implementation was checked against [FlagEmbedding M3](https://github.com/FlagOpen/FlagEmbedding/tree/897baa67c0ffddbfccf4e6776ff55310d67809d4), commit `897baa67c0ffddbfccf4e6776ff55310d67809d4`; legal evidence matching was informed by [KELLER](https://github.com/ChenlongDeng/KELLER/tree/a7cc63c022ba322ef113014e9d54f6f0c60d7654). The package does not include their pretrained model weights or claim to reproduce those systems.

## Dependencies and course materials

Python dependencies are installed separately using `requirements.txt` and retain their own distribution licenses. Official evaluation scripts and dataset descriptions originate from the CS781 project materials. Course datasets and the evaluation-only RoBERTa-large weights are not redistributed in this package. The retained foundation and selected task weights are this project's trained assets; this notice does not grant additional rights over any external material.
