"""Complete-token label evidence with ordered shared chunk context.

Every case owns an ordered run of lossless chunks. Token-level attention is
summarized inside each checkpointed encoder microbatch, then combined exactly
across chunks after the original mean/EOS BiLSTM supplies shared context.
Only original-case statute sets supervise training; labels never enter forward.
"""

import math
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.checkpoint import checkpoint

from .adaptation import apply_eva_lora
from .foundation import causal_features, load_own_foundation, pool_chunk_features


def token_attention_summary(values, scores, token_mask):
    """Return exact per-chunk token means and log normalizers.

    ``values`` is [chunks,tokens,width], ``scores`` [chunks,labels,tokens].
    Keeping each local log normalizer is essential: an ordinary average of
    independently normalized chunk means is a different attention mechanism.
    Production callers use FP32. This dtype-preserving primitive also permits
    literal FP64 output/gradient checks without changing the algorithm.
    """
    masked = scores.masked_fill(~token_mask[:, None], -torch.inf)
    log_masses = torch.logsumexp(masked, dim=-1)
    weights = torch.softmax(masked, dim=-1)
    local_values = torch.bmm(weights, values)
    return local_values, log_masses


def aggregate_label_summaries(local_values, log_masses, contexts, context_queries, case_offsets):
    """Exact complete-document attention with shared chunk-context conditioning.

    For token t in chunk c and label l the implicit global score is
    ``token_score[c,l,t] + dot(context_queries[l], contexts[c])`` and the
    value is ``token_value[c,t] + contexts[c]``. The supplied log masses
    make the compact result exactly equal to flattening all real tokens.
    """
    offsets = case_offsets.detach().to(device="cpu").tolist()
    scores = log_masses + torch.einsum("cd,ld->cl", contexts, context_queries)
    features, attentions = [], []
    # Avoid a [cases,max_chunks,labels,width] padded value allocation for a
    # batch containing one very long case and many short ones.
    for start, end in zip(offsets, offsets[1:]):
        weights = scores[start:end].transpose(0, 1).softmax(dim=-1)
        values = local_values[start:end] + contexts[start:end, None, :]
        features.append(torch.einsum("lc,cld->ld", weights, values))
        attentions.append(weights.transpose(0, 1))
    attention = nn.utils.rnn.pad_sequence(attentions, batch_first=True).transpose(1, 2)
    return torch.stack(features, dim=0), attention


class CaseHead(nn.Module):
    def __init__(self, model_width, hidden_size=256, dropout=0.1, prior_probabilities=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.token_norm = nn.LayerNorm(model_width)
        self.token_projection = nn.Linear(model_width, hidden_size)
        self.token_activation = nn.GELU()
        self.token_queries = nn.Parameter(torch.empty(100, hidden_size))
        nn.init.normal_(self.token_queries, std=1.0 / math.sqrt(hidden_size))
        self.input_norm = nn.LayerNorm(2 * model_width)
        self.projection = nn.Linear(2 * model_width, hidden_size)
        self.activation = nn.GELU()
        self.projection_dropout = nn.Dropout(dropout)
        self.context_lstm = nn.LSTM(
            hidden_size, hidden_size // 2, num_layers=1, batch_first=True, bidirectional=True
        )
        self.context_norm = nn.LayerNorm(hidden_size)
        self.context_queries = nn.Parameter(torch.empty(100, hidden_size))
        nn.init.normal_(self.context_queries, std=1.0 / math.sqrt(hidden_size))
        self.evidence_norm = nn.LayerNorm(hidden_size)
        self.output_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 100)
        nn.init.normal_(self.classifier.weight, std=0.02)
        with torch.no_grad():
            if prior_probabilities is None:
                self.classifier.bias.zero_()
            else:
                prior = torch.as_tensor(prior_probabilities, dtype=torch.float32)
                if (
                    prior.shape != (100,)
                    or not bool(torch.isfinite(prior).all())
                    or bool(((prior <= 0) | (prior >= 1)).any())
                ):
                    raise ValueError(
                        "TRAIN prior must have100 finite probabilities strictly between0and1"
                    )
                self.classifier.bias.copy_(torch.logit(prior))

    def summarize_tokens(self, hidden, lengths):
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            hidden = hidden.to(dtype=self.token_projection.weight.dtype)
            values = self.token_activation(self.token_projection(self.token_norm(hidden)))
            scores = torch.einsum("ctd,ld->clt", values, self.token_queries)
            positions = torch.arange(hidden.shape[1], device=hidden.device)[None]
            content = (positions > 0) & (positions < lengths[:, None] - 1)
            # Empty text uses its EOS state as the sole evidence token.
            mask = content | ((lengths == 2)[:, None] & (positions == lengths[:, None] - 1))
            return token_attention_summary(values, scores, mask)

    def chunk_context(self, pooled, case_offsets):
        counts = (case_offsets[1:] - case_offsets[:-1]).to(device="cpu", dtype=torch.long)
        width = int(counts.max())
        projected = self.projection_dropout(
            self.activation(self.projection(self.input_norm(pooled)))
        )
        # pad_sequence keeps gradients for every real chunk and original order.
        sequences = list(projected.split(counts.tolist(), dim=0))
        dense = nn.utils.rnn.pad_sequence(sequences, batch_first=True)
        packed = pack_padded_sequence(dense, counts, batch_first=True, enforce_sorted=False)
        encoded, _ = self.context_lstm(packed)
        encoded, _ = pad_packed_sequence(encoded, batch_first=True, total_length=width)
        encoded = self.context_norm(encoded)
        return torch.cat([row[:count] for row, count in zip(encoded, counts.tolist())], dim=0)

    def forward(self, pooled, case_offsets, token_values, token_log_masses):
        with torch.autocast(device_type=pooled.device.type, enabled=False):
            offsets = case_offsets
            dtype = self.projection.weight.dtype
            pooled = pooled.to(dtype=dtype)
            context = self.chunk_context(pooled, offsets)
            features, attention = aggregate_label_summaries(
                token_values.to(dtype=dtype),
                token_log_masses.to(dtype=dtype),
                context,
                self.context_queries,
                offsets,
            )
            features = self.evidence_norm(features)
            # Each statute scores its own evidence, with an independent logit.
            logits = (self.output_dropout(features) * self.classifier.weight[None]).sum(
                -1
            ) + self.classifier.bias
        return {"logits": logits, "chunk_attention": attention, "case_features": features}


class StatuteClassifier(nn.Module):
    def __init__(self, config, load_foundation=True, initialize_adapters=True):
        super().__init__()
        self.backbone, self.model_config, self.foundation_report = load_own_foundation(
            config, load_foundation
        )
        self.foundation_manifest_path = self.foundation_report["manifest_path"]
        inputs = config.get("input", {})
        self.maximum_length = inputs.get("chunk_length", 1024)
        if (
            type(self.maximum_length) is not int
            or not 2 <= self.maximum_length <= self.model_config.context_length
        ):
            raise ValueError("Declared chunk length must fit the original own model context")
        adaptation = config.get("adaptation", {})
        training = config.get("training", {})
        self.encoder_micro_chunks = training.get("encoder_micro_chunks", 1)
        self.encoder_checkpointing = training.get("gradient_checkpointing", True)
        if self.encoder_checkpointing and self.backbone.gradient_checkpointing:
            raise ValueError("Select whole-encoder OR per-block checkpointing, not both")
        self.adaptation_report = apply_eva_lora(
            self.backbone,
            rank=adaptation.get("rank", 16),
            alpha=adaptation.get("alpha", 16),
            seed=training.get("seed", 42),
            initialize=initialize_adapters,
            rank_pattern=adaptation.get("rank_pattern"),
            initialization_path=adaptation.get("initialization_path"),
            initialization_sha256=adaptation.get("initialization_sha256"),
            expected_foundation_sha256=self.foundation_report["checkpoint_sha256"],
        )
        head = config.get("head", {})
        hidden, dropout = head.get("hidden_size", 256), head.get("dropout", 0.1)
        prior = None
        if head.get("initial_bias", "train_prevalence") == "train_prevalence":
            manifest = json.loads(Path(config["data_manifest"]).read_text())
            prior = [
                n / manifest["splits"]["train"]["documents"]
                for n in manifest["train_label_frequencies"]
            ]
        elif head.get("initial_bias") != "zeros":
            raise ValueError("Unsupported label-bias initialization")
        with torch.random.fork_rng(devices=[]):
            # Preserve the caller's per-rank CUDA dropout stream while making
            # the newly initialized CPU head identical on every rank.
            torch.random.default_generator.manual_seed(training.get("seed", 42))
            self.head = CaseHead(self.model_config.d_model, hidden, float(dropout), prior)
        total = self.count_parameters()
        trained = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        head_trained = sum(
            parameter.numel() for parameter in self.head.parameters() if parameter.requires_grad
        )
        self.parameter_report = {
            "parameters": total,
            "trainable_parameters": trained,
            "trainable_head_parameters": head_trained,
            "trainable_backbone_adapter_parameters": trained - head_trained,
            "frozen_parameters": total - trained,
            "trainable_fraction": trained / total,
        }
        self._weights_ready = load_foundation

    def count_parameters(self):
        return sum(parameter.numel() for parameter in self.parameters())

    def head_parameters(self):
        yield from self.head.parameters()

    def load_state_dict(self, state_dict, strict=True, assign=False):
        result = super().load_state_dict(state_dict, strict=True, assign=assign)
        self._weights_ready = True
        return result

    def backbone_features(self, input_ids, lengths):
        if not self._weights_ready:
            raise RuntimeError("Production schema needs strict task restoration before use")
        if input_ids.ndim != 2 or input_ids.shape[1] > self.maximum_length:
            raise ValueError("Chunk input exceeds the declared input cap")
        return causal_features(self.backbone, self.model_config, input_ids, lengths)

    def _encode_pooled(self, input_ids, lengths):
        hidden = self.backbone_features(input_ids, lengths)
        with torch.autocast(device_type=input_ids.device.type, enabled=False):
            pooled = pool_chunk_features(hidden, lengths)
            token_values, token_log_masses = self.head.summarize_tokens(hidden, lengths)
            return pooled, token_values, token_log_masses

    def encode_chunks(self, input_ids, attention_mask):
        if (
            not isinstance(input_ids, torch.Tensor)
            or input_ids.ndim != 2
            or input_ids.dtype != torch.long
            or min(input_ids.shape) < 1
        ):
            raise ValueError("input_ids must be nonempty long [chunks,tokens]")
        if input_ids.shape[1] > self.maximum_length:
            raise ValueError("Chunk input exceeds its declared cap")
        if (
            not isinstance(attention_mask, torch.Tensor)
            or attention_mask.dtype != torch.bool
            or attention_mask.shape != input_ids.shape
            or attention_mask.device != input_ids.device
        ):
            raise ValueError("attention_mask must be matching device-local bool [chunks,tokens]")
        lengths = attention_mask.sum(dim=1).to(dtype=torch.long)
        expected = (
            torch.arange(input_ids.shape[1], device=input_ids.device)[None] < lengths[:, None]
        )
        if not torch.equal(expected, attention_mask):
            raise ValueError("Only a contiguous valid prefix and right padding are supported")
        pooled, token_values, token_log_masses = [], [], []
        for start in range(0, input_ids.shape[0], self.encoder_micro_chunks):
            local_lengths = lengths[start : start + self.encoder_micro_chunks]
            # Do not run needless trailing padding through the entire backbone.
            width = max(2, int(local_lengths.max()))
            local_ids = input_ids[start : start + self.encoder_micro_chunks, :width]
            if self.training and torch.is_grad_enabled() and self.encoder_checkpointing:
                value = checkpoint(
                    self._encode_pooled,
                    local_ids,
                    local_lengths,
                    use_reentrant=False,
                    preserve_rng_state=True,
                )
            else:
                value = self._encode_pooled(local_ids, local_lengths)
            pooled.append(value[0])
            token_values.append(value[1])
            token_log_masses.append(value[2])
        return (
            torch.cat(pooled, dim=0),
            torch.cat(token_values, dim=0),
            torch.cat(token_log_masses, dim=0),
        )

    @staticmethod
    def _validate_offsets(case_offsets, chunks):
        if (
            not isinstance(case_offsets, torch.Tensor)
            or case_offsets.ndim != 1
            or case_offsets.dtype != torch.long
            or case_offsets.numel() < 2
        ):
            raise ValueError("case_offsets must be long [cases+1]")
        offsets = case_offsets.detach().to(device="cpu")
        if (
            int(offsets[0]) != 0
            or int(offsets[-1]) != chunks
            or bool((offsets[1:] <= offsets[:-1]).any())
        ):
            raise ValueError(
                "Every case must own a nonempty contiguous chunk run, covering all chunks"
            )
        return offsets

    def forward(self, batch):
        input_ids = batch["input_ids"]
        offsets = self._validate_offsets(batch["case_offsets"], input_ids.shape[0])
        pooled, token_values, token_log_masses = self.encode_chunks(
            input_ids, batch["attention_mask"]
        )
        with torch.autocast(device_type=input_ids.device.type, enabled=False):
            return self.head(
                pooled.float(), offsets, token_values.float(), token_log_masses.float()
            )


def build_model(config, load_foundation=True, initialize_adapters=True):
    return StatuteClassifier(config, load_foundation, initialize_adapters)
