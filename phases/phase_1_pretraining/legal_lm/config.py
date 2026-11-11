"""Model and training settings for the Gated DeltaNet–attention hybrid."""

from dataclasses import asdict, dataclass, field


@dataclass
class ModelConfig:
    architecture: str = "gated_delta_hybrid"
    vocab_size: int = 32768
    context_length: int = 2560
    n_layers: int = 12
    d_model: int = 1024
    n_heads: int = 16
    d_ff: int = 2816
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    gdn_num_heads: int = 8
    gdn_head_dim: int = 128
    gdn_value_dim: int = 256
    conv_size: int = 4
    rope_theta: float = 10000.0
    gdn_backend: str = "fla"


@dataclass
class TrainConfig:
    seed: int = 42
    max_tokens: int = 200_000_000
    tokens_per_update: int = 32768
    micro_batch_size: int = 2
    learning_rate: float = 3e-4
    min_lr_ratio: float = 0.1
    warmup_fraction: float = 0.02
    betas: tuple = (0.9, 0.95)
    adam_epsilon: float = 1e-8
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    precision: str = "auto"
    gradient_checkpointing: bool = False
    trim_padding: bool = False
    eval_every_steps: int = 250
    eval_max_tokens: int = 2_000_000
    save_every_steps: int = 250
    save_every_minutes: float = 30.0
    log_every_steps: int = 10
    checkpoint_tokens: list = field(
        default_factory=lambda: [25_000_000, 50_000_000, 100_000_000, 200_000_000]
    )

    def __post_init__(self):
        self.betas = tuple(self.betas)
        self.checkpoint_tokens = sorted(set(self.checkpoint_tokens))


@dataclass
class RunConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self):
        return asdict(self)


def load_config(path):
    from .io import load_json

    raw = load_json(path)
    return RunConfig(
        ModelConfig(**raw.get("model", {})),
        TrainConfig(**raw.get("training", {})),
    )
