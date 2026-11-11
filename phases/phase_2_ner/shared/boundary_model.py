"""Add a separately initialized start/end boundary head."""
import torch
from torch import nn

from .ner_model import CharacterNER


class BoundaryCharacterNER(CharacterNER):
    def __init__(self, *args, boundary_seed=42, **kwargs):
        super().__init__(*args, **kwargs)
        # Keep the inherited head and subsequent dropout RNG identical to the
        # original model. The auxiliary head has its own reproducible CPU RNG.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(boundary_seed + 104729)
            self.boundary_head = nn.Linear(
                2 * self.head_config.bidirectional_lstm_hidden_per_direction, 2
            )
