import pytest
import torch
from torch import nn

from src.utils.model_layer_info import (
    get_all_layer_indices,
    get_num_layers,
    select_every_nth_layer,
    select_first_n_layers,
    select_last_n_layers,
    select_layer_fraction,
    select_middle_n_layers,
)


class TinyTransformer(nn.Module):
    """Minimal model with ``model.layers.N`` naming."""

    def __init__(self, num_layers: int = 4, hidden_size: int = 16):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [self._make_layer(hidden_size) for _ in range(num_layers)]
        )

    @staticmethod
    def _make_layer(hidden_size: int) -> nn.Module:
        layer = nn.Module()
        layer.self_attn = nn.Module()
        layer.self_attn.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        layer.self_attn.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        return layer

    def forward(self, x):
        for layer in self.model.layers:
            x = layer.self_attn.q_proj(x) + layer.self_attn.v_proj(x)
        return x


@pytest.fixture
def model():
    torch.manual_seed(42)
    return TinyTransformer(num_layers=8, hidden_size=16)


def test_get_all_layer_indices(model):
    assert get_all_layer_indices(model) == [0, 1, 2, 3, 4, 5, 6, 7]


def test_get_num_layers(model):
    assert get_num_layers(model) == 8


def test_select_every_nth_layer(model):
    assert select_every_nth_layer(model, n=2) == [0, 2, 4, 6]
    assert select_every_nth_layer(model, n=3) == [0, 3, 6]


def test_select_every_nth_layer_invalid(model):
    with pytest.raises(ValueError, match="n must be >= 1"):
        select_every_nth_layer(model, n=0)


def test_select_first_n_layers(model):
    assert select_first_n_layers(model, n=3) == [0, 1, 2]
    assert select_first_n_layers(model, n=0) == []


def test_select_last_n_layers(model):
    assert select_last_n_layers(model, n=3) == [5, 6, 7]
    assert select_last_n_layers(model, n=0) == []


def test_select_middle_n_layers(model):
    assert select_middle_n_layers(model, n=4) == [2, 3, 4, 5]
    assert select_middle_n_layers(model, n=3) == [2, 3, 4]


def test_select_middle_n_layers_exceeds_total(model):
    assert select_middle_n_layers(model, n=100) == [0, 1, 2, 3, 4, 5, 6, 7]


def test_select_layer_fraction(model):
    assert select_layer_fraction(model, 0.0) == []
    assert select_layer_fraction(model, 1.0) == [0, 1, 2, 3, 4, 5, 6, 7]
    half = select_layer_fraction(model, 0.5)
    assert len(half) == 4


def test_select_layer_fraction_invalid(model):
    with pytest.raises(ValueError, match="fraction must be between"):
        select_layer_fraction(model, 1.5)
