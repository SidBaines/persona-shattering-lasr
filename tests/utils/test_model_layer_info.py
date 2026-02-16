import pytest
import torch
from torch import nn

from src.utils.model_layer_info import (
    LAYER_PATTERNS,
    extract_layer_idx,
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


# ---------------------------------------------------------------------------
# extract_layer_idx tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, expected",
    [
        ("base_model.model.model.layers.3.self_attn.q_proj", 3),
        ("transformer.h.12.attn.c_attn", 12),
        ("encoder.layer.0.attention.self.query", 0),
        ("encoder.block.5.layer.0.SelfAttention", 5),
        ("decoder.block.7.layer.1.EncDecAttention", 7),
        ("transformer.blocks.2.attn.qkv_proj", 2),
        ("some.unknown.module.name", None),
        ("lm_head", None),
    ],
)
def test_extract_layer_idx(name, expected):
    assert extract_layer_idx(name) == expected


def test_layer_patterns_is_non_empty():
    """Sanity check that LAYER_PATTERNS is populated."""
    assert len(LAYER_PATTERNS) >= 6


# ---------------------------------------------------------------------------
# GPT-2-style model fixture for cross-architecture tests
# ---------------------------------------------------------------------------


class TinyGPT2(nn.Module):
    """Minimal model with ``transformer.h.N`` naming (GPT-2 style)."""

    def __init__(self, num_layers: int = 4, hidden_size: int = 16):
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.h = nn.ModuleList(
            [self._make_block(hidden_size) for _ in range(num_layers)]
        )

    @staticmethod
    def _make_block(hidden_size: int) -> nn.Module:
        block = nn.Module()
        block.attn = nn.Module()
        block.attn.c_attn = nn.Linear(hidden_size, hidden_size, bias=False)
        block.mlp = nn.Module()
        block.mlp.c_fc = nn.Linear(hidden_size, hidden_size, bias=False)
        return block

    def forward(self, x):
        for block in self.transformer.h:
            x = block.attn.c_attn(x) + block.mlp.c_fc(x)
        return x


@pytest.fixture
def gpt2_model():
    torch.manual_seed(42)
    return TinyGPT2(num_layers=6, hidden_size=16)


def test_get_all_layer_indices_gpt2(gpt2_model):
    assert get_all_layer_indices(gpt2_model) == [0, 1, 2, 3, 4, 5]


def test_get_num_layers_gpt2(gpt2_model):
    assert get_num_layers(gpt2_model) == 6


def test_get_all_layer_indices_custom_extractor(gpt2_model):
    """Custom extractor overrides the default pattern matching."""

    def custom(name: str) -> int | None:
        if "c_attn" in name:
            # Only count attention layers, remap to 100+block_idx
            idx = extract_layer_idx(name)
            return 100 + idx if idx is not None else None
        return None

    assert get_all_layer_indices(gpt2_model, layer_idx_extractor=custom) == [
        100,
        101,
        102,
        103,
        104,
        105,
    ]
