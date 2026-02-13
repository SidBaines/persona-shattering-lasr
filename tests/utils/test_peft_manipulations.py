import json
import warnings

import pytest
import torch
from peft import LoraConfig, get_peft_model
from torch import nn

from src.utils.peft_manipulations import LoraLayerZeroing, LoraRankReducer, LoraScaling

# ---------------------------------------------------------------------------
# Tiny model for real PEFT tests
# ---------------------------------------------------------------------------


class TinyTransformer(nn.Module):
    """Minimal model with ``model.layers.N`` naming for PEFT compatibility."""

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
        layer.mlp = nn.Module()
        layer.mlp.up_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        layer.mlp.down_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        return layer

    def forward(self, x):
        for layer in self.model.layers:
            attn_out = layer.self_attn.q_proj(x) + layer.self_attn.v_proj(x)
            mlp_out = layer.mlp.down_proj(layer.mlp.up_proj(x))
            x = attn_out + mlp_out
        return x


@pytest.fixture
def peft_model():
    """Tiny PeftModel with LoRA (r=8) on all projections, 4 layers."""
    torch.manual_seed(42)
    base = TinyTransformer(num_layers=4, hidden_size=16)
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(base, config)

    # Initialize lora_B with random values (PEFT initializes to zeros)
    # This gives the LoRA adapter a non-zero contribution for testing
    torch.manual_seed(123)
    for _name, module in _lora_modules(model):
        module.lora_B["default"].weight.data.normal_()

    return model


@pytest.fixture
def peft_model_custom_adapter():
    """PeftModel with a second adapter named 'custom'."""
    torch.manual_seed(42)
    base = TinyTransformer(num_layers=4, hidden_size=16)
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(base, config)
    custom_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model.add_adapter("custom", custom_config)

    # Initialize lora_B for both adapters
    torch.manual_seed(123)
    for _name, module in _lora_modules(model, "default"):
        module.lora_B["default"].weight.data.normal_()
    for _name, module in _lora_modules(model, "custom"):
        module.lora_B["custom"].weight.data.normal_()

    return model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _lora_modules(model, adapter="default"):
    """Yield (name, module) for LoRA modules with the given adapter."""
    for name, module in model.named_modules():
        if (
            hasattr(module, "lora_A")
            and hasattr(module, "lora_B")
            and adapter in module.lora_A
        ):
            yield name, module


def _snapshot_weights(model, adapter="default"):
    """Return {name: {A: tensor, B: tensor}} with cloned weights."""
    return {
        name: {
            "A": module.lora_A[adapter].weight.data.clone(),
            "B": module.lora_B[adapter].weight.data.clone(),
        }
        for name, module in _lora_modules(model, adapter)
    }


def _snapshot_scaling(model, adapter="default"):
    """Return {name: scaling_value} for all LoRA modules."""
    return {
        name: module.scaling[adapter] for name, module in _lora_modules(model, adapter)
    }


def _snapshot_ranks(model, adapter="default"):
    """Return {name: rank} for all LoRA modules."""
    return {name: module.r[adapter] for name, module in _lora_modules(model, adapter)}


def _get_layer_idx(name: str) -> int:
    """Extract layer index from module name."""
    return int(name.split("model.layers.")[-1].split(".")[0])


# ---------------------------------------------------------------------------
# LoraRankReducer tests
# ---------------------------------------------------------------------------


def test_rank_reducer_no_filters_all_modules(peft_model):
    """No filters should affect all modules."""
    reducer = LoraRankReducer(peft_model, adapter_name="default", new_rank=4)
    reducer.apply()

    for _name, module in _lora_modules(peft_model):
        assert module.lora_A["default"].weight.shape[0] == 4
        assert module.lora_B["default"].weight.shape[1] == 4
        assert module.r["default"] == 4

    assert peft_model.peft_config["default"].r == 4


def test_rank_reducer_target_modules_list_only(peft_model):
    """target_modules as list should filter by module type across all layers."""
    originals = _snapshot_weights(peft_model)

    reducer = LoraRankReducer(
        peft_model, adapter_name="default", new_rank=4, target_modules=["q_proj"]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reducer.apply()

    for name, module in _lora_modules(peft_model):
        if name.endswith("q_proj"):
            assert module.lora_A["default"].weight.shape[0] == 4
            assert module.r["default"] == 4
        else:
            assert torch.equal(
                module.lora_A["default"].weight.data, originals[name]["A"]
            )
            assert module.r["default"] == 8


def test_rank_reducer_layers_only(peft_model):
    """layers filter should affect all modules in specified layers."""
    originals = _snapshot_weights(peft_model)

    reducer = LoraRankReducer(
        peft_model, adapter_name="default", new_rank=4, layers=[0, 2]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reducer.apply()

    for name, module in _lora_modules(peft_model):
        layer_idx = _get_layer_idx(name)
        if layer_idx in (0, 2):
            assert module.lora_A["default"].weight.shape[0] == 4
        else:
            assert torch.equal(
                module.lora_A["default"].weight.data, originals[name]["A"]
            )


def test_rank_reducer_target_modules_and_layers_combined(peft_model):
    """Both filters should be ANDed together."""
    originals = _snapshot_weights(peft_model)

    reducer = LoraRankReducer(
        peft_model,
        adapter_name="default",
        new_rank=4,
        target_modules=["q_proj", "v_proj"],
        layers=[1, 3],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reducer.apply()

    for name, module in _lora_modules(peft_model):
        layer_idx = _get_layer_idx(name)
        is_target_module = name.endswith("q_proj") or name.endswith("v_proj")
        is_target_layer = layer_idx in (1, 3)

        if is_target_module and is_target_layer:
            assert module.lora_A["default"].weight.shape[0] == 4
        else:
            assert torch.equal(
                module.lora_A["default"].weight.data, originals[name]["A"]
            )


def test_rank_reducer_target_modules_dict_per_layer(peft_model):
    """Dict mode should allow different modules per layer."""
    originals = _snapshot_weights(peft_model)

    reducer = LoraRankReducer(
        peft_model,
        adapter_name="default",
        new_rank=4,
        target_modules={
            0: ["q_proj"],  # Layer 0: only q_proj
            1: ["v_proj"],  # Layer 1: only v_proj
            2: ["up_proj", "down_proj"],  # Layer 2: MLP projections
            # Layer 3: not specified, no changes
        },
    )
    reducer.apply()

    for name, module in _lora_modules(peft_model):
        layer_idx = _get_layer_idx(name)

        # Check which modules should be modified
        should_modify = False
        if layer_idx == 0 and name.endswith("q_proj"):
            should_modify = True
        elif layer_idx == 1 and name.endswith("v_proj"):
            should_modify = True
        elif layer_idx == 2 and (
            name.endswith("up_proj") or name.endswith("down_proj")
        ):
            should_modify = True

        if should_modify:
            assert module.lora_A["default"].weight.shape[0] == 4
            assert module.r["default"] == 4
        else:
            assert torch.equal(
                module.lora_A["default"].weight.data, originals[name]["A"]
            )
            assert module.r["default"] == 8


def test_rank_reducer_target_modules_dict_raises_with_layers(peft_model):
    """Dict mode + layers parameter should raise ValueError."""
    with pytest.raises(ValueError, match="Cannot specify both"):
        LoraRankReducer(
            peft_model,
            adapter_name="default",
            new_rank=4,
            target_modules={0: ["q_proj"]},
            layers=[0, 1],
        )


def test_rank_reducer_approximates_original_product(peft_model):
    """SVD approximation should preserve B@A product reasonably well."""
    # Initialize lora_B with random values (PEFT initializes to zeros)
    torch.manual_seed(123)
    for _name, module in _lora_modules(peft_model):
        module.lora_B["default"].weight.data.normal_()

    originals = {
        name: module.lora_B["default"].weight.data.clone()
        @ module.lora_A["default"].weight.data.clone()
        for name, module in _lora_modules(peft_model)
    }

    reducer = LoraRankReducer(peft_model, adapter_name="default", new_rank=4)
    reducer.apply()

    for name, module in _lora_modules(peft_model):
        new_product = (
            module.lora_B["default"].weight.data @ module.lora_A["default"].weight.data
        )
        error = (new_product - originals[name]).norm()
        # SVD approximation should have smaller error than original magnitude
        assert error < originals[name].norm()


def test_rank_reducer_restore_exact(peft_model):
    """restore() should return to exact original state."""
    originals = _snapshot_weights(peft_model)
    original_ranks = _snapshot_ranks(peft_model)

    reducer = LoraRankReducer(peft_model, adapter_name="default", new_rank=4)
    reducer.apply()
    reducer.restore()

    for name, module in _lora_modules(peft_model):
        assert torch.equal(module.lora_A["default"].weight.data, originals[name]["A"])
        assert torch.equal(module.lora_B["default"].weight.data, originals[name]["B"])
        assert module.r["default"] == original_ranks[name]

    assert peft_model.peft_config["default"].r == 8


def test_rank_reducer_is_applied_tracking(peft_model):
    """is_applied should track modification state."""
    reducer = LoraRankReducer(peft_model, adapter_name="default", new_rank=4)
    assert not reducer.is_applied

    reducer.apply()
    assert reducer.is_applied

    reducer.restore()
    assert not reducer.is_applied


def test_rank_reducer_idempotent_apply(peft_model):
    """Multiple apply() calls should produce same result."""
    reducer = LoraRankReducer(peft_model, adapter_name="default", new_rank=4)
    reducer.apply()
    after_first = _snapshot_weights(peft_model)

    reducer.apply()

    for name, module in _lora_modules(peft_model):
        assert torch.equal(module.lora_A["default"].weight.data, after_first[name]["A"])
        assert torch.equal(module.lora_B["default"].weight.data, after_first[name]["B"])


def test_rank_reducer_warns_about_peft_config_with_filters(peft_model):
    """Filtered modifications should warn about peft_config.r mismatch."""
    reducer = LoraRankReducer(
        peft_model, adapter_name="default", new_rank=4, target_modules=["q_proj"]
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        reducer.apply()
        assert len(w) == 1
        assert "peft_config.r" in str(w[0].message)

    # peft_config.r should NOT be updated when filters are active
    assert peft_model.peft_config["default"].r == 8


def test_rank_reducer_save_load_roundtrip(peft_model, tmp_path):
    """Reduced-rank adapter should save/load correctly."""
    reducer = LoraRankReducer(peft_model, adapter_name="default", new_rank=4)
    reducer.apply()

    save_dir = tmp_path / "adapter"
    peft_model.save_pretrained(str(save_dir))

    # Check config
    config_path = save_dir / "adapter_config.json"
    with open(config_path) as f:
        config_data = json.load(f)
    assert config_data["r"] == 4

    # Check saved weights
    from safetensors.torch import load_file

    state_dict = load_file(str(save_dir / "adapter_model.safetensors"))
    for key, tensor in state_dict.items():
        if "lora_A" in key:
            assert tensor.shape[0] == 4
        elif "lora_B" in key:
            assert tensor.shape[1] == 4


def test_rank_reducer_custom_adapter_isolation(peft_model_custom_adapter):
    """Modifying one adapter should not affect others."""
    model = peft_model_custom_adapter
    reducer = LoraRankReducer(model, adapter_name="custom", new_rank=4)
    reducer.apply()

    assert model.peft_config["custom"].r == 4
    assert model.peft_config["default"].r == 8

    for _name, module in _lora_modules(model, "custom"):
        assert module.lora_A["custom"].weight.shape[0] == 4
    for _name, module in _lora_modules(model, "default"):
        assert module.lora_A["default"].weight.shape[0] == 8


# ---------------------------------------------------------------------------
# LoraScaling tests
# ---------------------------------------------------------------------------


def test_scaling_no_filters_all_modules(peft_model):
    """No filters should scale all modules."""
    original_scaling = _snapshot_scaling(peft_model)

    scaler = LoraScaling(peft_model, adapter_name="default", scale_factor=2.0)
    scaler.apply()

    for name, module in _lora_modules(peft_model):
        expected = original_scaling[name] * 2.0
        assert module.scaling["default"] == pytest.approx(expected)


def test_scaling_target_modules_list_only(peft_model):
    """target_modules as list should filter by module type."""
    original_scaling = _snapshot_scaling(peft_model)

    scaler = LoraScaling(
        peft_model,
        adapter_name="default",
        scale_factor=5.0,
        target_modules=["q_proj"],
    )
    scaler.apply()

    for name, module in _lora_modules(peft_model):
        if name.endswith("q_proj"):
            assert module.scaling["default"] == pytest.approx(
                original_scaling[name] * 5.0
            )
        else:
            assert module.scaling["default"] == pytest.approx(original_scaling[name])


def test_scaling_layers_only(peft_model):
    """layers filter should affect all modules in specified layers."""
    original_scaling = _snapshot_scaling(peft_model)

    scaler = LoraScaling(
        peft_model, adapter_name="default", scale_factor=0.5, layers=[0]
    )
    scaler.apply()

    for name, module in _lora_modules(peft_model):
        layer_idx = _get_layer_idx(name)
        if layer_idx == 0:
            assert module.scaling["default"] == pytest.approx(
                original_scaling[name] * 0.5
            )
        else:
            assert module.scaling["default"] == pytest.approx(original_scaling[name])


def test_scaling_target_modules_dict_per_layer(peft_model):
    """Dict mode should allow different modules per layer."""
    original_scaling = _snapshot_scaling(peft_model)

    scaler = LoraScaling(
        peft_model,
        adapter_name="default",
        scale_factor=3.0,
        target_modules={
            0: ["q_proj"],
            1: ["v_proj"],
            2: ["up_proj", "down_proj"],
        },
    )
    scaler.apply()

    for name, module in _lora_modules(peft_model):
        layer_idx = _get_layer_idx(name)

        should_scale = False
        if layer_idx == 0 and name.endswith("q_proj"):
            should_scale = True
        elif layer_idx == 1 and name.endswith("v_proj"):
            should_scale = True
        elif layer_idx == 2 and (
            name.endswith("up_proj") or name.endswith("down_proj")
        ):
            should_scale = True

        if should_scale:
            assert module.scaling["default"] == pytest.approx(
                original_scaling[name] * 3.0
            )
        else:
            assert module.scaling["default"] == pytest.approx(original_scaling[name])


def test_scaling_negative_scale_factor(peft_model):
    """Negative scale factor should invert LoRA direction."""
    original_scaling = _snapshot_scaling(peft_model)

    scaler = LoraScaling(peft_model, adapter_name="default", scale_factor=-1.0)
    scaler.apply()

    for name, module in _lora_modules(peft_model):
        expected = original_scaling[name] * -1.0
        assert module.scaling["default"] == pytest.approx(expected)


def test_scaling_zero_scale_disables(peft_model):
    """Scale factor of 0 should disable adapter."""
    scaler = LoraScaling(peft_model, adapter_name="default", scale_factor=0.0)
    scaler.apply()

    for _name, module in _lora_modules(peft_model):
        assert module.scaling["default"] == 0.0


def test_scaling_restore_exact(peft_model):
    """restore() should return to exact original scaling."""
    original_scaling = _snapshot_scaling(peft_model)

    scaler = LoraScaling(peft_model, adapter_name="default", scale_factor=3.0)
    scaler.apply()
    scaler.restore()

    for name, module in _lora_modules(peft_model):
        assert module.scaling["default"] == pytest.approx(original_scaling[name])


def test_scaling_idempotent_apply(peft_model):
    """Multiple apply() calls should produce same result."""
    scaler = LoraScaling(peft_model, adapter_name="default", scale_factor=2.0)
    scaler.apply()
    after_first = _snapshot_scaling(peft_model)

    scaler.apply()

    for name, module in _lora_modules(peft_model):
        assert module.scaling["default"] == pytest.approx(after_first[name])


# ---------------------------------------------------------------------------
# LoraLayerZeroing tests
# ---------------------------------------------------------------------------


def test_zeroing_layers_only_zeros_specified(peft_model):
    """layers parameter should zero specified layers."""
    zeroer = LoraLayerZeroing(peft_model, adapter_name="default", layers=[1, 2, 3])
    zeroer.apply()

    for name, module in _lora_modules(peft_model):
        layer_idx = _get_layer_idx(name)

        if layer_idx in (1, 2, 3):
            # These layers should be zeroed (scaling = 0)
            # Weights themselves are not zeroed, just scaling
            assert module.scaling["default"] == 0.0
        else:
            # Layer 0 should be unchanged
            assert module.scaling["default"] != 0.0


def test_zeroing_target_modules_with_layers(peft_model):
    """Zero only specific modules in specific layers."""
    original_scaling = _snapshot_scaling(peft_model)

    zeroer = LoraLayerZeroing(
        peft_model,
        adapter_name="default",
        layers=[1, 2],
        target_modules=["q_proj"],
    )
    zeroer.apply()

    for name, module in _lora_modules(peft_model):
        layer_idx = _get_layer_idx(name)

        if layer_idx in (1, 2) and name.endswith("q_proj"):
            assert module.scaling["default"] == 0.0
        else:
            assert module.scaling["default"] == pytest.approx(original_scaling[name])


def test_zeroing_target_modules_dict_per_layer(peft_model):
    """Dict mode should zero different modules per layer."""
    original_scaling = _snapshot_scaling(peft_model)

    zeroer = LoraLayerZeroing(
        peft_model,
        adapter_name="default",
        target_modules={
            0: ["q_proj"],
            2: ["up_proj", "down_proj"],
        },
    )
    zeroer.apply()

    for name, module in _lora_modules(peft_model):
        layer_idx = _get_layer_idx(name)

        should_zero = False
        if layer_idx == 0 and name.endswith("q_proj"):
            should_zero = True
        elif layer_idx == 2 and (
            name.endswith("up_proj") or name.endswith("down_proj")
        ):
            should_zero = True

        if should_zero:
            assert module.scaling["default"] == 0.0
        else:
            assert module.scaling["default"] == pytest.approx(original_scaling[name])


def test_zeroing_restore(peft_model):
    """restore() should return scaling to original values."""
    original_scaling = _snapshot_scaling(peft_model)

    zeroer = LoraLayerZeroing(peft_model, adapter_name="default", layers=[1, 2, 3])
    zeroer.apply()
    zeroer.restore()

    for name, module in _lora_modules(peft_model):
        assert module.scaling["default"] == pytest.approx(original_scaling[name])


def test_zeroing_preserves_shapes_and_weights(peft_model):
    """Zeroing should only affect scaling, not shapes or weights."""
    originals = _snapshot_weights(peft_model)

    zeroer = LoraLayerZeroing(peft_model, adapter_name="default", layers=[1, 2, 3])
    zeroer.apply()

    for name, module in _lora_modules(peft_model):
        # Shapes should be unchanged
        assert module.lora_A["default"].weight.shape[0] == 8
        assert module.r["default"] == 8

        # Weights should be unchanged (only scaling is modified)
        assert torch.equal(module.lora_A["default"].weight.data, originals[name]["A"])
        assert torch.equal(module.lora_B["default"].weight.data, originals[name]["B"])


# ---------------------------------------------------------------------------
# Inference tests - verify actual model outputs
# ---------------------------------------------------------------------------


def test_inference_rank_reduction_changes_output(peft_model):
    """Rank reduction should change output due to SVD approximation error."""
    torch.manual_seed(42)
    x = torch.randn(2, 4, 16)  # (batch, seq, hidden)

    # Get original output
    with torch.no_grad():
        original_output = peft_model(x)

    # Reduce rank from 8 to 6
    # This creates ~17% approximation error per matrix (measured empirically)
    reducer = LoraRankReducer(peft_model, adapter_name="default", new_rank=6)
    reducer.apply()

    with torch.no_grad():
        reduced_output = peft_model(x)

    # Should change due to approximation error
    assert not torch.equal(original_output, reduced_output), (
        "Rank reduction should change output"
    )

    # Verify the change is reasonable: not zero (working) and not ~1.0 (random)
    relative_diff = (original_output - reduced_output).norm() / original_output.norm()
    # With 16 matrices (4 layers × 4 modules) each with ~17% error,
    # composing through activations, we expect significant accumulated error.
    # Empirically: ~50-60% is normal, ~100% would suggest random/broken behavior.
    assert 0.1 < relative_diff < 0.95, (
        f"Relative change {relative_diff:.2%} outside reasonable range"
    )


def test_inference_scaling_affects_output_magnitude(peft_model):
    """Scaling should affect the output."""
    torch.manual_seed(42)
    x = torch.randn(2, 4, 16)

    # Get base and original outputs
    peft_model.disable_adapter_layers()
    with torch.no_grad():
        base_output = peft_model(x)
    peft_model.enable_adapter_layers()

    with torch.no_grad():
        original_output = peft_model(x)

    original_distance = (original_output - base_output).norm()

    # Test scaling down moves toward base
    scaler = LoraScaling(peft_model, adapter_name="default", scale_factor=0.5)
    scaler.apply()

    with torch.no_grad():
        scaled_down_output = peft_model(x)

    scaled_down_distance = (scaled_down_output - base_output).norm()

    # Scaling down by 0.5 should reduce distance to base
    # (though not necessarily by exactly 0.5 due to multi-layer composition)
    assert scaled_down_distance < original_distance, (
        f"Scaling down should reduce distance: {scaled_down_distance:.1f} vs {original_distance:.1f}"
    )

    # Test scaling up moves away from base
    scaler.restore()
    scaler_up = LoraScaling(peft_model, adapter_name="default", scale_factor=2.0)
    scaler_up.apply()

    with torch.no_grad():
        scaled_up_output = peft_model(x)

    scaled_up_distance = (scaled_up_output - base_output).norm()

    # Scaling up by 2.0 should increase distance from base
    assert scaled_up_distance > original_distance, (
        f"Scaling up should increase distance: {scaled_up_distance:.1f} vs {original_distance:.1f}"
    )


def test_inference_zeroing_returns_to_base_model(peft_model):
    """Zeroing all layers should make output identical to base model."""
    torch.manual_seed(42)
    x = torch.randn(2, 4, 16)

    # Get base model output
    peft_model.disable_adapter_layers()
    with torch.no_grad():
        base_output = peft_model(x)
    peft_model.enable_adapter_layers()

    # Zero all layers
    zeroer = LoraLayerZeroing(peft_model, adapter_name="default", layers=[0, 1, 2, 3])
    zeroer.apply()

    with torch.no_grad():
        zeroed_output = peft_model(x)

    # Should match base model exactly
    assert torch.allclose(zeroed_output, base_output, rtol=1e-5, atol=1e-7)


def test_inference_partial_zeroing_partial_effect(peft_model):
    """Zeroing some layers should partially reduce LoRA effect."""
    torch.manual_seed(42)
    x = torch.randn(2, 4, 16)

    # Get base and original outputs
    peft_model.disable_adapter_layers()
    with torch.no_grad():
        base_output = peft_model(x)
    peft_model.enable_adapter_layers()

    with torch.no_grad():
        original_output = peft_model(x)

    original_lora_contribution = (original_output - base_output).norm()

    # Zero half the layers
    zeroer = LoraLayerZeroing(peft_model, adapter_name="default", layers=[2, 3])
    zeroer.apply()

    with torch.no_grad():
        partial_output = peft_model(x)

    partial_lora_contribution = (partial_output - base_output).norm()

    # LoRA contribution should be reduced but not zero
    assert partial_lora_contribution < original_lora_contribution
    assert partial_lora_contribution > 0


def test_inference_restore_returns_exact_output(peft_model):
    """After restore, outputs should be bit-exact identical."""
    torch.manual_seed(42)
    x = torch.randn(2, 4, 16)

    with torch.no_grad():
        original_output = peft_model(x)

    # Apply modification
    reducer = LoraRankReducer(peft_model, adapter_name="default", new_rank=4)
    reducer.apply()

    # Restore
    reducer.restore()

    with torch.no_grad():
        restored_output = peft_model(x)

    # Should be exactly identical
    assert torch.equal(restored_output, original_output)


def test_inference_dict_mode_selective_layers(peft_model):
    """Dict mode should only affect specified modules in specified layers."""
    torch.manual_seed(42)
    x = torch.randn(2, 4, 16)

    # Get original output
    with torch.no_grad():
        original_output = peft_model(x)

    # Reduce rank only in layer 0, q_proj
    reducer = LoraRankReducer(
        peft_model, adapter_name="default", new_rank=4, target_modules={0: ["q_proj"]}
    )
    reducer.apply()

    with torch.no_grad():
        modified_output = peft_model(x)

    # Output should change (layer 0 q_proj was modified)
    assert not torch.equal(original_output, modified_output)

    # But should be relatively similar (only one module in one layer changed)
    relative_diff = (original_output - modified_output).norm() / original_output.norm()
    assert relative_diff < 0.3  # Less than 30% relative change


def test_inference_negative_scaling_changes_direction(peft_model):
    """Negative scaling should significantly change the output direction."""
    torch.manual_seed(42)
    x = torch.randn(2, 4, 16)

    # Get base model output
    peft_model.disable_adapter_layers()
    with torch.no_grad():
        base_output = peft_model(x)
    peft_model.enable_adapter_layers()

    # Get original output with positive LoRA
    with torch.no_grad():
        original_output = peft_model(x)

    # Apply negative scaling
    scaler = LoraScaling(peft_model, adapter_name="default", scale_factor=-1.0)
    scaler.apply()

    with torch.no_grad():
        negatively_scaled_output = peft_model(x)

    # The output should be different from both base and original
    assert not torch.allclose(negatively_scaled_output, original_output, rtol=1e-2)
    assert not torch.allclose(negatively_scaled_output, base_output, rtol=1e-2)

    # The negatively scaled output should be "on the other side" of the base
    # i.e., if original pushed away from base, negative should push back
    # We can check this by seeing that the distance relationships change
    original_distance = (original_output - base_output).norm()
    negative_distance = (negatively_scaled_output - base_output).norm()

    # Both should have non-zero distance from base (LoRA is active)
    assert original_distance > 0
    assert negative_distance > 0


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


def test_integration_multiple_modifiers_chained(peft_model):
    """Multiple modifiers can be applied in sequence."""
    # First reduce rank
    reducer = LoraRankReducer(
        peft_model, adapter_name="default", new_rank=4, target_modules=["q_proj"]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reducer.apply()

    # Then scale
    scaler = LoraScaling(
        peft_model,
        adapter_name="default",
        scale_factor=2.0,
        target_modules=["v_proj"],
    )
    scaler.apply()

    # Check both modifications are active
    for name, module in _lora_modules(peft_model):
        if name.endswith("q_proj"):
            assert module.lora_A["default"].weight.shape[0] == 4
        if name.endswith("v_proj"):
            # Original scaling is lora_alpha / r = 16 / 8 = 2.0
            assert module.scaling["default"] == pytest.approx(4.0)


def test_integration_restore_order_independence(peft_model):
    """Restoring in any order should work."""
    originals = _snapshot_weights(peft_model)
    original_scaling = _snapshot_scaling(peft_model)

    reducer = LoraRankReducer(peft_model, adapter_name="default", new_rank=4)
    scaler = LoraScaling(peft_model, adapter_name="default", scale_factor=2.0)

    reducer.apply()
    scaler.apply()

    # Restore in reverse order
    scaler.restore()
    reducer.restore()

    for name, module in _lora_modules(peft_model):
        assert torch.equal(module.lora_A["default"].weight.data, originals[name]["A"])
        assert module.scaling["default"] == pytest.approx(original_scaling[name])


def test_integration_dict_with_many_layers(peft_model):
    """Dict mode should handle all layers with different configs."""
    reducer = LoraRankReducer(
        peft_model,
        adapter_name="default",
        new_rank=4,
        target_modules={
            0: ["q_proj"],
            1: ["v_proj"],
            2: ["up_proj"],
            3: ["down_proj"],
        },
    )
    reducer.apply()

    # Verify each layer has correct modules modified
    for name, module in _lora_modules(peft_model):
        layer_idx = _get_layer_idx(name)
        expected_modified = {
            0: "q_proj",
            1: "v_proj",
            2: "up_proj",
            3: "down_proj",
        }

        if name.endswith(expected_modified[layer_idx]):
            assert module.lora_A["default"].weight.shape[0] == 4
        else:
            assert module.lora_A["default"].weight.shape[0] == 8


def test_integration_empty_dict_modifies_nothing(peft_model):
    """Empty dict should not modify any modules."""
    originals = _snapshot_weights(peft_model)

    reducer = LoraRankReducer(
        peft_model, adapter_name="default", new_rank=4, target_modules={}
    )
    reducer.apply()

    for name, module in _lora_modules(peft_model):
        assert torch.equal(module.lora_A["default"].weight.data, originals[name]["A"])
        assert module.r["default"] == 8


# ---------------------------------------------------------------------------
# Pipeline tests
# ---------------------------------------------------------------------------


def test_pipeline_sequential_application_compounds(peft_model):
    """Pipeline applies modifications sequentially, not each to original."""
    from src.utils.peft_manipulations import LoraPipeline

    # Original scaling is 16/8 = 2.0
    # If we scale by 0.3 twice sequentially: 2.0 * 0.3 * 0.3 = 0.18
    # If each applied to original: 2.0 * 0.3 = 0.6 (wrong!)

    pipeline = LoraPipeline(
        peft_model,
        steps=[
            (LoraScaling, "default", {"scale_factor": 0.3}),
            (LoraScaling, "default", {"scale_factor": 0.3}),
        ],
    )

    pipeline.apply()

    # Verify compounding: 2.0 * 0.3 * 0.3 = 0.18
    for _name, module in _lora_modules(peft_model):
        assert torch.isclose(
            torch.tensor(module.scaling["default"]), torch.tensor(0.18), rtol=1e-5
        )


def test_pipeline_basic_single_adapter(peft_model):
    """Pipeline applies multiple modifications to a single adapter in sequence."""
    from src.utils.peft_manipulations import LoraPipeline

    # Create pipeline: reduce rank, then scale
    pipeline = LoraPipeline(
        peft_model,
        steps=[
            (LoraRankReducer, "default", {"new_rank": 4}),
            (LoraScaling, "default", {"scale_factor": 0.5}),
        ],
    )

    assert not pipeline.is_applied

    pipeline.apply()
    assert pipeline.is_applied

    # Check rank was reduced
    for _name, module in _lora_modules(peft_model):
        assert module.lora_A["default"].weight.shape[0] == 4
        assert module.r["default"] == 4

    # Check scaling: original 2.0, then scaled by 0.5 = 1.0
    for _name, module in _lora_modules(peft_model):
        assert module.scaling["default"] == 1.0


def test_pipeline_multi_adapter(peft_model_custom_adapter):
    """Pipeline can modify different adapters in different steps."""
    from src.utils.peft_manipulations import LoraPipeline

    pipeline = LoraPipeline(
        peft_model_custom_adapter,
        steps=[
            (LoraRankReducer, "default", {"new_rank": 4}),
            (LoraScaling, "custom", {"scale_factor": 0.1}),
        ],
    )

    pipeline.apply()

    # Check default adapter rank reduced
    for _name, module in _lora_modules(peft_model_custom_adapter, "default"):
        assert module.r["default"] == 4

    # Check custom adapter scaled
    for _name, module in _lora_modules(peft_model_custom_adapter, "custom"):
        # Original scaling was 16/8 = 2.0, now 2.0 * 0.1 = 0.2
        assert module.scaling["custom"] == 2.0 * 0.1


def test_pipeline_restore(peft_model):
    """Pipeline restore returns all adapters to original state."""
    from src.utils.peft_manipulations import LoraPipeline

    originals = _snapshot_weights(peft_model)

    pipeline = LoraPipeline(
        peft_model,
        steps=[
            (LoraRankReducer, "default", {"new_rank": 2}),
            (LoraScaling, "default", {"scale_factor": 3.0}),
        ],
    )

    pipeline.apply()
    pipeline.restore()

    assert not pipeline.is_applied

    # Check everything restored
    for name, module in _lora_modules(peft_model):
        assert torch.equal(module.lora_A["default"].weight.data, originals[name]["A"])
        assert torch.equal(module.lora_B["default"].weight.data, originals[name]["B"])
        assert module.r["default"] == 8
        assert module.scaling["default"] == 2.0


def test_pipeline_idempotent_apply(peft_model):
    """Calling apply() multiple times produces same result."""
    from src.utils.peft_manipulations import LoraPipeline

    pipeline = LoraPipeline(
        peft_model,
        steps=[
            (LoraRankReducer, "default", {"new_rank": 4}),
            (LoraScaling, "default", {"scale_factor": 0.5}),
        ],
    )

    pipeline.apply()
    first = _snapshot_weights(peft_model)

    pipeline.apply()
    second = _snapshot_weights(peft_model)

    for name in first:
        assert torch.equal(first[name]["A"], second[name]["A"])
        assert torch.equal(first[name]["B"], second[name]["B"])


def test_pipeline_with_filters(peft_model):
    """Pipeline steps can have different filters."""
    from src.utils.peft_manipulations import LoraPipeline

    pipeline = LoraPipeline(
        peft_model,
        steps=[
            (LoraRankReducer, "default", {"new_rank": 2, "layers": [0, 1]}),
            (
                LoraScaling,
                "default",
                {"scale_factor": 0.1, "target_modules": ["q_proj"]},
            ),
        ],
    )

    pipeline.apply()

    # Check layers 0-1 have rank 2, layers 2-3 have rank 8
    for name, module in _lora_modules(peft_model):
        layer_idx = int(name.split("model.layers.")[-1].split(".")[0])
        if layer_idx in [0, 1]:
            assert module.r["default"] == 2
        else:
            assert module.r["default"] == 8

    # Check q_proj modules have scaled scaling
    for name, module in _lora_modules(peft_model):
        if name.endswith("q_proj"):
            assert module.scaling["default"] == 2.0 * 0.1
        else:
            # Not scaled (original scaling)
            assert module.scaling["default"] == 2.0


def test_pipeline_validation_empty_steps():
    """Pipeline rejects empty steps list."""
    import torch.nn as nn

    from src.utils.peft_manipulations import LoraPipeline

    with pytest.raises(ValueError, match="at least one step"):
        LoraPipeline(nn.Module(), steps=[])


def test_pipeline_validation_invalid_modifier_class(peft_model):
    """Pipeline rejects non-BaseLoraModifier classes."""
    from src.utils.peft_manipulations import LoraPipeline

    class NotAModifier:
        pass

    with pytest.raises(TypeError, match="not a subclass of BaseLoraModifier"):
        LoraPipeline(peft_model, steps=[(NotAModifier, "default", {})])


def test_pipeline_validation_no_model_in_kwargs(peft_model):
    """Pipeline rejects kwargs containing 'model' or 'adapter_name'."""
    from src.utils.peft_manipulations import LoraPipeline

    with pytest.raises(ValueError, match="should not include 'model'"):
        LoraPipeline(
            peft_model,
            steps=[
                (LoraScaling, "default", {"scale_factor": 1.0, "model": peft_model})
            ],
        )

    with pytest.raises(ValueError, match="should not include.*'adapter_name'"):
        LoraPipeline(
            peft_model,
            steps=[
                (
                    LoraScaling,
                    "default",
                    {"scale_factor": 1.0, "adapter_name": "default"},
                )
            ],
        )


def test_pipeline_inference_compound_effects(peft_model):
    """Pipeline correctly compounds effects on model output."""
    torch.manual_seed(42)
    x = torch.randn(2, 4, 16)

    from src.utils.peft_manipulations import LoraPipeline

    # Get original output
    with torch.no_grad():
        original_output = peft_model(x)

    # Apply pipeline: rank reduction + scaling
    pipeline = LoraPipeline(
        peft_model,
        steps=[
            (LoraRankReducer, "default", {"new_rank": 6}),
            (LoraScaling, "default", {"scale_factor": 0.3}),
        ],
    )
    pipeline.apply()

    with torch.no_grad():
        pipeline_output = peft_model(x)

    # Output should change due to both rank reduction and scaling
    assert not torch.equal(original_output, pipeline_output)

    # Restore and verify
    pipeline.restore()
    with torch.no_grad():
        restored_output = peft_model(x)

    assert torch.equal(restored_output, original_output)


def test_pipeline_inference_scale_two_adapters_independently(
    peft_model_custom_adapter,
):
    """Pipeline scales two adapters independently, each affecting output correctly."""
    torch.manual_seed(42)
    x = torch.randn(2, 4, 16)

    from src.utils.peft_manipulations import LoraPipeline

    # Get base model output
    peft_model_custom_adapter.disable_adapter_layers()
    with torch.no_grad():
        base_output = peft_model_custom_adapter(x)

    # Get output with only default adapter
    peft_model_custom_adapter.enable_adapter_layers()
    peft_model_custom_adapter.set_adapter("default")
    with torch.no_grad():
        default_only_output = peft_model_custom_adapter(x)

    # Get output with only custom adapter
    peft_model_custom_adapter.set_adapter("custom")
    with torch.no_grad():
        custom_only_output = peft_model_custom_adapter(x)

    # Calculate original LoRA contributions
    default_contribution = (default_only_output - base_output).norm()
    custom_contribution = (custom_only_output - base_output).norm()

    # Both should have non-zero contributions
    assert default_contribution > 0
    assert custom_contribution > 0

    # Now scale both adapters with pipeline
    pipeline = LoraPipeline(
        peft_model_custom_adapter,
        steps=[
            (LoraScaling, "default", {"scale_factor": 0.5}),
            (LoraScaling, "custom", {"scale_factor": 0.2}),
        ],
    )
    pipeline.apply()

    # Check default adapter output changed
    peft_model_custom_adapter.set_adapter("default")
    with torch.no_grad():
        scaled_default_output = peft_model_custom_adapter(x)

    scaled_default_contribution = (scaled_default_output - base_output).norm()

    # Scaling down should reduce contribution
    assert scaled_default_contribution < default_contribution

    # Check custom adapter output changed
    peft_model_custom_adapter.set_adapter("custom")
    with torch.no_grad():
        scaled_custom_output = peft_model_custom_adapter(x)

    scaled_custom_contribution = (scaled_custom_output - base_output).norm()

    # Scaling down should reduce contribution
    assert scaled_custom_contribution < custom_contribution

    # Custom scaled more aggressively (0.2 vs 0.5), so should be closer to base
    assert scaled_custom_contribution < scaled_default_contribution

    # Restore and verify both adapters return to original
    pipeline.restore()

    peft_model_custom_adapter.set_adapter("default")
    with torch.no_grad():
        restored_default = peft_model_custom_adapter(x)

    peft_model_custom_adapter.set_adapter("custom")
    with torch.no_grad():
        restored_custom = peft_model_custom_adapter(x)

    assert torch.equal(restored_default, default_only_output)
    assert torch.equal(restored_custom, custom_only_output)
