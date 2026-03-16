import json
import warnings

import pytest
import torch
from peft import LoraConfig, get_peft_model
from torch import nn

from src.utils.model_layer_info import extract_layer_idx
from src.utils.peft_manipulations import (
    LoRaAdapterZeroing,
    LoRaPipeline,
    LoRaRankReducer,
    LoRaScaling,
    get_active_adapters,
    set_active_adapters,
)

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
# Single-linear model for exact forward-pass tests
# ---------------------------------------------------------------------------


class SingleLinear(nn.Module):
    """Bare single ``nn.Linear`` -- the simplest possible LoRA target.

    Base weight is set to identity so that ``forward(x) == x`` before LoRA.
    """

    def __init__(self, size: int = 4):
        super().__init__()
        self.proj = nn.Linear(size, size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


@pytest.fixture
def peft_single_linear():
    """Single Linear(4,4) with identity base and all-ones LoRA (r=1, alpha=1).

    Math
    ----
    * Base weight ``W = I₄``  →  ``base(x) = x``
    * LoRA: ``A = 1_(1×4)``, ``B = 1_(4×1)``  →  ``B·A = 1_(4×4)``
    * ``scaling = lora_alpha / r = 1``
    * Effective output: ``x + sum(x) · 1``

    For ``x = [1, 0, 0, 0]`` the expected output is ``[2, 1, 1, 1]``.
    """
    size = 4
    base = SingleLinear(size=size)
    base.proj.weight.data = torch.eye(size)

    config = LoraConfig(
        r=1,
        lora_alpha=1,
        target_modules=["proj"],
        bias="none",
    )
    model = get_peft_model(base, config)

    # Hard-code LoRA weights so every output is hand-computable
    for _, module in _lora_modules(model):
        module.lora_A["default"].weight.data = torch.ones(1, size)
        module.lora_B["default"].weight.data = torch.ones(size, 1)

    return model


@pytest.fixture
def peft_single_linear_two_adapters():
    """Single Linear(4,4) with two adapters having distinct LoRA weights.

    Math
    ----
    * Base weight ``W = I₄``
    * "default": ``A = 1_(1×4)``, ``B = 1_(4×1)``  →  contribution = ``[sum(x)] * 4``
    * "custom":  ``A = 1_(1×4)``, ``B = 2·1_(4×1)`` →  contribution = ``[2·sum(x)] * 4``

    For ``x = [1, 0, 0, 0]``:
      * "default" active → ``[2, 1, 1, 1]``
      * "custom"  active → ``[3, 2, 2, 2]``
    """
    size = 4
    base = SingleLinear(size=size)
    base.proj.weight.data = torch.eye(size)

    default_config = LoraConfig(
        r=1, lora_alpha=1, target_modules=["proj"], bias="none",
    )
    model = get_peft_model(base, default_config)

    # Hard-code "default" adapter: B @ A = ones(4,4)
    for _, module in _lora_modules(model, "default"):
        module.lora_A["default"].weight.data = torch.ones(1, size)
        module.lora_B["default"].weight.data = torch.ones(size, 1)

    # Add "custom" adapter with same shape but different B
    custom_config = LoraConfig(
        r=1, lora_alpha=1, target_modules=["proj"], bias="none",
    )
    model.add_adapter("custom", custom_config)

    # Hard-code "custom" adapter: B @ A = 2 * ones(4,4)
    for _, module in _lora_modules(model, "custom"):
        module.lora_A["custom"].weight.data = torch.ones(1, size)
        module.lora_B["custom"].weight.data = 2.0 * torch.ones(size, 1)

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
    idx = extract_layer_idx(name)
    assert idx is not None, f"Could not extract layer index from {name!r}"
    return idx


# ---------------------------------------------------------------------------
# LoraRankReducer tests
# ---------------------------------------------------------------------------


def test_rank_reducer_no_filters_all_modules(peft_model):
    """No filters should affect all modules."""
    reducer = LoRaRankReducer(peft_model, adapter_name="default", new_rank=4)
    reducer.apply()

    for _name, module in _lora_modules(peft_model):
        assert module.lora_A["default"].weight.shape[0] == 4
        assert module.lora_B["default"].weight.shape[1] == 4
        assert module.r["default"] == 4

    assert peft_model.peft_config["default"].r == 4


def test_rank_reducer_target_modules_list_only(peft_model):
    """target_modules as list should filter by module type across all layers."""
    originals = _snapshot_weights(peft_model)

    reducer = LoRaRankReducer(
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

    reducer = LoRaRankReducer(
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

    reducer = LoRaRankReducer(
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

    reducer = LoRaRankReducer(
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
        LoRaRankReducer(
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

    reducer = LoRaRankReducer(peft_model, adapter_name="default", new_rank=4)
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

    reducer = LoRaRankReducer(peft_model, adapter_name="default", new_rank=4)
    reducer.apply()
    reducer.restore()

    for name, module in _lora_modules(peft_model):
        assert torch.equal(module.lora_A["default"].weight.data, originals[name]["A"])
        assert torch.equal(module.lora_B["default"].weight.data, originals[name]["B"])
        assert module.r["default"] == original_ranks[name]

    assert peft_model.peft_config["default"].r == 8


def test_rank_reducer_is_applied_tracking(peft_model):
    """is_applied should track modification state."""
    reducer = LoRaRankReducer(peft_model, adapter_name="default", new_rank=4)
    assert not reducer.is_applied

    reducer.apply()
    assert reducer.is_applied

    reducer.restore()
    assert not reducer.is_applied


def test_rank_reducer_idempotent_apply(peft_model):
    """Multiple apply() calls should produce same result."""
    reducer = LoRaRankReducer(peft_model, adapter_name="default", new_rank=4)
    reducer.apply()
    after_first = _snapshot_weights(peft_model)

    reducer.apply()

    for name, module in _lora_modules(peft_model):
        assert torch.equal(module.lora_A["default"].weight.data, after_first[name]["A"])
        assert torch.equal(module.lora_B["default"].weight.data, after_first[name]["B"])


def test_rank_reducer_warns_about_peft_config_with_filters(peft_model):
    """Filtered modifications should warn about peft_config.r mismatch."""
    reducer = LoRaRankReducer(
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
    reducer = LoRaRankReducer(peft_model, adapter_name="default", new_rank=4)
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
    reducer = LoRaRankReducer(model, adapter_name="custom", new_rank=4)
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

    scaler = LoRaScaling(peft_model, adapter_name="default", scale_factor=2.0)
    scaler.apply()

    for name, module in _lora_modules(peft_model):
        expected = original_scaling[name] * 2.0
        assert module.scaling["default"] == pytest.approx(expected)


def test_scaling_negative_scale_factor(peft_model):
    """Negative scale factor should invert LoRA direction."""
    original_scaling = _snapshot_scaling(peft_model)

    scaler = LoRaScaling(peft_model, adapter_name="default", scale_factor=-1.0)
    scaler.apply()

    for name, module in _lora_modules(peft_model):
        expected = original_scaling[name] * -1.0
        assert module.scaling["default"] == pytest.approx(expected)


def test_scaling_zero_scale_disables(peft_model):
    """Scale factor of 0 should disable adapter."""
    scaler = LoRaScaling(peft_model, adapter_name="default", scale_factor=0.0)
    scaler.apply()

    for _name, module in _lora_modules(peft_model):
        assert module.scaling["default"] == 0.0


def test_scaling_restore_exact(peft_model):
    """restore() should return to exact original scaling."""
    original_scaling = _snapshot_scaling(peft_model)

    scaler = LoRaScaling(peft_model, adapter_name="default", scale_factor=3.0)
    scaler.apply()
    scaler.restore()

    for name, module in _lora_modules(peft_model):
        assert module.scaling["default"] == pytest.approx(original_scaling[name])


def test_scaling_idempotent_apply(peft_model):
    """Multiple apply() calls should produce same result."""
    scaler = LoRaScaling(peft_model, adapter_name="default", scale_factor=2.0)
    scaler.apply()
    after_first = _snapshot_scaling(peft_model)

    scaler.apply()

    for name, module in _lora_modules(peft_model):
        assert module.scaling["default"] == pytest.approx(after_first[name])


# ---------------------------------------------------------------------------
# LoraAdapterZeroing tests
# ---------------------------------------------------------------------------


def test_zeroing_layers_only_zeros_specified(peft_model):
    """layers parameter should zero specified layers."""
    zeroer = LoRaAdapterZeroing(peft_model, adapter_name="default", layers=[1, 2, 3])
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


def test_zeroing_restore(peft_model):
    """restore() should return scaling to original values."""
    original_scaling = _snapshot_scaling(peft_model)

    zeroer = LoRaAdapterZeroing(peft_model, adapter_name="default", layers=[1, 2, 3])
    zeroer.apply()
    zeroer.restore()

    for name, module in _lora_modules(peft_model):
        assert module.scaling["default"] == pytest.approx(original_scaling[name])


def test_zeroing_preserves_shapes_and_weights(peft_model):
    """Zeroing should only affect scaling, not shapes or weights."""
    originals = _snapshot_weights(peft_model)

    zeroer = LoRaAdapterZeroing(peft_model, adapter_name="default", layers=[1, 2, 3])
    zeroer.apply()

    for name, module in _lora_modules(peft_model):
        # Shapes should be unchanged
        assert module.lora_A["default"].weight.shape[0] == 8
        assert module.r["default"] == 8

        # Weights should be unchanged (only scaling is modified)
        assert torch.equal(module.lora_A["default"].weight.data, originals[name]["A"])
        assert torch.equal(module.lora_B["default"].weight.data, originals[name]["B"])


# ---------------------------------------------------------------------------
# Inference tests - multi-layer composition (TinyTransformer)
# ---------------------------------------------------------------------------


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
    zeroer = LoRaAdapterZeroing(peft_model, adapter_name="default", layers=[2, 3])
    zeroer.apply()

    with torch.no_grad():
        partial_output = peft_model(x)

    partial_lora_contribution = (partial_output - base_output).norm()

    # LoRA contribution should be reduced but not zero
    assert partial_lora_contribution < original_lora_contribution
    assert partial_lora_contribution > 0


def test_inference_dict_mode_selective_layers(peft_model):
    """Dict mode should only affect specified modules in specified layers."""
    torch.manual_seed(42)
    x = torch.randn(2, 4, 16)

    # Get original output
    with torch.no_grad():
        original_output = peft_model(x)

    # Reduce rank only in layer 0, q_proj
    reducer = LoRaRankReducer(
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


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


def test_integration_multiple_modifiers_chained(peft_model):
    """Multiple modifiers can be applied in sequence."""
    # First reduce rank
    reducer = LoRaRankReducer(
        peft_model, adapter_name="default", new_rank=4, target_modules=["q_proj"]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reducer.apply()

    # Then scale
    scaler = LoRaScaling(
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

    reducer = LoRaRankReducer(peft_model, adapter_name="default", new_rank=4)
    scaler = LoRaScaling(peft_model, adapter_name="default", scale_factor=2.0)

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
    reducer = LoRaRankReducer(
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

    reducer = LoRaRankReducer(
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

    # Original scaling is 16/8 = 2.0
    # If we scale by 0.3 twice sequentially: 2.0 * 0.3 * 0.3 = 0.18
    # If each applied to original: 2.0 * 0.3 = 0.6 (wrong!)

    pipeline = LoRaPipeline(
        peft_model,
        steps=[
            (LoRaScaling, "default", {"scale_factor": 0.3}),
            (LoRaScaling, "default", {"scale_factor": 0.3}),
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

    # Create pipeline: reduce rank, then scale
    pipeline = LoRaPipeline(
        peft_model,
        steps=[
            (LoRaRankReducer, "default", {"new_rank": 4}),
            (LoRaScaling, "default", {"scale_factor": 0.5}),
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

    pipeline = LoRaPipeline(
        peft_model_custom_adapter,
        steps=[
            (LoRaRankReducer, "default", {"new_rank": 4}),
            (LoRaScaling, "custom", {"scale_factor": 0.1}),
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

    originals = _snapshot_weights(peft_model)

    pipeline = LoRaPipeline(
        peft_model,
        steps=[
            (LoRaRankReducer, "default", {"new_rank": 2}),
            (LoRaScaling, "default", {"scale_factor": 3.0}),
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

    pipeline = LoRaPipeline(
        peft_model,
        steps=[
            (LoRaRankReducer, "default", {"new_rank": 4}),
            (LoRaScaling, "default", {"scale_factor": 0.5}),
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

    pipeline = LoRaPipeline(
        peft_model,
        steps=[
            (LoRaRankReducer, "default", {"new_rank": 2, "layers": [0, 1]}),
            (
                LoRaScaling,
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


    with pytest.raises(ValueError, match="at least one step"):
        LoRaPipeline(nn.Module(), steps=[])


def test_pipeline_validation_invalid_modifier_class(peft_model):
    """Pipeline rejects non-BaseLoraModifier classes."""

    class NotAModifier:
        pass

    with pytest.raises(TypeError, match="not a subclass of BaseLoraModifier"):
        LoRaPipeline(peft_model, steps=[(NotAModifier, "default", {})])


def test_pipeline_validation_no_model_in_kwargs(peft_model):
    """Pipeline rejects kwargs containing 'model' or 'adapter_name'."""

    with pytest.raises(ValueError, match="should not include 'model'"):
        LoRaPipeline(
            peft_model,
            steps=[
                (LoRaScaling, "default", {"scale_factor": 1.0, "model": peft_model})
            ],
        )

    with pytest.raises(ValueError, match="should not include.*'adapter_name'"):
        LoRaPipeline(
            peft_model,
            steps=[
                (
                    LoRaScaling,
                    "default",
                    {"scale_factor": 1.0, "adapter_name": "default"},
                )
            ],
        )


def test_pipeline_inference_compound_effects(peft_model):
    """Pipeline correctly compounds effects on model output."""
    torch.manual_seed(42)
    x = torch.randn(2, 4, 16)


    # Get original output
    with torch.no_grad():
        original_output = peft_model(x)

    # Apply pipeline: rank reduction + scaling
    pipeline = LoRaPipeline(
        peft_model,
        steps=[
            (LoRaRankReducer, "default", {"new_rank": 6}),
            (LoRaScaling, "default", {"scale_factor": 0.3}),
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
    pipeline = LoRaPipeline(
        peft_model_custom_adapter,
        steps=[
            (LoRaScaling, "default", {"scale_factor": 0.5}),
            (LoRaScaling, "custom", {"scale_factor": 0.2}),
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


# ---------------------------------------------------------------------------
# Exact forward-pass tests (SingleLinear + hard-coded LoRA)
# ---------------------------------------------------------------------------
#
# Base weight = I₄, LoRA B·A = 1_(4×4), scaling = 1.
# For x = [1, 0, 0, 0]:
#   base output      = [1, 0, 0, 0]
#   LoRA contribution = scaling · B·A·x = [1, 1, 1, 1]
#   full output       = [2, 1, 1, 1]

# Shared input & expected values used across tests
_X = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
_BASE = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
_LORA_CONTRIB = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
_FULL = _BASE + _LORA_CONTRIB  # [2, 1, 1, 1]


def test_fwd_base_is_identity(peft_single_linear):
    """With adapter disabled, output equals input (identity base weights)."""
    peft_single_linear.disable_adapter_layers()
    with torch.no_grad():
        out = peft_single_linear(_X)
    assert torch.equal(out, _BASE)


def test_fwd_lora_adds_contribution(peft_single_linear):
    """With adapter active, output = base + LoRA = [2, 1, 1, 1]."""
    with torch.no_grad():
        out = peft_single_linear(_X)
    assert torch.equal(out, _FULL)


def test_fwd_zeroing_disables_lora(peft_single_linear):
    """LoRaAdapterZeroing sets scaling to 0 → output = base."""
    zeroer = LoRaAdapterZeroing(peft_single_linear, adapter_name="default")
    zeroer.apply()
    with torch.no_grad():
        out = peft_single_linear(_X)
    assert torch.equal(out, _BASE)


def test_fwd_scaling_zero_disables(peft_single_linear):
    """LoRaScaling with factor 0 → output = base."""
    scaler = LoRaScaling(
        peft_single_linear, adapter_name="default", scale_factor=0.0
    )
    scaler.apply()
    with torch.no_grad():
        out = peft_single_linear(_X)
    assert torch.equal(out, _BASE)


def test_fwd_scaling_doubles(peft_single_linear):
    """scale_factor=2 → output = base + 2·LoRA = [3, 2, 2, 2]."""
    scaler = LoRaScaling(
        peft_single_linear, adapter_name="default", scale_factor=2.0
    )
    scaler.apply()
    expected = _BASE + 2.0 * _LORA_CONTRIB  # [3, 2, 2, 2]
    with torch.no_grad():
        out = peft_single_linear(_X)
    assert torch.equal(out, expected)


def test_fwd_scaling_half(peft_single_linear):
    """scale_factor=0.5 → output = base + 0.5·LoRA = [1.5, 0.5, 0.5, 0.5]."""
    scaler = LoRaScaling(
        peft_single_linear, adapter_name="default", scale_factor=0.5
    )
    scaler.apply()
    expected = _BASE + 0.5 * _LORA_CONTRIB  # [1.5, 0.5, 0.5, 0.5]
    with torch.no_grad():
        out = peft_single_linear(_X)
    assert torch.equal(out, expected)


def test_fwd_scaling_negative(peft_single_linear):
    """scale_factor=-1 → output = base - LoRA = [0, -1, -1, -1]."""
    scaler = LoRaScaling(
        peft_single_linear, adapter_name="default", scale_factor=-1.0
    )
    scaler.apply()
    expected = _BASE - _LORA_CONTRIB  # [0, -1, -1, -1]
    with torch.no_grad():
        out = peft_single_linear(_X)
    assert torch.equal(out, expected)


def test_fwd_restore_exact(peft_single_linear):
    """After scaling + restore, output returns to [2, 1, 1, 1] exactly."""
    scaler = LoRaScaling(
        peft_single_linear, adapter_name="default", scale_factor=0.0
    )
    scaler.apply()

    # Verify zeroed first
    with torch.no_grad():
        out = peft_single_linear(_X)
    assert torch.equal(out, _BASE)

    # Restore and verify original
    scaler.restore()
    with torch.no_grad():
        out = peft_single_linear(_X)
    assert torch.equal(out, _FULL)


def test_fwd_pipeline_compounds(peft_single_linear):
    """Pipeline scales compound: 1.0 * 3.0 * 0.5 = 1.5 → [2.5, 1.5, 1.5, 1.5]."""

    pipeline = LoRaPipeline(
        peft_single_linear,
        steps=[
            (LoRaScaling, "default", {"scale_factor": 3.0}),
            (LoRaScaling, "default", {"scale_factor": 0.5}),
        ],
    )
    pipeline.apply()

    expected = _BASE + 1.5 * _LORA_CONTRIB  # [2.5, 1.5, 1.5, 1.5]
    with torch.no_grad():
        out = peft_single_linear(_X)
    assert torch.equal(out, expected)

    # Restore and verify original output
    pipeline.restore()
    with torch.no_grad():
        out = peft_single_linear(_X)
    assert torch.equal(out, _FULL)


def test_fwd_pipeline_multi_adapter(peft_single_linear_two_adapters):
    """Pipeline scales two adapters independently, verified by exact outputs.

    * "default" scaled by 0.5 → contribution = 0.5 * [1,1,1,1]
    * "custom"  scaled by 3.0 → contribution = 3.0 * [2,2,2,2]
    * Both active → contributions stack additively
    """

    model = peft_single_linear_two_adapters

    pipeline = LoRaPipeline(
        model,
        steps=[
            (LoRaScaling, "default", {"scale_factor": 0.5}),
            (LoRaScaling, "custom", {"scale_factor": 3.0}),
        ],
    )
    pipeline.apply()

    # Check "default" alone: base + 0.5 * [1,1,1,1] = [1.5, 0.5, 0.5, 0.5]
    set_active_adapters(model, "default")
    with torch.no_grad():
        out = model(_X)
    expected_default = _BASE + 0.5 * _LORA_CONTRIB
    assert torch.equal(out, expected_default)

    # Check "custom" alone: base + 3.0 * 2 * [1,1,1,1] = [7, 6, 6, 6]
    set_active_adapters(model, "custom")
    with torch.no_grad():
        out = model(_X)
    expected_custom = _BASE + 3.0 * (2.0 * _LORA_CONTRIB)
    assert torch.equal(out, expected_custom)

    # Check both active: base + 0.5*[1,1,1,1] + 3.0*2*[1,1,1,1] = [7.5, 6.5, 6.5, 6.5]
    set_active_adapters(model, ["default", "custom"])
    with torch.no_grad():
        out = model(_X)
    expected_both = _BASE + 0.5 * _LORA_CONTRIB + 3.0 * (2.0 * _LORA_CONTRIB)
    assert torch.equal(out, expected_both)

    # Restore and verify all combinations return to unscaled outputs
    pipeline.restore()

    set_active_adapters(model, "default")
    with torch.no_grad():
        out = model(_X)
    assert torch.equal(out, _FULL)  # [2, 1, 1, 1]

    set_active_adapters(model, "custom")
    with torch.no_grad():
        out = model(_X)
    expected_custom_original = _BASE + 2.0 * _LORA_CONTRIB  # [3, 2, 2, 2]
    assert torch.equal(out, expected_custom_original)

    set_active_adapters(model, ["default", "custom"])
    with torch.no_grad():
        out = model(_X)
    expected_both_original = _BASE + _LORA_CONTRIB + 2.0 * _LORA_CONTRIB  # [4, 3, 3, 3]
    assert torch.equal(out, expected_both_original)


def test_fwd_raises_when_modifying_inactive_adapter(peft_single_linear_two_adapters):
    """Modifying an adapter that isn't active should raise ValueError."""
    model = peft_single_linear_two_adapters

    # Activate only "default"
    set_active_adapters(model, "default")
    assert get_active_adapters(model) == ["default"]

    # Modifying "custom" (inactive) should raise
    scaler = LoRaScaling(model, adapter_name="custom", scale_factor=2.0)
    with pytest.raises(ValueError, match="not currently active"):
        scaler.apply()

    # Modifying "default" (active) should NOT raise
    scaler2 = LoRaScaling(model, adapter_name="default", scale_factor=2.0)
    scaler2.apply()  # no exception


# ---------------------------------------------------------------------------
# GPT-2-style model for cross-architecture tests
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
        block.attn.c_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        block.mlp = nn.Module()
        block.mlp.c_fc = nn.Linear(hidden_size, hidden_size, bias=False)
        return block

    def forward(self, x):
        for block in self.transformer.h:
            x = block.attn.c_attn(x) + block.attn.c_proj(x) + block.mlp.c_fc(x)
        return x


@pytest.fixture
def peft_gpt2():
    """Tiny PeftModel with GPT-2-style naming and LoRA (r=8)."""
    torch.manual_seed(42)
    base = TinyGPT2(num_layers=4, hidden_size=16)
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["c_attn", "c_proj", "c_fc"],
        bias="none",
    )
    model = get_peft_model(base, config)

    torch.manual_seed(123)
    for _name, module in _lora_modules(model):
        module.lora_B["default"].weight.data.normal_()

    return model


# ---------------------------------------------------------------------------
# Cross-architecture tests (GPT-2)
# ---------------------------------------------------------------------------


def test_gpt2_rank_reducer_layers(peft_gpt2):
    """Layer filtering should work on GPT-2-style model (transformer.h.N)."""
    originals = _snapshot_weights(peft_gpt2)

    reducer = LoRaRankReducer(
        peft_gpt2, adapter_name="default", new_rank=4, layers=[0, 2]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reducer.apply()

    for name, module in _lora_modules(peft_gpt2):
        layer_idx = extract_layer_idx(name)
        if layer_idx in (0, 2):
            assert module.lora_A["default"].weight.shape[0] == 4
        else:
            assert torch.equal(
                module.lora_A["default"].weight.data, originals[name]["A"]
            )


def test_gpt2_dict_target_modules(peft_gpt2):
    """Dict-style target_modules should work on GPT-2-style model."""
    originals = _snapshot_weights(peft_gpt2)

    reducer = LoRaRankReducer(
        peft_gpt2,
        adapter_name="default",
        new_rank=4,
        target_modules={
            0: ["c_attn"],
            3: ["c_fc"],
        },
    )
    reducer.apply()

    for name, module in _lora_modules(peft_gpt2):
        layer_idx = extract_layer_idx(name)
        should_modify = (layer_idx == 0 and name.endswith("c_attn")) or (
            layer_idx == 3 and name.endswith("c_fc")
        )
        if should_modify:
            assert module.lora_A["default"].weight.shape[0] == 4
        else:
            assert torch.equal(
                module.lora_A["default"].weight.data, originals[name]["A"]
            )


def test_gpt2_zeroing_layers(peft_gpt2):
    """LoRaAdapterZeroing with layers filter should work on GPT-2-style model."""
    zeroer = LoRaAdapterZeroing(peft_gpt2, adapter_name="default", layers=[1, 2, 3])
    zeroer.apply()

    for name, module in _lora_modules(peft_gpt2):
        layer_idx = extract_layer_idx(name)
        if layer_idx in (1, 2, 3):
            assert module.scaling["default"] == 0.0
        else:
            assert module.scaling["default"] != 0.0


def test_gpt2_pipeline(peft_gpt2):
    """Pipeline should work on GPT-2-style model."""
    pipeline = LoRaPipeline(
        peft_gpt2,
        steps=[
            (LoRaRankReducer, "default", {"new_rank": 4}),
            (LoRaScaling, "default", {"scale_factor": 0.5}),
        ],
    )
    pipeline.apply()

    for _name, module in _lora_modules(peft_gpt2):
        assert module.lora_A["default"].weight.shape[0] == 4
        assert module.r["default"] == 4
        assert module.scaling["default"] == 1.0  # 2.0 * 0.5


def test_gpt2_restore(peft_gpt2):
    """Restore should work on GPT-2-style model."""
    originals = _snapshot_weights(peft_gpt2)

    reducer = LoRaRankReducer(peft_gpt2, adapter_name="default", new_rank=4)
    reducer.apply()
    reducer.restore()

    for name, module in _lora_modules(peft_gpt2):
        assert torch.equal(module.lora_A["default"].weight.data, originals[name]["A"])
        assert torch.equal(module.lora_B["default"].weight.data, originals[name]["B"])


# ---------------------------------------------------------------------------
# Validation error tests
# ---------------------------------------------------------------------------


def test_validation_bad_adapter_name(peft_model):
    """Non-existent adapter_name should raise ValueError."""
    with pytest.raises(ValueError, match="Adapter 'nonexistent' not found"):
        LoRaScaling(peft_model, adapter_name="nonexistent", scale_factor=1.0)


def test_validation_bad_adapter_name_rank_reducer(peft_model):
    """Non-existent adapter in LoRaRankReducer should raise ValueError."""
    with pytest.raises(ValueError, match="Available adapters"):
        LoRaRankReducer(peft_model, adapter_name="bogus", new_rank=4)


def test_validation_bad_adapter_name_zeroing(peft_model):
    """Non-existent adapter in LoRaAdapterZeroing should raise ValueError."""
    with pytest.raises(ValueError, match="Adapter 'nope' not found"):
        LoRaAdapterZeroing(peft_model, adapter_name="nope")


def test_validation_bad_adapter_in_pipeline(peft_model):
    """Non-existent adapter in a pipeline step should raise ValueError."""
    with pytest.raises(ValueError, match="Adapter 'ghost' not found"):
        LoRaPipeline(
            peft_model,
            steps=[(LoRaScaling, "ghost", {"scale_factor": 1.0})],
        )


def test_validation_bad_target_modules_list(peft_model):
    """target_modules list with typos should raise ValueError."""
    with pytest.raises(ValueError, match="target_modules=.*matched no LoRA"):
        LoRaScaling(
            peft_model,
            adapter_name="default",
            scale_factor=1.0,
            target_modules=["q_poj"],  # typo
        )


def test_validation_bad_target_modules_list_shows_available(peft_model):
    """Error message should list available module name suffixes."""
    with pytest.raises(ValueError, match="Available module name suffixes"):
        LoRaScaling(
            peft_model,
            adapter_name="default",
            scale_factor=1.0,
            target_modules=["nonexistent_proj"],
        )


def test_validation_out_of_range_layers(peft_model):
    """Out-of-range layer indices should raise ValueError."""
    with pytest.raises(ValueError, match="layers=.*matched no LoRA"):
        LoRaScaling(
            peft_model,
            adapter_name="default",
            scale_factor=1.0,
            layers=[99, 100],
        )


def test_validation_out_of_range_layers_shows_available(peft_model):
    """Error message should list available layer indices."""
    with pytest.raises(ValueError, match="Available layer indices"):
        LoRaScaling(
            peft_model,
            adapter_name="default",
            scale_factor=1.0,
            layers=[99],
        )


def test_validation_dict_target_modules_bad_keys(peft_model):
    """Dict target_modules with non-existent layer keys should raise ValueError."""
    with pytest.raises(ValueError, match="matched no LoRA"):
        LoRaRankReducer(
            peft_model,
            adapter_name="default",
            new_rank=4,
            target_modules={99: ["q_proj"]},
        )


def test_validation_empty_dict_no_error(peft_model):
    """Empty dict target_modules should NOT raise — intentional 'modify nothing'."""
    # Should not raise
    reducer = LoRaRankReducer(
        peft_model, adapter_name="default", new_rank=4, target_modules={}
    )
    reducer.apply()


# ---------------------------------------------------------------------------
# Unrecognized naming convention validation
# ---------------------------------------------------------------------------


class WeirdModel(nn.Module):
    """Model with non-standard naming that no pattern matches."""

    def __init__(self, hidden_size: int = 16):
        super().__init__()
        self.stack = nn.Module()
        self.stack.block_0 = nn.Module()
        self.stack.block_0.proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.stack.block_1 = nn.Module()
        self.stack.block_1.proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x):
        x = self.stack.block_0.proj(x)
        x = self.stack.block_1.proj(x)
        return x


@pytest.fixture
def peft_weird():
    """PeftModel with non-standard naming."""
    torch.manual_seed(42)
    base = WeirdModel(hidden_size=16)
    config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["proj"],
        bias="none",
    )
    model = get_peft_model(base, config)
    torch.manual_seed(123)
    for _name, module in _lora_modules(model):
        module.lora_B["default"].weight.data.normal_()
    return model


def test_validation_unrecognized_naming_with_layers(peft_weird):
    """layers filter on unrecognized naming should raise with helpful message."""
    with pytest.raises(ValueError, match="layer_idx_extractor"):
        LoRaScaling(
            peft_weird,
            adapter_name="default",
            scale_factor=1.0,
            layers=[0],
        )


def test_validation_unrecognized_naming_with_dict(peft_weird):
    """Dict target_modules on unrecognized naming should mention LAYER_PATTERNS."""
    with pytest.raises(ValueError, match="known pattern"):
        LoRaRankReducer(
            peft_weird,
            adapter_name="default",
            new_rank=2,
            target_modules={0: ["proj"]},
        )


def test_validation_unrecognized_naming_no_filter_ok(peft_weird):
    """No filter on unrecognized naming should work fine (no layer extraction needed)."""
    scaler = LoRaScaling(peft_weird, adapter_name="default", scale_factor=2.0)
    scaler.apply()

    for _name, module in _lora_modules(peft_weird):
        assert module.scaling["default"] == pytest.approx(4.0)  # 2.0 * 2.0


def test_validation_unrecognized_naming_target_modules_list_ok(peft_weird):
    """List target_modules on unrecognized naming should work (no layer extraction needed)."""
    scaler = LoRaScaling(
        peft_weird,
        adapter_name="default",
        scale_factor=0.5,
        target_modules=["proj"],
    )
    scaler.apply()

    for _name, module in _lora_modules(peft_weird):
        assert module.scaling["default"] == pytest.approx(1.0)  # 2.0 * 0.5


# ---------------------------------------------------------------------------
# Custom layer_idx_extractor tests
# ---------------------------------------------------------------------------


def _weird_extractor(name: str) -> int | None:
    """Extract layer index from WeirdModel's ``stack.block_N.`` naming."""
    if "stack.block_" not in name:
        return None
    return int(name.split("stack.block_")[-1].split(".")[0])


def test_custom_extractor_layers_filter(peft_weird):
    """Custom extractor enables layer filtering on non-standard models."""
    scaler = LoRaScaling(
        peft_weird,
        adapter_name="default",
        scale_factor=0.0,
        layers=[0],
        layer_idx_extractor=_weird_extractor,
    )
    scaler.apply()

    for name, module in _lora_modules(peft_weird):
        idx = _weird_extractor(name)
        if idx == 0:
            assert module.scaling["default"] == 0.0
        else:
            assert module.scaling["default"] != 0.0


def test_custom_extractor_dict_target_modules(peft_weird):
    """Custom extractor enables dict target_modules on non-standard models."""
    reducer = LoRaRankReducer(
        peft_weird,
        adapter_name="default",
        new_rank=2,
        target_modules={1: ["proj"]},
        layer_idx_extractor=_weird_extractor,
    )
    reducer.apply()

    for name, module in _lora_modules(peft_weird):
        idx = _weird_extractor(name)
        if idx == 1 and name.endswith("proj"):
            assert module.lora_A["default"].weight.shape[0] == 2
        else:
            assert module.lora_A["default"].weight.shape[0] == 4


def test_custom_extractor_restore(peft_weird):
    """Restore should work with custom extractor."""
    originals = _snapshot_weights(peft_weird)

    reducer = LoRaRankReducer(
        peft_weird,
        adapter_name="default",
        new_rank=2,
        target_modules={0: ["proj"]},
        layer_idx_extractor=_weird_extractor,
    )
    reducer.apply()
    reducer.restore()

    for name, module in _lora_modules(peft_weird):
        assert torch.equal(module.lora_A["default"].weight.data, originals[name]["A"])
        assert torch.equal(module.lora_B["default"].weight.data, originals[name]["B"])
