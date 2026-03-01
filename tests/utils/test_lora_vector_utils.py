import pytest
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch import nn

from src.utils.lora_vector_utils import (
    LoRaVector,
    LoRaVectorCollection,
    LoRaVectorSpace,
    cosine_similarity_matrix,
    gram_matrix,
)

# ---------------------------------------------------------------------------
# Tiny model (same pattern as test_peft_manipulations.py)
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


def _make_peft_model(rank: int = 8, lora_alpha: int = 16, seed: int = 42):
    """Create a tiny PeftModel with LoRA at the given rank."""
    torch.manual_seed(seed)
    base = TinyTransformer(num_layers=4, hidden_size=16)
    config = LoraConfig(
        r=rank,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(base, config)

    # Initialize lora_B with random values (PEFT initializes to zeros)
    torch.manual_seed(123)
    for _, module in model.named_modules():
        if hasattr(module, "lora_B") and "default" in module.lora_B:
            module.lora_B["default"].weight.data.normal_()

    return model


def _materialize_delta_w(vec: LoRaVector) -> torch.Tensor:
    """Materialize the full ∆W vector by computing B@A for each module and concatenating."""
    parts = []
    for name in vec.module_names:
        B, A = vec.factors[name]
        parts.append((B @ A).reshape(-1))
    return torch.cat(parts)


@pytest.fixture
def peft_model():
    return _make_peft_model(rank=8)


@pytest.fixture
def vec_a(peft_model):
    return LoRaVector.from_peft(peft_model, "default")


@pytest.fixture
def vec_b():
    model = _make_peft_model(rank=8, seed=99)
    return LoRaVector.from_peft(model, "default")


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_from_peft_basic(self, peft_model):
        vec = LoRaVector.from_peft(peft_model, "default")
        # 4 layers × 4 modules = 16 modules
        assert len(vec.module_names) == 16
        assert vec.max_rank == 8

    def test_from_peft_with_scaling(self, peft_model):
        vec_scaled = LoRaVector.from_peft(peft_model, "default", include_scaling=True)
        vec_plain = LoRaVector.from_peft(peft_model, "default", include_scaling=False)
        # lora_alpha=16, r=8 → scaling = 2.0
        for name in vec_scaled.module_names:
            B_scaled, _ = vec_scaled.factors[name]
            B_plain, _ = vec_plain.factors[name]
            torch.testing.assert_close(B_scaled, B_plain * 2.0)

    def test_from_peft_target_modules_filter(self, peft_model):
        vec = LoRaVector.from_peft(peft_model, "default", target_modules=["q_proj"])
        assert len(vec.module_names) == 4  # 4 layers × 1 module
        assert all("q_proj" in name for name in vec.module_names)

    def test_from_peft_layers_filter(self, peft_model):
        vec = LoRaVector.from_peft(peft_model, "default", layers=[0, 1])
        assert len(vec.module_names) == 8  # 2 layers × 4 modules

    def test_from_peft_invalid_adapter(self, peft_model):
        with pytest.raises(ValueError, match="not found"):
            LoRaVector.from_peft(peft_model, "nonexistent")

    def test_from_peft_no_match(self, peft_model):
        with pytest.raises(ValueError, match="No LoRA modules matched"):
            LoRaVector.from_peft(peft_model, "default", target_modules=["nonexistent"])

    def test_empty_factors_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            LoRaVector({})

    def test_module_names_sorted(self, vec_a):
        names = vec_a.module_names
        assert names == sorted(names)

    def test_total_params(self, vec_a):
        expected = 16 * (8 * 16 + 8 * 16)  # 16 modules, r=8, in=out=16
        assert vec_a.total_params == expected

    def test_vector_dim(self, vec_a):
        expected = 16 * (16 * 16)  # 16 modules, out=16, in=16
        assert vec_a.vector_dim == expected


# ---------------------------------------------------------------------------
# Arithmetic
# ---------------------------------------------------------------------------


class TestArithmetic:
    def test_add_materializes_correctly(self, vec_a, vec_b):
        """(a + b) materialized should equal materialized(a) + materialized(b)."""
        mat_a = _materialize_delta_w(vec_a)
        mat_b = _materialize_delta_w(vec_b)
        mat_sum = _materialize_delta_w(vec_a + vec_b)
        torch.testing.assert_close(mat_sum, mat_a + mat_b, atol=1e-5, rtol=1e-5)

    def test_sub_materializes_correctly(self, vec_a, vec_b):
        mat_a = _materialize_delta_w(vec_a)
        mat_b = _materialize_delta_w(vec_b)
        mat_diff = _materialize_delta_w(vec_a - vec_b)
        torch.testing.assert_close(mat_diff, mat_a - mat_b, atol=1e-5, rtol=1e-5)

    def test_neg_materializes_correctly(self, vec_a):
        mat_a = _materialize_delta_w(vec_a)
        mat_neg = _materialize_delta_w(-vec_a)
        torch.testing.assert_close(mat_neg, -mat_a, atol=1e-5, rtol=1e-5)

    def test_scale_materializes_correctly(self, vec_a):
        mat_a = _materialize_delta_w(vec_a)
        mat_scaled = _materialize_delta_w(vec_a * 3.5)
        torch.testing.assert_close(mat_scaled, mat_a * 3.5, atol=1e-5, rtol=1e-5)

    def test_rmul(self, vec_a):
        mat_a = _materialize_delta_w(vec_a)
        mat_scaled = _materialize_delta_w(2.0 * vec_a)
        torch.testing.assert_close(mat_scaled, mat_a * 2.0, atol=1e-5, rtol=1e-5)

    def test_add_rank_grows(self, vec_a, vec_b):
        result = vec_a + vec_b
        assert result.max_rank == vec_a.max_rank + vec_b.max_rank

    def test_scale_rank_unchanged(self, vec_a):
        result = vec_a * 2.0
        assert result.max_rank == vec_a.max_rank

    def test_incompatible_modules_raises(self, vec_a):
        # Create a vector with different module names
        factors = {f"different.{k}": v for k, v in vec_a.factors.items()}
        vec_different = LoRaVector(factors)
        with pytest.raises(ValueError, match="different module names"):
            vec_a + vec_different

    def test_sub_add_round_trip(self, vec_a, vec_b):
        """(a - b) + b should equal a."""
        result = (vec_a - vec_b) + vec_b
        sim = vec_a.cosine_similarity(result)
        assert sim == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class TestMetrics:
    def test_dot_matches_materialized(self, vec_a, vec_b):
        """Factored dot product should match materialized flat dot product."""
        mat_a = _materialize_delta_w(vec_a)
        mat_b = _materialize_delta_w(vec_b)
        expected = (mat_a @ mat_b).item()
        actual = vec_a.dot(vec_b)
        assert actual == pytest.approx(expected, rel=1e-4)

    def test_dot_self_matches_norm_squared(self, vec_a):
        mat_a = _materialize_delta_w(vec_a)
        expected = (mat_a @ mat_a).item()
        actual = vec_a.dot(vec_a)
        assert actual == pytest.approx(expected, rel=1e-4)

    def test_norm(self, vec_a):
        mat_a = _materialize_delta_w(vec_a)
        expected = mat_a.norm().item()
        actual = vec_a.norm
        assert actual == pytest.approx(expected, rel=1e-4)

    def test_cosine_similarity_self(self, vec_a):
        assert vec_a.cosine_similarity(vec_a) == pytest.approx(1.0, abs=1e-5)

    def test_cosine_similarity_matches_materialized(self, vec_a, vec_b):
        mat_a = _materialize_delta_w(vec_a)
        mat_b = _materialize_delta_w(vec_b)
        expected = (mat_a @ mat_b / (mat_a.norm() * mat_b.norm())).item()
        actual = vec_a.cosine_similarity(vec_b)
        assert actual == pytest.approx(expected, rel=1e-4)

    def test_linearity(self, vec_a, vec_b):
        """(a + b).dot(a) should equal a.dot(a) + b.dot(a)."""
        lhs = (vec_a + vec_b).dot(vec_a)
        rhs = vec_a.dot(vec_a) + vec_b.dot(vec_a)
        assert lhs == pytest.approx(rhs, rel=1e-4)

    def test_dot_commutative(self, vec_a, vec_b):
        assert vec_a.dot(vec_b) == pytest.approx(vec_b.dot(vec_a), rel=1e-5)

    def test_cosine_similarity_with_zero(self, vec_a):
        """Cosine similarity with zero vector should be 0."""
        zero = vec_a.zero_like()
        assert vec_a.cosine_similarity(zero) == 0.0
        assert zero.cosine_similarity(vec_a) == 0.0

    def test_norm_of_zero(self, vec_a):
        """Zero vector should have norm 0."""
        assert vec_a.zero_like().norm == 0.0


# ---------------------------------------------------------------------------
# Rank management
# ---------------------------------------------------------------------------


class TestRankManagement:
    def test_rank_reduce(self, vec_a):
        reduced = vec_a.rank_reduce(2)
        assert reduced.max_rank == 2
        # Should be a reasonable approximation
        sim = vec_a.cosine_similarity(reduced)
        assert sim > 0.5

    def test_rank_reduce_same_rank_exact(self, vec_a):
        """rank_reduce at the same rank should give cosine sim ≈ 1.0."""
        reduced = vec_a.rank_reduce(8)
        assert reduced.max_rank == 8
        sim = vec_a.cosine_similarity(reduced)
        assert sim == pytest.approx(1.0, abs=1e-3)

    def test_rank_reduce_invalid(self, vec_a):
        with pytest.raises(ValueError, match="new_rank must be >= 1"):
            vec_a.rank_reduce(0)

    def test_rank_reduce_higher_than_current_raises(self, vec_a):
        with pytest.raises(ValueError, match="new_rank must be <="):
            vec_a.rank_reduce(16)  # current rank is 8


# ---------------------------------------------------------------------------
# Model I/O
# ---------------------------------------------------------------------------


class TestModelIO:
    def test_round_trip(self, peft_model):
        """from_peft → write → from_peft should approximately recover."""
        vec = LoRaVector.from_peft(peft_model, "default")

        target = _make_peft_model(rank=8)
        vec.write_to_existing_peft_model(target, "default")

        vec_recovered = LoRaVector.from_peft(target, "default")
        sim = vec.cosine_similarity(vec_recovered)
        assert sim == pytest.approx(1.0, abs=1e-3)

    def test_write_to_lower_rank(self, peft_model):
        """Writing to a lower-rank model should truncate via SVD."""
        vec = LoRaVector.from_peft(peft_model, "default")

        target = _make_peft_model(rank=2)
        vec.write_to_existing_peft_model(target, "default", resize_peft_rank=False)

        for _, module in target.named_modules():
            if hasattr(module, "r") and "default" in getattr(module, "r", {}):
                assert module.r["default"] == 2

    def test_write_sets_scaling_to_one(self, peft_model):
        vec = LoRaVector.from_peft(peft_model, "default")

        target = _make_peft_model(rank=8)
        vec.write_to_existing_peft_model(target, "default")

        for _, module in target.named_modules():
            if hasattr(module, "scaling") and "default" in getattr(
                module, "scaling", {}
            ):
                assert module.scaling["default"] == pytest.approx(1.0)

    def test_forward_pass_after_write(self, peft_model):
        """Model should produce similar outputs after extract → write."""
        vec = LoRaVector.from_peft(peft_model, "default")

        x = torch.randn(2, 16)
        out_before = peft_model(x).detach()

        vec.write_to_existing_peft_model(peft_model, "default")
        out_after = peft_model(x).detach()

        torch.testing.assert_close(out_after, out_before, atol=1e-3, rtol=1e-3)

    def test_invalid_adapter_raises(self, peft_model, vec_a):
        with pytest.raises(ValueError, match="not found"):
            vec_a.write_to_existing_peft_model(peft_model, "nonexistent")


# ---------------------------------------------------------------------------
# to() device/dtype
# ---------------------------------------------------------------------------


class TestTo:
    def test_to_dtype(self, vec_a):
        vec_f64 = vec_a.to(dtype=torch.float64)
        for _, (B, A) in vec_f64.factors.items():
            assert B.dtype == torch.float64
            assert A.dtype == torch.float64


# ---------------------------------------------------------------------------
# gram_matrix
# ---------------------------------------------------------------------------


class TestGramMatrix:
    def test_shape(self, vec_a, vec_b):
        G = gram_matrix([vec_a, vec_b])
        assert G.shape == (2, 2)

    def test_symmetric(self, vec_a, vec_b):
        G = gram_matrix([vec_a, vec_b])
        torch.testing.assert_close(G, G.T)

    def test_diagonal_is_norm_squared(self, vec_a, vec_b):
        G = gram_matrix([vec_a, vec_b])
        assert G[0, 0].item() == pytest.approx(vec_a.dot(vec_a), rel=1e-5)
        assert G[1, 1].item() == pytest.approx(vec_b.dot(vec_b), rel=1e-5)

    def test_matches_materialized(self, vec_a, vec_b):
        G = gram_matrix([vec_a, vec_b])
        mat_a = _materialize_delta_w(vec_a)
        mat_b = _materialize_delta_w(vec_b)
        expected = torch.tensor(
            [
                [mat_a @ mat_a, mat_a @ mat_b],
                [mat_b @ mat_a, mat_b @ mat_b],
            ]
        )
        torch.testing.assert_close(G, expected, atol=1e-3, rtol=1e-3)


# ---------------------------------------------------------------------------
# LoRAVectorCollection
# ---------------------------------------------------------------------------


class TestLoRAVectorCollection:
    @pytest.fixture
    def vectors(self):
        """Create several distinct LoRAVectors."""
        vecs = []
        for seed in [42, 99, 7, 13, 55]:
            model = _make_peft_model(rank=8, seed=seed)
            vecs.append(LoRaVector.from_peft(model, "default"))
        return vecs

    def test_init_from_dict(self, vec_a, vec_b):
        coll = LoRaVectorCollection({"a": vec_a, "b": vec_b})
        assert coll.names == ["a", "b"]
        assert len(coll) == 2

    def test_init_from_list(self, vec_a, vec_b):
        coll = LoRaVectorCollection([vec_a, vec_b])
        assert coll.names == ["0", "1"]
        assert len(coll) == 2

    def test_getitem_by_name(self, vec_a, vec_b):
        coll = LoRaVectorCollection({"a": vec_a, "b": vec_b})
        assert coll["a"] is vec_a
        assert coll["b"] is vec_b

    def test_getitem_by_index(self, vec_a, vec_b):
        coll = LoRaVectorCollection({"a": vec_a, "b": vec_b})
        assert coll[0] is vec_a
        assert coll[1] is vec_b

    def test_getitem_invalid_name_raises(self, vec_a):
        coll = LoRaVectorCollection([vec_a])
        with pytest.raises(KeyError, match="No vector named"):
            coll["nonexistent"]

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            LoRaVectorCollection([])
        with pytest.raises(ValueError, match="must not be empty"):
            LoRaVectorCollection({})

    def test_gram_matrix_cached(self, vec_a, vec_b):
        coll = LoRaVectorCollection([vec_a, vec_b])
        G1 = coll.gram_matrix()
        G2 = coll.gram_matrix()
        assert G1 is G2  # same object — cached

    def test_gram_matrix_matches_standalone(self, vec_a, vec_b):
        coll = LoRaVectorCollection([vec_a, vec_b])
        G_coll = coll.gram_matrix()
        G_standalone = gram_matrix([vec_a, vec_b])
        torch.testing.assert_close(G_coll, G_standalone)

    def test_cosine_similarity_matrix(self, vec_a, vec_b):
        coll = LoRaVectorCollection([vec_a, vec_b])
        C = coll.cosine_similarity_matrix()
        assert C.shape == (2, 2)
        # Diagonal should be 1.0
        assert C[0, 0].item() == pytest.approx(1.0, abs=1e-5)
        assert C[1, 1].item() == pytest.approx(1.0, abs=1e-5)

    def test_pca_returns_correct_types(self, vectors):
        coll = LoRaVectorCollection(vectors)
        result = coll.pca(n_dims=3)
        assert isinstance(result.space, LoRaVectorSpace)
        assert result.input_coords.shape == (5, 3)
        assert len(result.eigenvalues) == 5
        assert len(result.explained_variance) == 3

    def test_pca_default_n_dims(self, vectors):
        """Default n_dims should equal number of vectors."""
        coll = LoRaVectorCollection(vectors)
        result = coll.pca()
        # Centering reduces rank by 1, so n_actual = n - 1
        assert result.input_coords.shape[0] == 5
        assert result.input_coords.shape[1] <= 5

    def test_pca_n_dims_one(self, vectors):
        """n_dims=1 should work and return a single component."""
        coll = LoRaVectorCollection(vectors)
        result = coll.pca(n_dims=1)
        assert result.input_coords.shape == (5, 1)
        assert result.n_dims == 1
        assert len(result.pc_names) == 1

    def test_pca_too_many_dims_raises(self, vectors):
        coll = LoRaVectorCollection(vectors)
        with pytest.raises(ValueError, match="cannot exceed"):
            coll.pca(n_dims=10)


# ---------------------------------------------------------------------------
# LoRAVectorSpace / PCA
# ---------------------------------------------------------------------------


class TestLoRAVectorSpace:
    @pytest.fixture
    def vectors(self):
        """Create several distinct LoRAVectors for PCA testing."""
        vecs = []
        for seed in [42, 99, 7, 13, 55]:
            model = _make_peft_model(rank=8, seed=seed)
            vecs.append(LoRaVector.from_peft(model, "default"))
        return vecs

    def test_init_from_dict(self, vec_a, vec_b):
        space = LoRaVectorSpace({"pc0": vec_a, "pc1": vec_b})
        assert space.names == ["pc0", "pc1"]
        assert space.n_dims == 2

    def test_init_from_list(self, vec_a, vec_b):
        space = LoRaVectorSpace([vec_a, vec_b])
        assert space.names == ["0", "1"]
        assert space.n_dims == 2

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            LoRaVectorSpace([])
        with pytest.raises(ValueError, match="must not be empty"):
            LoRaVectorSpace({})

    def test_pca_basis_orthonormal(self, vectors):
        """Basis vectors should be approximately orthonormal."""
        coll = LoRaVectorCollection(vectors)
        result = coll.pca(n_dims=3)
        space = result.space
        for i in range(space.n_dims):
            assert space.basis[i].norm == pytest.approx(1.0, abs=1e-3)
            for j in range(i + 1, space.n_dims):
                dot = space.basis[i].dot(space.basis[j])
                assert dot == pytest.approx(0.0, abs=1e-2)

    def test_pca_explained_variance_sums_to_at_most_one(self, vectors):
        coll = LoRaVectorCollection(vectors)
        result = coll.pca(n_dims=3)
        assert result.explained_variance.sum().item() <= 1.0 + 1e-6

    def test_pca_eigenvalues_descending(self, vectors):
        coll = LoRaVectorCollection(vectors)
        result = coll.pca(n_dims=3)
        eigs = result.eigenvalues
        for i in range(len(eigs) - 1):
            assert eigs[i] >= eigs[i + 1] - 1e-6

    def test_project_reconstruct_round_trip(self, vectors):
        """project → reconstruct with n_dims < n should be approximate."""
        coll = LoRaVectorCollection(vectors)
        result = coll.pca(n_dims=4)
        space = result.space

        coords = space.project(vectors[0])
        reconstructed = space.reconstruct(coords)

        sim = vectors[0].cosine_similarity(reconstructed)
        assert sim > 0.9

    def test_reconstruct_project_round_trip(self, vectors):
        """reconstruct → project should recover the original coordinates."""
        coll = LoRaVectorCollection(vectors)
        result = coll.pca(n_dims=3)
        space = result.space

        coords = torch.tensor([1.5, -0.3, 0.7])
        reconstructed = space.reconstruct(coords)
        recovered = space.project(reconstructed)
        torch.testing.assert_close(recovered, coords, atol=1e-3, rtol=1e-3)

    def test_full_rank_pca_round_trip(self, vectors):
        """With n_dims = n_vectors, project → reconstruct should be ≈ exact."""
        coll = LoRaVectorCollection(vectors)
        result = coll.pca(n_dims=5)
        space = result.space

        for v in vectors:
            coords = space.project(v)
            reconstructed = space.reconstruct(coords)
            sim = v.cosine_similarity(reconstructed)
            assert sim == pytest.approx(1.0, abs=1e-2)

    def test_shift_changes_coordinate(self, vectors):
        """Shifting along dim 0 should change that coordinate by exactly the delta."""
        coll = LoRaVectorCollection(vectors)
        result = coll.pca(n_dims=3)
        space = result.space

        original_coords = space.project(vectors[0])
        shifted = space.shift(vectors[0], {0: 1.0})
        shifted_coords = space.project(shifted)

        delta_0 = shifted_coords[0] - original_coords[0]
        assert delta_0.item() == pytest.approx(1.0, abs=1e-3)

        # Other coordinates should be unchanged
        for i in range(1, 3):
            delta_i = shifted_coords[i] - original_coords[i]
            assert delta_i.item() == pytest.approx(0.0, abs=1e-3)

    def test_shift_preserves_out_of_basis(self, vectors):
        """Shift forward then back should recover the original (lossless)."""
        coll = LoRaVectorCollection(vectors)
        result = coll.pca(n_dims=3)
        space = result.space

        shifted = space.shift(vectors[0], {0: 5.0, 1: -3.0})
        recovered = space.shift(shifted, {0: -5.0, 1: 3.0})

        sim = vectors[0].cosine_similarity(recovered)
        assert sim == pytest.approx(1.0, abs=1e-3)

    def test_shift_invalid_dim_raises(self, vectors):
        coll = LoRaVectorCollection(vectors)
        result = coll.pca(n_dims=2)
        with pytest.raises(ValueError, match="out of range"):
            result.space.shift(vectors[0], {5: 1.0})

    def test_project_all(self, vectors):
        coll = LoRaVectorCollection(vectors)
        result = coll.pca(n_dims=3)
        coords = result.space.project_all(vectors)
        assert coords.shape == (5, 3)
        for i, v in enumerate(vectors):
            single = result.space.project(v)
            torch.testing.assert_close(coords[i], single, atol=1e-4, rtol=1e-4)

    def test_reconstruct_wrong_shape_raises(self, vectors):
        coll = LoRaVectorCollection(vectors)
        result = coll.pca(n_dims=3)
        with pytest.raises(ValueError, match="Expected coords"):
            result.space.reconstruct(torch.zeros(5))

    def test_normalize_true_coords_match_project(self, vectors):
        """With normalize=True, input_coords[i] should match space.project(v_i)."""
        coll = LoRaVectorCollection(vectors)
        result = coll.pca(n_dims=3, normalize=True)
        assert result.normalized is True
        for i, v in enumerate(vectors):
            projected = result.space.project(v)
            torch.testing.assert_close(
                result.input_coords[i], projected, atol=1e-3, rtol=1e-3
            )

    def test_normalize_false_coords_match_project(self, vectors):
        """With normalize=False, input_coords[i] should match space.project(v_i)."""
        coll = LoRaVectorCollection(vectors)
        result = coll.pca(n_dims=3, normalize=False)
        assert result.normalized is False
        for i, v in enumerate(vectors):
            projected = result.space.project(v)
            torch.testing.assert_close(
                result.input_coords[i], projected, atol=1e-3, rtol=1e-3
            )

    def test_normalize_true_unit_variance(self, vectors):
        """With normalize=True, each PC column should have approximately unit variance."""
        coll = LoRaVectorCollection(vectors)
        result = coll.pca(normalize=True)
        for k in range(result.n_dims):
            col = result.input_coords[:, k]
            variance = col.var(correction=0).item()
            assert variance == pytest.approx(1.0, abs=0.1), (
                f"PC{k} variance = {variance:.4f}, expected ~1.0"
            )

    def test_pca_result_metadata(self, vectors):
        """PCAResult should have correct metadata fields."""
        coll = LoRaVectorCollection(
            {"alpha": v for v in vectors[:1]}
            | {f"v{i}": v for i, v in enumerate(vectors[1:])}
        )
        result = coll.pca(n_dims=3)
        assert result.input_names == coll.names
        assert result.pc_names == ["PC0", "PC1", "PC2"]
        assert result.n_dims == 3
        assert len(result.pc_scales) == 3
        assert (result.pc_scales > 0).all()

    def test_shift_works_with_normalize_true(self, vectors):
        """Shift delta=1 should change that coordinate by 1 in normalized space."""
        coll = LoRaVectorCollection(vectors)
        result = coll.pca(n_dims=3, normalize=True)
        space = result.space

        original_coords = space.project(vectors[0])
        shifted = space.shift(vectors[0], {0: 1.0})
        shifted_coords = space.project(shifted)

        delta_0 = shifted_coords[0] - original_coords[0]
        assert delta_0.item() == pytest.approx(1.0, abs=1e-3)
        for i in range(1, 3):
            delta_i = shifted_coords[i] - original_coords[i]
            assert delta_i.item() == pytest.approx(0.0, abs=1e-3)


# ---------------------------------------------------------------------------
# PCA matches materialized kernel PCA
# ---------------------------------------------------------------------------


class TestPCAMatchesMaterialized:
    def test_gram_matrix_matches(self):
        """The factored Gram matrix should match the materialized one."""
        vectors = []
        for seed in [42, 99, 7]:
            model = _make_peft_model(rank=4, seed=seed)
            vectors.append(LoRaVector.from_peft(model, "default"))

        G_factored = gram_matrix(vectors)

        mats = [_materialize_delta_w(v) for v in vectors]
        G_materialized = torch.zeros(3, 3)
        for i in range(3):
            for j in range(3):
                G_materialized[i, j] = mats[i] @ mats[j]

        torch.testing.assert_close(G_factored, G_materialized, atol=1e-3, rtol=1e-3)

    def test_pca_coords_match_materialized(self):
        """PCA coordinates should match kernel PCA on materialized vectors."""
        vectors = []
        for seed in [42, 99, 7, 13]:
            model = _make_peft_model(rank=4, seed=seed)
            vectors.append(LoRaVector.from_peft(model, "default"))

        coll = LoRaVectorCollection(vectors)
        result = coll.pca(n_dims=3, normalize=False)

        # Do the same PCA manually on materialized vectors
        mats = [_materialize_delta_w(v) for v in vectors]
        n = len(mats)
        G = torch.zeros(n, n)
        for i in range(n):
            for j in range(n):
                G[i, j] = mats[i] @ mats[j]

        ones_n = torch.ones(n, n) / n
        G_centered = G - ones_n @ G - G @ ones_n + ones_n @ G @ ones_n
        eigenvalues, eigenvectors = torch.linalg.eigh(G_centered)
        eigenvalues = eigenvalues.flip(0)
        eigenvectors = eigenvectors.flip(1)
        eigs_safe = eigenvalues.clamp(min=0)
        coords_manual = eigenvectors[:, :3] * eigs_safe[:3].sqrt()

        # With normalize=False, coords should match raw kernel PCA coords
        for k in range(3):
            col_result = result.input_coords[:, k]
            col_manual = coords_manual[:, k]
            # Eigenvectors can have flipped sign; check cosine sim ≈ ±1
            sim = torch.dot(col_result, col_manual) / (
                col_result.norm() * col_manual.norm() + 1e-10
            )
            assert abs(sim.item()) > 0.99, (
                f"PC{k} coordinates don't match (cosine sim = {sim.item():.4f})"
            )
            # Values should match (up to sign flip)
            sign = 1.0 if sim > 0 else -1.0
            torch.testing.assert_close(
                col_result, sign * col_manual, atol=1e-3, rtol=1e-3
            )


# ---------------------------------------------------------------------------
# zero_like
# ---------------------------------------------------------------------------


class TestZeroLike:
    def test_zero_like_structure(self, vec_a):
        """zero_like should have the same module names and compatible shapes."""
        zero = vec_a.zero_like()
        assert zero.module_names == vec_a.module_names
        for name in zero.module_names:
            B_z, A_z = zero.factors[name]
            B_a, A_a = vec_a.factors[name]
            # Same out/in dimensions, rank can differ (rank-1 suffices for zeros)
            assert B_z.shape[0] == B_a.shape[0]
            assert A_z.shape[1] == A_a.shape[1]

    def test_zero_like_materializes_to_zeros(self, vec_a):
        """zero_like should materialize to all zeros."""
        zero = vec_a.zero_like()
        mat = _materialize_delta_w(zero)
        assert mat.abs().max().item() == pytest.approx(0.0, abs=1e-10)

    def test_add_zero_like_identity(self, vec_a):
        """vec + vec.zero_like() should equal vec."""
        result = vec_a + vec_a.zero_like()
        sim = vec_a.cosine_similarity(result)
        assert sim == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# cosine_similarity_matrix standalone
# ---------------------------------------------------------------------------


class TestCosineSimilarityMatrix:
    def test_matches_pairwise(self):
        """Standalone cosine_similarity_matrix should match pairwise calls."""
        vectors = []
        for seed in [42, 99, 7]:
            model = _make_peft_model(rank=4, seed=seed)
            vectors.append(LoRaVector.from_peft(model, "default"))

        C = cosine_similarity_matrix(vectors)
        for i in range(len(vectors)):
            for j in range(len(vectors)):
                expected = vectors[i].cosine_similarity(vectors[j])
                assert C[i, j].item() == pytest.approx(expected, abs=1e-4)


# ---------------------------------------------------------------------------
# LoRAVectorSpace with center/scale
# ---------------------------------------------------------------------------


class TestLoRAVectorSpaceCenterScale:
    def test_project_reconstruct_with_center(self, vec_a, vec_b):
        """Space with center should subtract/add it during project/reconstruct."""
        space = LoRaVectorSpace([vec_a], center=vec_b)

        coords = space.project(vec_a)
        reconstructed = space.reconstruct(coords)

        sim = vec_a.cosine_similarity(reconstructed)
        assert sim == pytest.approx(1.0, abs=1e-2)

    def test_no_center_no_scale_unchanged(self, vec_a, vec_b):
        """Without center/scale, behavior is plain dot-product projection."""
        space = LoRaVectorSpace([vec_a])
        coords = space.project(vec_a)
        # Projection of vec onto itself (unit basis) = norm²
        assert coords[0].item() == pytest.approx(vec_a.dot(vec_a), rel=1e-4)

    def test_scale_without_center(self, vec_a, vec_b):
        """Scale without center should divide/multiply coordinates."""
        scale = torch.tensor([2.0])
        space = LoRaVectorSpace([vec_a], scale=scale)
        coords = space.project(vec_a)
        # Without scale: dot(a, a) = norm². With scale: norm² / 2.0
        assert coords[0].item() == pytest.approx(vec_a.dot(vec_a) / 2.0, rel=1e-4)

        # reconstruct should invert project
        reconstructed = space.reconstruct(coords)
        sim = vec_a.cosine_similarity(reconstructed)
        assert sim == pytest.approx(1.0, abs=1e-3)


# ---------------------------------------------------------------------------
# Multi-adapter isolation and scaling correctness
# ---------------------------------------------------------------------------


def _add_adapter(model, adapter_name: str, rank: int, lora_alpha: int, seed: int):
    """Add a second adapter to an existing PeftModel and randomize its lora_B."""
    config = LoraConfig(
        r=rank,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model.add_adapter(adapter_name, config)

    torch.manual_seed(seed)
    for _, module in model.named_modules():
        if hasattr(module, "lora_B") and adapter_name in module.lora_B:
            module.lora_B[adapter_name].weight.data.normal_()


def _snapshot_adapter_weights(model, adapter_name: str) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Clone all lora_A and lora_B weights for an adapter."""
    snapshot = {}
    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and adapter_name in module.lora_A:
            a = module.lora_A[adapter_name].weight.data.clone()
            b = module.lora_B[adapter_name].weight.data.clone()
            snapshot[name] = (b, a)
    return snapshot


class TestMultiAdapterAndScaling:
    """Tests for multi-adapter isolation and scaling correctness."""

    def test_write_to_adapter_does_not_corrupt_other_adapter(self):
        """Writing to 'default' must not change 'other' adapter weights."""
        model = _make_peft_model(rank=8, lora_alpha=16, seed=42)
        _add_adapter(model, "other", rank=4, lora_alpha=8, seed=77)

        # Snapshot 'other' before we touch 'default'
        snapshot = _snapshot_adapter_weights(model, "other")

        # Modify and write back to 'default'
        vec = LoRaVector.from_peft(model, "default")
        modified = vec * 2.0
        modified.write_to_existing_peft_model(model, "default")

        # Verify 'other' is bitwise identical
        for name, (b_before, a_before) in snapshot.items():
            module = dict(model.named_modules())[name]
            a_after = module.lora_A["other"].weight.data
            b_after = module.lora_B["other"].weight.data
            assert torch.equal(a_before, a_after), f"lora_A changed for {name}"
            assert torch.equal(b_before, b_after), f"lora_B changed for {name}"

    def test_from_peft_reads_correct_adapter(self):
        """Extracting from different adapters should give different vectors."""
        model = _make_peft_model(rank=4, lora_alpha=8, seed=42)
        _add_adapter(model, "other", rank=4, lora_alpha=8, seed=99)

        vec_default = LoRaVector.from_peft(model, "default")
        vec_other = LoRaVector.from_peft(model, "other")

        # They should be materially different
        sim = vec_default.cosine_similarity(vec_other)
        assert abs(sim) < 0.99, f"Vectors from different adapters too similar: {sim}"

        # Each vector's unscaled factors should match the model's raw weights
        for adapter_name, vec in [("default", vec_default), ("other", vec_other)]:
            vec_unscaled = LoRaVector.from_peft(model, adapter_name, include_scaling=False)
            for name in vec_unscaled.module_names:
                B_vec, A_vec = vec_unscaled.factors[name]
                module = dict(model.named_modules())[name]
                A_model = module.lora_A[adapter_name].weight.data
                B_model = module.lora_B[adapter_name].weight.data
                assert torch.allclose(A_vec, A_model, atol=1e-6), f"A mismatch for {name}"
                assert torch.allclose(B_vec, B_model, atol=1e-6), f"B mismatch for {name}"

    @pytest.mark.parametrize(
        "rank, lora_alpha, expected_scaling",
        [
            (4, 4, 1.0),
            (4, 8, 2.0),
            (8, 32, 4.0),
            (4, 2, 0.5),
        ],
    )
    def test_scaling_round_trip_various_ratios(self, rank, lora_alpha, expected_scaling):
        """Extract → write → extract round-trip preserves direction and magnitude."""
        model = _make_peft_model(rank=rank, lora_alpha=lora_alpha, seed=42)
        vec_original = LoRaVector.from_peft(model, "default")

        vec_original.write_to_existing_peft_model(model, "default")
        vec_after = LoRaVector.from_peft(model, "default")

        sim = vec_original.cosine_similarity(vec_after)
        assert sim == pytest.approx(1.0, abs=1e-4), f"Direction changed: sim={sim}"

        norm_original = vec_original.norm
        norm_after = vec_after.norm
        assert norm_after == pytest.approx(norm_original, rel=1e-4), (
            f"Norm changed: {norm_original} → {norm_after}"
        )

    def test_scaling_absorbed_correctly(self):
        """With scaling=2.0, B_scaled should be exactly 2.0 * B_unscaled."""
        model = _make_peft_model(rank=4, lora_alpha=8, seed=42)  # scaling = 8/4 = 2.0

        vec_scaled = LoRaVector.from_peft(model, "default", include_scaling=True)
        vec_unscaled = LoRaVector.from_peft(model, "default", include_scaling=False)

        for name in vec_scaled.module_names:
            B_scaled, A_scaled = vec_scaled.factors[name]
            B_unscaled, A_unscaled = vec_unscaled.factors[name]

            # A should be identical regardless of scaling
            assert torch.allclose(A_scaled, A_unscaled, atol=1e-6), (
                f"A differs for {name} — scaling should not affect A"
            )
            # B_scaled should be 2.0 * B_unscaled
            assert torch.allclose(B_scaled, 2.0 * B_unscaled, atol=1e-6), (
                f"B scaling incorrect for {name}"
            )

    def test_write_does_not_double_scale(self):
        """Forward pass must be identical before and after write round-trip."""
        model = _make_peft_model(rank=4, lora_alpha=8, seed=42)  # scaling = 2.0
        model.eval()

        x = torch.randn(1, 16)
        with torch.no_grad():
            output_before = model(x).clone()

        # Extract (absorbs scaling into B) and write back (sets scaling=1.0)
        vec = LoRaVector.from_peft(model, "default")
        vec.write_to_existing_peft_model(model, "default")

        with torch.no_grad():
            output_after = model(x)

        assert torch.allclose(output_before, output_after, atol=1e-5), (
            f"Forward pass changed after write round-trip. "
            f"Max diff: {(output_before - output_after).abs().max().item()}"
        )


# ---------------------------------------------------------------------------
# Rank handling and write_to_existing_peft_model rank behavior
# ---------------------------------------------------------------------------


class TestRankAndWriteBehavior:
    """Tests for rank changes during write_to_existing_peft_model and rank_reduce."""

    def test_write_after_explicit_rank_reduce(self):
        """rank_reduce(2) then write should produce rank-2 weights in the model."""
        model = _make_peft_model(rank=8, lora_alpha=16, seed=42)
        vec = LoRaVector.from_peft(model, "default")

        vec.rank_reduce(2).write_to_existing_peft_model(model, "default")

        for _, module in model.named_modules():
            if hasattr(module, "lora_A") and "default" in module.lora_A:
                assert module.lora_A["default"].weight.shape[0] == 2
                assert module.lora_B["default"].weight.shape[1] == 2

    def test_write_bloated_rank_truncates_without_resize(self):
        """Adding two rank-8 vectors gives rank 16; truncates when resize_peft_rank=False."""
        model_a = _make_peft_model(rank=8, lora_alpha=16, seed=42)
        model_b = _make_peft_model(rank=8, lora_alpha=16, seed=99)
        vec_a = LoRaVector.from_peft(model_a, "default")
        vec_b = LoRaVector.from_peft(model_b, "default")

        combined = vec_a + vec_b
        assert combined.max_rank == 16

        target = _make_peft_model(rank=8, lora_alpha=16, seed=0)
        combined.write_to_existing_peft_model(target, "default", resize_peft_rank=False)

        # Tensor shapes should match model rank (8), not vector rank (16)
        for _, module in target.named_modules():
            if hasattr(module, "lora_A") and "default" in module.lora_A:
                assert module.lora_A["default"].weight.shape[0] == 8
                assert module.lora_B["default"].weight.shape[1] == 8

        # Should still be a reasonable approximation
        recovered = LoRaVector.from_peft(target, "default")
        sim = combined.cosine_similarity(recovered)
        assert sim > 0.5, f"Truncated vector too different: sim={sim}"

    def test_write_with_resize_preserves_full_rank(self):
        """resize_peft_rank=True should expand model to match vector rank."""
        model_a = _make_peft_model(rank=8, lora_alpha=16, seed=42)
        model_b = _make_peft_model(rank=8, lora_alpha=16, seed=99)
        vec_a = LoRaVector.from_peft(model_a, "default")
        vec_b = LoRaVector.from_peft(model_b, "default")

        combined = vec_a + vec_b
        assert combined.max_rank == 16

        target = _make_peft_model(rank=8, lora_alpha=16, seed=0)
        combined.write_to_existing_peft_model(target, "default", resize_peft_rank=True)

        # Model should now have rank 16
        for _, module in target.named_modules():
            if hasattr(module, "lora_A") and "default" in module.lora_A:
                assert module.lora_A["default"].weight.shape[0] == 16
                assert module.lora_B["default"].weight.shape[1] == 16
                assert module.r["default"] == 16

        # Round-trip should be lossless
        recovered = LoRaVector.from_peft(target, "default")
        sim = combined.cosine_similarity(recovered)
        assert sim == pytest.approx(1.0, abs=1e-4), f"Resize lost info: sim={sim}"

    def test_write_with_resize_shrinks_to_vector_rank(self):
        """resize_peft_rank=True should shrink model when vector rank < model rank."""
        model = _make_peft_model(rank=8, lora_alpha=16, seed=42)
        vec = LoRaVector.from_peft(model, "default")
        reduced = vec.rank_reduce(4)

        target = _make_peft_model(rank=8, lora_alpha=16, seed=0)
        reduced.write_to_existing_peft_model(target, "default", resize_peft_rank=True)

        for _, module in target.named_modules():
            if hasattr(module, "lora_A") and "default" in module.lora_A:
                assert module.lora_A["default"].weight.shape[0] == 4
                assert module.lora_B["default"].weight.shape[1] == 4
                assert module.r["default"] == 4

        recovered = LoRaVector.from_peft(target, "default")
        sim = reduced.cosine_similarity(recovered)
        assert sim == pytest.approx(1.0, abs=1e-4)

    def test_write_with_resize_forward_pass(self):
        """Forward pass should work after resizing to a different rank."""
        model = _make_peft_model(rank=8, lora_alpha=16, seed=42)
        vec_a = LoRaVector.from_peft(model, "default")
        vec_b = LoRaVector.from_peft(_make_peft_model(rank=8, lora_alpha=16, seed=99), "default")

        combined = vec_a + vec_b  # rank 16
        combined.write_to_existing_peft_model(model, "default", resize_peft_rank=True)

        model.eval()
        x = torch.randn(1, 16)
        with torch.no_grad():
            output = model(x)

        assert not torch.isnan(output).any(), "NaN in output after resize write"
        assert not torch.isinf(output).any(), "Inf in output after resize write"

    def test_tensor_shapes_consistent_after_resize(self):
        """After resize, module.r, lora_A shape, and lora_B shape should all agree."""
        model = _make_peft_model(rank=8, lora_alpha=16, seed=42)
        vec = LoRaVector.from_peft(model, "default")

        # Test both expansion (rank 8 → 16) and shrinking (rank 8 → 3)
        for target_rank, use_add in [(16, True), (3, False)]:
            target = _make_peft_model(rank=8, lora_alpha=16, seed=0)
            if use_add:
                test_vec = vec + vec  # rank 16
            else:
                test_vec = vec.rank_reduce(target_rank)

            test_vec.write_to_existing_peft_model(target, "default", resize_peft_rank=True)

            for _, module in target.named_modules():
                if hasattr(module, "lora_A") and "default" in module.lora_A:
                    r_metadata = module.r["default"]
                    a_rank = module.lora_A["default"].weight.shape[0]
                    b_rank = module.lora_B["default"].weight.shape[1]
                    assert r_metadata == a_rank == b_rank == target_rank, (
                        f"Inconsistent ranks: r={r_metadata}, A={a_rank}, B={b_rank}, "
                        f"expected={target_rank}"
                    )

    def test_repeated_add_reduce_accumulates_error(self):
        """Repeated add→reduce with aggressive rank reduction loses information."""
        # The tiny model has hidden_size=16, so ΔW is at most rank 16.
        # We use rank-8 vectors and reduce to rank 2 after each addition
        # to force genuine information loss.
        vecs = []
        for seed in range(6):
            model = _make_peft_model(rank=8, lora_alpha=16, seed=seed)
            vecs.append(LoRaVector.from_peft(model, "default"))

        # Exact sum (keeps all rank)
        exact = vecs[0]
        for v in vecs[1:]:
            exact = exact + v

        # Sum with aggressive rank reduction (rank 2) after each addition
        accumulated = vecs[0].rank_reduce(2)
        for v in vecs[1:]:
            accumulated = (accumulated + v).rank_reduce(2)

        sim = exact.cosine_similarity(accumulated)
        # Compressing to rank 2 after each step should lose significant info
        assert sim < 0.99, (
            f"Expected meaningful error from repeated rank-2 reduction, got sim={sim}"
        )
        # But SVD keeps the dominant directions, so it shouldn't be garbage
        assert sim > 0.3, f"Too much error accumulated: sim={sim}"

    def test_rank_reduce_quality_monotonic(self):
        """Higher target rank should give better approximation."""
        model = _make_peft_model(rank=8, lora_alpha=16, seed=42)
        vec = LoRaVector.from_peft(model, "default")

        sim_rank6 = vec.cosine_similarity(vec.rank_reduce(6))
        sim_rank2 = vec.cosine_similarity(vec.rank_reduce(2))

        assert sim_rank6 > sim_rank2, (
            f"Rank 6 ({sim_rank6}) should approximate better than rank 2 ({sim_rank2})"
        )


# ---------------------------------------------------------------------------
# Config sync after write
# ---------------------------------------------------------------------------


class TestConfigSync:
    """Tests for peft_config syncing after write_to_existing_peft_model."""

    def test_write_default_resizes(self):
        """Default resize_peft_rank=True should expand model to match vector rank."""
        model_a = _make_peft_model(rank=8, lora_alpha=16, seed=42)
        model_b = _make_peft_model(rank=8, lora_alpha=16, seed=99)
        vec_a = LoRaVector.from_peft(model_a, "default")
        vec_b = LoRaVector.from_peft(model_b, "default")

        combined = vec_a + vec_b  # rank 16
        target = _make_peft_model(rank=8, lora_alpha=16, seed=0)
        combined.write_to_existing_peft_model(target, "default")  # default resize=True

        for _, module in target.named_modules():
            if hasattr(module, "lora_A") and "default" in module.lora_A:
                assert module.lora_A["default"].weight.shape[0] == 16
                assert module.r["default"] == 16

    def test_peft_config_r_updated_after_resize(self):
        """peft_config.r should match the vector's rank after write."""
        model_a = _make_peft_model(rank=8, lora_alpha=16, seed=42)
        model_b = _make_peft_model(rank=8, lora_alpha=16, seed=99)
        combined = LoRaVector.from_peft(model_a, "default") + LoRaVector.from_peft(model_b, "default")

        target = _make_peft_model(rank=8, lora_alpha=16, seed=0)
        combined.write_to_existing_peft_model(target, "default")

        assert target.peft_config["default"].r == 16

    def test_peft_config_r_updated_after_shrink(self):
        """peft_config.r should shrink when writing a lower-rank vector."""
        model = _make_peft_model(rank=8, lora_alpha=16, seed=42)
        vec = LoRaVector.from_peft(model, "default").rank_reduce(4)

        target = _make_peft_model(rank=8, lora_alpha=16, seed=0)
        vec.write_to_existing_peft_model(target, "default")

        assert target.peft_config["default"].r == 4

    def test_peft_config_lora_alpha_synced(self):
        """peft_config.lora_alpha should equal r after write (scaling = 1.0)."""
        model = _make_peft_model(rank=8, lora_alpha=16, seed=42)
        vec = LoRaVector.from_peft(model, "default")

        target = _make_peft_model(rank=8, lora_alpha=16, seed=0)
        vec.write_to_existing_peft_model(target, "default")

        config = target.peft_config["default"]
        assert config.lora_alpha == config.r, (
            f"lora_alpha ({config.lora_alpha}) should equal r ({config.r})"
        )


# ---------------------------------------------------------------------------
# to_peft_model
# ---------------------------------------------------------------------------


class TestToPeftModel:
    """Tests for LoRaVector.to_peft_model."""

    def test_to_peft_model_basic(self):
        """to_peft_model should create a PeftModel with correct weights."""
        model = _make_peft_model(rank=8, lora_alpha=16, seed=42)
        vec = LoRaVector.from_peft(model, "default")

        base = TinyTransformer(num_layers=4, hidden_size=16)
        peft_model = vec.to_peft_model(base)

        assert isinstance(peft_model, PeftModel)

        # Round-trip: extract vector back and check similarity
        recovered = LoRaVector.from_peft(peft_model, "default")
        sim = vec.cosine_similarity(recovered)
        assert sim == pytest.approx(1.0, abs=1e-4), f"Round-trip lost info: sim={sim}"

        # Config should reflect vector's rank
        config = peft_model.peft_config["default"]
        assert config.r == vec.max_rank
        assert config.lora_alpha == vec.max_rank

    def test_to_peft_model_infers_target_modules(self):
        """to_peft_model should infer target_modules from module name suffixes."""
        model = _make_peft_model(rank=4, lora_alpha=8, seed=42)
        vec = LoRaVector.from_peft(model, "default")

        base = TinyTransformer(num_layers=4, hidden_size=16)
        peft_model = vec.to_peft_model(base)

        config = peft_model.peft_config["default"]
        expected = {"q_proj", "v_proj", "up_proj", "down_proj"}
        assert set(config.target_modules) == expected

    def test_to_peft_model_forward_pass(self):
        """PeftModel from to_peft_model should produce valid output."""
        model = _make_peft_model(rank=4, lora_alpha=8, seed=42)
        vec = LoRaVector.from_peft(model, "default")

        base = TinyTransformer(num_layers=4, hidden_size=16)
        peft_model = vec.to_peft_model(base)
        peft_model.eval()

        x = torch.randn(1, 16)
        with torch.no_grad():
            output = peft_model(x)

        assert not torch.isnan(output).any(), "NaN in output"
        assert not torch.isinf(output).any(), "Inf in output"


# ---------------------------------------------------------------------------
# to_file
# ---------------------------------------------------------------------------


class TestToFile:
    """Tests for LoRaVector.to_file."""

    def test_to_file_round_trip(self, tmp_path):
        """to_file → from_file should recover the same vector."""
        model = _make_peft_model(rank=8, lora_alpha=16, seed=42)
        vec = LoRaVector.from_peft(model, "default")

        adapter_dir = tmp_path / "adapter"
        vec.to_file(adapter_dir)
        recovered = LoRaVector.from_file(adapter_dir)

        sim = vec.cosine_similarity(recovered)
        assert sim == pytest.approx(1.0, abs=1e-4), f"Round-trip lost info: sim={sim}"

    def test_to_file_creates_correct_files(self, tmp_path):
        """to_file should create adapter_config.json and adapter_model.safetensors."""
        import json

        model = _make_peft_model(rank=4, lora_alpha=8, seed=42)
        vec = LoRaVector.from_peft(model, "default")

        adapter_dir = tmp_path / "adapter"
        vec.to_file(adapter_dir)

        assert (adapter_dir / "adapter_config.json").exists()
        assert (adapter_dir / "adapter_model.safetensors").exists()

        with open(adapter_dir / "adapter_config.json") as f:
            config = json.load(f)

        assert config["peft_type"] == "LORA"
        assert config["task_type"] == "CAUSAL_LM"
        assert config["r"] == vec.max_rank
        assert config["lora_alpha"] == vec.max_rank
        assert set(config["target_modules"]) == {"q_proj", "v_proj", "up_proj", "down_proj"}

    def test_to_file_weight_keys(self, tmp_path):
        """Safetensors keys should match {module_name}.lora_{A,B}.weight pattern."""
        from safetensors.torch import safe_open

        model = _make_peft_model(rank=4, lora_alpha=8, seed=42)
        vec = LoRaVector.from_peft(model, "default")

        adapter_dir = tmp_path / "adapter"
        vec.to_file(adapter_dir)

        with safe_open(str(adapter_dir / "adapter_model.safetensors"), framework="pt") as f:
            keys = set(f.keys())

        # Each module should have .lora_A.weight and .lora_B.weight
        for name in vec.module_names:
            assert f"{name}.lora_A.weight" in keys, f"Missing A key for {name}"
            assert f"{name}.lora_B.weight" in keys, f"Missing B key for {name}"

        # Total keys = 2 * number of modules
        assert len(keys) == 2 * len(vec.module_names)
