import pytest
import torch
from peft import LoraConfig, get_peft_model
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
        """from_peft → write_to_model → from_peft should approximately recover."""
        vec = LoRaVector.from_peft(peft_model, "default")

        target = _make_peft_model(rank=8)
        vec.write_to_model(target, "default")

        vec_recovered = LoRaVector.from_peft(target, "default")
        sim = vec.cosine_similarity(vec_recovered)
        assert sim == pytest.approx(1.0, abs=1e-3)

    def test_write_to_lower_rank(self, peft_model):
        """Writing to a lower-rank model should truncate via SVD."""
        vec = LoRaVector.from_peft(peft_model, "default")

        target = _make_peft_model(rank=2)
        vec.write_to_model(target, "default")

        for _, module in target.named_modules():
            if hasattr(module, "r") and "default" in getattr(module, "r", {}):
                assert module.r["default"] == 2

    def test_write_sets_scaling_to_one(self, peft_model):
        vec = LoRaVector.from_peft(peft_model, "default")

        target = _make_peft_model(rank=8)
        vec.write_to_model(target, "default")

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

        vec.write_to_model(peft_model, "default")
        out_after = peft_model(x).detach()

        torch.testing.assert_close(out_after, out_before, atol=1e-3, rtol=1e-3)

    def test_invalid_adapter_raises(self, peft_model, vec_a):
        with pytest.raises(ValueError, match="not found"):
            vec_a.write_to_model(peft_model, "nonexistent")


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
