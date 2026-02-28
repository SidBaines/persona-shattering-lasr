import pytest
import torch
from peft import LoraConfig, get_peft_model
from torch import nn

from src.utils.lora_vector_utils import (
    LoRAVector,
    LoRAVectorCollection,
    LoRAVectorSpace,
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


def _materialize_delta_w(vec: LoRAVector) -> torch.Tensor:
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
    return LoRAVector.from_peft(peft_model, "default")


@pytest.fixture
def vec_b():
    model = _make_peft_model(rank=8, seed=99)
    return LoRAVector.from_peft(model, "default")


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_from_peft_basic(self, peft_model):
        vec = LoRAVector.from_peft(peft_model, "default")
        # 4 layers × 4 modules = 16 modules
        assert len(vec.module_names) == 16
        assert vec.max_rank == 8

    def test_from_peft_with_scaling(self, peft_model):
        vec_scaled = LoRAVector.from_peft(peft_model, "default", include_scaling=True)
        vec_plain = LoRAVector.from_peft(peft_model, "default", include_scaling=False)
        # lora_alpha=16, r=8 → scaling = 2.0
        for name in vec_scaled.module_names:
            B_scaled, _ = vec_scaled.factors[name]
            B_plain, _ = vec_plain.factors[name]
            torch.testing.assert_close(B_scaled, B_plain * 2.0)

    def test_from_peft_target_modules_filter(self, peft_model):
        vec = LoRAVector.from_peft(peft_model, "default", target_modules=["q_proj"])
        assert len(vec.module_names) == 4  # 4 layers × 1 module
        assert all("q_proj" in name for name in vec.module_names)

    def test_from_peft_layers_filter(self, peft_model):
        vec = LoRAVector.from_peft(peft_model, "default", layers=[0, 1])
        assert len(vec.module_names) == 8  # 2 layers × 4 modules

    def test_from_peft_invalid_adapter(self, peft_model):
        with pytest.raises(ValueError, match="not found"):
            LoRAVector.from_peft(peft_model, "nonexistent")

    def test_from_peft_no_match(self, peft_model):
        with pytest.raises(ValueError, match="No LoRA modules matched"):
            LoRAVector.from_peft(peft_model, "default", target_modules=["nonexistent"])

    def test_empty_factors_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            LoRAVector({})

    def test_module_names_sorted(self, vec_a):
        names = vec_a.module_names
        assert names == sorted(names)

    def test_total_params(self, vec_a):
        expected = 16 * (8 * 16 + 8 * 16)  # 16 modules, r=8, in=out=16
        assert vec_a.total_params == expected


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
        vec_different = LoRAVector(factors)
        with pytest.raises(ValueError, match="different module names"):
            vec_a + vec_different


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
        actual = vec_a.norm()
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

    def test_rank_reduce_preserves_when_already_given_rank(self, vec_a):
        reduced = vec_a.rank_reduce(8)  # higher than current rank 8
        assert reduced.max_rank == 8  # unchanged

    def test_rank_reduce_invalid(self, vec_a):
        with pytest.raises(ValueError, match="new_rank must be >= 1"):
            vec_a.rank_reduce(0)


# ---------------------------------------------------------------------------
# Model I/O
# ---------------------------------------------------------------------------


class TestModelIO:
    def test_round_trip(self, peft_model):
        """from_peft → write_to_model → from_peft should approximately recover."""
        vec = LoRAVector.from_peft(peft_model, "default")

        target = _make_peft_model(rank=8)
        vec.write_to_model(target, "default")

        vec_recovered = LoRAVector.from_peft(target, "default")
        sim = vec.cosine_similarity(vec_recovered)
        assert sim == pytest.approx(1.0, abs=1e-3)

    def test_write_to_lower_rank(self, peft_model):
        """Writing to a lower-rank model should truncate via SVD."""
        vec = LoRAVector.from_peft(peft_model, "default")

        target = _make_peft_model(rank=2)
        vec.write_to_model(target, "default")

        for _, module in target.named_modules():
            if hasattr(module, "r") and "default" in getattr(module, "r", {}):
                assert module.r["default"] == 2

    def test_write_sets_scaling_to_one(self, peft_model):
        vec = LoRAVector.from_peft(peft_model, "default")

        target = _make_peft_model(rank=8)
        vec.write_to_model(target, "default")

        for _, module in target.named_modules():
            if hasattr(module, "scaling") and "default" in getattr(
                module, "scaling", {}
            ):
                assert module.scaling["default"] == pytest.approx(1.0)

    def test_forward_pass_after_write(self, peft_model):
        """Model should produce similar outputs after extract → write."""
        vec = LoRAVector.from_peft(peft_model, "default")

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
            vecs.append(LoRAVector.from_peft(model, "default"))
        return vecs

    def test_init_from_dict(self, vec_a, vec_b):
        coll = LoRAVectorCollection({"a": vec_a, "b": vec_b})
        assert coll.names == ["a", "b"]
        assert len(coll) == 2

    def test_init_from_list(self, vec_a, vec_b):
        coll = LoRAVectorCollection([vec_a, vec_b])
        assert coll.names == ["0", "1"]
        assert len(coll) == 2

    def test_getitem_by_name(self, vec_a, vec_b):
        coll = LoRAVectorCollection({"a": vec_a, "b": vec_b})
        assert coll["a"] is vec_a
        assert coll["b"] is vec_b

    def test_getitem_by_index(self, vec_a, vec_b):
        coll = LoRAVectorCollection({"a": vec_a, "b": vec_b})
        assert coll[0] is vec_a
        assert coll[1] is vec_b

    def test_getitem_invalid_name_raises(self, vec_a):
        coll = LoRAVectorCollection([vec_a])
        with pytest.raises(KeyError, match="No vector named"):
            coll["nonexistent"]

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            LoRAVectorCollection([])
        with pytest.raises(ValueError, match="must not be empty"):
            LoRAVectorCollection({})

    def test_gram_matrix_cached(self, vec_a, vec_b):
        coll = LoRAVectorCollection([vec_a, vec_b])
        G1 = coll.gram_matrix()
        G2 = coll.gram_matrix()
        assert G1 is G2  # same object — cached

    def test_gram_matrix_matches_standalone(self, vec_a, vec_b):
        coll = LoRAVectorCollection([vec_a, vec_b])
        G_coll = coll.gram_matrix()
        G_standalone = gram_matrix([vec_a, vec_b])
        torch.testing.assert_close(G_coll, G_standalone)

    def test_cosine_similarity_matrix(self, vec_a, vec_b):
        coll = LoRAVectorCollection([vec_a, vec_b])
        C = coll.cosine_similarity_matrix()
        assert C.shape == (2, 2)
        # Diagonal should be 1.0
        assert C[0, 0].item() == pytest.approx(1.0, abs=1e-5)
        assert C[1, 1].item() == pytest.approx(1.0, abs=1e-5)

    def test_pca_returns_correct_types(self, vectors):
        coll = LoRAVectorCollection(vectors)
        result = coll.pca(n_dims=3)
        assert isinstance(result.space, LoRAVectorSpace)
        assert result.coords.shape == (5, 3)
        assert len(result.eigenvalues) == 5
        assert len(result.explained_variance) == 3

    def test_pca_too_many_dims_raises(self, vectors):
        coll = LoRAVectorCollection(vectors)
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
            vecs.append(LoRAVector.from_peft(model, "default"))
        return vecs

    def test_init_from_dict(self, vec_a, vec_b):
        space = LoRAVectorSpace({"pc0": vec_a, "pc1": vec_b})
        assert space.names == ["pc0", "pc1"]
        assert space.n_dims == 2

    def test_init_from_list(self, vec_a, vec_b):
        space = LoRAVectorSpace([vec_a, vec_b])
        assert space.names == ["0", "1"]
        assert space.n_dims == 2

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            LoRAVectorSpace([])
        with pytest.raises(ValueError, match="must not be empty"):
            LoRAVectorSpace({})

    def test_pca_basis_orthonormal(self, vectors):
        """Basis vectors should be approximately orthonormal."""
        coll = LoRAVectorCollection(vectors)
        result = coll.pca(n_dims=3)
        space = result.space
        for i in range(space.n_dims):
            assert space.basis[i].norm() == pytest.approx(1.0, abs=1e-3)
            for j in range(i + 1, space.n_dims):
                dot = space.basis[i].dot(space.basis[j])
                assert dot == pytest.approx(0.0, abs=1e-2)

    def test_pca_explained_variance_sums_to_at_most_one(self, vectors):
        coll = LoRAVectorCollection(vectors)
        result = coll.pca(n_dims=3)
        assert result.explained_variance.sum().item() <= 1.0 + 1e-6

    def test_pca_eigenvalues_descending(self, vectors):
        coll = LoRAVectorCollection(vectors)
        result = coll.pca(n_dims=3)
        eigs = result.eigenvalues
        for i in range(len(eigs) - 1):
            assert eigs[i] >= eigs[i + 1] - 1e-6

    def test_project_reconstruct_round_trip(self, vectors):
        """project → reconstruct should approximately recover the vector
        when using enough dimensions."""
        coll = LoRAVectorCollection(vectors)
        result = coll.pca(n_dims=4)
        space = result.space

        coords = space.project(vectors[0])
        reconstructed = space.reconstruct(coords)

        sim = vectors[0].cosine_similarity(reconstructed)
        assert sim > 0.5

    def test_shift_changes_coordinate(self, vectors):
        """Shifting along dim 0 should change that coordinate by the delta."""
        coll = LoRAVectorCollection(vectors)
        result = coll.pca(n_dims=3)
        space = result.space

        original_coords = space.project(vectors[0])
        shifted = space.shift(vectors[0], {0: 1.0})
        shifted_coords = space.project(shifted)

        delta_0 = shifted_coords[0] - original_coords[0]
        assert delta_0.item() == pytest.approx(1.0, abs=0.1)

    def test_shift_invalid_dim_raises(self, vectors):
        coll = LoRAVectorCollection(vectors)
        result = coll.pca(n_dims=2)
        with pytest.raises(ValueError, match="out of range"):
            result.space.shift(vectors[0], {5: 1.0})

    def test_project_all(self, vectors):
        coll = LoRAVectorCollection(vectors)
        result = coll.pca(n_dims=3)
        coords = result.space.project_all(vectors)
        assert coords.shape == (5, 3)
        for i, v in enumerate(vectors):
            single = result.space.project(v)
            torch.testing.assert_close(coords[i], single, atol=1e-4, rtol=1e-4)

    def test_reconstruct_wrong_shape_raises(self, vectors):
        coll = LoRAVectorCollection(vectors)
        result = coll.pca(n_dims=3)
        with pytest.raises(ValueError, match="Expected coords"):
            result.space.reconstruct(torch.zeros(5))


# ---------------------------------------------------------------------------
# PCA matches materialized kernel PCA
# ---------------------------------------------------------------------------


class TestPCAMatchesMaterialized:
    def test_gram_matrix_matches(self):
        """The factored Gram matrix should match the materialized one."""
        vectors = []
        for seed in [42, 99, 7]:
            model = _make_peft_model(rank=4, seed=seed)
            vectors.append(LoRAVector.from_peft(model, "default"))

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
            vectors.append(LoRAVector.from_peft(model, "default"))

        coll = LoRAVectorCollection(vectors)
        result = coll.pca(n_dims=3)

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

        for k in range(3):
            col_result = result.coords[:, k]
            col_manual = coords_manual[:, k]
            sim = torch.dot(col_result, col_manual) / (
                col_result.norm() * col_manual.norm() + 1e-10
            )
            assert abs(sim.item()) > 0.99, (
                f"PC{k} coordinates don't match (cosine sim = {sim.item():.4f})"
            )
