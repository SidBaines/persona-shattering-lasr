"""LoRA vectors in factored (B, A) form for memory-efficient vector space operations.

Instead of materializing full ∆W = B @ A matrices (which can be billions of
elements for large models), this module keeps LoRA weights in their low-rank
factored form and performs all operations — dot products, addition, scaling,
PCA — directly on the small factor matrices.

Memory: O(n_modules × r × (d_in + d_out)) instead of O(n_modules × d_in × d_out).
For rank 16, that's roughly a 250x reduction.

Loss only occurs when writing back to a model with a fixed rank r via truncated
SVD. All arithmetic in the vector space is exact.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from safetensors.torch import safe_open
from torch import Tensor, nn

from src.utils.linalg import reduce_lora_rank_efficient
from src.utils.model_layer_info import LayerIdxExtractor, extract_layer_idx
from src.utils.peft_manipulations import _iter_all_lora_modules, _matches_target_modules


class LoRAVector:
    """A point in LoRA weight space, stored in factored (B, A) form.

    Each instance holds a dict mapping module names to (B, A) factor pairs,
    where B is (out_features, r) and A is (r, in_features). The rank r can
    vary per module and grows after addition/subtraction.

    Args:
        factors: Mapping of module_name -> (B, A) tensor pairs.
    """

    def __init__(self, factors: dict[str, tuple[Tensor, Tensor]]) -> None:
        if not factors:
            raise ValueError("factors dict must not be empty")
        self._factors = dict(factors)

    @classmethod
    def from_peft(
        cls,
        model: PeftModel,
        adapter_name: str,
        *,
        include_scaling: bool = True,
        target_modules: list[str] | dict[int, list[str]] | None = None,
        layers: list[int] | None = None,
        layer_idx_extractor: LayerIdxExtractor | None = None,
    ) -> LoRAVector:
        """Extract a LoRAVector from a PeftModel's adapter.

        Args:
            model: A PEFT model with LoRA adapters.
            adapter_name: Which adapter to extract.
            include_scaling: If True, absorb the PEFT scaling factor
                (lora_alpha / r) into B.
            target_modules: Filter which modules to include (see
                BaseLoRaModifier for semantics).
            layers: If provided, only include modules in these layer indices.
            layer_idx_extractor: Custom layer index extraction function.

        Returns:
            A LoRAVector with the adapter's factors.
        """
        if adapter_name not in model.peft_config:
            available = list(model.peft_config.keys())
            raise ValueError(
                f"Adapter '{adapter_name}' not found. Available: {available}"
            )

        extractor = layer_idx_extractor or extract_layer_idx
        layers_set = set(layers) if layers is not None else None

        factors: dict[str, tuple[Tensor, Tensor]] = {}
        for name, module in _iter_all_lora_modules(model, adapter_name):
            # Apply filtering
            if isinstance(target_modules, dict):
                layer_idx = extractor(name)
                if layer_idx is None or layer_idx not in target_modules:
                    continue
                if not _matches_target_modules(name, target_modules[layer_idx]):
                    continue
            else:
                if target_modules is not None:
                    if not _matches_target_modules(name, target_modules):
                        continue
                if layers_set is not None:
                    layer_idx = extractor(name)
                    if layer_idx is None or layer_idx not in layers_set:
                        continue

            A = module.lora_A[adapter_name].weight.data.clone()
            B = module.lora_B[adapter_name].weight.data.clone()

            if include_scaling:
                scaling = module.scaling[adapter_name]
                B = scaling * B

            factors[name] = (B, A)

        if not factors:
            raise ValueError(
                "No LoRA modules matched the given filters. "
                "Check target_modules and layers arguments."
            )

        return cls(factors)

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        target_modules: list[str] | None = None,
        layers: list[int] | None = None,
        layer_idx_extractor: LayerIdxExtractor | None = None,
    ) -> LoRAVector:
        """Load a LoRAVector from a saved adapter directory.

        Reads adapter_config.json for config and adapter_model.safetensors
        for weights. Absorbs the scaling factor (lora_alpha / r) into B.

        Args:
            path: Path to adapter directory containing adapter_config.json
                and adapter_model.safetensors.
            target_modules: Filter which modules to include by name suffix.
            layers: If provided, only include modules in these layer indices.
            layer_idx_extractor: Custom layer index extraction function.

        Returns:
            A LoRAVector with the adapter's factors.
        """
        path = Path(path)
        config_path = path / "adapter_config.json"
        weights_path = path / "adapter_model.safetensors"

        if not config_path.exists():
            raise FileNotFoundError(f"No adapter_config.json found at {config_path}")
        if not weights_path.exists():
            raise FileNotFoundError(
                f"No adapter_model.safetensors found at {weights_path}"
            )

        with open(config_path) as f:
            config = json.load(f)

        r = config["r"]
        lora_alpha = config.get("lora_alpha", r)
        scaling = lora_alpha / r

        extractor = layer_idx_extractor or extract_layer_idx
        layers_set = set(layers) if layers is not None else None

        factors: dict[str, tuple[Tensor, Tensor]] = {}
        with safe_open(str(weights_path), framework="pt") as f:
            # Discover all module names from the weight keys
            all_keys = list(f.keys())
            module_names = sorted(
                {
                    k.removesuffix(".lora_A.weight").removesuffix(".lora_B.weight")
                    for k in all_keys
                }
            )

            for module_name in module_names:
                a_key = f"{module_name}.lora_A.weight"
                b_key = f"{module_name}.lora_B.weight"
                if a_key not in all_keys or b_key not in all_keys:
                    continue

                # Apply filtering
                if target_modules is not None:
                    if not _matches_target_modules(module_name, target_modules):
                        continue
                if layers_set is not None:
                    layer_idx = extractor(module_name)
                    if layer_idx is None or layer_idx not in layers_set:
                        continue

                A = f.get_tensor(a_key)
                B = f.get_tensor(b_key)
                B = scaling * B

                factors[module_name] = (B, A)

        if not factors:
            raise ValueError(
                "No LoRA modules matched the given filters. "
                "Check target_modules and layers arguments."
            )

        return cls(factors)

    @classmethod
    def from_hub(
        cls,
        repo_id: str,
        *,
        subfolder: str | None = None,
        target_modules: list[str] | None = None,
        layers: list[int] | None = None,
        layer_idx_extractor: LayerIdxExtractor | None = None,
        revision: str | None = None,
    ) -> LoRAVector:
        """Load a LoRAVector from a HuggingFace Hub repository.

        Downloads the adapter files and delegates to :meth:`from_file`.

        Args:
            repo_id: HuggingFace repo ID (e.g. "user/model-lora-adapters").
            subfolder: Subfolder within the repo containing the adapter
                (e.g. "sarcasm" for a multi-adapter repo).
            target_modules: Filter which modules to include by name suffix.
            layers: If provided, only include modules in these layer indices.
            layer_idx_extractor: Custom layer index extraction function.
            revision: Git revision (branch, tag, or commit hash) to download.

        Returns:
            A LoRAVector with the adapter's factors.
        """
        allow_patterns = [f"{subfolder}/*"] if subfolder else None
        local_path = Path(
            snapshot_download(
                repo_id,
                allow_patterns=allow_patterns,
                revision=revision,
            )
        )
        adapter_path = local_path / subfolder if subfolder else local_path
        return cls.from_file(
            adapter_path,
            target_modules=target_modules,
            layers=layers,
            layer_idx_extractor=layer_idx_extractor,
        )

    def zero_like(self) -> LoRAVector:
        """Return a zero vector with the same module structure.

        Each module gets zero-valued (B, A) factors with rank 1 and the
        same (out_features, in_features) dimensions as this vector.
        """
        factors = {}
        for name, (B, A) in self._factors.items():
            factors[name] = (
                torch.zeros(B.shape[0], 1, dtype=B.dtype, device=B.device),
                torch.zeros(1, A.shape[1], dtype=A.dtype, device=A.device),
            )
        return LoRAVector(factors)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def factors(self) -> dict[str, tuple[Tensor, Tensor]]:
        """The underlying (B, A) factor pairs, keyed by module name."""
        return self._factors

    @property
    def module_names(self) -> list[str]:
        """Sorted list of module names."""
        return sorted(self._factors.keys())

    @property
    def max_rank(self) -> int:
        """Maximum rank across all modules."""
        return max(B.shape[1] for B, _ in self._factors.values())

    @property
    def total_params(self) -> int:
        """Total number of parameters stored (sum of r*(in+out) per module)."""
        total = 0
        for B, A in self._factors.values():
            total += B.numel() + A.numel()
        return total

    # ------------------------------------------------------------------
    # Arithmetic (exact, no SVD)
    # ------------------------------------------------------------------

    def _validate_compatible(self, other: LoRAVector) -> None:
        """Check that two vectors have the same modules and compatible shapes."""
        if self._factors.keys() != other._factors.keys():
            missing = self._factors.keys() ^ other._factors.keys()
            raise ValueError(
                f"LoRAVectors have different module names. "
                f"Symmetric difference: {sorted(missing)}"
            )
        for name in self._factors:
            B1, A1 = self._factors[name]
            B2, A2 = other._factors[name]
            # Check ∆W shape compatibility (out_features and in_features)
            if B1.shape[0] != B2.shape[0] or A1.shape[1] != A2.shape[1]:
                raise ValueError(
                    f"Incompatible ∆W shapes for module '{name}': "
                    f"({B1.shape[0]}, {A1.shape[1]}) vs ({B2.shape[0]}, {A2.shape[1]})"
                )

    def __add__(self, other: LoRAVector) -> LoRAVector:
        self._validate_compatible(other)
        factors = {}
        for name in self._factors:
            B1, A1 = self._factors[name]
            B2, A2 = other._factors[name]
            factors[name] = (
                torch.cat([B1, B2], dim=1),
                torch.cat([A1, A2], dim=0),
            )
        return LoRAVector(factors)

    def __sub__(self, other: LoRAVector) -> LoRAVector:
        self._validate_compatible(other)
        factors = {}
        for name in self._factors:
            B1, A1 = self._factors[name]
            B2, A2 = other._factors[name]
            factors[name] = (
                torch.cat([B1, B2], dim=1),
                torch.cat([A1, -A2], dim=0),
            )
        return LoRAVector(factors)

    def __neg__(self) -> LoRAVector:
        factors = {}
        for name, (B, A) in self._factors.items():
            factors[name] = (-B, A)
        return LoRAVector(factors)

    def __mul__(self, scalar: float) -> LoRAVector:
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        factors = {}
        for name, (B, A) in self._factors.items():
            factors[name] = (scalar * B, A)
        return LoRAVector(factors)

    def __rmul__(self, scalar: float) -> LoRAVector:
        return self.__mul__(scalar)

    # ------------------------------------------------------------------
    # Metrics (never materialize full ∆W)
    # ------------------------------------------------------------------

    def dot(self, other: LoRAVector) -> float:
        """Compute the dot product ⟨self, other⟩ in ∆W space.

        Uses the identity: ⟨B₁A₁, B₂A₂⟩ = Σᵢ tr(A₂ᵢ A₁ᵢᵀ · B₁ᵢᵀ B₂ᵢ).
        Cost: O(r₁ r₂ (in + out)) per module instead of O(r in out).
        """
        self._validate_compatible(other)
        total = 0.0
        for name in self._factors:
            B1, A1 = self._factors[name]
            B2, A2 = other._factors[name]
            # B1^T B2 is (r1, r2), A2 A1^T is (r2, r1)
            BtB = B1.T @ B2  # (r1, r2)
            AAt = A2 @ A1.T  # (r2, r1)
            # tr(AAt @ BtB) = tr((r2, r1) @ (r1, r2)) = sum of elementwise product
            total += (AAt * BtB.T).sum().item()
        return total

    def norm(self) -> float:
        """Frobenius norm of the ∆W vector."""
        return math.sqrt(self.dot(self))

    def cosine_similarity(self, other: LoRAVector) -> float:
        """Cosine similarity between two LoRAVectors in ∆W space."""
        d = self.dot(other)
        n1 = self.norm()
        n2 = other.norm()
        if n1 == 0.0 or n2 == 0.0:
            return 0.0
        return d / (n1 * n2)

    # ------------------------------------------------------------------
    # Rank management
    # ------------------------------------------------------------------

    def rank_reduce(self, new_rank: int) -> LoRAVector:
        """Return a new LoRAVector with each module's rank reduced via SVD.

        Uses the efficient QR-based rank reduction from src.utils.linalg
        (no full ∆W materialization).

        Args:
            new_rank: Target rank for all modules.

        Returns:
            New LoRAVector with reduced rank.
        """
        if new_rank < 1:
            raise ValueError(f"new_rank must be >= 1, got {new_rank=}")

        factors = {}
        for name, (B, A) in self._factors.items():
            current_rank = A.shape[0]
            if new_rank > current_rank:
                raise ValueError(f"new_rank must be <= {current_rank=} got {new_rank=}")
            if current_rank <= new_rank:
                factors[name] = (B.clone(), A.clone())
            else:
                new_A, new_B = reduce_lora_rank_efficient(A, B, new_rank=new_rank)
                factors[name] = (new_B, new_A)
        return LoRAVector(factors)

    # ------------------------------------------------------------------
    # Model I/O
    # ------------------------------------------------------------------

    def write_to_model(
        self,
        model: PeftModel,
        adapter_name: str,
        *,
        rank: int | None = None,
    ) -> None:
        """Write this vector's factors into a PeftModel's adapter.

        If a module's internal rank exceeds the model's rank for that module,
        truncated SVD is used to fit. Sets scaling to 1.0.

        Args:
            model: PEFT model to write into (modified in-place).
            adapter_name: Which adapter to overwrite.
            rank: If specified, reduce all modules to this rank before writing.
        """
        if adapter_name not in model.peft_config:
            available = list(model.peft_config.keys())
            raise ValueError(
                f"Adapter '{adapter_name}' not found. Available: {available}"
            )

        vec = self.rank_reduce(rank) if rank is not None else self

        modules = dict(model.named_modules())
        for name, (B, A) in vec._factors.items():
            if name not in modules:
                raise ValueError(f"Module '{name}' not found in model")

            module = modules[name]
            model_rank = module.r[adapter_name]

            if A.shape[0] > model_rank:
                A, B = reduce_lora_rank_efficient(A, B, new_rank=model_rank)

            module.lora_A[adapter_name].weight = nn.Parameter(
                A.to(
                    device=module.lora_A[adapter_name].weight.device,
                    dtype=module.lora_A[adapter_name].weight.dtype,
                )
            )
            module.lora_B[adapter_name].weight = nn.Parameter(
                B.to(
                    device=module.lora_B[adapter_name].weight.device,
                    dtype=module.lora_B[adapter_name].weight.dtype,
                )
            )
            module.scaling[adapter_name] = 1.0

    def to(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> LoRAVector:
        """Return a new LoRAVector with tensors moved/cast.

        Args:
            device: Target device.
            dtype: Target dtype.

        Returns:
            New LoRAVector with moved/cast tensors.
        """
        factors = {}
        for name, (B, A) in self._factors.items():
            factors[name] = (
                B.to(device=device, dtype=dtype),
                A.to(device=device, dtype=dtype),
            )
        return LoRAVector(factors)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        n = len(self._factors)
        ranks = [A.shape[0] for _, (_, A) in sorted(self._factors.items())]
        min_r, max_r = min(ranks), max(ranks)
        rank_str = str(min_r) if min_r == max_r else f"{min_r}-{max_r}"
        return f"LoRAVector(modules={n}, rank={rank_str}, params={self.total_params:,})"


# ---------------------------------------------------------------------------
# Standalone helpers
# ---------------------------------------------------------------------------


def gram_matrix(vectors: list[LoRAVector]) -> Tensor:
    """Compute the n x n Gram matrix of pairwise dot products.

    Args:
        vectors: List of LoRAVector instances (must all be compatible).

    Returns:
        Symmetric tensor of shape (n, n) where G[i,j] = vectors[i].dot(vectors[j]).
    """
    n = len(vectors)
    G = torch.zeros(n, n)
    for i in range(n):
        G[i, i] = vectors[i].dot(vectors[i])
        for j in range(i):
            val = vectors[i].dot(vectors[j])
            G[i, j] = val
            G[j, i] = val
    return G


def cosine_similarity_matrix(
    vectors: list[LoRAVector],
    *,
    _gram: Tensor | None = None,
) -> Tensor:
    """Compute the n x n pairwise cosine similarity matrix.

    Args:
        vectors: List of LoRAVector instances (must all be compatible).
        _gram: Precomputed Gram matrix (internal use by LoRAVectorCollection).

    Returns:
        Symmetric tensor of shape (n, n) where C[i,j] is the cosine
        similarity between vectors[i] and vectors[j].
    """
    G = _gram if _gram is not None else gram_matrix(vectors)
    return _cosine_from_gram(G)


def _cosine_from_gram(G: Tensor) -> Tensor:
    """Convert a Gram matrix to a cosine similarity matrix."""
    norms = torch.sqrt(torch.diag(G))
    norms = norms.clamp(min=1e-10)
    return G / norms.unsqueeze(1) / norms.unsqueeze(0)


# ---------------------------------------------------------------------------
# LoRAVectorCollection
# ---------------------------------------------------------------------------


@dataclass
class PCAResult:
    """Result from LoRAVectorCollection.pca, bundling metadata."""

    space: LoRAVectorSpace
    coords: Tensor  # (n_vectors, n_dims) — coordinates of input vectors
    eigenvalues: Tensor  # all eigenvalues (for scree plots)
    explained_variance: Tensor  # fraction explained per retained component


class LoRAVectorCollection:
    """A named set of LoRAVectors with cached bulk analysis operations.

    Lazily computes the Gram matrix and reuses it for cosine similarity
    and PCA, avoiding redundant pairwise dot products.

    Args:
        vectors: Either a dict mapping names to LoRAVectors, or a list
            (in which case names default to "0", "1", "2", ...).
    """

    def __init__(self, vectors: dict[str, LoRAVector] | list[LoRAVector]) -> None:
        if isinstance(vectors, dict):
            if not vectors:
                raise ValueError("vectors dict must not be empty")
            self._names = list(vectors.keys())
            self._vectors = list(vectors.values())
        else:
            if not vectors:
                raise ValueError("vectors list must not be empty")
            self._names = [str(i) for i in range(len(vectors))]
            self._vectors = list(vectors)

        self._gram_cache: Tensor | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def names(self) -> list[str]:
        """Names in insertion/index order."""
        return list(self._names)

    @property
    def vectors(self) -> list[LoRAVector]:
        """Vectors in matching order."""
        return list(self._vectors)

    def __len__(self) -> int:
        return len(self._vectors)

    def __getitem__(self, key: str | int) -> LoRAVector:
        if isinstance(key, int):
            return self._vectors[key]
        try:
            idx = self._names.index(key)
        except ValueError:
            raise KeyError(f"No vector named '{key}'") from None
        return self._vectors[idx]

    # ------------------------------------------------------------------
    # Analysis (Gram matrix computed once, cached)
    # ------------------------------------------------------------------

    def gram_matrix(self) -> Tensor:
        """Compute (or return cached) n x n Gram matrix of pairwise dot products."""
        if self._gram_cache is None:
            self._gram_cache = gram_matrix(self._vectors)
        return self._gram_cache

    def cosine_similarity_matrix(self) -> Tensor:
        """Compute n x n pairwise cosine similarity matrix (reuses cached Gram)."""
        return _cosine_from_gram(self.gram_matrix())

    def pca(self, n_dims: int | None = None) -> PCAResult:
        """Perform kernel PCA on the collection (reuses cached Gram matrix).

        Args:
            n_dims: Number of principal components to retain.
                Defaults to len(vectors).

        Returns:
            PCAResult containing the space, coordinates, eigenvalues,
            and explained variance.
        """
        n = len(self._vectors)
        if n_dims is None:
            n_dims = n
        elif n_dims < 1:
            raise ValueError(f"n_dims must be >= 1, got {n_dims}")
        elif n_dims > n:
            raise ValueError(f"n_dims ({n_dims}) cannot exceed number of vectors ({n})")

        G = self.gram_matrix()

        # Center the Gram matrix (kernel PCA centering)
        ones_n = torch.ones(n, n) / n
        G_centered = G - ones_n @ G - G @ ones_n + ones_n @ G @ ones_n

        # Eigendecompose (eigh returns ascending order)
        eigenvalues, eigenvectors = torch.linalg.eigh(G_centered)
        eigenvalues = eigenvalues.flip(0)
        eigenvectors = eigenvectors.flip(1)

        # Compute coordinates
        eigenvalues_safe = eigenvalues.clamp(min=0)
        total_var = eigenvalues_safe.sum()
        coords = eigenvectors[:, :n_dims] * eigenvalues_safe[:n_dims].sqrt()

        # Explained variance
        if total_var > 0:
            explained = eigenvalues_safe[:n_dims] / total_var
        else:
            explained = torch.zeros(n_dims)

        # Construct orthonormal basis vectors.
        # Each PC direction is a linear combination of the input vectors:
        #   basis_k = Σᵢ eigenvectors[i, k] * vectors[i]
        # Then normalize to unit length.
        basis: list[LoRAVector] = []
        for k in range(n_dims):
            if eigenvalues_safe[k] <= 0:
                break

            coeffs = eigenvectors[:, k]
            basis_vec = coeffs[0].item() * self._vectors[0]
            for i in range(1, n):
                basis_vec = basis_vec + (coeffs[i].item() * self._vectors[i])

            vec_norm = basis_vec.norm()
            if vec_norm > 0:
                basis_vec = (1.0 / vec_norm) * basis_vec

            basis.append(basis_vec)

        space = LoRAVectorSpace(basis)
        return PCAResult(
            space=space,
            coords=coords,
            eigenvalues=eigenvalues,
            explained_variance=explained,
        )

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"LoRAVectorCollection(n={len(self._vectors)}, names={self._names})"


# ---------------------------------------------------------------------------
# LoRAVectorSpace
# ---------------------------------------------------------------------------


class LoRAVectorSpace:
    """An orthonormal basis in LoRA weight space.

    Provides projection, reconstruction, and shifting operations.
    Typically created via :meth:`LoRAVectorCollection.pca`.

    Args:
        basis: Either a dict mapping names to basis vectors, or a list
            (in which case names default to "0", "1", "2", ...).
    """

    def __init__(self, basis: dict[str, LoRAVector] | list[LoRAVector]) -> None:
        if isinstance(basis, dict):
            if not basis:
                raise ValueError("basis must not be empty")
            self._names = list(basis.keys())
            self._basis = list(basis.values())
        else:
            if not basis:
                raise ValueError("basis must not be empty")
            self._names = [str(i) for i in range(len(basis))]
            self._basis = list(basis)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def names(self) -> list[str]:
        """Names of the basis vectors."""
        return list(self._names)

    @property
    def basis(self) -> list[LoRAVector]:
        """The orthonormal basis vectors."""
        return self._basis

    @property
    def n_dims(self) -> int:
        """Number of basis dimensions."""
        return len(self._basis)

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------

    def project(self, vec: LoRAVector) -> Tensor:
        """Project a LoRAVector onto this basis.

        Args:
            vec: The vector to project.

        Returns:
            Coordinates tensor of shape (n_dims,).
        """
        coords = torch.tensor([vec.dot(b) for b in self._basis])
        return coords

    def project_all(self, vectors: list[LoRAVector]) -> Tensor:
        """Project multiple LoRAVectors onto this basis.

        Args:
            vectors: List of vectors to project.

        Returns:
            Coordinates tensor of shape (n_vectors, n_dims).
        """
        return torch.stack([self.project(v) for v in vectors])

    def reconstruct(self, coords: Tensor) -> LoRAVector:
        """Reconstruct a LoRAVector from coordinates in this basis.

        Args:
            coords: Coordinates tensor of shape (n_dims,).

        Returns:
            The reconstructed LoRAVector.
        """
        if coords.shape != (self.n_dims,):
            raise ValueError(
                f"Expected coords of shape ({self.n_dims},), got {coords.shape}"
            )

        result = coords[0].item() * self._basis[0]
        for i in range(1, self.n_dims):
            result = result + (coords[i].item() * self._basis[i])
        return result

    def shift(self, vec: LoRAVector, deltas: dict[int, float]) -> LoRAVector:
        """Shift a vector along specific basis dimensions.

        Projects the vector, adds deltas to specified coordinates, and
        reconstructs.

        Args:
            vec: The vector to shift.
            deltas: Mapping of dimension index to shift amount.
                E.g. {0: +0.5, 4: -0.3} shifts along PC1 and PC5.

        Returns:
            The shifted LoRAVector.
        """
        for dim in deltas:
            if dim < 0 or dim >= self.n_dims:
                raise ValueError(f"Dimension {dim} out of range [0, {self.n_dims})")

        coords = self.project(vec)
        for dim, delta in deltas.items():
            coords[dim] += delta
        return self.reconstruct(coords)
        return self.reconstruct(coords)
        return self.reconstruct(coords)
