"""Tests that low-rank operations match their naive (materialized) equivalents.

Each test generates random B (d_out, r) and A (r, d_in) factor pairs,
computes a metric both ways — via the efficient low-rank trace trick and
via explicitly forming the full dW = B @ A matrix — and checks they agree.
"""

import numpy as np
import pytest

from experiments.compare_checkpoints import (
    low_rank_cos_sim,
    low_rank_frob_diff,
    low_rank_frob_norm,
    principal_angles_low_rank,
)


# ---------------------------------------------------------------------------
# Naive (materialized) reference implementations
# ---------------------------------------------------------------------------


def naive_frob_norm(B, A, scale):
    """||scale * B @ A||_F by materializing the full matrix."""
    dW = B @ A
    return abs(scale) * np.linalg.norm(dW, "fro")


def naive_frob_diff(B1, A1, B2, A2, scale):
    """||scale * (B1 A1 - B2 A2)||_F by materializing both full matrices."""
    dW1 = B1 @ A1
    dW2 = B2 @ A2
    return abs(scale) * np.linalg.norm(dW1 - dW2, "fro")


def naive_cos_sim(B1, A1, B2, A2):
    """Cosine similarity by materializing both full matrices and flattening."""
    dW1 = (B1 @ A1).ravel()
    dW2 = (B2 @ A2).ravel()
    dot = np.dot(dW1, dW2)
    return dot / (np.linalg.norm(dW1) * np.linalg.norm(dW2) + 1e-12)


def naive_principal_angles(B1, A1, B2, A2):
    """Principal angles between column spaces of B1@A1 and B2@A2.

    Uses full SVD on the materialized matrices to get column-space bases.
    """
    dW1 = B1 @ A1
    dW2 = B2 @ A2
    Q1, _ = np.linalg.qr(dW1)
    Q2, _ = np.linalg.qr(dW2)
    # Take only the first r columns (rank of dW)
    r = B1.shape[1]
    Q1 = Q1[:, :r]
    Q2 = Q2[:, :r]
    M = Q1.T @ Q2
    svals = np.linalg.svd(M, compute_uv=False)
    svals = np.clip(svals, 0.0, 1.0)
    return np.degrees(np.arccos(svals))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture(
    params=[
        (128, 64, 4),   # small, typical LoRA rank
        (256, 128, 8),  # medium
        (64, 64, 2),    # square, rank-2
        (512, 256, 16), # larger
    ],
    ids=["128x64_r4", "256x128_r8", "64x64_r2", "512x256_r16"],
)
def factor_pair(request, rng):
    """Returns (B, A) where B is (d_out, r) and A is (r, d_in)."""
    d_out, d_in, r = request.param
    B = rng.standard_normal((d_out, r))
    A = rng.standard_normal((r, d_in))
    return B, A


@pytest.fixture
def two_factor_pairs(request, rng):
    """Returns ((B1, A1), (B2, A2)) with same outer dimensions but independent random values."""
    d_out, d_in, r = 128, 64, 4
    B1 = rng.standard_normal((d_out, r))
    A1 = rng.standard_normal((r, d_in))
    B2 = rng.standard_normal((d_out, r))
    A2 = rng.standard_normal((r, d_in))
    return (B1, A1), (B2, A2)


@pytest.fixture(
    params=[
        (128, 64, 4),
        (256, 128, 8),
        (512, 256, 16),
    ],
    ids=["128x64_r4", "256x128_r8", "512x256_r16"],
)
def two_factor_pairs_sized(request, rng):
    """Returns ((B1, A1), (B2, A2)) parametrized over sizes."""
    d_out, d_in, r = request.param
    B1 = rng.standard_normal((d_out, r))
    A1 = rng.standard_normal((r, d_in))
    B2 = rng.standard_normal((d_out, r))
    A2 = rng.standard_normal((r, d_in))
    return (B1, A1), (B2, A2)


# ---------------------------------------------------------------------------
# Tests: Frobenius norm
# ---------------------------------------------------------------------------


class TestFrobNorm:
    def test_matches_naive(self, factor_pair):
        B, A = factor_pair
        for scale in [1.0, 2.0, 0.5]:
            expected = naive_frob_norm(B, A, scale)
            actual = low_rank_frob_norm(B, A, scale)
            np.testing.assert_allclose(actual, expected, rtol=1e-6)

    def test_zero_matrix(self, rng):
        B = np.zeros((64, 4))
        A = rng.standard_normal((4, 32))
        assert low_rank_frob_norm(B, A, 1.0) == pytest.approx(0.0)

    def test_scale_linearity(self, factor_pair):
        B, A = factor_pair
        n1 = low_rank_frob_norm(B, A, 1.0)
        n3 = low_rank_frob_norm(B, A, 3.0)
        np.testing.assert_allclose(n3, 3.0 * n1, rtol=1e-6)


# ---------------------------------------------------------------------------
# Tests: Frobenius difference
# ---------------------------------------------------------------------------


class TestFrobDiff:
    def test_matches_naive(self, two_factor_pairs_sized):
        (B1, A1), (B2, A2) = two_factor_pairs_sized
        for scale in [1.0, 2.0]:
            expected = naive_frob_diff(B1, A1, B2, A2, scale)
            actual = low_rank_frob_diff(B1, A1, B2, A2, scale)
            np.testing.assert_allclose(actual, expected, rtol=1e-6)

    def test_identical_matrices_give_zero(self, factor_pair):
        B, A = factor_pair
        diff = low_rank_frob_diff(B, A, B, A, 1.0)
        assert diff == pytest.approx(0.0, abs=1e-8)

    def test_triangle_inequality(self, rng):
        """||dW1 - dW2|| <= ||dW1|| + ||dW2|| (triangle inequality)."""
        d_out, d_in, r = 128, 64, 4
        B1, A1 = rng.standard_normal((d_out, r)), rng.standard_normal((r, d_in))
        B2, A2 = rng.standard_normal((d_out, r)), rng.standard_normal((r, d_in))
        diff = low_rank_frob_diff(B1, A1, B2, A2, 1.0)
        n1 = low_rank_frob_norm(B1, A1, 1.0)
        n2 = low_rank_frob_norm(B2, A2, 1.0)
        assert diff <= n1 + n2 + 1e-8


# ---------------------------------------------------------------------------
# Tests: Cosine similarity
# ---------------------------------------------------------------------------


class TestCosSim:
    def test_matches_naive(self, two_factor_pairs_sized):
        (B1, A1), (B2, A2) = two_factor_pairs_sized
        expected = naive_cos_sim(B1, A1, B2, A2)
        actual = low_rank_cos_sim(B1, A1, B2, A2)
        np.testing.assert_allclose(actual, expected, rtol=1e-6)

    def test_identical_matrices_give_one(self, factor_pair):
        B, A = factor_pair
        cos = low_rank_cos_sim(B, A, B, A)
        np.testing.assert_allclose(cos, 1.0, atol=1e-8)

    def test_negated_gives_minus_one(self, factor_pair):
        B, A = factor_pair
        cos = low_rank_cos_sim(B, A, -B, A)
        np.testing.assert_allclose(cos, -1.0, atol=1e-8)

    def test_bounded(self, two_factor_pairs):
        (B1, A1), (B2, A2) = two_factor_pairs
        cos = low_rank_cos_sim(B1, A1, B2, A2)
        assert -1.0 - 1e-8 <= cos <= 1.0 + 1e-8


# ---------------------------------------------------------------------------
# Tests: Principal angles
# ---------------------------------------------------------------------------


class TestPrincipalAngles:
    def test_matches_naive(self, two_factor_pairs_sized):
        (B1, A1), (B2, A2) = two_factor_pairs_sized
        expected = naive_principal_angles(B1, A1, B2, A2)
        actual = principal_angles_low_rank(B1, A1, B2, A2)
        np.testing.assert_allclose(actual, expected, atol=0.5)  # degrees

    def test_identical_subspaces_give_zero_angles(self, rng):
        d_out, d_in, r = 128, 64, 4
        B = rng.standard_normal((d_out, r))
        A1 = rng.standard_normal((r, d_in))
        A2 = rng.standard_normal((r, d_in))
        # Same B → same column space → angles should be 0
        angles = principal_angles_low_rank(B, A1, B, A2)
        np.testing.assert_allclose(angles, 0.0, atol=1e-6)

    def test_angles_in_valid_range(self, two_factor_pairs):
        (B1, A1), (B2, A2) = two_factor_pairs
        angles = principal_angles_low_rank(B1, A1, B2, A2)
        assert np.all(angles >= -1e-6)
        assert np.all(angles <= 90.0 + 1e-6)
