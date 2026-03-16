import pytest
import torch

from src.utils.linalg import reduce_lora_rank_efficient, reduce_rank


@pytest.fixture
def matrix():
    torch.manual_seed(0)
    return torch.randn(6, 4)


@pytest.fixture
def lora_pair():
    """Returns A (r, n) and B (m, r) with r << m, n."""
    torch.manual_seed(0)
    m, n, r = 64, 48, 8
    B = torch.randn(m, r)
    A = torch.randn(r, n)
    return A, B


# --- reduce_rank tests ---


def test_reduce_rank_output_shape(matrix):
    result = reduce_rank(matrix, new_rank=2)
    assert result.shape == matrix.shape


def test_reduce_rank_output_rank(matrix):
    result = reduce_rank(matrix, new_rank=2)
    assert torch.linalg.matrix_rank(result) == 2


def test_reduce_rank_full_rank_reconstructs(matrix):
    result = reduce_rank(matrix, new_rank=4)
    assert torch.allclose(result, matrix, atol=1e-5)


def test_reduce_rank_eckart_young(matrix):
    """Frobenius error equals sqrt(sum of squared tail singular values) — Eckart-Young theorem."""
    new_rank = 2
    S = torch.linalg.svdvals(matrix)
    expected_error = S[new_rank:].pow(2).sum().sqrt()
    actual_error = (matrix - reduce_rank(matrix, new_rank=new_rank)).norm()
    assert torch.allclose(actual_error, expected_error, atol=1e-5)


def test_reduce_rank_invalid_too_large(matrix):
    with pytest.raises(ValueError, match="new_rank"):
        reduce_rank(matrix, new_rank=5)


def test_reduce_rank_invalid_zero(matrix):
    with pytest.raises(ValueError, match="new_rank"):
        reduce_rank(matrix, new_rank=0)


def test_reduce_rank_invalid_negative(matrix):
    with pytest.raises(ValueError, match="new_rank"):
        reduce_rank(matrix, new_rank=-1)


# --- reduce_lora_rank_efficient tests ---


def test_reduce_lora_rank_matches_svd(lora_pair):
    """reduce_lora_rank_efficient(A, B, k) reconstructs B@A the same as reduce_rank(B@A, k)."""
    A, B = lora_pair
    new_rank = 4

    new_A, new_B = reduce_lora_rank_efficient(A, B, new_rank)
    assert torch.allclose(new_B @ new_A, reduce_rank(B @ A, new_rank), atol=1e-4)


def test_reduce_lora_rank_matches_svd_rank_1(lora_pair):
    A, B = lora_pair
    new_rank = 1

    new_A, new_B = reduce_lora_rank_efficient(A, B, new_rank)
    assert torch.allclose(new_B @ new_A, reduce_rank(B @ A, new_rank), atol=1e-4)


def test_reduce_lora_rank_output_shapes(lora_pair):
    A, B = lora_pair
    m, n = B.shape[0], A.shape[1]
    new_rank = 3

    new_A, new_B = reduce_lora_rank_efficient(A, B, new_rank)

    assert new_A.shape == (new_rank, n)
    assert new_B.shape == (m, new_rank)


def test_reduce_lora_rank_invalid_too_large(lora_pair):
    A, B = lora_pair  # current rank r = 8
    with pytest.raises(ValueError, match="new_rank"):
        reduce_lora_rank_efficient(A, B, new_rank=9)


def test_reduce_lora_rank_invalid_zero(lora_pair):
    A, B = lora_pair
    with pytest.raises(ValueError, match="new_rank"):
        reduce_lora_rank_efficient(A, B, new_rank=0)


def test_reduce_lora_rank_invalid_negative(lora_pair):
    A, B = lora_pair
    with pytest.raises(ValueError, match="new_rank"):
        reduce_lora_rank_efficient(A, B, new_rank=-1)
