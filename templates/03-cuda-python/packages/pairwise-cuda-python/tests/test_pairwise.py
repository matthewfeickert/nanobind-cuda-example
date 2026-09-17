import numpy as np
import pairwise_cuda_python
import pytest
from scipy.spatial.distance import cdist


@pytest.fixture
def rng():
    return np.random.default_rng(20260912)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(("n", "m", "d"), [(1, 1, 1), (7, 5, 3), (257, 130, 16)])
def test_matches_scipy(rng, dtype, n, m, d):
    x = rng.normal(size=(n, d)).astype(dtype)
    y = rng.normal(size=(m, d)).astype(dtype)
    got = pairwise_cuda_python.pairwise_distances(x, y)
    assert got.dtype == dtype
    assert got.shape == (n, m)
    rtol = 1e-5 if dtype == np.float32 else 1e-12
    np.testing.assert_allclose(got, cdist(x, y), rtol=rtol, atol=0)


def test_y_defaults_to_x(rng):
    x = rng.normal(size=(20, 4))
    got = pairwise_cuda_python.pairwise_distances(x)
    np.testing.assert_allclose(got, cdist(x, x), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.diag(got), 0.0, atol=1e-12)


def test_non_contiguous_and_integer_inputs(rng):
    # Fortran-ordered integer input must be normalised before hitting the kernel.
    x = np.asfortranarray(rng.integers(0, 10, size=(9, 3)))
    got = pairwise_cuda_python.pairwise_distances(x)
    assert got.dtype == np.float32
    np.testing.assert_allclose(got, cdist(x, x), rtol=1e-5)


def test_feature_mismatch_raises(rng):
    with pytest.raises(ValueError, match="same number of columns"):
        pairwise_cuda_python.pairwise_distances(
            rng.normal(size=(3, 2)), rng.normal(size=(3, 5))
        )
