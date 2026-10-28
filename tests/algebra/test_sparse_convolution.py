"""Compare integer-key convolution with an independent tuple-exponent oracle."""
import random

import pytest

import css_decoupling.algebra.laurent as laurent
from css_decoupling.algebra.laurent import Matrix, Ring, zeros


def tuple_product(A, B):
    """Independent entry-by-entry polynomial convolution, including cancellations."""
    entries = {}
    for i in range(A.nrows):
        for j in range(B.ncols):
            terms = {}
            for k in range(A.ncols):
                for a, c in A[i, k].terms.items():
                    for b, d in B[k, j].terms.items():
                        exponent = tuple(a[axis] + b[axis] for axis in range(A.ring.D))
                        terms[exponent] = (terms.get(exponent, 0) + c * d) % A.ring.p
            if any(terms.values()):
                entries[i, j] = terms
    return Matrix(A.ring, A.nrows, B.ncols, entries)


def random_matrix(ring, rows, cols, rng, *, terms=17, shift=0):
    entries = {}
    for i in range(rows):
        for j in range(cols):
            if rng.random() < 0.3:
                continue
            support = {tuple(rng.randrange(-3, 4) + shift * (axis + 1)
                             for axis in range(ring.D)): rng.randrange(1, ring.p)
                       for _ in range(terms)}
            entries[i, j] = support
    return Matrix(ring, rows, cols, entries)


@pytest.mark.parametrize("D", [2, 3, 5, 20])
@pytest.mark.parametrize("p", [2, 3, 2147483647])
def test_random_products_with_large_negative_exponents(D, p, monkeypatch):
    monkeypatch.setattr(laurent, "_CONVOLUTION_MIN_PRODUCTS", 0)
    monkeypatch.setattr(laurent, "_CONVOLUTION_REUSE_FACTOR", 0)
    ring = Ring(p, D)
    rng = random.Random(991 + D + p)
    for seed in range(3):
        A = random_matrix(ring, 3, 4, rng, shift=-(10**30 + seed))
        B = random_matrix(ring, 4, 2, rng, shift=10**20)
        assert A * B == tuple_product(A, B)


def test_no_carry_between_axes_at_maximal_digits(monkeypatch):
    monkeypatch.setattr(laurent, "_CONVOLUTION_MIN_PRODUCTS", 0)
    monkeypatch.setattr(laurent, "_CONVOLUTION_REUSE_FACTOR", 0)
    ring = Ring(5, 3)
    A = Matrix.from_rows(ring, [[ring({(-8, 7, -2): 1, (13, 20, -2): 4})]])
    B = Matrix.from_rows(ring, [[ring({(-9, -12, 90): 2, (15, 5, 90): 3})]])
    assert A * B == tuple_product(A, B)
    assert (A * B)[0, 0] == ring({(-17, -5, 88): 2, (7, 12, 88): 3,
                                 (4, 8, 88): 3, (28, 25, 88): 2})


@pytest.mark.parametrize("p", [2, 3, 5])
def test_cancellation_across_inner_matrix_indices(p, monkeypatch):
    monkeypatch.setattr(laurent, "_CONVOLUTION_MIN_PRODUCTS", 0)
    monkeypatch.setattr(laurent, "_CONVOLUTION_REUSE_FACTOR", 0)
    ring = Ring(p, 5)
    f = ring({(i, -i, 2 * i, 0, 1): (i % (p - 1)) + 1 for i in range(70)})
    g = ring({(-i, 2 * i, i, 3, -4): 1 for i in range(70)})
    A = Matrix.from_rows(ring, [[f, f]])
    B = Matrix.from_rows(ring, [[g], [-g]])
    assert A * B == zeros(ring, 1, 1)
    assert not (A * B).entries


def test_excessive_integer_key_width_falls_back_exactly(monkeypatch):
    ring = Ring(5, 2)
    wide = 1 << 5000
    f = ring({(wide + i, -wide): i % 4 + 1 for i in range(65)} |
             {(-wide, wide): 1})
    g = ring({(-wide + i, wide): 1 for i in range(65)} |
             {(wide, -wide): 1})
    A, B = Matrix.from_rows(ring, [[f]]), Matrix.from_rows(ring, [[g]])
    assert laurent._encoded_matrix_product(A, B, {0: [(0, g)]}) is None
    assert A * B == tuple_product(A, B)


@pytest.mark.parametrize("shape", [(0, 4, 3), (3, 0, 2), (2, 4, 0)])
def test_empty_shapes_remain_exact(shape):
    ring = Ring(3, 5)
    rows, inner, cols = shape
    A, B = zeros(ring, rows, inner), zeros(ring, inner, cols)
    assert A * B == zeros(ring, rows, cols)


def test_tiny_product_does_not_use_encoding(monkeypatch):
    monkeypatch.setattr(laurent, "_encoded_matrix_product",
                        lambda *args: pytest.fail("Small product unexpectedly encoded"))
    ring = Ring(3, 2)
    A = Matrix.from_rows(ring, [[ring.variable(0) - 1]])
    assert A * A == tuple_product(A, A)


def test_thousand_term_support_uses_encoding(monkeypatch):
    calls = []
    real = laurent._encoded_matrix_product

    def record(*args):
        calls.append(True)
        return real(*args)
    monkeypatch.setattr(laurent, "_encoded_matrix_product", record)
    ring = Ring(3, 3)
    f = ring({(i, -i, 10**20): i % 2 + 1 for i in range(1000)})
    g = ring({(-i, i, -10**30): 1 for i in range(1000)})
    A, B = Matrix.from_rows(ring, [[f]]), Matrix.from_rows(ring, [[g]])
    assert A * B == tuple_product(A, B)
    assert calls
