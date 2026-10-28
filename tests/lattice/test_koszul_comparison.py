"""Appendix B2 path comparison, signs and output-sensitive construction."""
from itertools import combinations
from math import comb

import pytest
from sympy import Matrix as IntegerMatrix

from css_decoupling.algebra.exterior import pullback_coefficients
from css_decoupling.algebra.laurent import Matrix, Ring, eye
from css_decoupling.certificates.koszul_comparison import verify_koszul_comparison
from css_decoupling.errors import ResourceLimit
from css_decoupling.lattice.koszul_comparison import koszul_comparison
from css_decoupling.lattice.retained import Lattice


def _tilted(D):
    return [[2 * (i == j) + (j == i + 1) - (i == j + 1) for j in range(D)] for i in range(D)]


@pytest.mark.parametrize("p,D", [(2, 2), (3, 2), (5, 2), (2, 3), (3, 3), (5, 3),
                                 (2, 4), (3, 4), (5, 4), (2, 5), (3, 5), (5, 5)])
def test_all_requested_degrees_have_chain_equations_and_integer_minors(p, D):
    ring, basis = Ring(p, D), _tilted(D)
    result = koszul_comparison(ring, basis, degrees=range(D + 1))
    assert verify_koszul_comparison(result).valid
    for degree, matrix in result.maps.items():
        subsets = tuple(combinations(range(D), degree))
        augmented = result.augmentation(degree)
        for i, I in enumerate(subsets):
            for j, J in enumerate(subsets):
                minor = int(IntegerMatrix([[basis[a][b] for b in J] for a in I]).det())
                assert augmented[i, j] == minor % p
        # Compare with the existing form pullback, including multiple copies.
        coefficients = Matrix(ring, 2, len(subsets) * 2,
                              {(r, 2 * s + c): r + 2 * s + c + 1 for r in range(2)
                               for s in range(len(subsets)) for c in range(2)})
        actual = pullback_coefficients(coefficients, degree, 2, basis)
        for r in range(2):
            for j in range(len(subsets)):
                for c in range(2):
                    assert actual[r, 2 * j + c] == sum(
                        (coefficients[r, 2 * i + c] * augmented[i, j] for i in range(len(subsets))), ring.zero())


def test_negative_path_orientation_is_laurent_not_reciprocal_rational():
    ring = Ring(5, 2)
    x, y = ring.variables
    result = koszul_comparison(ring, [[-2, 1], [1, -1]], degrees=(2,))
    assert result.path_map == Matrix.from_rows(ring, [[-x ** -1 - x ** -2, 1],
                                                       [x ** -2, -x * y ** -1]])
    assert result.chi(2)[0, 0] == (1 + x ** -1) * y ** -1 - x ** -2
    assert verify_koszul_comparison(result).valid


def test_default_and_sparse_degree_selection_do_not_build_whole_exterior_algebra():
    ring = Ring(3, 18)
    identity = [[int(i == j) for j in range(18)] for i in range(18)]
    default = koszul_comparison(ring, identity)
    assert set(default.maps) == {0, 1}
    high = koszul_comparison(ring, identity, degrees=(18,))
    assert set(high.maps) == {0, 17, 18}
    assert high.chi(17) == eye(ring, 18)
    assert high.chi(18) == eye(ring, 1)
    assert verify_koszul_comparison(high).valid
    with pytest.raises(ValueError, match="not materialized"):
        high.chi(9)


def test_high_dimension_small_requested_degree_and_huge_index_never_use_cosets(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("No coset or restriction-of-scalars expansion is permitted")

    monkeypatch.setattr(Lattice, "__init__", forbidden)
    ring = Ring(5, 40)
    basis = [[2 * (i == j) for j in range(40)] for i in range(40)]
    result = koszul_comparison(ring, basis, max_terms=200, max_entries=4000)
    assert set(result.maps) == {0, 1}
    assert sum(len(f.terms) for f in result.path_map.entries.values()) == 80
    assert 2 ** 40 > 10 ** 12
    assert verify_koszul_comparison(result, max_terms=300, max_entries=4000).valid


def test_scalar_inclusion_and_composition_after_augmentation():
    ring = Ring(5, 3)
    A = IntegerMatrix([[2, 1, 0], [-1, 2, 1], [0, -1, 2]])
    B = IntegerMatrix([[0, 1, 0], [1, 0, -1], [0, 0, 1]])
    left = koszul_comparison(ring, A, degrees=(0, 2, 3))
    right = koszul_comparison(ring, B, degrees=(0, 2, 3))
    composed = koszul_comparison(ring, A * B, degrees=(0, 2, 3))
    for n in (0, 1, 2, 3):
        assert composed.augmentation(n) == left.augmentation(n) * right.augmentation(n)
    # Canonical coordinate paths need not compose strictly as Laurent chains;
    # exterior pullback after augmentation does compose strictly.
    only_zero = koszul_comparison(ring, A, degrees=())
    assert only_zero.maps == {0: eye(ring, 1)}
    assert verify_koszul_comparison(only_zero).valid


def test_caps_precede_long_paths_and_large_exterior_output():
    with pytest.raises(ResourceLimit, match="max_terms"):
        koszul_comparison(Ring(3, 2), [[10 ** 100, 0], [0, 1]], max_terms=100)
    D = 30
    identity = [[int(i == j) for j in range(D)] for i in range(D)]
    with pytest.raises(ResourceLimit, match="max_entries"):
        koszul_comparison(Ring(3, D), identity, degrees=(15,), max_entries=10000)
    with pytest.raises(ResourceLimit, match="max_products"):
        koszul_comparison(Ring(3, 4), _tilted(4), degrees=(4,), max_products=100)
    # Compact inputs can still have an unavoidably large explicit determinant.
    with pytest.raises(ResourceLimit, match="max_terms"):
        koszul_comparison(Ring(3, 10), [[2 * (i == j) for j in range(10)] for i in range(10)],
                           degrees=(10,), max_terms=200)


def test_berkowitz_counts_zero_scans_and_terminates_zero_krylov_vectors():
    from css_decoupling.lattice.koszul_comparison import _Limits, _determinant
    from css_decoupling.algebra.laurent import Poly
    from unittest.mock import patch

    ring, D = Ring(3, 2), 30
    rows = [[ring(int(i == j)) for j in range(D)] for i in range(D)]
    calls = 0
    original = Poly.__bool__

    def counted(poly):
        nonlocal calls
        calls += 1
        return original(poly)

    limits = _Limits(10000, 10000, 100000)
    with patch.object(Poly, "__bool__", counted):
        assert _determinant(rows, ring, limits) == 1
    # A zero Krylov vector is terminated, rather than scanning all later
    # matrix powers. The previous implementation performed >200,000 scans.
    assert calls < 20000 and calls < limits.products
    zero = [[ring.zero() for _ in range(D)] for _ in range(D)]
    with pytest.raises(ResourceLimit, match="max_products"):
        _determinant(zero, ring, _Limits(10000, 10000, 28000))


@pytest.mark.parametrize("basis,degrees,kwargs", [
    ([[1, 0], [0, 0]], (1,), {}), ([[1, True], [0, 1]], (1,), {}),
    ([[1], [0]], (1,), {}), ([[1, 0], [0, 1]], (True,), {}),
    ([[1, 0], [0, 1]], (-1,), {}), ([[1, 0], [0, 1]], (3,), {}),
    ([[1, 0], [0, 1]], (1,), {"max_terms": 0}),
    ([[1, 0], [0, 1]], (1,), {"max_entries": True}),
    ([[1, 0], [0, 1]], (1,), {"max_products": 1.5}),
])
def test_invalid_input_is_not_resource_or_mathematical_obstruction(basis, degrees, kwargs):
    with pytest.raises(ValueError):
        koszul_comparison(Ring(3, 2), basis, degrees=degrees, **kwargs)
