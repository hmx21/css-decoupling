"""Exterior evaluations agree with independent determinants and composition."""
from itertools import combinations
from math import comb

import pytest
from sympy import Matrix as IntegerMatrix

from css_decoupling.algebra.exterior import pullback_coefficients, subset_at
from css_decoupling.algebra.laurent import Ring, Matrix, zeros
from css_decoupling.decoupling.gluing import GluingClass
from css_decoupling.errors import ResourceLimit


@pytest.mark.parametrize("p,D,degree", [(2, 3, 2), (3, 4, 3), (5, 5, 2), (2147483659, 3, 2)])
def test_matrix_valued_restriction_matches_integer_minors(p, D, degree):
    ring = Ring(p, D)
    subsets = list(combinations(range(D), degree))
    C = Matrix(ring, 2, len(subsets) * 2,
               {(i, 2 * k + s): (i + 2) * (k + 1) + s for i in range(2)
                for k in range(len(subsets)) for s in range(2)})
    M = [[(i + 2) * (j + 1) - 3 * (i == j) for j in range(D)] for i in range(D)]
    cls = GluingClass(degree, 1, 2, 2, C)
    answer = cls.restrict(M)
    for j, J in enumerate(subsets):
        for row in range(2):
            for source in range(2):
                expected = sum(int(IntegerMatrix([[M[a][b] for b in J] for a in I]).det())
                               * C[row, 2 * k + source].constant() for k, I in enumerate(subsets)) % p
                assert answer.coefficients[row, 2 * j + source] == expected
    assert cls.coefficients == C


def test_composition_and_odd_prime_orientation():
    ring = Ring(5, 3)
    cls = GluingClass(2, 1, 1, 1, Matrix.from_rows(ring, [[1, 2, 3]]))
    A = IntegerMatrix([[1, 2, -1], [0, 1, 1], [3, 0, 1]])
    B = IntegerMatrix([[0, 1, 0], [1, 0, 0], [0, 2, 5]])
    assert cls.restrict(A).restrict(B).coefficients == cls.restrict(A * B).coefficients
    assert cls.evaluate([(1, 0, 0), (0, 1, 0)]) == Matrix.from_rows(ring, [[1]])
    assert cls.evaluate([(0, 1, 0), (1, 0, 0)]) == Matrix.from_rows(ring, [[4]])
    assert not cls.evaluate([(1, 2, 3), (2, 4, 6)])
    assert cls.vanishes_on([[1], [0], [0]])
    assert cls.vanishes_on([[], [], []])


def test_sparse_high_degree_does_not_build_all_source_subsets():
    ring = Ring(3, 32)
    C = Matrix(ring, 1, comb(32, 16), {(0, 0): 1})
    cls = GluingClass(16, 1, 1, 1, C)
    assert cls.terms() == ((tuple(range(16)), 0, 0, 1),)
    vectors = [tuple(int(i == j) for i in range(32)) for j in range(16)]
    assert cls.evaluate(vectors, max_exterior_entries=1) == Matrix.from_rows(ring, [[1]])
    W = [[int(i == j) for j in range(1, 32)] for i in range(32)]
    assert cls.vanishes_on(W, max_exterior_entries=1)
    identity = [[int(i == j) for j in range(32)] for i in range(32)]
    with pytest.raises(ResourceLimit, match="Exterior target"):
        cls.restrict(identity, max_exterior_entries=100)


def test_zero_structural_degrees_and_unranking():
    for D in range(1, 7):
        for q in range(D + 1):
            assert [subset_at(j, D, q) for j in range(comb(D, q))] == list(combinations(range(D), q))
    ring = Ring(5, 2)
    C = zeros(ring, 3, 0)
    assert pullback_coefficients(C, 3, 2, [[1, 0], [0, 1]]).shape == (3, 0)


@pytest.mark.parametrize("bad", [[[1, True], [0, 1]], [[1, 0.5], [0, 1]], [[1], [0, 1]]])
def test_noninteger_or_ragged_pullback_is_rejected(bad):
    ring = Ring(5, 2)
    with pytest.raises(ValueError):
        pullback_coefficients(Matrix.from_rows(ring, [[1]]), 2, 1, bad)


def test_invalid_coefficient_shapes_and_resource_caps():
    ring = Ring(3, 3)
    with pytest.raises(ValueError, match="declared shape"):
        GluingClass(2, 1, 1, 1, Matrix.from_rows(ring, [[1, 2]]))
    with pytest.raises(ValueError, match="constant"):
        GluingClass(2, 1, 1, 1, Matrix.from_rows(ring, [[ring.variable(0), 0, 0]]))
    cls = GluingClass(2, 1, 1, 1, Matrix.from_rows(ring, [[1, 2, 0]]))
    with pytest.raises(ValueError, match="dimensions"):
        cls.evaluate([(1, 0, 0)])
    with pytest.raises(ValueError, match="cap"):
        cls.restrict([[1, 0, 0], [0, 1, 0], [0, 0, 1]], max_exterior_entries=True)
