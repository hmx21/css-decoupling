"""Sparse pack/restrict assembly agrees with the independent additive route."""
from unittest.mock import patch

import pytest

from css_decoupling.algebra.laurent import Matrix, Ring, zeros
from css_decoupling.lattice.retained import Lattice
from css_decoupling.lattice.restriction_backend import unpack_columns


def _additive_pack(A, lattice):
    out = zeros(A.ring, A.nrows * lattice.index, A.ncols)
    for (i, j), value in A.entries.items():
        for exponent, coefficient in value.terms.items():
            row, shift = lattice.reduce(exponent)
            out[i * lattice.index + row, j] += A.ring.monomial(shift, coefficient)
    return out


def _additive_restriction(A, lattice):
    out = zeros(A.ring, A.nrows * lattice.index, A.ncols * lattice.index)
    for (i, j), value in A.entries.items():
        for exponent, coefficient in value.terms.items():
            for column, representative in enumerate(lattice.representatives):
                row, shift = lattice.reduce(tuple(a + b for a, b in zip(exponent, representative)))
                out[i * lattice.index + row, j * lattice.index + column] += A.ring.monomial(
                    shift, coefficient)
    return out


@pytest.mark.parametrize('prime', [2, 3, 5, 2147483659])
@pytest.mark.parametrize('basis', [((2, 1), (1, -1)), ((0, -1), (1, 0)), ((-2, 0), (0, 1))])
def test_negative_and_skew_assembly_matches_independent_monomial_addition(prime, basis):
    R, lattice = Ring(prime, 2), Lattice(basis)
    x, y = R.variables
    A = Matrix.from_rows(R, [[x ** -3 + 2 * y, y ** -2 - x],
                            [x ** 4 * y ** -1 - 3, x * y + 2]])
    assert lattice.pack(A) == _additive_pack(A, lattice)
    assert lattice.restrict(A) == _additive_restriction(A, lattice)
    assert unpack_columns(lattice.pack(A), lattice) == A


def test_per_call_geometry_cache_scales_with_unique_support_not_matrix_repetitions():
    R, lattice = Ring(3, 2), Lattice(((2, 1), (1, -1)))
    support = {(i, j): 1 for i in range(-2, 3) for j in range(-1, 2)}
    value = R(support)
    A = Matrix(R, 9, 7, {(i, j): value for i in range(9) for j in range(7)})
    with patch.object(lattice, 'reduce', wraps=lattice.reduce) as geometry:
        packed = lattice.pack(A)
    assert geometry.call_count == len(support)
    translated = {tuple(a + b for a, b in zip(exponent, representative))
                  for exponent in support for representative in lattice.representatives}
    with patch.object(lattice, 'reduce', wraps=lattice.reduce) as geometry:
        restricted = lattice.restrict(A)
    assert geometry.call_count == len(translated)
    assert packed == _additive_pack(A, lattice)
    assert restricted == _additive_restriction(A, lattice)


def test_geometry_is_not_cached_across_lattice_mutations():
    R = Ring(5, 2)
    x, y = R.variables
    A = Matrix.from_rows(R, [[x ** -2 + y, x * y ** -1]])
    lattice = Lattice(((2, 0), (0, 1)))
    first = lattice.pack(A)
    lattice.restrict(A)
    # Replace all geometry consistently, retaining the same Python object.
    changed = Lattice(((1, 0), (0, -2)))
    lattice.__dict__.update(changed.__dict__)
    assert lattice.pack(A) == _additive_pack(A, changed)
    assert lattice.restrict(A) == _additive_restriction(A, changed)
    assert lattice.pack(A) != first


def test_wide_exponents_and_empty_matrices_do_not_scan_boxes_or_expand_empty_cells():
    R = Ring(3, 2)
    huge = 10 ** 30
    A = Matrix.from_rows(R, [[R.monomial((huge, -huge), 2) + R.monomial((-huge, huge))]])
    lattice = Lattice(((2, 1), (1, -1)))
    assert lattice.pack(A) == _additive_pack(A, lattice)
    assert lattice.restrict(A) == _additive_restriction(A, lattice)
    large = Lattice(((10 ** 6, 0), (0, 1)), max_index=None)
    for rows, columns in [(0, 3), (2, 0), (2, 3)]:
        empty = zeros(R, rows, columns)
        assert large.pack(empty).shape == (rows * large.index, columns)
        assert large.restrict(empty).shape == (rows * large.index, columns * large.index)
        assert large._representatives is None
