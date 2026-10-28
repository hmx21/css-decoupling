from itertools import product

import pytest

from css_decoupling import Ring, Matrix, block, zeros
from css_decoupling.decoupling.analysis import analyze_profile
from css_decoupling.decoupling.gluing import GluingClass
from css_decoupling.errors import ResourceLimit
from css_decoupling.models.examples import glued_css, toric_css
from css_decoupling.topology.complexes import koszul
from css_decoupling.topology.periodic import (
    periodic_code, periodic_matrix, periodic_from_gluing, periodic_profile,
)


def diagonal(lengths):
    return [[a if i == j else 0 for j in range(len(lengths))]
            for i, a in enumerate(lengths)]


def dense_rectangular(A, lengths):
    """Independent rectangular group-ring evaluation, with coordinate moduli."""
    cells = tuple(product(*(range(L) for L in lengths)))
    indices = {a: i for i, a in enumerate(cells)}
    N = len(cells)
    out = [[0] * (A.ncols * N) for _ in range(A.nrows * N)]
    for (i, j), f in A.entries.items():
        for exponent, coefficient in f.terms.items():
            for k, cell in enumerate(cells):
                target = tuple((a + b) % L for a, b, L in zip(cell, exponent, lengths))
                row, column = i * N + indices[target], j * N + k
                out[row][column] = (out[row][column] + coefficient) % A.ring.p
    return out


def dense_rank(rows, p):
    rows = [list(row) for row in rows]
    pivot = 0
    for column in range(len(rows[0]) if rows else 0):
        found = next((i for i in range(pivot, len(rows)) if rows[i][column] % p), None)
        if found is None:
            continue
        rows[pivot], rows[found] = rows[found], rows[pivot]
        inverse = pow(rows[pivot][column], -1, p)
        for i in range(pivot + 1, len(rows)):
            coefficient = rows[i][column] * inverse % p
            rows[i] = [(a - coefficient * b) % p for a, b in zip(rows[i], rows[pivot])]
        pivot += 1
    return pivot


def shared_css(ring):
    """Explicit 9-qudit shared-target example, independent of fast contraction."""
    A, B = koszul(ring, 1), koszul(ring, 2)
    c12, c13 = Matrix.from_rows(ring, [[-1, 0, 0]]), Matrix.from_rows(ring, [[0, -1, 0]])
    d12 = Matrix.from_rows(ring, [[0], [0], [1]])
    d13 = Matrix.from_rows(ring, [[0], [-1], [0]])
    HX = block([[A.d(0), c12, c13],
                [zeros(ring, 3, 3), B.d(0), zeros(ring, 3, 3)],
                [zeros(ring, 3, 3), zeros(ring, 3, 3), B.d(0)]])
    Zdag = block([[A.d(-1), d12, d13],
                  [zeros(ring, 3, 3), B.d(-1), zeros(ring, 3, 1)],
                  [zeros(ring, 3, 3), zeros(ring, 3, 1), B.d(-1)]])
    return HX, Zdag.dagger()


@pytest.mark.parametrize("p", [2, 3, 5])
def test_periodic_odd_prime_rectangular_independent_dense(p):
    HX, HZ = glued_css(Ring(p, 3), coefficient=2 if p != 2 else 1)
    lengths = (p, 1, 2)
    result = periodic_code(HX, HZ, diagonal(lengths))
    X, Z = dense_rectangular(HX, lengths), dense_rectangular(HZ, lengths)
    assert result.HX_periodic.dense() == X
    assert result.HZ_periodic.dense() == Z
    assert result.rank_x == dense_rank(X, p)
    assert result.rank_z == dense_rank(Z, p)
    assert result.logical_qudits == 6
    assert result.upper_bound is None and result.decouplable is None


@pytest.mark.parametrize("p", [2, 3, 5])
def test_shared_matrix_valued_restrictions_can_be_nonzero_proportional(p):
    ring = Ring(p, 3)
    cls = GluingClass(2, 1, 2, 1, Matrix(ring, 1, 6, {(0, 0): 1, (0, 3): 1}))
    # On the rank-two mod-p image both forms become eta_12; incoming rank
    # is one, even though both coefficients remain nonzero.
    skew = [[1, 0, 0], [0, 1, 0], [0, 1, p]]
    quick = periodic_from_gluing([cls], {1: 1, 2: 2}, skew)
    finite = periodic_code(*shared_css(ring), skew)
    assert (quick.incoming_rank, quick.outgoing_rank, quick.logical_qudits) == (1, 1, 7)
    assert finite.logical_qudits == quick.logical_qudits
    ordinary = periodic_from_gluing([cls], {1: 1, 2: 2}, diagonal([1, 1, 1]))
    assert (ordinary.incoming_rank, ordinary.outgoing_rank, ordinary.logical_qudits) == (2, 1, 6)
    assert quick.upper_bound is None and quick.decouplable is None


@pytest.mark.parametrize("D,i,j,axes", [(4, 1, 3, (0, 1, 2)), (5, 2, 4, (0, 2, 4))])
def test_nonadjacent_ext_and_shifted_lower_degree(D, i, j, axes):
    from itertools import combinations
    ring = Ring(3, D)
    exterior = tuple(combinations(range(D), j - i + 1))
    cls = GluingClass(j, i, 1, 1, Matrix(ring, 1, len(exterior), {(0, exterior.index(axes)): 2}))
    B = diagonal([1] * D)
    quick = periodic_from_gluing([cls], {i: 1, j: 1}, B)
    finite = periodic_code(*glued_css(ring, i, j, axes=axes, coefficient=2), B)
    assert quick.logical_qudits == finite.logical_qudits
    B[axes[0]][axes[0]] = 3
    assert periodic_from_gluing([cls], {i: 1, j: 1}, B).logical_qudits == quick.model_total


def test_nontrivial_translation_actions_and_physical_profile_binding():
    ring = Ring(3, 2)
    x, y = ring.variables
    HX = Matrix.from_rows(ring, [[x + 1, y + 1]])
    HZ = Matrix.from_rows(ring, [[-(y + 1)], [x + 1]]).dagger()
    analysis = analyze_profile(HX, HZ)
    small = periodic_profile(analysis, diagonal([1, 1]))
    assert (small.method, small.logical_qudits, small.upper_bound, small.decouplable) == ("single-layer", 0, 2, False)
    skew = [[1, 0], [1, 2]]
    large = periodic_profile(analysis, skew)
    assert (large.method, large.logical_qudits, large.decouplable) == ("single-layer", 2, True)
    assert large.logical_qudits == periodic_code(HX, HZ, skew).logical_qudits


def test_verified_fast_count_large_index_does_not_expand_cells():
    analysis = analyze_profile(*glued_css(Ring(2, 3)))
    huge = diagonal([1, 2 * 10**30, 1])
    result = periodic_profile(analysis, huge, max_qudits=1)
    assert result.method == "two-layer" and result.logical_qudits == 6
    assert result.physical_qudits == 12 * 10**30
    assert result.decouplable is True


def test_initial_lattice_is_not_silently_replaced_by_lambda_zero():
    analysis = analyze_profile(*glued_css(Ring(2, 3)), lattice_basis=diagonal([2, 1, 1]))
    # This period kills the true gluing but is not contained in the base on
    # which the supplied analysis recorded its already-zero class.
    result = periodic_profile(analysis, diagonal([1, 2, 1]))
    assert result.method == "finite" and result.logical_qudits == 6
    result = periodic_profile(analysis, diagonal([1, 1, 1]))
    assert result.logical_qudits == 4 and not result.decouplable


def test_multilayer_uses_full_periodic_matrix_and_never_pairwise_sum():
    ring = Ring(2, 4)
    checks = toric_css(ring, {1: 1, 2: 1, 3: 1})
    result = periodic_profile(analyze_profile(*checks), diagonal([1] * 4))
    assert result.method == "finite" and result.logical_qudits == 14
    with pytest.raises(ValueError, match="at most two"):
        periodic_from_gluing([], {1: 1, 2: 1, 3: 1}, diagonal([1] * 4), prime=2)


def test_periodic_negative_exponents_cancellations_and_lattice_basis_change():
    ring = Ring(5, 2)
    x, y = ring.variables
    A = Matrix.from_rows(ring, [[x**-2 + 2 * y - 3, x * y**-1]])
    B = [[2, 1], [0, 3]]
    changed = [[2, 3], [0, 3]]  # right multiply B by an integral shear
    r1 = periodic_code(A, zeros(ring, 0, 2), B)
    r2 = periodic_code(A, zeros(ring, 0, 2), changed)
    assert r1.logical_qudits == r2.logical_qudits
    assert periodic_matrix(Matrix.from_rows(ring, [[x - 1]]), diagonal([1, 1])).entries == {}


def test_resource_caps_and_bad_periods_are_not_mathematical_zero():
    HX, HZ = glued_css(Ring(3, 3))
    with pytest.raises(ResourceLimit):
        periodic_code(HX, HZ, diagonal([3, 3, 3]), max_qudits=10)
    with pytest.raises(ResourceLimit):
        periodic_code(HX, HZ, diagonal([1, 1, 1]), max_matrix_entries=1)
    with pytest.raises(ValueError):
        periodic_code(HX, HZ, diagonal([1, 0, 1]))
    with pytest.raises(ValueError):
        periodic_code(HX, HZ, diagonal([1, True, 1]))
    cls = GluingClass(2, 1, 1, 1, Matrix(HX.ring, 1, 3, {(0, 0): 1}))
    with pytest.raises(ResourceLimit):
        periodic_from_gluing([cls], {1: 1, 2: 1}, diagonal([1] * 3), max_exterior_entries=1)
    with pytest.raises(ValueError, match="contained"):
        periodic_from_gluing([cls], {1: 1, 2: 1}, diagonal([1] * 3),
                            base_lattice_basis=diagonal([3, 1, 1]))
