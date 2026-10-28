"""Actual Appendix-C free complements, including noncoordinate summands."""

import pytest

from css_decoupling import Matrix, Ring, decouple_general as decouple
from css_decoupling.algebra.laurent import eye
from css_decoupling.algebra.modules import get_backend, solve, using_backend
from css_decoupling.algebra.projective import free_basis
from css_decoupling.certificates.projective import verify_projective_basis
from css_decoupling.decoupling.comparison import stably_trivialize_idempotent
from css_decoupling.errors import ResourceLimit
from css_decoupling.models.examples import glued_css


@pytest.mark.parametrize("p", [2, 3, 5])
def test_primitive_rank_one_basis_without_any_unit_entry(p):
    ring = Ring(p, 3)
    x, y, z = ring.variables
    f = x + y * z**-2
    P = Matrix.from_rows(ring, [[1 + f], [f]])
    Q = Matrix.from_rows(ring, [[1 - f, f]])
    E = P * Q
    assert not any(entry.is_unit for entry in E.entries.values())
    result = free_basis(E)
    assert result['rank'] == 1
    assert result['odd_free_rank'] == 0
    assert result['steps'] == [{'method': 'primitive-column-bezout', 'rank_removed': 1}]
    assert verify_projective_basis(E, result['basis'], result['coordinates'])


def test_syzygy_basis_when_every_idempotent_column_is_nonunimodular():
    ring = Ring(5, 2)
    x, y = ring.variables
    v = Matrix.from_rows(ring, [[x - 1], [y - 1], [3 - x - y]])
    E = eye(ring, 3) - v * Matrix.from_rows(ring, [[1, 1, 1]])
    count, remainder, _, _ = get_backend(ring).reduce_units(E)
    assert count == 0 and remainder
    # All three columns vanish at a torus point: respectively (2,1),
    # (1,2), (1,1).  No original column alone can be part of a free basis.
    for j, values in enumerate([(2, 1), (1, 2), (1, 1)]):
        for entry in E[:, j:j + 1].entries.values():
            assert sum(c * pow(values[0], e[0], 5) * pow(values[1], e[1], 5)
                       for e, c in entry.terms.items()) % 5 == 0
    result = free_basis(E)
    assert result['rank'] == 2
    assert result['steps'] == [{'method': 'syzygy-generator-reduction', 'rank_removed': 2}]
    assert verify_projective_basis(E, result['basis'], result['coordinates'])


def test_actual_basis_removes_former_extra_parity_summand():
    ring = Ring(5, 2)
    x, y = ring.variables
    P = Matrix.from_rows(ring, [[1, x], [y, 1 + x*y]])
    Pinv = Matrix.from_rows(ring, [[1 + x*y, -x], [-y, 1]])
    E = P * Matrix(ring, 2, 2, {(0, 0): 1}) * Pinv
    former = stably_trivialize_idempotent(E, [E, P[:, 1:]], solve)
    current = free_basis(E)
    assert (former['even_free_rank'], former['odd_free_rank']) == (2, 1)
    assert (current['even_free_rank'], current['odd_free_rank']) == (1, 0)
    assert verify_projective_basis(E, current['basis'], current['coordinates'])


def test_basis_verifier_rejects_a_tampered_coordinate_map():
    ring = Ring(3, 2)
    E = Matrix.from_rows(ring, [[1, ring.variable(0)], [0, 0]])
    witness = free_basis(E)
    Q = witness['coordinates'].copy()
    Q[0, 0] += 1
    with pytest.raises(ValueError, match='inverse factorization'):
        verify_projective_basis(E, witness['basis'], Q)


def _scrambled_rank_two_projector(*, dense):
    ring = Ring(5, 2)
    x, y = ring.variables
    v = Matrix.from_rows(ring, [[x - 1], [y - 1], [3 - x - y]])
    F = eye(ring, 3) - v * Matrix.from_rows(ring, [[1, 1, 1]])
    B, Binv = eye(ring, 3), eye(ring, 3)
    operations = [(1, 0, x - 1), (2, 1, x*y - 1)]
    if dense:
        operations += [(0, 2, x*x + y - 2), (2, 0, x*y - 1)]
    for i, j, f in operations:
        A = Matrix(ring, 3, 3, {(i, j): f})
        B, Binv = (eye(ring, 3) + A) * B, Binv * (eye(ring, 3) - A)
    E = B * F * Binv
    known_P = B * Matrix.from_rows(ring, [[1, 0], [0, 1], [-1, -1]])
    known_Q = F[:2, :] * Binv
    assert verify_projective_basis(E, known_P, known_Q)
    return E


def test_nontrivial_higher_rank_reduction_returns_an_actual_basis():
    E = _scrambled_rank_two_projector(dense=False)
    with using_backend('sympy'):
        result = free_basis(E, qs_options={'timeout': 20})
        assert result['rank'] == 2 and result['odd_free_rank'] == 0
        assert verify_projective_basis(E, result['basis'], result['coordinates'])


def test_dense_scrambled_projector_resource_stop_is_not_a_partial_basis():
    # This former optional-CAS case exhausted a 120-second native QS budget
    # in an isolated run with an independent 180-second process-tree deadline.
    # Retain its exact input and independently known basis, but test a
    # deterministic operation stop instead of requiring a long timed solve.
    E = _scrambled_rank_two_projector(dense=True)
    with using_backend('sympy'), pytest.raises(ResourceLimit, match='operation'):
        free_basis(E, qs_options={'max_steps': 0, 'timeout': 5})


@pytest.mark.integration
@pytest.mark.parametrize('dimension,upper', [(3, 2), (4, 3)])
def test_real_odd_prime_gluing_uses_paper_complement_ranks(dimension, upper):
    ring = Ring(3, dimension)
    lattice = [[3 if i == j == 0 else int(i == j)
                for j in range(dimension)] for i in range(dimension)]
    result = decouple(*glued_css(ring, 1, upper), lattice_basis=lattice, backend='sympy',
                     resolution_timeout=20, comparison_method='free-basis')
    comparison = result.trace['comparison']
    basis = comparison['trivialization']
    assert result.free_T == result.trace['restricted'].rank(2)
    assert result.free_C == basis['rank']
    assert basis['odd_free_rank'] == 0
    assert verify_projective_basis(comparison['idempotent'], basis['basis'], basis['coordinates'])
