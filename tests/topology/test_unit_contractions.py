"""Unit-disk retractions and complete residual contractions, with signs."""
from unittest.mock import Mock

import pytest

from css_decoupling.algebra.laurent import Matrix, Ring, eye, zeros
from css_decoupling.algebra.modules import NoLaurentSolution, solve
from css_decoupling.errors import ResourceLimit
from css_decoupling.topology.complexes import (
    CochainComplex, CochainMap, direct_sum_complexes, disk, identity_map,
    koszul, mapping_cone, _solve_contraction_descending, validate_homotopy,
)
from css_decoupling.topology.contractions import cancel_unit_disks, solve_contraction_with_units


def unchanged(C):
    return dict(C.ranks), {n: d.key() for n, d in C.differentials.items()}


@pytest.mark.parametrize("prime", [2, 3, 5])
def test_a_laurent_unit_disk_needs_no_module_solve(prime):
    ring = Ring(prime, 3)
    unit = (prime-1)*ring.variable(0)**-3*ring.variable(2)**2
    C = CochainComplex(ring, {-2: 1, -1: 1}, {-2: Matrix.from_rows(ring, [[unit]])})
    before = unchanged(C)
    solver = Mock(side_effect=AssertionError('unit disk requested module algebra'))
    result = solve_contraction_with_units(C, solver, residual_solver=solver)
    assert result[-1] == Matrix.from_rows(ring, [[unit**-1]])
    assert unchanged(C) == before
    assert not solver.called


@pytest.mark.parametrize("prime", [2, 3, 5])
def test_unit_schur_complement_has_exact_chain_maps_and_signed_homotopy(prime):
    ring = Ring(prime, 2)
    x, y = ring.variables
    # Its unit determinant is not used as an assumed certificate.
    A = Matrix.from_rows(ring, [[x**-2, y+1], [x+y, x*x*(x+y)*(y+1)+2*x**-2]])
    C = CochainComplex(ring, {0: 2, 1: 2}, {0: A})
    before = unchanged(C)
    reduced = cancel_unit_disks(C)
    assert reduced.validate()
    assert reduced.steps
    assert unchanged(C) == before
    for n in C.degrees:
        assert reduced.projection.at(n)*reduced.inclusion.at(n) == eye(ring, reduced.residual.rank(n))


@pytest.mark.parametrize("prime", [2, 3, 5])
def test_basis_changed_identity_cone_contracts_and_visits_high_degree_first(prime):
    ring = Ring(prime, 3)
    C = koszul(ring, 1)
    # A monomial on one coordinate and a transvection give genuine Laurent
    # basis changes, preserving all cochain degrees and odd-prime signs.
    U = eye(ring, C.rank(0))
    U[0, 0] = ring.variable(0)**-1
    U[0, 1] = ring.variable(1)+2
    inverse = solve(U, eye(ring, U.nrows))
    maps = {n: U if n == 0 else eye(ring, C.rank(n)) for n in C.degrees}
    inverses = {n: inverse if n == 0 else eye(ring, C.rank(n)) for n in C.degrees}
    T = CochainComplex(ring, C.ranks, {
        n: maps.get(n+1, zeros(ring, 0, 0))*C.d(n)*inverses[n] for n in C.degrees})
    cone = mapping_cone(CochainMap(C, T, maps))
    before = unchanged(cone)
    reduction = cancel_unit_disks(cone)
    assert not any(reduction.residual.ranks.values())
    assert [p.degree for p in reduction.steps] == sorted((p.degree for p in reduction.steps), reverse=True)
    h = solve_contraction_with_units(cone, solve, residual_solver=_solve_contraction_descending)
    validate_homotopy(cone, cone, identity_map(cone), CochainMap(cone, cone, {}), h)
    assert unchanged(cone) == before


def test_no_units_falls_back_to_the_complete_residual_solver():
    ring = Ring(5, 2)
    x = ring.variable(0)
    C = CochainComplex(ring, {0: 1, 1: 1}, {0: Matrix.from_rows(ring, [[x+1]])})
    complete = Mock(wraps=_solve_contraction_descending)
    with pytest.raises(NoLaurentSolution):
        solve_contraction_with_units(C, solve, residual_solver=complete)
    assert complete.call_count == 1
    assert complete.call_args.args[0].d(0) == C.d(0)


def test_no_units_can_still_have_a_valid_contraction_and_is_not_rejected():
    ring = Ring(2, 2)
    x = ring.variable(0)
    # Bézout: (x+1)*x + (x*x+x+1) = 1. Neither differential has a unit entry.
    a, b = x+1, x*x+x+1
    C = CochainComplex(ring, {-1: 1, 0: 2, 1: 1}, {
        -1: Matrix.from_rows(ring, [[b], [-a]]),
        0: Matrix.from_rows(ring, [[a, b]]),
    })
    assert not cancel_unit_disks(C).steps
    complete = Mock(wraps=_solve_contraction_descending)
    h = solve_contraction_with_units(C, solve, residual_solver=complete)
    assert complete.call_count == 1
    validate_homotopy(C, C, identity_map(C), CochainMap(C, C, {}), h)


def test_partial_cancellation_still_solves_and_pulls_back_the_residual():
    ring = Ring(3, 2)
    x = ring.variable(0)
    # a and b differ by a unit, so their two-generator Koszul complex is exact.
    a, b = x+1, x+2
    residual = CochainComplex(ring, {-1: 1, 0: 2, 1: 1}, {
        -1: Matrix.from_rows(ring, [[-b], [a]]), 0: Matrix.from_rows(ring, [[a, b]])})
    C = direct_sum_complexes(residual, disk(ring, 0, 2))
    reduction = cancel_unit_disks(C)
    assert len(reduction.steps) == 2
    assert sum(reduction.residual.ranks.values()) == sum(residual.ranks.values())
    complete = Mock(wraps=_solve_contraction_descending)
    h = solve_contraction_with_units(C, solve, residual_solver=complete)
    assert complete.call_count == 1
    validate_homotopy(C, C, identity_map(C), CochainMap(C, C, {}), h)


def test_window_uses_the_boundary_disk_without_claiming_lower_contractibility():
    ring = Ring(5, 2)
    obstruction = CochainComplex(ring, {-3: 1}, {})
    C = direct_sum_complexes(obstruction, disk(ring, -1), disk(ring, 1))
    reduction = cancel_unit_disks(C, min_degree=0)
    assert [p.degree for p in reduction.steps] == [1, -1]
    assert reduction.residual.rank(-3) == 1
    h = solve_contraction_with_units(C, solve, residual_solver=_solve_contraction_descending, min_degree=0)
    assert set(h) == {n for n in C.degrees if n >= 0}
    validate_homotopy(C, C, identity_map(C), CochainMap(C, C, {}), h, min_degree=0)
    with pytest.raises(NoLaurentSolution):
        solve_contraction_with_units(C, solve, residual_solver=_solve_contraction_descending)


def test_units_below_requested_window_are_not_cancelled():
    ring = Ring(3, 2)
    C = direct_sum_complexes(disk(ring, -3), disk(ring, 0))
    result = cancel_unit_disks(C, min_degree=0)
    assert [p.degree for p in result.steps] == [0]
    assert result.residual.rank(-3) == result.residual.rank(-2) == 1


def test_residual_resource_failure_propagates_and_false_homotopy_is_rejected():
    ring = Ring(3, 2)
    x = ring.variable(0)
    C = CochainComplex(ring, {0: 1, 1: 1}, {0: Matrix.from_rows(ring, [[x+1]])})
    with pytest.raises(ResourceLimit, match='budget'):
        solve_contraction_with_units(C, solve, residual_solver=Mock(side_effect=ResourceLimit('budget')))
    with pytest.raises(ValueError, match='homotopy'):
        solve_contraction_with_units(C, solve, residual_solver=lambda *a, **k: {})


def test_corrupted_retraction_is_rejected_by_actual_identities():
    ring = Ring(3, 2)
    C = disk(ring, 0)
    reduction = cancel_unit_disks(C)
    reduction.homotopy[1] = zeros(ring, 1, 1)
    with pytest.raises(ValueError, match='homotopy'):
        reduction.validate()


@pytest.mark.parametrize('value', [True, 0.5, '0'])
def test_invalid_window_bounds_are_rejected(value):
    with pytest.raises(ValueError, match='min_degree'):
        cancel_unit_disks(disk(Ring(3, 2), 0), min_degree=value)
