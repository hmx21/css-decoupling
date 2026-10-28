"""Actual Gamma equations and strong sections for the checked toric solver.

All witnesses are checked against the original Laurent differential. These
small tests do not stand in for either real four-dimensional color-code run.
"""
from unittest.mock import Mock

import pytest

from css_decoupling.algebra.bounded_linear import augmentation
from css_decoupling.algebra.laurent import Matrix, Ring, eye, zeros
from css_decoupling.algebra.modules import solve
from css_decoupling.decoupling.toric_comparison import toric_cone_solver
from css_decoupling.errors import NoLaurentSolution, ResourceLimit
from css_decoupling.topology.complexes import (
    CochainComplex, CochainMap, direct_sum_complexes, disk, identity_map,
    koszul, mapping_cone, toric_stack,
)
from css_decoupling.topology.direct_splitting import solve_strong_contraction
from css_decoupling.topology.trivial_cohomology import (
    trivial_cohomology, trivial_cohomology_layer,
)


def _changed_comparison(prime, dimension, copies=2, degree=None):
    R = Ring(prime, dimension)
    degree = min(2, dimension - 1) if degree is None else degree
    T = koszul(R, degree, copies)
    raw = direct_sum_complexes(T, disk(R, degree - 1), disk(R, min(T.degrees) - 1))
    changes, inverses = {}, {}
    for n in raw.degrees:
        size = raw.rank(n)
        U, V = eye(R, size), eye(R, size)
        if size:
            unit = -R.variable(0) ** -1
            U[0, 0], V[0, 0] = unit, unit ** -1
            if size > 1:
                shift = R.variable(dimension - 1) ** -1 + 1
                U[0, size - 1], V[0, size - 1] = shift, -unit ** -1 * shift
        assert U * V == V * U == eye(R, size)
        changes[n], inverses[n] = U, V
    C = CochainComplex(R, raw.ranks, {
        n: changes.get(n + 1, eye(R, 0)) * raw.d(n) * inverses[n]
        for n in raw.degrees})
    # A Laurent polynomial congruent to 1 modulo the augmentation ideal
    # changes actual representatives by boundaries, without being a unit.
    scalar = 1 + (R.variable(0) ** -2 - 1) * (R.variable(1) - 1)
    assert not scalar.is_unit
    maps = {}
    for n in T.degrees:
        mixed = zeros(R, T.rank(n), T.rank(n))
        for index in range(0, T.rank(n), copies):
            if copies == 1:
                mixed[index, index] = scalar
            else:
                mixed[index, index + 1] = scalar
                mixed[index + 1, index] = -scalar
                mixed[index + 1, index + 1] = scalar
        maps[n] = changes[n][:, :T.rank(n)] * mixed
    phi = CochainMap(T, C, maps)
    layer = trivial_cohomology_layer(C, degree)
    assert layer.dimension == copies
    return phi, {degree: layer}


def _check_strong_in_original_coordinates(G, h):
    for n in G.degrees:
        hn = h.get(n, zeros(G.ring, G.rank(n - 1), G.rank(n)))
        following = h.get(n + 1, zeros(G.ring, G.rank(n), G.rank(n + 1)))
        assert hn.shape == (G.rank(n - 1), G.rank(n))
        assert G.d(n - 1) * hn + following * G.d(n) == eye(G.ring, G.rank(n))
        assert hn * following == zeros(G.ring, G.rank(n - 1), G.rank(n + 1))


@pytest.mark.parametrize("dimension", [2, 3, 4, 5])
@pytest.mark.parametrize("prime", [2, 3, 5])
def test_actual_perturbed_multicopy_gamma_has_complete_strong_sections(prime, dimension):
    phi, layers = _changed_comparison(prime, dimension)
    G, j = mapping_cone(phi), next(iter(layers))
    before = ({n: A.key() for n, A in phi.components.items()},
              {n: A.key() for n, A in phi.target.differentials.items()},
              layers[j].representatives.key())
    solver = toric_cone_solver(phi, layers, solve)
    assert solver is not None
    starts = []

    def observe(event):
        if event["event"] == "solve-start":
            starts.append(event["degree"])
            if event["degree"] == j - 1:
                bottom = event["R"][phi.target.rank(j - 1):, :]
                # At this degree target d is zero. Its augmentation must
                # vanish by the actual comparison's cohomology injection.
                assert not augmentation(bottom)

    h = solve_strong_contraction(G, solver, observer=observe)
    _check_strong_in_original_coordinates(G, h)
    assert starts == list(reversed(G.degrees))
    assert before == ({n: A.key() for n, A in phi.components.items()},
                      {n: A.key() for n, A in phi.target.differentials.items()},
                      layers[j].representatives.key())


@pytest.mark.parametrize("dimension,degree", [(3, 2), (4, 3), (5, 4)])
def test_high_form_top_degrees_keep_the_full_negative_part(dimension, degree):
    phi, layers = _changed_comparison(3, dimension, copies=1, degree=degree)
    G = mapping_cone(phi)
    h = solve_strong_contraction(G, toric_cone_solver(phi, layers, solve))
    _check_strong_in_original_coordinates(G, h)
    assert min(h) == min(G.degrees)
    assert max(h) == degree


@pytest.mark.parametrize("prime", [2, 3, 5])
def test_block_solver_lifts_actual_multi_rhs_equations_with_negative_exponents(prime):
    phi, layers = _changed_comparison(prime, 3)
    G, j, R = mapping_cone(phi), next(iter(layers)), phi.source.ring
    solver = toric_cone_solver(phi, layers, solve)
    for n in G.degrees:
        if n >= j:
            continue
        A = G.d(n - 1)
        witness = zeros(R, A.ncols, 3)
        if A.ncols:
            witness[0, 0] = R.variable(0) ** -3
            witness[A.ncols - 1, 1] = 2 * R.variable(1) ** -1
            witness[A.ncols // 2, 2] = R.variable(2) + R.variable(0) ** -1
        B = A * witness
        X = solver(A, B)
        assert X.ring == R
        assert X.shape == witness.shape
        assert A * X == B


@pytest.mark.parametrize("prime", [2, 3, 5])
def test_degree_below_top_does_not_mistake_zero_target_d_for_zero_augmentation(prime):
    R = Ring(prime, 3)
    T = koszul(R, 2)
    # This is deliberately NOT a quasi-isomorphism. At n=j-1 its cone has
    # cycles with nonzero target augmentation and no preimage.
    phi = CochainMap(T, T, {})
    layers = trivial_cohomology(T)
    G = mapping_cone(phi)
    n = 1
    B = zeros(R, G.rank(n), 1)
    B[T.rank(n), 0] = R.variable(0) ** -2
    assert not G.d(n) * B
    assert augmentation(B[T.rank(n):, :]) == eye(R, 1)
    backend = Mock(wraps=solve)
    adapted = toric_cone_solver(phi, layers, backend)
    assert adapted is not None
    with pytest.raises(NoLaurentSolution):
        adapted(G.d(n - 1), B)
    assert backend.called
    # This failure must go to the full original operator, not be reported
    # solely because the explicit target preimage could not be constructed.
    assert backend.call_args.args[0] == G.d(n - 1)
    assert backend.call_args.args[1] == B


@pytest.mark.parametrize("prime", [2, 3, 5])
def test_bottom_source_zero_rank_and_empty_unknowns_are_exact(prime):
    R = Ring(prime, 4)
    T = koszul(R, 2, copies=2)
    phi = identity_map(T)
    G = mapping_cone(phi)
    adapted = toric_cone_solver(phi, trivial_cohomology(T), solve)
    bottom = min(G.degrees)
    assert T.rank(bottom) == 0
    assert G.rank(bottom - 1) == 0
    B = zeros(R, G.rank(bottom), 2)
    X = adapted(G.d(bottom - 1), B)
    assert X == zeros(R, 0, 2)
    assert G.d(bottom - 1) * X == B
    n = bottom + 1
    assert T.rank(n - 1) == 0
    witness = Matrix(R, G.rank(n - 1), 2, {(0, 0): R.variable(3) ** -2, (1, 1): -1})
    B = G.d(n - 1) * witness
    X = adapted(G.d(n - 1), B)
    assert G.d(n - 1) * X == B
    assert X.shape == witness.shape


@pytest.mark.parametrize("kind", ["multilayer", "non-top-differential", "higher-term", "changed-target"])
def test_inapplicable_presentations_request_the_general_route(kind):
    R = Ring(3, 3)
    if kind == "multilayer":
        C = toric_stack(R, {1: 1, 2: 1})
        phi, layers = identity_map(C), trivial_cohomology(C)
    else:
        T = koszul(R, 1)
        if kind == "changed-target":
            changed = direct_sum_complexes(T, disk(R, -2))
            phi, layers = identity_map(changed), trivial_cohomology(changed)
        else:
            extra = disk(R, 1 if kind == "non-top-differential" else 2)
            C = direct_sum_complexes(T, extra)
            phi = CochainMap(T, C, {n: eye(R, T.rank(n)).vstack(
                zeros(R, C.rank(n) - T.rank(n), T.rank(n))) for n in T.degrees})
            layers = trivial_cohomology(C)
    forbidden = Mock(side_effect=AssertionError("inapplicable adapter invoked its solver"))
    assert toric_cone_solver(phi, layers, forbidden) is None
    assert not forbidden.called


def test_unrelated_left_operator_preserves_the_selected_backend():
    R = Ring(5, 2)
    T = koszul(R, 1)
    A = Matrix.from_rows(R, [[R.variable(0) ** -4]])
    B = Matrix.from_rows(R, [[R.variable(1), 2]])
    backend = Mock(wraps=solve)
    adapted = toric_cone_solver(identity_map(T), trivial_cohomology(T), backend)
    X = adapted(A, B)
    assert A * X == B
    backend.assert_called_once_with(A, B)


@pytest.mark.parametrize("kind", ["zero", "wrong-shape", "wrong-ring"])
def test_bad_source_block_solutions_do_not_become_gamma_witnesses(kind):
    R = Ring(5, 3)
    T = koszul(R, 2)
    phi, layers = identity_map(T), trivial_cohomology(T)
    G, n = mapping_cone(phi), 1
    Y = Matrix(R, G.rank(n - 1), 1, {(0, 0): R.variable(0) ** -1})
    B = G.d(n - 1) * Y
    assert B

    def bad(A, rhs):
        if kind == "wrong-shape":
            return zeros(R, A.ncols + 1, rhs.ncols)
        if kind == "wrong-ring":
            return zeros(Ring(3, 3), A.ncols, rhs.ncols)
        return zeros(R, A.ncols, rhs.ncols)

    adapted = toric_cone_solver(phi, layers, bad)
    with pytest.raises(ValueError):
        adapted(G.d(n - 1), B)


def test_target_lift_budget_falls_back_to_the_same_exact_full_operator():
    R = Ring(5, 2)
    T = koszul(R, 1)
    phi, layers = identity_map(T), trivial_cohomology(T)
    G, n = mapping_cone(phi), 0
    Y = Matrix(R, G.rank(n - 1), 1, {(T.rank(n - 1), 0): R.variable(0) ** -3})
    B = G.d(n - 1) * Y
    backend = Mock(wraps=solve)
    adapted = toric_cone_solver(phi, layers, backend, max_terms=1)
    X = adapted(G.d(n - 1), B)
    assert G.d(n - 1) * X == B
    assert any(call.args[0] == G.d(n - 1) and call.args[1] == B for call in backend.call_args_list)


def test_unresolved_backend_resource_stop_is_not_changed_to_nonmembership():
    R = Ring(5, 2)
    T = koszul(R, 1)
    phi, layers = identity_map(T), trivial_cohomology(T)
    G, n = mapping_cone(phi), 0
    Y = Matrix(R, G.rank(n - 1), 1, {(0, 0): R.variable(0) ** -1})
    B = G.d(n - 1) * Y
    failure = ResourceLimit("native backend deliberate resource cap")
    adapted = toric_cone_solver(phi, layers, Mock(side_effect=failure))
    with pytest.raises(ResourceLimit) as caught:
        adapted(G.d(n - 1), B)
    assert caught.value is failure


def test_failed_source_choice_can_still_lift_through_the_original_gamma_operator():
    R = Ring(3, 3)
    T = koszul(R, 2)
    phi, layers = identity_map(T), trivial_cohomology(T)
    G, n = mapping_cone(phi), 1
    Y = Matrix(R, G.rank(n - 1), 1, {(0, 0): R.variable(0) ** -1})
    B = G.d(n - 1) * Y
    attempts = []

    def limited_choice(A, rhs):
        attempts.append(A)
        if A == T.d(n - 1):
            raise NoLaurentSolution("test backend rejects this particular block lift")
        return solve(A, rhs)

    adapted = toric_cone_solver(phi, layers, limited_choice)
    X = adapted(G.d(n - 1), B)
    assert G.d(n - 1) * X == B
    assert attempts == [T.d(n - 1), G.d(n - 1)]
