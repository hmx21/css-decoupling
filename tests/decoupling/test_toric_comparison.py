"""Explicit Koszul target inverses retain full source and cone identities."""
import pytest

from css_decoupling.algebra.laurent import Ring, eye, zeros
from css_decoupling.algebra.modules import solve
from css_decoupling.decoupling.toric_comparison import (
    toric_comparison_inverse, cone_contraction_from_inverse,
)
from css_decoupling.topology.complexes import (
    CochainComplex, CochainMap, direct_sum_complexes, disk, homotopy_inverse,
    identity_map, mapping_cone, toric_stack, validate_homotopy,
)
from css_decoupling.topology.trivial_cohomology import trivial_cohomology


def changed_comparison(prime, dimension, degree, copies):
    R = Ring(prime, dimension)
    T = toric_stack(R, {degree: copies})
    raw = direct_sum_complexes(T, disk(R, degree - 1), disk(R, min(T.degrees) - 1))
    U, V = {}, {}
    for n in raw.degrees:
        U[n] = eye(R, raw.rank(n))
        if n != degree and raw.rank(n) >= 2:
            U[n][0, 0] = R.variable(0)**-1
            U[n][0, -1] = 2 + R.variable(dimension - 1)
        V[n] = solve(U[n], eye(R, raw.rank(n)))
    C = CochainComplex(R, raw.ranks, {
        n: U.get(n + 1, zeros(R, 0, 0)) * raw.d(n) * V[n] for n in raw.degrees})
    phi = CochainMap(T, C, {n: U[n][:, :T.rank(n)] for n in C.degrees})
    return phi, trivial_cohomology(C)


@pytest.mark.parametrize("p,D,j,t", [(2, 3, 1, 1), (3, 4, 2, 2), (5, 4, 1, 2)])
def test_target_preimages_give_full_inverse_homotopies_and_cone(p, D, j, t):
    phi, layers = changed_comparison(p, D, j, t)
    inverse = toric_comparison_inverse(phi, layers, solve)
    assert inverse is not None
    C, T = phi.target, phi.source
    validate_homotopy(C, C, identity_map(C), CochainMap(C, C, {
        n: phi.at(n) * inverse.psi.at(n) for n in C.degrees}), inverse.hC)
    validate_homotopy(T, T, identity_map(T), CochainMap(T, T, {
        n: inverse.psi.at(n) * phi.at(n) for n in T.degrees}), inverse.hT)
    G = mapping_cone(phi)
    validate_homotopy(G, G, identity_map(G), CochainMap(G, G, {}), inverse.cone_contraction)


@pytest.mark.parametrize("p", [2, 3, 5])
def test_cone_alignment_formula_checks_a_separately_constructed_inverse(p):
    phi, _ = changed_comparison(p, 3, 1, 1)
    psi, hC, hT = homotopy_inverse(phi, solve)
    aligned, contraction = cone_contraction_from_inverse(phi, psi, hC, hT)
    G = mapping_cone(phi)
    assert validate_homotopy(G, G, identity_map(G), CochainMap(G, G, {}), contraction)
    assert set(aligned) == set(phi.source.degrees)


def test_nonapplicable_multilayer_or_unmatched_projection_is_not_assumed():
    R = Ring(3, 3)
    C = toric_stack(R, {1: 1, 2: 1})
    assert toric_comparison_inverse(identity_map(C), trivial_cohomology(C), solve) is None
    phi, layers = changed_comparison(3, 3, 1, 1)
    layers[1].quotient.projection = zeros(R, 1, phi.target.rank(1))
    assert toric_comparison_inverse(phi, layers, solve) is None


def test_incorrect_source_homotopy_solver_is_rejected():
    phi, layers = changed_comparison(3, 3, 1, 1)
    with pytest.raises(ArithmeticError, match="Source comparison"):
        toric_comparison_inverse(phi, layers, lambda A, B: zeros(A.ring, A.ncols, B.ncols))


@pytest.mark.parametrize("p", [2, 3, 5])
def test_top_quotient_inverse_allows_boundary_changes_and_mixed_copies(p):
    R = Ring(p, 3)
    C = toric_stack(R, {2: 2})
    # This polynomial need not be a Laurent unit, but its augmentation is
    # one. It changes representatives by boundaries and induces an iso.
    q = 1 + (R.variable(0) - 1) * (R.variable(1) - 1)
    maps = {}
    for n in C.degrees:
        A = zeros(R, C.rank(n), C.rank(n))
        for i in range(0, C.rank(n), 2):
            A[i, i + 1] = q
            A[i + 1, i] = -q
            A[i + 1, i + 1] = q
        maps[n] = A
    phi = CochainMap(C, C, maps)
    inverse = toric_comparison_inverse(phi, trivial_cohomology(C), solve)
    assert inverse is not None
    G = mapping_cone(phi)
    assert validate_homotopy(G, G, identity_map(G), CochainMap(G, G, {}),
                            inverse.cone_contraction)


def test_top_map_with_singular_augmentation_requests_general_route():
    R = Ring(3, 3)
    C = toric_stack(R, {1: 1})
    phi = CochainMap(C, C, {n: (R.variable(0) - 1) * eye(R, C.rank(n))
                            for n in C.degrees})
    assert toric_comparison_inverse(phi, trivial_cohomology(C), solve) is None
