"""Sign, homotopy, and stable-projective checks independent of examples."""

import pytest

from css_decoupling.algebra.laurent import Matrix, Ring, direct_sum, eye, zeros
from css_decoupling.decoupling.comparison import (
    stable_cokernel_isomorphism, stably_trivialize_idempotent,
)
from css_decoupling.topology.complexes import (
    CochainComplex, CochainMap, direct_sum_complexes, disk, homotopy_inverse,
    homotopy_inverse_window, identity_map, koszul, koszul_window, mapping_cone,
    solve_contraction, solve_matrix_equations, toric_stack, toric_window,
)
from css_decoupling.lattice.retained import ResourceLimit
from css_decoupling.algebra.modules import NoLaurentSolution, solve
from css_decoupling.topology.resolutions import image_resolution
from css_decoupling.decoupling.stabilization import LiftData, stabilize


@pytest.mark.parametrize("dimension,prime", [(2, 2), (3, 3), (4, 5), (5, 3)])
def test_koszul_odd_characteristic_signs_and_every_degree(dimension, prime):
    from math import comb
    ring = Ring(prime, dimension)
    for degree in range(1, dimension):
        C = koszul(ring, degree, copies=2)
        assert C.rank(0) == 2 * comb(dimension, degree)
        assert C.rank(1) == 2 * comb(dimension, degree - 1)
        assert C.validate()
        # The wedge e_0 ^ e_1 maps to (x_0-1)e_1 - (x_1-1)e_0.
        d2 = C.d(degree - 2)
        assert d2[0, 0] == -(ring.variable(1) - 1)
        assert d2[2, 0] == ring.variable(0) - 1


def test_multilayer_stack_and_negative_degrees():
    ring = Ring(5, 4)
    C = toric_stack(ring, {1: 2, 3: 1})
    assert C.rank(0) == 12
    assert min(C.degrees) == -3
    assert max(C.degrees) == 3
    assert C.validate()
    assert toric_stack(ring, {}).rank(0) == 0


def test_cone_contracts_a_laurent_basis_changed_equivalence():
    ring = Ring(3, 2)
    T = koszul(ring)
    B = Matrix(ring, 2, 2, {(0, 0): 1, (1, 1): 1,
                            (0, 1): ring.variable(0) ** -1 + ring.variable(1)})
    Binv = 2 * eye(ring, 2) - B
    changes = {n: B if n == 0 else eye(ring, T.rank(n)) for n in T.degrees}
    inverses = {n: Binv if n == 0 else eye(ring, T.rank(n)) for n in T.degrees}
    C = CochainComplex(ring, T.ranks, {
        n: changes.get(n + 1, zeros(ring, 0, 0)) * T.d(n) * inverses[n]
        for n in T.degrees})
    phi = CochainMap(T, C, changes)
    psi, hc, ht = homotopy_inverse(phi, solve)
    assert psi.at(0) == Binv
    assert mapping_cone(phi).validate()
    assert set(hc) == set(C.degrees)
    assert set(ht) == set(T.degrees)


def test_non_quasi_isomorphism_is_rejected_by_contraction():
    ring = Ring(3, 2)
    C = CochainComplex(ring, {1: 1}, {})
    with pytest.raises(NoLaurentSolution):
        solve_contraction(C, solve)


def test_simultaneous_matrix_equations_have_no_polynomial_degree_cutoff():
    ring = Ring(5, 2)
    x = ring.variable(0)
    wanted = Matrix(ring, 1, 2, {(0, 0): x ** -11, (0, 1): x ** 13 + 2})
    B = Matrix(ring, 2, 2, {(0, 0): 1, (1, 1): 1, (0, 1): x})
    answer = solve_matrix_equations(ring, {"X": (1, 2), "Y": (1, 2)}, [
        ([(eye(ring, 1), "X", B), (-eye(ring, 1), "Y", eye(ring, 2))], zeros(ring, 1, 2)),
        ([(eye(ring, 1), "X", eye(ring, 2))], wanted),
    ], solve)
    assert answer['X'] == wanted
    assert answer['Y'] == wanted * B


def test_projective_stabilization_uses_resolution_without_a_free_basis():
    ring = Ring(5, 2)
    x, y = ring.variables
    P = Matrix(ring, 2, 2, {(0, 0): 1, (0, 1): x,
                           (1, 0): y, (1, 1): 1 + x * y})
    Pinv = Matrix(ring, 2, 2, {(0, 0): 1 + x * y, (0, 1): -x,
                              (1, 0): -y, (1, 1): 1})
    E = P * Matrix(ring, 2, 2, {(0, 0): 1}) * Pinv
    # Deliberately retain two redundant generators for a rank-one image.
    # The parity construction must handle the nonzero first syzygy term.
    result = stably_trivialize_idempotent(E, [E, P[:, 1:]], solve)
    assert result['odd_free_rank'] == 1
    assert result['even_free_rank'] == 2
    assert result['from_free'] * result['to_free'] == direct_sum(E, eye(ring, 1))
    assert result['to_free'] * result['from_free'] == eye(ring, 2)


def test_false_projective_resolution_is_not_trusted():
    ring = Ring(3, 2)
    E = Matrix(ring, 2, 2, {(0, 0): 1})
    with pytest.raises(NoLaurentSolution):
        stably_trivialize_idempotent(E, [E], solve)


def test_koszul_window_avoids_full_exterior_algebra_allocation():
    ring = Ring(3, 32)
    with pytest.raises(ResourceLimit):
        koszul(ring, degree=1, max_total_rank=1000)
    C = koszul_window(ring, degree=1, max_total_rank=1000)
    assert sum(C.ranks.values()) == 1 + 32 + 496
    assert min(C.degrees) == -1
    comparison = toric_window(ring, {1: 2}, lowest_degree=0, max_total_rank=100)
    assert sum(comparison.ranks.values()) == 66
    with pytest.raises(ResourceLimit):
        toric_window(ring, {1: 2}, max_total_rank=1000)


def test_window_comparison_retains_physical_boundary_without_degree_zero_inverse():
    ring = Ring(3, 4)
    Tfull = koszul_window(ring, degree=1)
    Cfull = direct_sum_complexes(Tfull, disk(ring, -1), disk(ring, 1))

    def positive(C):
        return CochainComplex(ring, {n: r for n, r in C.ranks.items() if n >= 0},
                              {n: d for n, d in C.differentials.items() if n >= 0})

    C, T = positive(Cfull), positive(Tfull)
    phi = CochainMap(T, C, {
        n: eye(ring, T.rank(n)).vstack(zeros(ring, C.rank(n) - T.rank(n), T.rank(n)))
        for n in C.degrees})
    psi, hc, ht = homotopy_inverse_window(phi, physical_boundary=Cfull.d(-1), solve=solve)
    assert 0 not in hc and 0 not in ht
    result = stable_cokernel_isomorphism(phi, psi, hc, ht, solve, image_resolution, window=True)
    LiftData(**{name: result[name] for name in LiftData.__dataclass_fields__}).validate()
    # The added Z-fixed physical ancilla creates degree-zero cohomology in
    # Cpos.  No homotopy inverse in that degree exists or is being asserted.
    with pytest.raises(ValueError, match='homotopy in degree 0'):
        stable_cokernel_isomorphism(phi, psi, hc, ht, solve, image_resolution)
    with pytest.raises(NoLaurentSolution):
        homotopy_inverse_window(phi, physical_boundary=zeros(ring, C.rank(0), 0), solve=solve)


@pytest.mark.parametrize('prime', [2, 3, 5])
def test_appendix_c_stable_comparison_with_redundant_checks_and_disks(prime):
    ring = Ring(prime, 3)
    T = koszul(ring, degree=2)
    C0 = direct_sum_complexes(T, disk(ring, 1), disk(ring, 2), disk(ring, 0))
    # Mix the added free directions with toric coordinates in every degree.
    changes, inverses = {}, {}
    for n in C0.degrees:
        r = C0.rank(n)
        shear = Matrix(ring, r, r, {(0, r - 1): ring.variable(0) ** -1}) if r > 1 else zeros(ring, r, r)
        changes[n], inverses[n] = eye(ring, r) + shear, eye(ring, r) - shear
    C = CochainComplex(ring, C0.ranks, {
        n: changes.get(n + 1, zeros(ring, 0, 0)) * C0.d(n) * inverses[n]
        for n in C0.degrees})
    phi = CochainMap(T, C, {
        n: changes[n] * eye(ring, T.rank(n)).vstack(
            zeros(ring, C.rank(n) - T.rank(n), T.rank(n)))
        for n in C.degrees})
    psi, hc, ht = homotopy_inverse(phi, solve)
    result = stable_cokernel_isomorphism(phi, psi, hc, ht, solve, image_resolution)
    lifts = LiftData(**{name: result[name] for name in LiftData.__dataclass_fields__})
    lifts.validate()
    stable = stabilize(lifts)
    assert stable.dThat * stable.U == stable.V * stable.dChat
    assert stable.Uinv * stable.U == eye(ring, stable.U.ncols)
    assert result['free_C'] >= 0 and result['free_T'] >= 0
