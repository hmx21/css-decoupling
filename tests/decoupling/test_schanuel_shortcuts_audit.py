"""Independent equations for reused Gamma contractions and target preimages."""
from copy import deepcopy

import pytest

from css_decoupling.algebra.bounded_linear import augmentation
from css_decoupling.algebra.laurent import Matrix, Ring, block, direct_sum, eye, zeros
from css_decoupling.algebra.modules import GeneralBackend
from css_decoupling.certificates.schanuel import verify_schanuel
from css_decoupling.decoupling.comparison import paper_stable_cokernel_isomorphism
from css_decoupling.decoupling.schanuel import stable_complement_from_cone
from css_decoupling.decoupling.toric_comparison import (
    cone_contraction_from_inverse, toric_comparison_inverse,
)
from css_decoupling.errors import ResourceLimit
from css_decoupling.topology.complexes import (
    CochainComplex, CochainMap, direct_sum_complexes, disk, identity_map,
    koszul, mapping_cone, validate_homotopy,
)
from css_decoupling.topology.trivial_cohomology import trivial_cohomology


def _comparison(prime=3, dimension=4, degree=2, *, perturb=False):
    R = Ring(prime, dimension)
    T = koszul(R, degree)
    extra = direct_sum_complexes(disk(R, 2), disk(R, -2))
    raw = direct_sum_complexes(T, extra)
    change, inverse = {}, {}
    for n in raw.degrees:
        shear = zeros(R, raw.rank(n), raw.rank(n))
        if raw.rank(n) > 1:
            shear[0, raw.rank(n) - 1] = R.variable(0) ** -1 + 2 * R.variable(1)
        change[n], inverse[n] = eye(R, raw.rank(n)) + shear, eye(R, raw.rank(n)) - shear
    C = CochainComplex(R, raw.ranks, {
        n: change.get(n + 1, eye(R, raw.rank(n + 1))) * raw.d(n) * inverse[n]
        for n in raw.degrees})
    phi = CochainMap(T, C, {
        n: change[n] * eye(R, T.rank(n)).vstack(zeros(R, extra.rank(n), T.rank(n)))
        for n in C.degrees})
    psi = CochainMap(C, T, {
        n: eye(R, T.rank(n)).hstack(zeros(R, T.rank(n), extra.rank(n))) * inverse[n]
        for n in C.degrees})
    hC = {}
    for n in C.degrees:
        h_extra = eye(R, 1) if n in (-1, 3) else zeros(R, extra.rank(n - 1), extra.rank(n))
        hC[n] = change.get(n - 1, eye(R, C.rank(n - 1))) * direct_sum(
            zeros(R, T.rank(n - 1), T.rank(n)), h_extra) * inverse[n]
    if perturb:
        # Add delta(a), a degree -2 map. This changes hC without changing
        # d*hC+hC*d, and exercises contractions not already square-zero.
        a = Matrix(R, C.rank(0), C.rank(2), {
            (0, 0): R.variable(0) ** -1 + R.variable(1),
            (0, C.rank(2) - 1): 1})
        hC[1] = hC[1] - a * C.d(1)
        hC[2] = hC[2] + C.d(0) * a
    hT, contraction = cone_contraction_from_inverse(phi, psi, hC, {})
    return phi, psi, hC, hT, contraction


def _check_original_cone_frame(phi, frame):
    C, T, R = phi.target, phi.source, phi.source.ring
    low = min((0, *C.degrees, *T.degrees))
    ranks = {n: (C.rank(n) if low <= n <= 1 else 0) +
             ((T.rank(n + 1) + (C.rank(2) if n == 0 else 0))
              if low <= n + 1 <= 1 else 0)
             for n in range(low - 1, 2)}
    cone, h = frame['cone'], frame['contraction']
    assert cone.ranks == ranks
    assert frame['even_free_rank'] == sum(rank for n, rank in ranks.items() if n % 2 == 0)
    assert frame['odd_free_rank'] == sum(rank for n, rank in ranks.items() if n % 2)
    assert frame['idempotent']  # Q is actually nonzero in these examples.
    section, projection = frame['section'], frame['projection']
    for n in cone.degrees:
        following = h.get(n + 1, zeros(R, cone.rank(n), cone.rank(n + 1)))
        previous = h.get(n - 1, zeros(R, cone.rank(n - 2), cone.rank(n - 1)))
        P = section * projection if n == 0 else zeros(R, cone.rank(n), cone.rank(n))
        assert cone.d(n - 1) * h[n] + following * cone.d(n) == eye(R, cone.rank(n)) - P
        assert not previous * h[n]
    assert not h[0] * section
    assert not projection * h[1]
    assert frame['to_free'] * frame['from_free'] == eye(R, frame['even_free_rank'])
    assert frame['from_free'] * frame['to_free'] == direct_sum(
        frame['idempotent'], eye(R, frame['odd_free_rank']))


@pytest.mark.parametrize('prime,dimension,degree,perturb', [
    (2, 2, 1, False), (3, 4, 2, False), (5, 5, 3, False),
    (3, 4, 2, True), (5, 4, 2, True),
])
def test_supplied_contraction_keeps_nonzero_Q_and_all_original_parity_ranks(
        prime, dimension, degree, perturb):
    phi, psi, hC, hT, H = _comparison(prime, dimension, degree, perturb=perturb)

    # A legacy weak witness is checked, then the original C4 cone is split
    # afresh. It is not normalized or frozen as higher-degree data.
    backend = GeneralBackend(phi.source.ring)
    frame = stable_complement_from_cone(phi, psi, hC, hT, backend.solve,
                                       comparison_contraction=H, strategy="direct")
    _check_original_cone_frame(phi, frame)
    backend = GeneralBackend(phi.source.ring)
    result = paper_stable_cokernel_isomorphism(
        phi, psi, hC, hT, backend.solve, comparison_contraction=H,
        contraction_strategy="direct")
    cancelled = len(result['unit_cancellation']['directions'])
    assert result['free_C'] == frame['even_free_rank'] - cancelled
    assert result['free_T'] == phi.target.rank(2) + frame['odd_free_rank'] - cancelled
    assert verify_schanuel(result).valid


@pytest.mark.parametrize('mutation', ['zero-top', 'missing-top', 'other-Gamma'])
def test_supplied_contraction_tamper_is_rejected_against_the_actual_Gamma_cone(mutation):
    phi, psi, hC, hT, H = _comparison()
    bad = {n: value.copy() for n, value in H.items()}
    top = max(mapping_cone(phi).degrees)
    if mutation == 'zero-top':
        bad[top] = zeros(phi.source.ring, *bad[top].shape)
    elif mutation == 'missing-top':
        del bad[top]
    else:
        x = phi.source.ring.variable(0)
        other_phi = CochainMap(phi.source, phi.target, {n: x * phi.at(n) for n in phi.degrees})
        other_psi = CochainMap(psi.source, psi.target, {n: x ** -1 * psi.at(n) for n in psi.degrees})
        _, bad = cone_contraction_from_inverse(other_phi, other_psi, hC, hT)
        other_cone = mapping_cone(other_phi)
        assert validate_homotopy(other_cone, other_cone, identity_map(other_cone),
                                CochainMap(other_cone, other_cone, {}), bad)
    with pytest.raises(ValueError):
        stable_complement_from_cone(phi, psi, hC, hT, GeneralBackend(phi.source.ring).solve,
                                    comparison_contraction=bad)


def test_explicit_target_preimages_are_checked_on_all_actual_lift_equations():
    phi, psi, hC, hT, H = _comparison(5, 4, 2, perturb=True)
    backend, calls = GeneralBackend(phi.source.ring), []

    def target(rhs):
        calls.append(rhs)
        assert rhs.nrows == phi.source.rank(1)
        return backend.solve(phi.source.d(0), rhs)

    result = paper_stable_cokernel_isomorphism(phi, psi, hC, hT, backend.solve,
        comparison_contraction=H, target_preimage=target)
    assert len(calls) == 2
    assert verify_schanuel(result).valid

    def false_target(rhs):
        return zeros(rhs.ring, phi.source.rank(0), rhs.ncols)

    with pytest.raises(ValueError, match='Explicit toric lift'):
        paper_stable_cokernel_isomorphism(phi, psi, hC, hT, backend.solve,
            comparison_contraction=H, target_preimage=false_target)

    def exhausted(rhs):
        raise ResourceLimit('target preimage budget')

    with pytest.raises(ResourceLimit, match='target preimage budget'):
        paper_stable_cokernel_isomorphism(phi, psi, hC, hT, backend.solve,
            comparison_contraction=H, target_preimage=exhausted)


@pytest.mark.parametrize('prime,dimension,degree', [(2, 3, 1), (3, 4, 2), (5, 3, 1)])
def test_top_augmentation_inverse_allows_copy_mixing_and_boundary_changes(prime, dimension, degree):
    R = Ring(prime, dimension)
    T = koszul(R, degree, copies=2)
    x = R.variable(0)
    mixed = Matrix.from_rows(R, [[0, 1], [-1, 1]]) + (x - 1) * eye(R, 2)
    components = {n: direct_sum(*(mixed for _ in range(T.rank(n) // 2)))
                  if T.rank(n) else zeros(R, 0, 0) for n in T.degrees}
    phi = CochainMap(T, T, components)
    layers = trivial_cohomology(T)
    top = layers[degree].quotient.projection * phi.at(degree)
    assert top != eye(R, 2)
    assert augmentation(top) != eye(R, 2)
    inverse = toric_comparison_inverse(phi, layers, GeneralBackend(R).solve)
    assert inverse is not None
    validate_homotopy(T, T, identity_map(T), CochainMap(T, T, {
        n: phi.at(n) * inverse.psi.at(n) for n in T.degrees}), inverse.hC)
    validate_homotopy(T, T, identity_map(T), CochainMap(T, T, {
        n: inverse.psi.at(n) * phi.at(n) for n in T.degrees}), inverse.hT)
    cone = mapping_cone(phi)
    validate_homotopy(cone, cone, identity_map(cone), CochainMap(cone, cone, {}),
                      inverse.cone_contraction)
    singular = CochainMap(T, T, {n: (x - 1) * eye(R, T.rank(n)) for n in T.degrees})
    assert toric_comparison_inverse(singular, layers, GeneralBackend(R).solve) is None
