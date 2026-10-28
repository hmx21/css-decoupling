"""Distinguish Appendix-C structure from automatic free-model choices."""

import pytest

from css_decoupling import Matrix, Ring, decouple_general as decouple
from css_decoupling.algebra.laurent import block, direct_sum, eye, zeros
from css_decoupling.algebra.modules import solve
from css_decoupling.certificates.projective import verify_projective_basis
from css_decoupling.decoupling.comparison import stable_cokernel_isomorphism
from css_decoupling.models.examples import toric_css
from css_decoupling.topology.complexes import (
    CochainComplex, CochainMap, direct_sum_complexes, koszul, mapping_cone,
)


def _split_injection_comparison(J, Q):
    """Package ker(Q) into the actual complement of a free-model comparison."""
    ring, e, o = J.ring, J.nrows, J.ncols
    assert Q * J == eye(ring, o)
    F = eye(ring, e) - J * Q
    A = CochainComplex(ring, {1: o, 2: e, 3: e, 4: o}, {1: J, 2: F, 3: Q})
    hA = {2: Q, 3: F, 4: J}
    for n in A.degrees:
        h = hA.get(n, zeros(ring, A.rank(n - 1), A.rank(n)))
        next_h = hA.get(n + 1, zeros(ring, A.rank(n), A.rank(n + 1)))
        assert A.d(n - 1) * h + next_h * A.d(n) == eye(ring, A.rank(n))
    T = koszul(ring, degree=1)
    C = direct_sum_complexes(T, A)
    phi = CochainMap(T, C, {
        n: eye(ring, T.rank(n)).vstack(zeros(ring, A.rank(n), T.rank(n)))
        for n in C.degrees})
    psi = CochainMap(C, T, {
        n: eye(ring, T.rank(n)).hstack(zeros(ring, T.rank(n), A.rank(n)))
        for n in C.degrees})
    hC = {n: direct_sum(zeros(ring, T.rank(n - 1), T.rank(n)),
                        hA.get(n, zeros(ring, A.rank(n - 1), A.rank(n))))
          for n in C.degrees}
    # The full cone, not merely a physical truncation, comes with an explicit
    # contraction. This verifies every extra datum used in Appendix C.
    G = mapping_cone(phi)
    hG = {n: block([[hC.get(n, zeros(ring, C.rank(n - 1), C.rank(n))),
                    zeros(ring, C.rank(n - 1), T.rank(n + 1))],
                   [psi.at(n), zeros(ring, T.rank(n), T.rank(n + 1))]])
          for n in G.degrees}
    for n in G.degrees:
        next_h = hG.get(n + 1, zeros(ring, G.rank(n), G.rank(n + 1)))
        assert G.d(n - 1) * hG[n] + next_h * G.d(n) == eye(ring, G.rank(n))
    return C, T, phi, psi, hC, F


def test_actual_appendix_complement_can_equal_a_no_unit_idempotent():
    # Use D=4 so the displayed positive model also lies within degrees <=D.
    ring = Ring(5, 4)
    x, y = ring.variables[:2]
    J = Matrix.from_rows(ring, [[x - 1], [y - 1], [3 - x - y]])
    Q = Matrix.from_rows(ring, [[1, 1, 1]])
    C, T, phi, psi, hC, F = _split_injection_comparison(J, Q)
    comparison = stable_cokernel_isomorphism(phi, psi, hC, {}, solve)
    assert comparison['idempotent'] == F
    assert not any(f.is_unit for f in F.entries.values())
    assert (comparison['free_C'], comparison['free_T']) == (2, 3)
    witness = comparison['trivialization']
    assert verify_projective_basis(F, witness['basis'], witness['coordinates'])
    # A only alters the freely chosen positive model: its physical checks are
    # just TC1 with an extra zero X check. This is not a raw-pipeline failure.
    base_X, base_Z = toric_css(ring, {1: 1})
    assert C.d(0) == base_X.vstack(zeros(ring, 1, base_X.ncols))
    assert C.d(-1).dagger() == base_Z


def test_general_qs_complement_feeds_the_actual_appendix_and_theta(monkeypatch):
    from css_decoupling.algebra.projective import free_basis
    from css_decoupling.decoupling.stabilization import LiftData, stabilize
    from css_decoupling.certificates.comparison import certificate_dict, verify_certificate

    ring = Ring(3, 4)
    x, y = ring.variables[:2]
    a, b = x - 1, y - 1
    J = Matrix.from_rows(ring, [[b*b], [1+a*b], [a*a-a]])
    Q = Matrix.from_rows(ring, [[a*a, 1-a*b, 0]])
    C, T, phi, psi, hC, E = _split_injection_comparison(J, Q)
    trace = []

    def forced_general(F, **options):
        assert F == E
        return free_basis(F, method='quillen-suslin',
                          image_resolution=lambda _: [F, J], qs_options={'trace': trace})

    monkeypatch.setattr('css_decoupling.algebra.projective.free_basis', forced_general)
    comparison = stable_cokernel_isomorphism(phi, psi, hC, {}, solve)
    witness = comparison['trivialization']
    assert (comparison['free_C'], comparison['free_T']) == (2, 3)
    assert witness['odd_free_rank'] == 0
    assert witness['general_constructions'][0]['cancellation']['cancelled_free_rank'] == 1
    assert trace and len(trace[0]['patch_denominators']) == 2
    assert verify_projective_basis(E, witness['basis'], witness['coordinates'])
    lifts = LiftData(**{name: comparison[name] for name in
                       ('dC', 'dT', 'phi0', 'psi0', 'phi1', 'psi1', 'etaC', 'etaT')})
    # Independent serialized check includes the actual SUM/SWAP gate product.
    assert verify_certificate(certificate_dict(stabilize(lifts))).valid


@pytest.mark.integration
@pytest.mark.parametrize('scramble', [False, True])
def test_raw_check_pipeline_computes_nonzero_complement_after_redundant_mixing(scramble, monkeypatch):
    ring = Ring(5, 2)
    x, y = ring.variables
    values = [x*x - 1, y*y - 1, x*y - 1, x + y - 1,
              5 - x*x - y*y - x*y - x - y]
    v = Matrix.from_rows(ring, [[f] for f in values])
    assert sum(values, ring.zero()) == 1
    if scramble:
        B = eye(ring, 5)
        for i, j, f in [(0, 1, x+y-2), (1, 2, x*y-1), (2, 3, x*x+y-2),
                        (3, 4, x-y), (4, 0, x**-1+y)]:
            B = (eye(ring, 5) + Matrix(ring, 5, 5, {(i, j): f})) * B
        v = B * v
    # This real pipeline family has a terminal third degree. Its contraction
    # must supply the Appendix-C basis before a redundant projective syzygy
    # computation can cause support explosion.
    def no_generic_basis(*args, **kwargs):
        raise AssertionError('The terminal cone contraction already supplies the basis')
    monkeypatch.setattr('css_decoupling.algebra.projective.free_basis', no_generic_basis)
    HX, HZ = toric_css(ring, {1: 1})
    result = decouple(v * HX, HZ, backend='sympy', resolution_timeout=20, comparison_method='free-basis')
    comparison = result.trace['comparison']
    basis = comparison['trivialization']
    assert result.multiplicities == {1: 1}
    assert result.free_T == result.trace['restricted'].rank(2)
    assert result.free_C == basis['rank']
    assert basis['steps'][0]['method'] == 'terminal-contraction'
    assert verify_projective_basis(comparison['idempotent'], basis['basis'], basis['coordinates'])
