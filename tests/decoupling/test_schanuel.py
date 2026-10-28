"""C4 cone construction, retaining stable summands without any QS call."""
import pytest

from css_decoupling.algebra.laurent import Matrix, Ring, direct_sum, eye, zeros
from css_decoupling.algebra.modules import solve
from css_decoupling.certificates.schanuel import verify_schanuel
from css_decoupling.decoupling.comparison import paper_stable_cokernel_isomorphism
from css_decoupling.decoupling.stabilization import LiftData, stabilize
from css_decoupling.errors import ResourceLimit
from css_decoupling.topology.complexes import (
    CochainComplex, CochainMap, direct_sum_complexes, disk, koszul,
)


def comparison_example(prime=3, dimension=3, *, mix=False, extra_negative=False,
                       source_disk=False, no_unit=False, toric_degree=1):
    R = Ring(prime, dimension)
    base = koszul(R, toric_degree)
    if no_unit:
        x, y = R.variables[:2]
        J = Matrix.from_rows(R, [[x - 1], [y - 1], [3 - x - y]])
        Q = Matrix.from_rows(R, [[1, 1, 1]])
        F = eye(R, 3) - J * Q
        A = CochainComplex(R, {1: 1, 2: 3, 3: 3, 4: 1}, {1: J, 2: F, 3: Q})
        hA = {2: Q, 3: F, 4: J}
    else:
        A = disk(R, 2)
        hA = {3: eye(R, 1)}
    if extra_negative:
        A0 = A
        A = direct_sum_complexes(A, disk(R, -2))
        hA = {n: direct_sum(hA.get(n, zeros(R, A0.rank(n - 1), A0.rank(n))),
                            eye(R, 1) if n == -1 else zeros(
                                R, int(n - 1 in (-2, -1)), int(n in (-2, -1))))
              for n in A.degrees}
    B = disk(R, 1) if source_disk else CochainComplex(R, {}, {})
    T = direct_sum_complexes(base, B)
    C0 = direct_sum_complexes(base, A)
    changes, inverses = {}, {}
    for n in C0.degrees:
        r = C0.rank(n)
        shear = Matrix(R, r, r, {(0, r - 1): R.variable(0) ** -1 + R.variable(1)}) if mix and r > 1 else zeros(R, r, r)
        changes[n], inverses[n] = eye(R, r) + shear, eye(R, r) - shear
    C = CochainComplex(R, C0.ranks, {
        n: changes.get(n + 1, eye(R, C0.rank(n + 1))) * C0.d(n) * inverses[n]
        for n in C0.degrees})
    phi = CochainMap(T, C, {
        n: changes.get(n, eye(R, C.rank(n))) * direct_sum(
            eye(R, base.rank(n)), zeros(R, A.rank(n), B.rank(n))) for n in set(C.degrees) | set(T.degrees)})
    psi = CochainMap(C, T, {
        n: direct_sum(eye(R, base.rank(n)), zeros(R, B.rank(n), A.rank(n))) *
           inverses.get(n, eye(R, C.rank(n))) for n in set(C.degrees) | set(T.degrees)})
    hc = {n: changes.get(n - 1, eye(R, C.rank(n - 1))) * direct_sum(
        zeros(R, base.rank(n - 1), base.rank(n)),
        hA.get(n, zeros(R, A.rank(n - 1), A.rank(n)))) * inverses[n] for n in C.degrees}
    ht = {n: direct_sum(zeros(R, base.rank(n - 1), base.rank(n)),
                       eye(R, 1) if source_disk and n == 2 else zeros(R, B.rank(n - 1), B.rank(n)))
          for n in T.degrees}
    return phi, psi, hc, ht


@pytest.mark.parametrize('prime', [2, 3, 5])
@pytest.mark.parametrize('dimension', [2, 3, 4, 5])
def test_retained_stable_complement_all_primes_dimensions(prime, dimension):
    phi, psi, hc, ht = comparison_example(prime, dimension)
    result = paper_stable_cokernel_isomorphism(phi, psi, hc, ht, solve)
    frame = result['schanuel']
    assert frame['idempotent'] == eye(phi.source.ring, 1)
    assert frame['odd_free_rank'] > 0
    cancelled = len(result['unit_cancellation']['directions'])
    assert result['free_T'] == phi.target.rank(2) + frame['odd_free_rank'] - cancelled
    assert result['free_C'] == frame['even_free_rank'] - cancelled
    assert 'basis' not in frame
    assert verify_schanuel(result).valid
    LiftData(**{name: result[name] for name in LiftData.__dataclass_fields__}).validate()


@pytest.mark.parametrize('prime', [2, 3, 5])
def test_full_negative_terms_and_both_homotopies_with_laurent_mixing(prime):
    phi, psi, hc, ht = comparison_example(prime, 3, mix=True, extra_negative=True, source_disk=True)
    result = paper_stable_cokernel_isomorphism(phi, psi, hc, ht, solve)
    assert min(result['schanuel']['cone'].degrees) == -3
    assert verify_schanuel(result).valid


def test_nonunit_complement_uses_only_existing_complex(monkeypatch):
    import css_decoupling.algebra.projective as projective
    import css_decoupling.algebra.resolutions as resolutions
    import css_decoupling.algebra.quillen_suslin as qs
    import css_decoupling.topology.resolutions as top_resolutions
    def forbidden(*args, **kwargs):
        raise AssertionError('C4 must not find a Q basis or a new Q resolution')
    monkeypatch.setattr(projective, 'free_basis', forbidden)
    monkeypatch.setattr(resolutions, 'image_resolution', forbidden)
    monkeypatch.setattr(top_resolutions, 'image_resolution', forbidden)
    for name in ('complete_unimodular_row', 'quillen_suslin_basis'):
        if hasattr(qs, name):
            monkeypatch.setattr(qs, name, forbidden)
    phi, psi, hc, ht = comparison_example(5, 4, no_unit=True)
    result = paper_stable_cokernel_isomorphism(phi, psi, hc, ht, solve)
    assert not any(f.is_unit for f in result['idempotent'].entries.values())
    assert result['schanuel']['odd_free_rank'] > 0
    assert verify_schanuel(result).valid


def test_stable_cokernel_feeds_sum_swap_construction():
    from css_decoupling.certificates.comparison import verify_certificate
    phi, psi, hc, ht = comparison_example(3, 2, mix=True)
    result = paper_stable_cokernel_isomorphism(phi, psi, hc, ht, solve)
    lifts = LiftData(**{name: result[name] for name in LiftData.__dataclass_fields__})
    assert verify_certificate(stabilize(lifts)).valid


def test_empty_complex_and_zero_complement():
    R = Ring(3, 2)
    C = CochainComplex(R, {}, {})
    phi = CochainMap(C, C, {})
    result = paper_stable_cokernel_isomorphism(phi, phi, {}, {}, solve)
    assert result['free_C'] == result['free_T'] == 0
    assert verify_schanuel(result).valid


def test_invalid_homotopy_and_solver_are_rejected():
    phi, psi, hc, ht = comparison_example()
    with pytest.raises(ValueError, match='homotopy'):
        paper_stable_cokernel_isomorphism(phi, psi, {}, ht, solve)
    with pytest.raises(ValueError):
        paper_stable_cokernel_isomorphism(phi, psi, hc, ht,
                                         lambda A, B: zeros(A.ring, A.ncols, B.ncols))


def test_resource_failure_propagates_as_unfinished():
    phi, psi, hc, ht = comparison_example()
    def exhausted(A, B):
        raise ResourceLimit('test backend budget')
    with pytest.raises(ResourceLimit, match='budget'):
        paper_stable_cokernel_isomorphism(phi, psi, hc, ht, exhausted)


@pytest.mark.parametrize('dimension,degree', [(4, 2), (5, 3)])
def test_higher_toric_excitation_layers(dimension, degree):
    phi, psi, hc, ht = comparison_example(3, dimension, toric_degree=degree, mix=True)
    result = paper_stable_cokernel_isomorphism(phi, psi, hc, ht, solve)
    assert phi.source.rank(2) > 0
    assert result['schanuel']['odd_free_rank'] > 0
    assert verify_schanuel(result).valid


@pytest.mark.parametrize('prime', [3, 5])
def test_direct_sections_keep_projective_retract_and_original_parity(prime, monkeypatch):
    """C4 must directly split the original cone, without the old weak retract."""
    import css_decoupling.decoupling.schanuel as schanuel
    phi, psi, hc, ht = comparison_example(prime, 3, mix=True,
                                         extra_negative=True, source_disk=True)
    def forbidden(*args, **kwargs):
        raise AssertionError('Old weak-contraction/normalization route was used')
    monkeypatch.setattr(schanuel, '_contract_around_projector', forbidden)
    frame = schanuel.stable_complement_from_cone(phi, psi, hc, ht, solve, strategy="direct")
    cone, R, h = frame['cone'], phi.target.ring, frame['contraction']
    s, q = frame['section'], frame['projection']
    for n in cone.degrees:
        following = h.get(n+1, zeros(R, cone.rank(n), cone.rank(n+1)))
        previous = h.get(n-1, zeros(R, cone.rank(n-2), cone.rank(n-1)))
        P = s*q if n == 0 else zeros(R, cone.rank(n), cone.rank(n))
        assert cone.d(n-1)*h[n] + following*cone.d(n) == eye(R, cone.rank(n))-P
        assert not previous*h[n]
    assert not h[0]*s and not q*h[1]
    assert frame['to_free'] * frame['from_free'] == eye(R, frame['even_free_rank'])
    assert frame['from_free'] * frame['to_free'] == direct_sum(
        frame['idempotent'], eye(R, frame['odd_free_rank']))
