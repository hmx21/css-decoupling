"""Strong C4 frames from normalized weak contractions, with direct optional."""
import json

import pytest

from css_decoupling.algebra.laurent import Matrix, Ring, direct_sum, eye, zeros
from css_decoupling.algebra.modules import solve
from css_decoupling.certificates.schanuel import schanuel_certificate_dict, verify_schanuel
from css_decoupling.decoupling.comparison import paper_stable_cokernel_isomorphism
from css_decoupling.decoupling import schanuel
from css_decoupling.decoupling.toric_comparison import cone_contraction_from_inverse
from css_decoupling.errors import ResourceLimit
from css_decoupling.topology.complexes import (
    CochainComplex, CochainMap, direct_sum_complexes, disk, identity_map,
    koszul, mapping_cone, validate_homotopy,
)


def _comparison(prime, dimension):
    """A Laurent-mixed comparison with nonzero Q and an extra negative disk."""
    R = Ring(prime, dimension)
    T = koszul(R, min(2, dimension - 1))
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
    hT, H = cone_contraction_from_inverse(phi, psi, hC, {})
    return phi, psi, hC, hT, H


def _check_frame(phi, frame):
    C, T, R = phi.target, phi.source, phi.source.ring
    low = min((0, *C.degrees, *T.degrees))
    ranks = {n: (C.rank(n) if low <= n <= 1 else 0) +
             ((T.rank(n + 1) + (C.rank(2) if n == 0 else 0))
              if low <= n + 1 <= 1 else 0)
             for n in range(low - 1, 2)}
    cone, h = frame['cone'], frame['contraction']
    s, q, E = frame['section'], frame['projection'], frame['idempotent']
    assert E
    assert cone.ranks == ranks
    assert frame['even_free_rank'] == sum(r for n, r in ranks.items() if n % 2 == 0)
    assert frame['odd_free_rank'] == sum(r for n, r in ranks.items() if n % 2)
    for n in cone.degrees:
        following = h.get(n + 1, zeros(R, cone.rank(n), cone.rank(n + 1)))
        previous = h.get(n - 1, zeros(R, cone.rank(n - 2), cone.rank(n - 1)))
        P = s * q if n == 0 else zeros(R, cone.rank(n), cone.rank(n))
        assert cone.d(n - 1) * h[n] + following * cone.d(n) == eye(R, cone.rank(n)) - P
        assert not previous * h[n]
    assert not h[0] * s and not q * h[1]
    assert frame['to_free'] * frame['from_free'] == eye(R, frame['even_free_rank'])
    assert frame['from_free'] * frame['to_free'] == direct_sum(E, eye(R, frame['odd_free_rank']))


@pytest.mark.parametrize('prime', [2, 3, 5])
@pytest.mark.parametrize('dimension', [2, 3, 4, 5])
@pytest.mark.parametrize('supplied', [False, True])
def test_normalized_keeps_full_strong_frame(prime, dimension, supplied):
    phi, psi, hC, hT, H = _comparison(prime, dimension)
    frame = schanuel.stable_complement_from_cone(
        phi, psi, hC, hT, solve, comparison_contraction=H if supplied else None,
        strategy='normalized')
    _check_frame(phi, frame)


def _weak_gamma(phi, H):
    """Add a degree-minus-two Hom boundary without changing dH+Hd."""
    G, R = mapping_cone(phi), phi.source.ring
    a = Matrix(R, G.rank(0), G.rank(2), {(0, 0): R.variable(0) ** -1 + R.variable(1)})
    weak = {n: value.copy() for n, value in H.items()}
    weak[1] = weak[1] - a * G.d(1)
    weak[2] = weak[2] + G.d(0) * a
    validate_homotopy(G, G, identity_map(G), CochainMap(G, G, {}), weak)
    assert any(weak.get(n - 1, zeros(R, G.rank(n - 2), G.rank(n - 1))) * weak[n]
               for n in G.degrees)
    return weak


@pytest.mark.parametrize('prime', [2, 3, 5])
def test_genuinely_weak_gamma_is_normalized_before_transfer(prime):
    phi, psi, hC, hT, H = _comparison(prime, 3)
    frame = schanuel.stable_complement_from_cone(
        phi, psi, hC, hT, solve, comparison_contraction=_weak_gamma(phi, H))
    _check_frame(phi, frame)


def test_normalization_cannot_be_replaced_by_an_unchecked_weak_witness(monkeypatch):
    phi, psi, hC, hT, H = _comparison(3, 3)
    weak = _weak_gamma(phi, H)
    monkeypatch.setattr(schanuel, '_contracting_sandwich', lambda H, d: H.copy())
    with pytest.raises(ArithmeticError, match='square-zero'):
        schanuel.stable_complement_from_cone(
            phi, psi, hC, hT, solve, comparison_contraction=weak)


@pytest.mark.parametrize('strategy', ['normalized', 'direct'])
def test_strategy_uses_its_selected_construction(strategy, monkeypatch):
    from css_decoupling.topology import direct_splitting
    phi, psi, hC, hT, H = _comparison(5, 3)

    def forbidden(*args, **kwargs):
        raise AssertionError('Unselected Schanuel strategy executed')

    if strategy == 'normalized':
        monkeypatch.setattr(direct_splitting, 'solve_strong_contraction', forbidden)
        options = {}  # Also establish the low-level default, not just explicit dispatch.
    else:
        monkeypatch.setattr(schanuel, '_contracting_sandwich', forbidden)
        monkeypatch.setattr(schanuel, '_contract_around_projector', forbidden)
        options = {'strategy': strategy}
    frame = schanuel.stable_complement_from_cone(
        phi, psi, hC, hT, solve, comparison_contraction=H, **options)
    _check_frame(phi, frame)


@pytest.mark.parametrize('strategy', ['normalized', 'direct'])
def test_supplied_gamma_tamper_is_not_a_fallback(strategy):
    phi, psi, hC, hT, H = _comparison(3, 3)
    top = max(n for n, value in H.items() if value)
    bad = dict(H)
    bad[top] = zeros(phi.source.ring, *H[top].shape)
    with pytest.raises(ValueError):
        schanuel.stable_complement_from_cone(
            phi, psi, hC, hT, solve, comparison_contraction=bad, strategy=strategy)


@pytest.mark.parametrize('strategy', ['normalized', 'direct'])
def test_backend_budget_propagates_without_switching_strategy(strategy):
    phi, psi, hC, hT, _ = _comparison(3, 3)

    def exhausted(A, B):
        raise ResourceLimit('explicit test backend budget')

    with pytest.raises(ResourceLimit, match='explicit test backend budget'):
        schanuel.stable_complement_from_cone(phi, psi, hC, hT, exhausted, strategy=strategy)


@pytest.mark.parametrize('strategy', ['', 'auto', 'legacy_combined', None, True, []])
def test_invalid_strategy_fails_before_algebra(strategy):
    with pytest.raises(ValueError, match='Schanuel strategy'):
        schanuel.stable_complement_from_cone(None, None, None, None, None, strategy=strategy)


def test_normalized_frame_serializes_under_unchanged_schema_and_rejects_tamper():
    phi, psi, hC, hT, H = _comparison(3, 3)
    result = paper_stable_cokernel_isomorphism(
        phi, psi, hC, hT, solve, comparison_contraction=_weak_gamma(phi, H))
    saved = json.loads(json.dumps(schanuel_certificate_dict(result)))
    assert saved['version'] == 2
    assert verify_schanuel(saved).valid
    saved['ranks']['odd'] += 1
    assert not verify_schanuel(saved).valid


@pytest.mark.parametrize("prime,dimension", [(2, 3), (3, 3), (5, 4)])
def test_unit_cancellation_shrinks_frame_and_keeps_both_inverses(prime, dimension):
    phi, psi, hC, hT, H = _comparison(prime, dimension)
    full = paper_stable_cokernel_isomorphism(phi, psi, hC, hT, solve,
                                             comparison_contraction=_weak_gamma(phi, H), cancel_units=False)
    reduced = paper_stable_cokernel_isomorphism(phi, psi, hC, hT, solve,
                                                comparison_contraction=_weak_gamma(phi, H))
    frame, cut = full['schanuel'], reduced['unit_cancellation']
    r = len(cut['directions'])
    assert full['unit_cancellation']['directions'] == []
    assert full['free_C'] == frame['even_free_rank'] and full['free_T'] == phi.target.rank(2) + frame['odd_free_rank']
    assert 0 < r <= frame['odd_free_rank'] and len(set(cut['directions'])) == r
    assert (reduced['free_C'], reduced['free_T']) == (full['free_C'] - r, full['free_T'] - r)
    R, E = phi.source.ring, frame['idempotent']
    assert cut['to_free'] * cut['from_free'] == eye(R, reduced['free_C'])
    assert cut['from_free'] * cut['to_free'] == direct_sum(E, eye(R, frame['odd_free_rank'] - r))
    for result in (full, reduced):
        saved = json.loads(json.dumps(schanuel_certificate_dict(result)))
        assert verify_schanuel(saved).valid


def test_unit_cancelled_certificate_rejects_tamper_and_reads_version_one():
    from css_decoupling.certificates.schanuel import CONVENTIONS
    phi, psi, hC, hT, H = _comparison(3, 3)
    reduced = paper_stable_cokernel_isomorphism(phi, psi, hC, hT, solve,
                                                comparison_contraction=_weak_gamma(phi, H))
    saved = json.loads(json.dumps(schanuel_certificate_dict(reduced)))
    for change in ("claim", "matrix", "direction"):
        bad = json.loads(json.dumps(saved))
        if change == "claim":
            bad['ranks']['cancelled'] += 1
            bad['ranks']['free_C'] -= 1
            bad['ranks']['free_T'] -= 1
        elif change == "matrix":
            bad['unit_cancellation']['to_free'] = bad['frame']['to_free']
        else:
            bad['unit_cancellation']['directions'][0] = bad['unit_cancellation']['directions'][-1] + 10 ** 6
        assert not verify_schanuel(bad).valid
    full = paper_stable_cokernel_isomorphism(phi, psi, hC, hT, solve,
                                             comparison_contraction=_weak_gamma(phi, H), cancel_units=False)
    old = json.loads(json.dumps(schanuel_certificate_dict(full)))
    old.pop('unit_cancellation')
    old['ranks'].pop('cancelled')
    old['version'], old['conventions'] = 1, dict(CONVENTIONS)
    assert verify_schanuel(old).valid
