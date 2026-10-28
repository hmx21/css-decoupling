"""Known C4 sections retain every equation, resource fallback and stable term."""
import json
from unittest.mock import Mock

import pytest

from css_decoupling.algebra.laurent import Matrix, Ring, direct_sum, eye, zeros
from css_decoupling.algebra.modules import GeneralBackend
from css_decoupling.certificates.schanuel import (
    schanuel_certificate_dict, verify_schanuel,
)
from css_decoupling.decoupling.comparison import paper_stable_cokernel_isomorphism
from css_decoupling.topology.complexes import (
    CochainComplex, CochainMap, direct_sum_complexes, disk, identity_map,
    koszul, validate_homotopy,
)
from css_decoupling.topology.direct_splitting import (
    observe_direct_splitting, solve_strong_contraction,
)


def _weak_comparison(prime, dimension):
    """A changed Laurent presentation with nonzero Q and non-square-zero hC."""
    R = Ring(prime, dimension)
    T = koszul(R, min(2, dimension - 1))
    extra = direct_sum_complexes(disk(R, 2), disk(R, -2))
    raw = direct_sum_complexes(T, extra)
    change, inverse = {}, {}
    for n in raw.degrees:
        shear = zeros(R, raw.rank(n), raw.rank(n))
        if raw.rank(n) > 1:
            shear[0, -1] = R.variable(0) ** -1 + 2 * R.variable(1)
        change[n] = eye(R, raw.rank(n)) + shear
        inverse[n] = eye(R, raw.rank(n)) - shear
        assert change[n] * inverse[n] == eye(R, raw.rank(n))
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
    # Add delta(a); this preserves dhC+hCd but intentionally spoils hC^2=0.
    a = Matrix(R, C.rank(0), C.rank(2), {
        (0, 0): R.variable(0) ** -1 + R.variable(1),
        (0, C.rank(2) - 1): 1})
    hC[1] = hC[1] - a * C.d(1)
    hC[2] = hC[2] + C.d(0) * a
    # psi*phi=I; psi*hC*phi is a generally nonzero homotopy of zero.
    hT = {n: psi.at(n - 1) * hC.get(n, zeros(R, C.rank(n - 1), C.rank(n))) * phi.at(n)
          for n in T.degrees}
    validate_homotopy(C, C, identity_map(C), CochainMap(C, C, {
        n: phi.at(n) * psi.at(n) for n in C.degrees}), hC)
    validate_homotopy(T, T, identity_map(T), CochainMap(T, T, {
        n: psi.at(n) * phi.at(n) for n in T.degrees}), hT)
    assert any(hC.get(n - 1, zeros(R, C.rank(n - 2), C.rank(n - 1))) * hn
               for n, hn in hC.items())
    return phi, psi, hC, hT


def _assert_strong_equalities(C, h, s=None, q=None):
    """Check the original matrices independently of the construction API."""
    R = C.ring
    for n in C.degrees:
        hn = h.get(n, zeros(R, C.rank(n - 1), C.rank(n)))
        following = h.get(n + 1, zeros(R, C.rank(n), C.rank(n + 1)))
        P = s * q if n == 0 and s is not None else zeros(R, C.rank(n), C.rank(n))
        assert C.d(n - 1) * hn + following * C.d(n) == eye(R, C.rank(n)) - P
        assert not hn * following
    if s is not None:
        assert not h.get(0, zeros(R, C.rank(-1), C.rank(0))) * s
        assert not q * h.get(1, zeros(R, C.rank(0), C.rank(1)))


@pytest.mark.parametrize("dimension", (2, 3, 4, 5))
@pytest.mark.parametrize("prime", (2, 3, 5))
def test_schanuel_known_top_accepts_weak_inverse_and_retains_full_certificate(prime, dimension):
    phi, psi, hC, hT = _weak_comparison(prime, dimension)
    C, R = phi.target, phi.target.ring
    events = []
    with observe_direct_splitting(events.append):
        result = paper_stable_cokernel_isomorphism(
            phi, psi, hC, hT, GeneralBackend(R).solve, contraction_strategy="direct")
    accepted = [event for event in events
                if event["event"] == "solve-complete"
                and event.get("strategy") == "verified-known-section"]
    assert len(accepted) == 1 and accepted[0]["degree"] == 1
    assert accepted[0]["h"] == hC[1].vstack(psi.at(1), C.d(1))
    frame = result["schanuel"]
    cone = frame["cone"]
    assert frame["idempotent"]  # Q is genuinely nonzero.
    assert frame["even_free_rank"] == sum(cone.rank(n) for n in cone.degrees if n % 2 == 0)
    assert frame["odd_free_rank"] == sum(cone.rank(n) for n in cone.degrees if n % 2)
    cancelled = len(result["unit_cancellation"]["directions"])
    assert result["free_C"] == frame["even_free_rank"] - cancelled
    assert result["free_T"] == C.rank(2) + frame["odd_free_rank"] - cancelled
    _assert_strong_equalities(cone, frame["contraction"], frame["section"], frame["projection"])
    serialized = json.loads(json.dumps(schanuel_certificate_dict(result)))
    checked = verify_schanuel(serialized)
    assert checked.valid, checked.errors


def _violating_seed(R, condition):
    H = CochainComplex(R, {0: 1}, {})
    if condition in ("left", "projection"):
        C = direct_sum_complexes(disk(R, 0), H)
        s = Matrix(R, 2, 1, {(1, 0): 1})
        q = s.transpose()
        X = (zeros(R, 2, 1) if condition == "left"
             else Matrix(R, 2, 1, {(0, 0): 1, (1, 0): 1}))
        if condition == "projection":
            assert C.d(0) * X == eye(R, 1) and q * X
        else:
            assert C.d(0) * X != eye(R, 1) and not q * X
        return C, {1: X}, dict(section=s, projection=q, idempotent=eye(R, 1))
    if condition == "section":
        C = direct_sum_complexes(disk(R, -2), disk(R, -1), H)
        s = Matrix(R, 2, 1, {(1, 0): 1})
        q = s.transpose()
        X = Matrix(R, 2, 2, {(1, 0): 1, (0, 1): 1})
        assert C.d(-1) * X == eye(R, 2) - s * q and X * s
        return C, {0: X}, dict(section=s, projection=q, idempotent=eye(R, 1))
    C = direct_sum_complexes(disk(R, -1), disk(R, 0), disk(R, 1))
    following = Matrix(R, 2, 1, {(1, 0): 1})
    X = Matrix(R, 2, 2, {(1, 0): 1, (0, 1): 1})
    assert C.d(0) * X == eye(R, 2) - following * C.d(1) and X * following
    return C, {2: following, 1: X}, {}


@pytest.mark.parametrize("prime", (2, 3, 5))
@pytest.mark.parametrize("condition", ("left", "following", "section", "projection"))
def test_known_sections_reject_each_original_constraint(prime, condition):
    C, seeds, options = _violating_seed(Ring(prime, 3), condition)
    backend = Mock(side_effect=AssertionError("Invalid seed must not reach the backend"))
    with pytest.raises(ValueError, match="Known section violates an original constraint"):
        solve_strong_contraction(C, backend, initial_h=seeds, **options)
    backend.assert_not_called()


@pytest.mark.parametrize("kind", ("lower-only", "gap", "bool", "outside", "ring", "shape"))
def test_known_sections_reject_malformed_prefix_or_matrix(kind):
    R = Ring(3, 3)
    C = direct_sum_complexes(disk(R, -1), disk(R, 0))
    top = Matrix(R, 2, 1, {(1, 0): 1})
    malformed = {
        "lower-only": {0: zeros(R, 1, 2)},
        "gap": {1: top, -1: zeros(R, 0, 1)},
        "bool": {True: top},
        "outside": {2: zeros(R, 1, 0)},
        "ring": {1: zeros(Ring(5, 3), 2, 1)},
        "shape": {1: zeros(R, 1, 1)},
    }[kind]
    backend = Mock(side_effect=AssertionError("Malformed seed must not reach the backend"))
    with pytest.raises(ValueError):
        solve_strong_contraction(C, backend, initial_h=malformed)
    backend.assert_not_called()


@pytest.mark.parametrize("prime", (2, 3, 5))
@pytest.mark.parametrize("budget", ("support", "product"))
def test_resource_declined_seed_clears_dependent_lower_seed_and_falls_back(prime, budget):
    R = Ring(prime, 3)
    f = sum((R.variable(0) ** i for i in range(20)), R.zero())
    if budget == "support":
        C = direct_sum_complexes(disk(R, -1), disk(R, 0))
        large = Matrix(R, 2, 1, {(0, 0): f, (1, 0): 1})
        caps = dict(max_terms=8)
    else:
        C = CochainComplex(R, {-1: 1, 0: 2, 1: 1}, {
            -1: Matrix(R, 2, 1, {(0, 0): 1, (1, 0): -1}),
            0: Matrix(R, 1, 2, {(0, 0): 1, (0, 1): 1})})
        large = Matrix(R, 2, 1, {(0, 0): f, (1, 0): 1 - f})
        caps = dict(max_product_work=5)
    assert C.d(0) * large == eye(R, 1)  # It is valid, merely unaffordable.
    events = []
    backend = Mock(wraps=GeneralBackend(R).solve)
    h = solve_strong_contraction(C, backend,
        initial_h={1: large, 0: zeros(R, 1, 2)}, observer=events.append, **caps)
    assert backend.call_count > 0
    assert [e["degree"] for e in events if e["event"] == "known-section-declined"] == [1]
    assert not any(e["event"] == "solve-complete" and e.get("strategy") == "verified-known-section"
                   for e in events)
    assert h[0]  # The supplied zero lower seed was discarded, not accepted.
    _assert_strong_equalities(C, h)
