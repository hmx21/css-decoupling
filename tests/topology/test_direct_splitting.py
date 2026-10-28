"""Independent exact checks for descending C4 sections, including bad witnesses.

These finite examples test implementation behavior; they are not an existence
proof or an acceptance substitute for either four-dimensional color code.
"""
import json
from unittest.mock import Mock

import pytest

from css_decoupling.algebra.bounded_linear import BoundedLinearBackend
from css_decoupling.algebra.laurent import Matrix, Ring, eye, zeros
from css_decoupling.algebra.modules import solve
from css_decoupling.certificates.comparison import _matrix_dict, _matrix_from_dict
from css_decoupling.errors import NoLaurentSolution, ResourceLimit
from css_decoupling.topology.complexes import (
    CochainComplex, direct_sum_complexes, disk,
)
from css_decoupling.topology.direct_splitting import (
    observe_direct_splitting, solve_strong_contraction, validate_strong_contraction,
)


def _unchanged(C):
    return dict(C.ranks), {n: value.key() for n, value in C.differentials.items()}


def _check_actual_equalities(C, h, s=None, q=None, e=None):
    """Check original matrices directly, without invoking either tested API."""
    R = C.ring
    for n in range(min(C.degrees, default=0) - 1,
                   max(C.degrees, default=0) + 2):
        hn = h.get(n, zeros(R, C.rank(n - 1), C.rank(n)))
        hnext = h.get(n + 1, zeros(R, C.rank(n), C.rank(n + 1)))
        assert hn.ring == R
        assert hn.shape == (C.rank(n - 1), C.rank(n))
        P = s * q if n == 0 and s is not None else zeros(R, C.rank(n), C.rank(n))
        assert C.d(n - 1) * hn + hnext * C.d(n) == eye(R, C.rank(n)) - P
        assert hn * hnext == zeros(R, C.rank(n - 1), C.rank(n + 1))
    if s is not None:
        assert q * s == e
        assert e * e == e
        assert s * e == s
        assert e * q == q
        assert C.d(0) * s == zeros(R, C.rank(1), s.ncols)
        assert q * C.d(-1) == zeros(R, q.nrows, C.rank(-1))
        assert h.get(0, zeros(R, C.rank(-1), C.rank(0))) * s == zeros(R, C.rank(-1), s.ncols)
        assert q * h.get(1, zeros(R, C.rank(0), C.rank(1))) == zeros(R, q.nrows, C.rank(1))


def _basis_changed_disks(R, with_Q):
    """An independently specified split complex using every Laurent variable."""
    pieces = [disk(R, n) for n in range(-R.D, R.D)]
    if with_Q:
        pieces.append(CochainComplex(R, {0: 1}, {}))
    original = direct_sum_complexes(*pieces)
    changes, inverses = {}, {}
    for n in original.degrees:
        rank = original.rank(n)
        unit = -R.variable(n % R.D) ** (-1 if n % 2 else 2)
        D, Dinv = eye(R, rank), eye(R, rank)
        D[0, 0], Dinv[0, 0] = unit, unit ** -1
        S, Sinv = eye(R, rank), eye(R, rank)
        if rank > 1:
            a = R.variable(0) ** -2 + sum(
                ((-1) ** i) * R.variable(i) for i in range(R.D))
            S[0, rank - 1], Sinv[0, rank - 1] = a, -a
        changes[n], inverses[n] = S * D, Dinv * Sinv
        assert changes[n] * inverses[n] == eye(R, rank)
        assert inverses[n] * changes[n] == eye(R, rank)
    C = CochainComplex(R, original.ranks, {
        n: changes.get(n + 1, eye(R, 0)) * original.d(n) * inverses[n]
        for n in original.degrees})
    if not with_Q:
        return C, None, None, None
    # The same rank-one Q uses two redundant coordinates with a non-diagonal
    # idempotent. No free basis of Q is supplied to the implementation.
    t, u = R.variable(0) ** -1 + R.variable(1), R.variable(0) + R.variable(1) ** -1
    v = Matrix.from_rows(R, [[1], [t]])
    w = Matrix.from_rows(R, [[1 - u * t, u]])
    e = v * w
    assert w * v == eye(R, 1)
    i = Matrix(R, original.rank(0), 1, {(original.rank(0) - 1, 0): 1})
    r = Matrix(R, 1, original.rank(0), {(0, original.rank(0) - 1): 1})
    s, q = changes[0] * i * w, v * r * inverses[0]
    return C, s, q, e


@pytest.mark.parametrize("dimension", [2, 3, 4, 5])
@pytest.mark.parametrize("prime", [2, 3, 5])
@pytest.mark.parametrize("with_Q", [False, True])
def test_descending_sections_all_dimensions_and_primes(dimension, prime, with_Q):
    C, s, q, e = _basis_changed_disks(Ring(prime, dimension), with_Q)
    before = _unchanged(C)
    h = solve_strong_contraction(C, solve, section=s, projection=q, idempotent=e)
    _check_actual_equalities(C, h, s, q, e)
    assert validate_strong_contraction(C, h, section=s, projection=q, idempotent=e)
    assert _unchanged(C) == before
    assert any(any(k < 0 for k in exponent)
               for A in C.differentials.values() for f in A.entries.values() for exponent in f.terms)


@pytest.mark.parametrize("prime", [2, 3, 5])
def test_exact_complex_without_any_laurent_unit_entry(prime):
    R = Ring(prime, 4)
    x = R.variable(0)
    a, b = x + 1, x * x + x + 1
    # b - x*a = 1 in every characteristic. Thus this three-term complex
    # has a handwritten contraction, even though its entries are not units.
    C = CochainComplex(R, {-1: 1, 0: 2, 1: 1}, {
        -1: Matrix.from_rows(R, [[-b], [a]]),
        0: Matrix.from_rows(R, [[a, b]]),
    })
    assert all(len(f.terms) > 1 for A in C.differentials.values() for f in A.entries.values())
    known = {0: Matrix.from_rows(R, [[-1, -x]]), 1: Matrix.from_rows(R, [[-x], [1]])}
    _check_actual_equalities(C, known)
    result = solve_strong_contraction(C, solve)
    _check_actual_equalities(C, result)


def test_non_square_zero_valid_lift_is_not_frozen_for_later_degrees(monkeypatch):
    # Exercise the retained weak/joint fallback when the positive quotient
    # strategy is unavailable; quotient-first otherwise avoids this weak X.
    monkeypatch.setattr("css_decoupling.topology.direct_splitting._right_annihilator_by_units",
                        lambda *args: None)
    R = Ring(5, 3)
    C = direct_sum_complexes(disk(R, 0), disk(R, 1), disk(R, 2))
    injected = []

    def noncanonical_solver(A, B):
        candidate = solve(A, B)
        if A == C.d(1) and B.shape == (2, 2) and not injected:
            # This extra map from the top disk to the bottom disk is killed
            # by d, so it passes A*X=B but breaks X*hnext=0.
            candidate[0, 1] = 1
            assert A * candidate == B
            assert candidate * Matrix.from_rows(R, [[0], [1]]) != zeros(R, 2, 1)
            injected.append(candidate)
        return candidate

    h = solve_strong_contraction(C, noncanonical_solver)
    assert injected
    _check_actual_equalities(C, h)


def test_validator_rejects_a_genuine_weak_contraction_with_nonzero_square():
    R = Ring(3, 2)
    C = direct_sum_complexes(disk(R, 0), disk(R, 1), disk(R, 2))
    weak = {
        1: Matrix.from_rows(R, [[1, 0]]),
        2: Matrix.from_rows(R, [[0, 1], [1, 0]]),
        3: Matrix.from_rows(R, [[0], [1]]),
    }
    for n in C.degrees:
        hn = weak.get(n, zeros(R, C.rank(n - 1), C.rank(n)))
        hnext = weak.get(n + 1, zeros(R, C.rank(n), C.rank(n + 1)))
        assert C.d(n - 1) * hn + hnext * C.d(n) == eye(R, C.rank(n))
    assert weak[1] * weak[2] != zeros(R, 1, 2)
    with pytest.raises(ValueError):
        validate_strong_contraction(C, weak)


@pytest.mark.parametrize("ranks", [{}, {-4: 0, 0: 0, 5: 0}])
def test_empty_complex_and_explicit_empty_degrees(ranks):
    R = Ring(3, 5)
    C = CochainComplex(R, ranks, {})
    h = solve_strong_contraction(C, solve)
    _check_actual_equalities(C, h)
    assert validate_strong_contraction(C, h)


def test_only_degree_zero_projective_homology_needs_no_nonzero_homotopy():
    R = Ring(5, 2)
    C = CochainComplex(R, {0: 1}, {})
    s = Matrix.from_rows(R, [[1, R.variable(0) ** -1]])
    q = Matrix.from_rows(R, [[1], [0]])
    e = q * s
    h = solve_strong_contraction(C, solve, section=s, projection=q, idempotent=e)
    _check_actual_equalities(C, h, s, q, e)
    assert all(not value for value in h.values())


@pytest.mark.parametrize("degree", [-2, 0, 2])
def test_unaccounted_homology_is_not_a_successful_contraction(degree):
    R = Ring(3, 3)
    C = direct_sum_complexes(disk(R, 1), CochainComplex(R, {degree: 1}, {}))
    with pytest.raises(NoLaurentSolution):
        solve_strong_contraction(C, solve)


def test_projective_coordinates_identifying_only_part_of_homology_are_rejected():
    R = Ring(5, 4)
    C = CochainComplex(R, {0: 2}, {})
    s, q, e = Matrix.from_rows(R, [[1], [0]]), Matrix.from_rows(R, [[1, 0]]), eye(R, 1)
    assert q * s == e
    with pytest.raises(NoLaurentSolution):
        solve_strong_contraction(C, solve, section=s, projection=q, idempotent=e)


@pytest.mark.parametrize("supplied", ["section", "projection", "idempotent"])
def test_partial_projective_data_is_rejected(supplied):
    R = Ring(3, 2)
    C = CochainComplex(R, {0: 1}, {})
    with pytest.raises(ValueError):
        solve_strong_contraction(C, solve, **{supplied: eye(R, 1)})


@pytest.mark.parametrize("kind", ["wrong-ring", "wrong-shape", "wrong-idempotent", "non-idempotent"])
def test_bad_projective_coordinate_data(kind):
    R = Ring(5, 2)
    C = CochainComplex(R, {0: 1}, {})
    s, q = Matrix.from_rows(R, [[1, 0]]), Matrix.from_rows(R, [[1], [0]])
    e = q * s
    if kind == "wrong-ring":
        s = Matrix.from_rows(Ring(3, 2), [[1, 0]])
    elif kind == "wrong-shape":
        q = eye(R, 2)
    elif kind == "wrong-idempotent":
        e = eye(R, 2)
    else:
        q = 2 * q
        e = q * s
        assert e * e != e
    with pytest.raises(ValueError):
        solve_strong_contraction(C, solve, section=s, projection=q, idempotent=e)
    with pytest.raises(ValueError):
        validate_strong_contraction(C, {}, section=s, projection=q, idempotent=e)


@pytest.mark.parametrize("kind", ["section-not-cycle", "projection-not-cocycle", "se-not-s", "eq-not-q"])
def test_projective_equations_beyond_qs_equal_e_are_enforced(kind):
    R = Ring(5, 3)
    C = direct_sum_complexes(disk(R, -1), disk(R, 0), CochainComplex(R, {0: 1}, {}))
    s = Matrix.from_rows(R, [[0, 0], [0, 0], [1, 0]])
    q = Matrix.from_rows(R, [[0, 0, 1], [0, 0, 0]])
    e = q * s
    if kind == "section-not-cycle":
        s[1, 0] = 1
    elif kind == "projection-not-cocycle":
        q[0, 0] = 1
    elif kind == "se-not-s":
        s[0, 1] = 1
    else:
        q[1, 1] = 1
    assert q * s == e
    assert e * e == e
    with pytest.raises(ValueError):
        solve_strong_contraction(C, solve, section=s, projection=q, idempotent=e)


@pytest.mark.parametrize("kind", ["zero", "wrong-shape", "wrong-ring"])
def test_false_solver_results_are_rejected(kind):
    R = Ring(3, 2)
    C = disk(R, 0)

    def false_solver(A, B):
        if kind == "wrong-shape":
            return zeros(R, A.ncols + 1, B.ncols)
        if kind == "wrong-ring":
            return eye(Ring(5, 2), 1)
        return zeros(R, A.ncols, B.ncols)

    with pytest.raises(ValueError):
        solve_strong_contraction(C, false_solver)


@pytest.mark.parametrize("kind", ["changed-entry", "missing-component", "wrong-shape", "wrong-ring", "extra-degree"])
def test_serializable_homotopy_tampering_is_rejected(kind):
    R = Ring(5, 2)
    C = disk(R, 0)
    h = {1: eye(R, 1)}
    assert validate_strong_contraction(C, h)
    if kind == "changed-entry":
        h[1][0, 0] = R.variable(0) ** -1
    elif kind == "missing-component":
        h.clear()
    elif kind == "wrong-shape":
        h[1] = zeros(R, 2, 1)
    elif kind == "wrong-ring":
        h[1] = eye(Ring(3, 2), 1)
    else:
        h[7] = eye(R, 1)
    with pytest.raises(ValueError):
        validate_strong_contraction(C, h)


def test_cannot_supply_an_old_unnormalized_homotopy_as_a_frozen_prefix():
    C = disk(Ring(3, 2), 0)
    with pytest.raises(TypeError):
        solve_strong_contraction(C, solve, homotopy={1: eye(C.ring, 1)})


def test_backend_resource_failure_remains_incomplete():
    C = disk(Ring(3, 2), 0)
    failure = ResourceLimit("finite support budget exhausted; mathematical answer unknown")
    with pytest.raises(ResourceLimit) as caught:
        solve_strong_contraction(C, Mock(side_effect=failure))
    assert caught.value is failure


def test_bounded_backend_cap_is_not_relabelled_as_nonmembership():
    R = Ring(3, 2)
    x = R.variable(0)
    a, b = x + 1, x * x + x + 1
    C = CochainComplex(R, {-1: 1, 0: 2, 1: 1}, {
        -1: Matrix.from_rows(R, [[-b], [a]]), 0: Matrix.from_rows(R, [[a, b]])})
    backend = BoundedLinearBackend(R, max_degree=0, max_unknowns=0)
    with pytest.raises(ResourceLimit):
        solve_strong_contraction(C, backend.solve)


@pytest.mark.parametrize("limit", ["max_unknowns", "max_terms"])
def test_direct_splitting_resource_preflight(limit):
    C, s, q, e = _basis_changed_disks(Ring(3, 2), True)
    with pytest.raises(ResourceLimit):
        solve_strong_contraction(C, solve, section=s, projection=q, idempotent=e, **{limit: 1})


@pytest.mark.parametrize("limit", ["max_unknowns", "max_terms"])
@pytest.mark.parametrize("value", [True, -1, 1.5, "10"])
def test_invalid_resource_caps_are_rejected(limit, value):
    with pytest.raises(ValueError):
        solve_strong_contraction(disk(Ring(3, 2), 0), solve, **{limit: value})


@pytest.mark.parametrize("prime", [2, 3, 5])
def test_dagger_reflection_preserves_odd_prime_and_negative_exponent_conventions(prime):
    C, s, q, e = _basis_changed_disks(Ring(prime, 4), True)
    D = CochainComplex(C.ring, {-n: r for n, r in C.ranks.items()}, {
        -n - 1: value.dagger() for n, value in C.differentials.items()})
    sd, qd, ed = q.dagger(), s.dagger(), e.dagger()
    h = solve_strong_contraction(D, solve, section=sd, projection=qd, idempotent=ed)
    _check_actual_equalities(D, h, sd, qd, ed)
    reflected = {1 - n: value.dagger() for n, value in h.items()}
    _check_actual_equalities(C, reflected, s, q, e)


def test_finite_gaussian_backend_constructs_nonzero_redundant_Q_without_groebner():
    R = Ring(5, 5)
    C = direct_sum_complexes(disk(R, -1), disk(R, 0), CochainComplex(R, {0: 1}, {}))
    s = Matrix.from_rows(R, [[0, 0], [0, 0], [1, 2]])
    q = Matrix.from_rows(R, [[0, 0, 1], [0, 0, 0]])
    e = q * s
    backend = BoundedLinearBackend(R)
    h = solve_strong_contraction(C, backend.solve, section=s, projection=q, idempotent=e)
    _check_actual_equalities(C, h, s, q, e)


def test_observer_exposes_portable_equations_and_only_completed_higher_sections():
    C, s, q, e = _basis_changed_disks(Ring(5, 3), True)
    events = []
    with observe_direct_splitting(events.append):
        h = solve_strong_contraction(C, solve, section=s, projection=q, idempotent=e)
    starts = [event for event in events if event["event"] == "solve-start"]
    assert [event["degree"] for event in starts] == list(reversed(C.degrees))
    for event in starts:
        n = event["degree"]
        assert all(k > n for k in event["completed_h"])
        assert event["C"] is C
        # Ordinary JSON, rather than pickle or machine-specific files, must
        # retain exactly the original four Laurent equations at this stage.
        payload = json.loads(json.dumps({key: _matrix_dict(event[key])
                                       for key in ("dprev", "hnext", "s", "q", "R")}))
        A = {key: _matrix_from_dict(value, C.ring, key) for key, value in payload.items()}
        assert A["dprev"] * h[n] == A["R"]
        assert not h[n] * A["hnext"]
        assert not h[n] * A["s"]
        assert not A["q"] * h[n]
    # The context observer must not leak into an independent later run.
    old_count = len(events)
    solve_strong_contraction(disk(C.ring, 0), solve)
    assert len(events) == old_count


def test_resource_stop_keeps_the_actual_pending_equation_available():
    R = Ring(3, 4)
    C = direct_sum_complexes(disk(R, 0), disk(R, 1), disk(R, 2))
    events = []
    calls = []

    def stop_after_first(A, B):
        calls.append((A, B))
        if len(calls) >= 2:
            raise ResourceLimit("deliberate test stop after the highest section")
        return solve(A, B)

    with pytest.raises(ResourceLimit):
        solve_strong_contraction(C, stop_after_first, observer=events.append)
    pending = [event for event in events if event["event"] == "solve-start"][-1]
    assert pending["degree"] == 2
    assert set(pending["completed_h"]) == {3}
    assert pending["dprev"] == calls[-1][0]
    assert pending["R"] == calls[-1][1]
    assert not any(event["event"] == "solve-complete" and event["degree"] == 2 for event in events)


def test_json_round_trip_homotopy_is_rechecked_and_one_coefficient_tamper_fails():
    C, s, q, e = _basis_changed_disks(Ring(5, 2), True)
    h = solve_strong_contraction(C, solve, section=s, projection=q, idempotent=e)
    payload = json.loads(json.dumps([[n, _matrix_dict(A)] for n, A in h.items()]))
    restored = {n: _matrix_from_dict(A, C.ring, "h") for n, A in payload}
    _check_actual_equalities(C, restored, s, q, e)
    assert validate_strong_contraction(C, restored, section=s, projection=q, idempotent=e)
    restored[max(C.degrees)][0, 0] += C.ring.variable(1) ** -3
    with pytest.raises(ValueError):
        validate_strong_contraction(C, restored, section=s, projection=q, idempotent=e)
