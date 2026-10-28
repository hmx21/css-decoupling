"""Independent A-path checks: quotient before weak solve, with honest reuse."""
import pytest

from css_decoupling.algebra.laurent import Matrix, Ring, eye, zeros
from css_decoupling.algebra.modules import solve
from css_decoupling.errors import NoLaurentSolution, ResourceLimit
from css_decoupling.topology.complexes import CochainComplex, disk
from css_decoupling.topology.direct_splitting import solve_strong_contraction, structural_sections


def family(D, p, with_Q=False, section_shift=0):
    ring = Ring(p, D)
    v, t = ring.variable(0), ring.variable(1)
    u = v**-1 + ring.variable(D-1) + section_shift
    ranks = {0: 2, 1: 2, 2: 1}
    differentials = {0: Matrix.from_rows(ring, [[v, 0], [0, 0]]),
                     1: Matrix.from_rows(ring, [[0, t]])}
    if not with_Q:
        ranks[-1] = 1
        differentials[-1] = Matrix.from_rows(ring, [[0], [1]])
    C = CochainComplex(ring, ranks, differentials)
    if with_Q:
        s = Matrix.from_rows(ring, [[0, 0], [1, 1]])
        q = Matrix.from_rows(ring, [[0, 1], [0, 0]])
        e = Matrix.from_rows(ring, [[1, 1], [0, 0]])
    else:
        s, q, e = zeros(ring, 2, 0), zeros(ring, 0, 2), zeros(ring, 0, 0)
    top = Matrix.from_rows(ring, [[u], [t**-1]])
    N = Matrix.from_rows(ring, [[1, -u*t]])
    middle = Matrix.from_rows(ring, [[v**-1, -v**-1*u*t], [0, 0]])
    known = {2: top, 1: middle,
             0: zeros(ring, 0, 2) if with_Q else Matrix.from_rows(ring, [[0, 1]])}
    left = C.d(0).vstack(q) if q else C.d(0)
    rhs = eye(ring, 2)-top*C.d(1)
    full_right = rhs.vstack(zeros(ring, q.nrows, 2)) if q else rhs
    return dict(C=C, ring=ring, s=s, q=q, e=e, top=top, N=N,
                middle=middle, known=known, left=left, rhs=rhs, full_right=full_right)


def options(f):
    return dict(section=f["s"], projection=f["q"], idempotent=f["e"])


def actual_strong_equalities(f, h):
    C, ring, s, q, e = (f[key] for key in ("C", "ring", "s", "q", "e"))
    assert q*s == e and e*e == e and s*e == s and e*q == q
    assert not C.d(0)*s and not q*C.d(-1)
    for n in C.degrees:
        hn = h.get(n, zeros(ring, C.rank(n-1), C.rank(n)))
        following = h.get(n+1, zeros(ring, C.rank(n), C.rank(n+1)))
        P = s*q if n == 0 else zeros(ring, C.rank(n), C.rank(n))
        assert hn.shape == (C.rank(n-1), C.rank(n)) and hn.ring == ring
        assert C.d(n-1)*hn + following*C.d(n) == eye(ring, C.rank(n))-P
        assert not hn*following
    assert not h.get(0, zeros(ring, C.rank(-1), C.rank(0)))*s
    assert not q*h.get(1, zeros(ring, C.rank(0), C.rank(1)))


def guarded_solver(f, calls, *, target_reply=None, allow_full=False):
    def backend(A, B):
        if A == f["C"].d(1):
            assert B == eye(f["ring"], 1)
            return f["top"].copy()
        if A == f["left"]:
            calls.append(B.copy())
            if not allow_full and B.ncols != 1:
                raise AssertionError("An available right quotient was preceded by a weak full-RHS solve")
            if B.ncols == 1:
                assert B == f["full_right"][:, [0]]
            if target_reply is not None:
                return target_reply(A, B)
        return solve(A, B)
    return backend


@pytest.mark.parametrize("D", [2, 3, 4, 5])
@pytest.mark.parametrize("p", [2, 3, 5])
@pytest.mark.parametrize("with_Q", [False, True])
def test_complete_right_quotient_precedes_any_weak_full_rhs_and_preserves_full_strong_h(D, p, with_Q):
    f = family(D, p, with_Q)
    actual_strong_equalities(f, f["known"])
    before = (dict(f["C"].ranks), {n: A.key() for n, A in f["C"].differentials.items()},
              tuple(f[key].key() for key in ("s", "q", "e")))
    calls, events = [], []
    h = solve_strong_contraction(f["C"], guarded_solver(f, calls), observer=events.append, **options(f))
    assert calls and all(B.ncols == 1 for B in calls)
    assert any(event["event"] == "right-quotient-ready" and event["degree"] == 1 for event in events)
    actual_strong_equalities(f, h)
    assert before == (dict(f["C"].ranks), {n: A.key() for n, A in f["C"].differentials.items()},
                      tuple(f[key].key() for key in ("s", "q", "e")))


@pytest.mark.parametrize("prime", [2, 3, 5])
def test_current_structural_positive_quotient_is_reused_after_finite_support_miss(monkeypatch, prime):
    import css_decoupling.algebra.structural_c4 as structural
    import css_decoupling.topology.direct_splitting as direct
    f = family(4, prime, True)
    wanted = f["top"].hstack(zeros(f["ring"], 2, 2))
    computations = []
    first, second = structural.right_annihilator_by_units, direct._right_annihilator_by_units

    def structural_helper(B, **kwargs):
        if B == wanted:
            computations.append("structural")
        return first(B, **kwargs)

    def direct_helper(B, *args):
        if B == wanted:
            computations.append("outer")
        return second(B, *args)

    monkeypatch.setattr(structural, "right_annihilator_by_units", structural_helper)
    monkeypatch.setattr(direct, "_right_annihilator_by_units", direct_helper)
    calls, events = [], []
    with structural_sections(max_radius=0, try_sparse_seed=False):
        h = solve_strong_contraction(f["C"], guarded_solver(f, calls), observer=events.append, **options(f))
    assert computations == ["structural"]
    assert any(event["event"] == "structural-incomplete" and event["degree"] == 1 for event in events)
    assert any(event["event"] == "right-quotient-ready" and event["degree"] == 1 and event["reused"] for event in events)
    actual_strong_equalities(f, h)


def test_reused_quotient_local_resource_failure_goes_to_weak_fallback_without_repeating_elimination(monkeypatch):
    import css_decoupling.topology.direct_splitting as direct
    f = family(3, 5, True)
    wanted = f["top"].hstack(zeros(f["ring"], 2, 2))
    original = direct._right_annihilator_by_units

    def no_repeat(B, *args):
        if B == wanted:
            raise AssertionError("A complete reused quotient was redundantly eliminated after local failure")
        return original(B, *args)

    monkeypatch.setattr(direct, "_right_annihilator_by_units", no_repeat)
    calls, events = [], []

    def reply(A, B):
        if B.ncols == 1:
            raise ResourceLimit("one local reduced-solve budget is exhausted")
        return f["middle"].copy()

    with structural_sections(max_radius=0, try_sparse_seed=False):
        h = solve_strong_contraction(f["C"], guarded_solver(f, calls, target_reply=reply, allow_full=True),
                                     observer=events.append, **options(f))
    assert [B.ncols for B in calls] == [1, 2]
    assert any(event["event"] == "right-quotient-declined" and event["degree"] == 1 and event["reused"] for event in events)
    actual_strong_equalities(f, h)


def test_reduced_backend_nonmembership_is_not_swallowed_as_a_local_budget_miss():
    f = family(3, 3)
    failure = NoLaurentSolution("backend reports an exact negative rather than a resource miss")
    calls = []

    def negative(A, B):
        raise failure

    with pytest.raises(NoLaurentSolution) as caught:
        solve_strong_contraction(f["C"], guarded_solver(f, calls, target_reply=negative, allow_full=True), **options(f))
    assert caught.value is failure
    assert [B.ncols for B in calls] == [1]


def test_same_complex_with_a_different_accepted_high_section_does_not_reuse_stale_factors():
    first, second = family(3, 5, True), family(3, 5, True, section_shift=1)
    assert first["C"].differentials == second["C"].differentials
    assert first["top"] != second["top"] and first["rhs"] != second["rhs"]
    results = []
    for f in (first, second):
        events = []
        with structural_sections(max_radius=0, try_sparse_seed=False):
            h = solve_strong_contraction(f["C"], guarded_solver(f, []), observer=events.append, **options(f))
        actual_strong_equalities(f, h)
        records = [event for event in events if event["event"] == "right-quotient-ready" and event["degree"] == 1]
        assert len(records) == 1 and records[0]["quotient"] == f["N"]
        results.append(h[1])
    assert results[0] != results[1]


@pytest.mark.parametrize("change", ["constraints", "rhs", "N", "N-ring", "N-shape", "missing-source",
                                    "duplicate-index", "boolean-index", "float-index", "negative-index",
                                    "out-of-range-index", "wrong-surviving-index"])
def test_reused_quotient_source_factors_and_indices_are_not_trusted(monkeypatch, change):
    import css_decoupling.algebra.structural_c4 as structural
    f = family(3, 5, True)
    original = structural.solve_structural_section

    def forged(d, R, hnext, s, q, Pprev, **kwargs):
        if d != f["C"].d(0):
            return original(d, R, hnext, s, q, Pprev, **kwargs)
        payload = dict(constraints=hnext.hstack(s), rhs=R.copy(), N=f["N"].copy(), active=(0,))
        assert not payload["N"]*payload["constraints"]
        assert R[:, [0]]*payload["N"] == R
        if change in ("constraints", "rhs"):
            payload[change][0, 0] += 1
        elif change == "N":
            payload["N"][0, 1] += 1
        elif change == "N-ring":
            payload["N"] = zeros(Ring(3, 3), 1, 2)
        elif change == "N-shape":
            payload["N"] = zeros(f["ring"], 0, 2)
        elif change == "missing-source":
            payload.pop("constraints")
        else:
            payload["active"] = {"duplicate-index": (0, 0), "boolean-index": (False,),
                                 "float-index": (0.0,), "negative-index": (-2,),
                                 "out-of-range-index": (2,), "wrong-surviving-index": (1,)}[change]
        failure = ResourceLimit("deliberate finite-support stop with a corrupted carrier")
        failure.structural_quotient = payload
        raise failure

    monkeypatch.setattr(structural, "solve_structural_section", forged)
    calls, events = [], []
    with structural_sections(max_radius=0, try_sparse_seed=False):
        with pytest.raises((ValueError, ArithmeticError)) as caught:
            solve_strong_contraction(f["C"], guarded_solver(f, calls), observer=events.append, **options(f))
    assert not isinstance(caught.value, NoLaurentSolution)
    assert not calls
    assert not any(event["event"] == "solve-complete" and event["degree"] == 1 for event in events)


@pytest.mark.parametrize("prime", [2, 3, 5])
def test_no_complete_unit_quotient_preserves_the_exact_original_fallback(prime):
    ring = Ring(prime, 3)
    x = ring.variable(0)
    a, b = x+1, x*x+x+1
    C = CochainComplex(ring, {0: 1, 1: 2, 2: 1}, {
        0: Matrix.from_rows(ring, [[1], [x]]), 1: Matrix.from_rows(ring, [[-x, 1]])})
    top = Matrix.from_rows(ring, [[a], [b]])
    assert all(not value.is_unit for value in top.entries.values())
    calls = []

    def backend(A, B):
        if A == C.d(1):
            return top.copy()
        if A == C.d(0):
            calls.append(B.ncols)
        return solve(A, B)

    h = solve_strong_contraction(C, backend)
    assert calls == [2]
    f = dict(C=C, ring=ring, s=zeros(ring, 1, 0), q=zeros(ring, 0, 1), e=zeros(ring, 0, 0))
    actual_strong_equalities(f, h)


@pytest.mark.parametrize("kind", ["zero", "wrong-shape", "wrong-ring"])
def test_bad_reduced_left_solver_answer_is_rejected_before_accepting_h(kind):
    f = family(3, 5, True)

    def bad(A, B):
        if kind == "wrong-shape":
            return zeros(f["ring"], A.ncols+1, B.ncols)
        if kind == "wrong-ring":
            return zeros(Ring(3, 3), A.ncols, B.ncols)
        return zeros(f["ring"], A.ncols, B.ncols)

    events = []
    with pytest.raises(ValueError):
        solve_strong_contraction(f["C"], guarded_solver(f, [], target_reply=bad), observer=events.append, **options(f))
    assert not any(event["event"] == "solve-complete" and event["degree"] == 1 for event in events)


@pytest.mark.parametrize("limit", ["product", "terms"])
def test_local_quotient_budget_declines_before_big_restore_and_allows_smaller_fallback(monkeypatch, limit):
    f = family(3, 5)
    x = f["ring"].variable(0)
    W = sum((x**j for j in range(50 if limit == "product" else 10)), f["ring"].zero())
    Y = Matrix.from_rows(f["ring"], [[x**-1], [W]])
    assert f["C"].d(0)*Y == f["rhs"][:, [0]]
    multiplied = []
    original = Matrix.__mul__

    def watched(self, other):
        if self == Y and isinstance(other, Matrix) and other == f["N"]:
            multiplied.append(True)
            raise AssertionError("Over-budget Y*N was evaluated before its guard")
        return original(self, other)

    monkeypatch.setattr(Matrix, "__mul__", watched)

    def reply(A, B):
        return Y.copy() if B.ncols == 1 else f["middle"].copy()

    calls, events = [], []
    budgets = dict(max_product_work=100) if limit == "product" else dict(max_terms=8)
    h = solve_strong_contraction(f["C"], guarded_solver(f, calls, target_reply=reply, allow_full=True),
                                 observer=events.append, **options(f), **budgets)
    assert [B.ncols for B in calls] == [1, 2]
    assert not multiplied
    assert any(event["event"] == "right-quotient-declined" and event["degree"] == 1 for event in events)
    actual_strong_equalities(f, h)


def test_quotient_backend_resource_failure_can_fall_back_but_never_becomes_nonmembership():
    f = family(2, 3)
    calls = []

    def reply(A, B):
        if B.ncols == 1:
            raise ResourceLimit("declared reduced-route budget")
        return f["middle"].copy()

    h = solve_strong_contraction(f["C"], guarded_solver(f, calls, target_reply=reply, allow_full=True), **options(f))
    assert [B.ncols for B in calls] == [1, 2]
    actual_strong_equalities(f, h)


def test_zero_product_cap_never_emits_a_completed_degree():
    events = []
    with structural_sections():
        with pytest.raises(ResourceLimit):
            solve_strong_contraction(disk(Ring(3, 2), 0), solve, max_product_work=0, observer=events.append)
    assert not any(event["event"] == "solve-complete" for event in events)


def test_empty_active_columns_for_pure_redundant_Q_are_valid():
    ring = Ring(5, 4)
    C = CochainComplex(ring, {0: 1}, {})
    s, q = Matrix.from_rows(ring, [[1, 1]]), Matrix.from_rows(ring, [[1], [0]])
    f = dict(C=C, ring=ring, s=s, q=q, e=q*s)
    events = []
    h = solve_strong_contraction(C, solve, observer=events.append, **options(f))
    assert any(event["event"] == "right-quotient-ready" and event["active"] == () for event in events)
    actual_strong_equalities(f, h)
    assert all(not A for A in h.values())


def test_empty_complex_never_requests_a_backend_solve():
    C = CochainComplex(Ring(3, 5), {}, {})
    def forbidden(*args):
        raise AssertionError("Empty complex requested a solve")
    assert solve_strong_contraction(C, forbidden) == {}
