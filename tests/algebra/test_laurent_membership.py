"""Independent C replay checks against the unchanged SymPy weak normal form.

The random ordered generators test replay equivalence, not GB completeness.
Negative fixtures use an explicitly complete localized principal-image basis.
"""
from copy import deepcopy
import json
import random

import pytest
from sympy.polys.agca.modules import ModuleOrder
from sympy.polys.distributedmodules import sdm_nf_buchberger
from sympy.polys.orderings import monomial_key

from css_decoupling.algebra import laurent_membership as lm
from css_decoupling.algebra import modules
from css_decoupling.algebra.laurent import Matrix, Ring, eye, zeros
from css_decoupling.algebra.modules import GeneralBackend
from css_decoupling.certificates.comparison import _matrix_dict
from css_decoupling.certificates.structural_c4 import (
    structural_section_dict, verify_structural_section,
)
from css_decoupling.errors import NoLaurentSolution, ResourceLimit


def sdm(backend, entries):
    return sorted([(m, backend.field(c)) for m, c in entries.items()
                   if c % backend.ring.p],
                  key=lambda pair: backend.order(pair[0]), reverse=True)


def project(ring, value, rows, active=None):
    entries = {}
    for (component, *powers), coefficient in value:
        row = component if active is None else active[component]
        if row >= rows:
            continue
        exponent = tuple(a-powers[-1] for a in powers[:-1])
        at = row, 0
        entries[at] = entries.get(at, ring.zero()) + ring.monomial(exponent, int(coefficient))
    return Matrix(ring, rows, 1, entries)


def quotient_matrix(ring, value, rows):
    entries = {}
    for (index, exponent), coefficient in value.items():
        at = index, 0
        entries[at] = entries.get(at, ring.zero()) + ring.monomial(exponent, coefficient)
    return Matrix(ring, rows, 1, entries)


def compare_weak_nf(backend, f, basis, rows):
    ring = backend.ring
    transition = [[((i,)+(0,)*(ring.D+1), backend.field.one)] for i in range(len(basis))]
    expected, phantom = sdm_nf_buchberger(
        f, basis, backend.order, backend.field, phantom=([], transition))
    prepared = lm.prepare_basis(basis, backend.order, ring.p, dimension=ring.D, rows=rows)
    actual, quotient, stats = lm.divide(f, prepared)
    assert actual == [(m, int(c) % ring.p) for m, c in expected]
    Q = quotient_matrix(ring, quotient, len(basis))
    assert Q == -project(ring, phantom, len(basis))
    G = zeros(ring, rows, 0).hstack(*(project(ring, g, rows) for g in basis))
    assert project(ring, f, rows) == project(ring, actual, rows) + G*Q
    return actual, Q, stats


def principal_fixture(dimension=3, prime=5, permuted=False):
    ring = Ring(prime, dimension)
    backend = GeneralBackend(ring)
    x, y = ring.variable(0), ring.variable(dimension-1)
    a = x-1
    A = Matrix.from_rows(ring, [[0, a, y**-1*a]])
    zero = (0,)*(dimension+1)
    xp = (1,)+(0,)*dimension
    mp = (0,)+(1,)*dimension
    scale = prime-1
    # Leading monomials x1 and z*x2*...*xD are coprime. These two
    # generators form a complete GB of (x1-1, z*prod(x)-1).
    basis = [sdm(backend, {(0,)+xp: scale, (0,)+zero: -scale}),
             sdm(backend, {(0,)+mp: 1, (0,)+zero: -1})]
    active = [3, 1, 2] if permuted else [1, 2, 3]
    original_position, relation_position = active.index(1), active.index(3)
    transition = [sdm(backend, {(original_position,)+zero: scale}),
                  sdm(backend, {(relation_position,)+zero: 1,
                                (original_position,)+mp: -1})]
    coefficients = [x**-2, ring.zero(), y**-2+1, x*x+y]
    X = Matrix(ring, 3, len(coefficients), {(1, j): f for j, f in enumerate(coefficients) if f})
    return backend, A, A*X, (basis, transition, active), X


def old_solution(backend, A, B, data):
    basis, transition, active = data
    columns = []
    for column in B.columns():
        remainder, phantom = sdm_nf_buchberger(
            backend.vector_sdm(column), basis, backend.order, backend.field,
            phantom=([], transition))
        assert not remainder
        columns.append(-project(A.ring, phantom, A.ncols, active))
    return zeros(A.ring, A.ncols, 0).hstack(*columns)


def replay(backend, A, B, data, **caps):
    return lm.membership_columns(A, B, data, order=backend.order,
                                 vector_sdm=backend.vector_sdm, **caps)


def force_complete_replay(monkeypatch, backend, data=None):
    monkeypatch.setattr("css_decoupling.algebra.sparse_linear.sparse_binary_solution", lambda *args: None)
    monkeypatch.setattr(backend, "_solve_polynomial", lambda *args: None)
    if data is not None:
        monkeypatch.setattr(backend, "_basis", lambda A: data)


@pytest.mark.parametrize("dimension", [2, 3, 4, 5])
@pytest.mark.parametrize("prime", [2, 3, 5])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_fixed_random_laurent_inputs_match_full_sympy_remainder_and_positive_quotient(dimension, prime, seed):
    ring = Ring(prime, dimension)
    backend = GeneralBackend(ring)
    rng = random.Random(9191+1000*dimension+100*prime+seed)
    basis = []
    for component in (0, 1):
        for variable in (0, 1, dimension):
            exponent = [0]*(dimension+1)
            exponent[variable] = 1
            basis.append(sdm(backend, {(component,)+tuple(exponent): rng.randrange(1, prime),
                                       (component,)+(0,)*(dimension+1): rng.randrange(1, prime)}))
    values = []
    for _ in (0, 1):
        polynomial = ring.zero()
        for _ in range(8):
            exponent = tuple(rng.randint(-2, 2) for _ in range(dimension))
            polynomial += ring.monomial(exponent, rng.randrange(1, prime))
        values.append(polynomial)
    compare_weak_nf(backend, backend.vector_sdm(values), basis, 2)


@pytest.mark.parametrize("prime", [2, 3, 5])
def test_stale_heap_cancellation_and_first_divisor_are_exact(prime):
    backend = GeneralBackend(Ring(prime, 3))
    basis = [sdm(backend, {(0, 2, 0, 0, 0): 1, (0, 1, 0, 0, 0): -1, (0, 0, 0, 0, 0): -1}),
             sdm(backend, {(0, 1, 0, 0, 0): 1, (0, 0, 0, 0, 0): -1})]
    f = sdm(backend, {(0, 80, 0, 0, 0): 1, (0, 65, 0, 0, 0): -1, (0, 32, 0, 0, 0): 1})
    _, _, stats = compare_weak_nf(backend, f, basis, 1)
    assert stats["steps"] > 20
    assert stats["peak_queue_entries"] > stats["peak_remainder_terms"]
    g = basis[1]
    _, Q, _ = compare_weak_nf(backend, g, [g, g], 1)
    assert Q[0, 0] == 1 and not Q[1, 0]


@pytest.mark.parametrize("name", ["lex", "grlex", "grevlex"])
@pytest.mark.parametrize("top", [False, True])
def test_supported_global_orders_match_sympy(name, top):
    backend = GeneralBackend(Ring(5, 2))
    backend.order = ModuleOrder(monomial_key("lex"), monomial_key(name), top)
    basis = [sdm(backend, {(0, 1, 0, 0): 2, (0, 0, 1, 0): 1}),
             sdm(backend, {(1, 0, 0, 1): 3, (1, 0, 0, 0): -1})]
    f = backend.vector_sdm([backend.ring.variable(0)**-2+1, backend.ring.variable(1)**-1])
    compare_weak_nf(backend, f, basis, 2)


@pytest.mark.parametrize("dimension", [2, 3, 4, 5])
@pytest.mark.parametrize("prime", [2, 3, 5])
@pytest.mark.parametrize("permuted", [False, True])
def test_complete_basis_restores_zero_redundant_and_localization_coordinates(dimension, prime, permuted):
    backend, A, B, data, expected = principal_fixture(dimension, prime, permuted)
    events = []
    columns = list(replay(backend, A, B, data, observer=events.append))
    assert [j for j, _ in columns] == list(range(B.ncols))
    answer = zeros(A.ring, A.ncols, 0).hstack(*(x for _, x in columns))
    assert answer == expected == old_solution(backend, A, B, data)
    assert A*answer == B
    assert events[-1]["event"] == "complete"
    assert events[-1]["stats"]["columns_verified"] == B.ncols


@pytest.mark.parametrize("bad", ["duplicate", "missing", "negative", "extra", "boolean"])
def test_invalid_active_source_binding_is_not_a_resource_fallback(bad):
    backend, A, B, (basis, transition, active), _ = principal_fixture()
    active = {"duplicate": [1, 1, 3], "missing": [1, 3], "negative": [-1, 2, 3],
              "extra": [1, 2, 4], "boolean": [True, 2, 3]}[bad]
    with pytest.raises(ValueError):
        list(replay(backend, A, B, (basis, transition, active)))


@pytest.mark.parametrize("bad", ["empty", "reverse", "duplicate", "negative", "width", "component", "zero", "float", "boolean"])
def test_canonical_basis_errors_are_rejected(bad):
    backend = GeneralBackend(Ring(3, 2))
    good = sdm(backend, {(0, 1, 0, 0): 1, (0, 0, 0, 0): -1})
    vector = {"empty": [], "reverse": good[::-1], "duplicate": [good[0], good[0]],
              "negative": [((0, -1, 0, 0), 1)], "width": [((0, 1, 0), 1)],
              "component": [((1, 0, 0, 0), 1)], "zero": [((0, 0, 0, 0), 0)],
              "float": [((0, 0, 0, 0), 1.0)], "boolean": [((0, 0, 0, 0), True)]}[bad]
    with pytest.raises(ValueError):
        lm.prepare_basis([vector], backend.order, 3, dimension=2, rows=1)


def test_input_canonical_validation_empty_modules_and_unsupported_order():
    backend = GeneralBackend(Ring(3, 2))
    prepared = lm.prepare_basis([], backend.order, 3, dimension=2, rows=0)
    assert lm.divide([], prepared)[:2] == ([], {})
    prepared = lm.prepare_basis([], backend.order, 3, dimension=2, rows=1)
    f = [((0, 0, 0, 0), 1)]
    assert lm.divide(f, prepared)[:2] == (f, {})
    with pytest.raises(ValueError):
        lm.divide(f+f, prepared)
    with pytest.raises(ResourceLimit):
        lm.prepare_basis([], backend.elimination_order, 3, dimension=2, rows=1)


@pytest.mark.parametrize("bad", ["noncanonical", "wrong-image", "wrong-count"])
def test_transition_is_validated_before_any_rhs_is_yielded(bad):
    backend, A, B, (basis, transition, active), _ = principal_fixture()
    transition = deepcopy(transition)
    if bad == "noncanonical":
        transition[1].reverse()
    elif bad == "wrong-image":
        transition[0] = sdm(backend, {(0, 1, 0, 0, 0): 1})
    else:
        transition.pop()
    events = []
    with pytest.raises((ValueError, ArithmeticError)):
        list(replay(backend, A, B, (basis, transition, active), observer=events.append))
    assert not any(event["event"] == "column-verified" for event in events)


@pytest.mark.parametrize("stop", [0, 1, 2, 3])
def test_resource_fallback_preserves_verified_prefix_including_zero_rhs(monkeypatch, stop):
    backend, A, B, data, expected = principal_fixture()
    force_complete_replay(monkeypatch, backend, data)
    calls = []
    original_nf = modules.sdm_nf_buchberger

    def partial(*args, **kwargs):
        for j in range(stop):
            x = expected[:, j:j+1]
            assert A*x == B[:, j:j+1]
            yield j, x
        raise ResourceLimit("injected current-column stop")

    def tracked(f, *args, **kwargs):
        calls.append((list(f), "phantom" in kwargs))
        return original_nf(f, *args, **kwargs)

    monkeypatch.setattr(modules, "membership_columns", partial)
    monkeypatch.setattr(modules, "sdm_nf_buchberger", tracked)
    assert backend.solve(A, B) == expected
    assert calls == [(backend.vector_sdm([B[i, j] for i in range(B.nrows)]), True)
                     for j in range(stop, B.ncols)]
    assert backend._last_membership_stats["fallback_column"] == stop


@pytest.mark.parametrize("stop", [0, 2, 3])
def test_verification_interruption_never_yields_the_unverified_current_column(monkeypatch, stop):
    backend, A, B, data, expected = principal_fixture()
    original_product = lm._product

    def interrupted(left, right, budget):
        if left is A and right.shape == (A.ncols, 1) and right == expected[:, stop:stop+1]:
            raise ResourceLimit("injected before actual A*x verification")
        return original_product(left, right, budget)

    monkeypatch.setattr(lm, "_product", interrupted)
    seen = []
    with pytest.raises(ResourceLimit) as caught:
        for j, column in replay(backend, A, B, data):
            seen.append(j)
            assert A*column == B[:, j:j+1]
    assert seen == list(range(stop))
    assert caught.value.membership_stats["columns_verified"] == stop
    assert caught.value.membership_stats["current_column"] == stop


def test_reduction_budget_is_cumulative_across_rhs_columns():
    backend, A, _, data, _ = principal_fixture()
    B = Matrix.from_rows(A.ring, [[A[0, 1], A[0, 1], A[0, 1]]])
    seen = []
    with pytest.raises(ResourceLimit) as caught:
        for j, X in replay(backend, A, B, data, max_steps=2):
            assert A*X == B[:, j:j+1]
            seen.append(j)
    assert seen == [0, 1]
    stats = caught.value.membership_stats
    assert (stats["steps"], stats["columns_verified"], stats["current_column"]) == (2, 2, 2)


def test_actual_cumulative_budget_fallback_replays_only_the_remaining_original_column(monkeypatch):
    backend, A, _, data, _ = principal_fixture()
    B = Matrix.from_rows(A.ring, [[A[0, 1], A[0, 1], A[0, 1]]])
    force_complete_replay(monkeypatch, backend, data)
    original_membership, original_nf = modules.membership_columns, modules.sdm_nf_buchberger
    calls = []

    def limited(*args, **kwargs):
        return original_membership(*args, **kwargs, max_steps=2)

    def tracked(f, *args, **kwargs):
        calls.append((list(f), "phantom" in kwargs))
        return original_nf(f, *args, **kwargs)

    monkeypatch.setattr(modules, "membership_columns", limited)
    monkeypatch.setattr(modules, "sdm_nf_buchberger", tracked)
    X = backend.solve(A, B)
    assert A*X == B
    assert calls == [(backend.vector_sdm([B[0, 2]]), True)]
    assert backend._last_membership_stats["columns_verified"] == 2
    assert backend._last_membership_stats["fallback_column"] == 2


def test_corrupt_positive_quotient_cannot_be_yielded_before_original_equation_recheck(monkeypatch):
    backend, A, B, data, _ = principal_fixture()
    original_divide = lm.divide
    events = []

    def corrupt(*args, **kwargs):
        remainder, quotient, stats = original_divide(*args, **kwargs)
        assert not remainder and quotient
        return remainder, {}, stats

    monkeypatch.setattr(lm, "divide", corrupt)
    with pytest.raises(ArithmeticError, match="A\\*x_j=B_j"):
        list(replay(backend, A, B, data, observer=events.append))
    assert not any(event["event"] == "column-verified" for event in events)


def test_outer_unit_reduction_retains_the_actual_original_laurent_equation(monkeypatch):
    backend, A0, B0, data, _ = principal_fixture()
    ring = A0.ring
    u, v = ring.variable(0)**-2, ring.variable(1)
    A = Matrix.from_rows(ring, [[u, v, 0, 0], [0, 0, A0[0, 1], A0[0, 2]]])
    B = Matrix(ring, 2, B0.ncols, {(1, j): B0[0, j] for j in range(B0.ncols)})
    B[0, 0] = 1
    force_complete_replay(monkeypatch, backend, data)
    X = backend.solve(A, B)
    assert A*X == B
    assert X[0, 0] == u**-1
    assert backend._last_membership_stats["event"] == "complete"


@pytest.mark.parametrize("cap", ["max_work", "max_terms", "max_product_work"])
def test_preparation_resource_failure_yields_no_prefix(cap):
    backend, A, B, data, _ = principal_fixture()
    events = []
    with pytest.raises(ResourceLimit) as caught:
        list(replay(backend, A, B, data, observer=events.append, **{cap: 0}))
    assert caught.value.membership_stats["columns_verified"] == 0
    assert events[-1]["event"] == "resource-incomplete"
    assert not any(event["event"] == "column-start" for event in events)


def test_product_budget_preflight_precedes_actual_multiplication(monkeypatch):
    ring = Ring(5, 3)
    A, X = eye(ring, 1), eye(ring, 1)

    def forbidden(*args):
        raise AssertionError("Matrix multiplication happened before the work preflight")

    monkeypatch.setattr(Matrix, "__mul__", forbidden)
    with pytest.raises(ResourceLimit, match="product-work"):
        lm._product(A, X, lm._Budget(max_product_work=0))


@pytest.mark.parametrize("prime", [2, 3, 5])
def test_negative_replay_rechecks_original_rhs_with_old_nf_without_phantom(monkeypatch, prime):
    backend, A, _, data, _ = principal_fixture(prime=prime)
    x = A.ring.variable(0)
    B = Matrix.from_rows(A.ring, [[x*x+x+1 if prime == 2 else x*x+1]])
    force_complete_replay(monkeypatch, backend, data)
    calls = []
    original_nf = modules.sdm_nf_buchberger

    def tracked(f, *args, **kwargs):
        calls.append((list(f), kwargs))
        return original_nf(f, *args, **kwargs)

    monkeypatch.setattr(modules, "sdm_nf_buchberger", tracked)
    with pytest.raises(NoLaurentSolution):
        backend.solve(A, B)
    assert calls == [(backend.vector_sdm([B[0, 0]]), {})]
    assert len(calls[0][0]) > 1  # The canonical weak remainder has one term.


@pytest.mark.parametrize("mode", ["old-zero", "different-nonzero", "empty", "wrong-column"])
def test_tampered_negative_marker_never_becomes_a_negative_certificate(monkeypatch, mode):
    backend, A, original_B, data, _ = principal_fixture(prime=5)
    B = original_B[:, :1] if mode == "old-zero" else Matrix.from_rows(A.ring, [[A.ring.variable(0)**2+1]])
    force_complete_replay(monkeypatch, backend, data)
    marker = [] if mode == "empty" else [((0,)+(0,)*(A.ring.D+1), 1)]

    def corrupt(*args, **kwargs):
        raise lm._NonzeroRemainder(1 if mode == "wrong-column" else 0, marker)
        yield  # A generator must raise when consumed, not when constructed.

    monkeypatch.setattr(modules, "membership_columns", corrupt)
    with pytest.raises(ArithmeticError) as caught:
        backend.solve(A, B)
    if mode != "wrong-column":
        assert caught.value.membership_mismatch["original"] == backend.vector_sdm([B[0, 0]])


@pytest.mark.parametrize("mode", ["skip", "duplicate", "early", "boolean", "wrong-shape", "bad-value"])
def test_malformed_or_unverified_generator_results_are_not_success(monkeypatch, mode):
    backend, A, B, data, expected = principal_fixture()
    force_complete_replay(monkeypatch, backend, data)

    def corrupt(*args, **kwargs):
        if mode == "early":
            return
        if mode == "duplicate":
            yield 0, expected[:, :1]
            yield 0, expected[:, :1]
            return
        if mode == "skip":
            yield 1, expected[:, 1:2]
            return
        if mode == "boolean":
            yield False, expected[:, :1]
            return
        if mode == "wrong-shape":
            yield 0, expected
            return
        for j in range(B.ncols):
            yield j, zeros(A.ring, A.ncols, 1)

    monkeypatch.setattr(modules, "membership_columns", corrupt)
    with pytest.raises(ArithmeticError):
        backend.solve(A, B)


@pytest.mark.parametrize("dimension", [2, 3, 4, 5])
@pytest.mark.parametrize("prime", [2, 3, 5])
def test_actual_replay_section_round_trip_and_source_bound_certificate_tampering(monkeypatch, dimension, prime):
    ring = Ring(prime, dimension)
    backend = GeneralBackend(ring)
    x = ring.variable(0)
    # The original nonunit columns generate 1: (x^2-x+1)-x*(x-1)=1.
    d = Matrix.from_rows(ring, [[x-1, x*x-x+1]])
    factors = dict(d=d, R=eye(ring, 1), hnext=zeros(ring, 1, 1),
                   s=zeros(ring, 1, 2), q=zeros(ring, 0, 2), Pprev=zeros(ring, 2, 2))
    force_complete_replay(monkeypatch, backend)
    X = backend.solve(d, factors["R"])
    assert backend._last_membership_stats["event"] == "complete"
    assert d*X == factors["R"]
    assert not X*factors["hnext"] and not X*factors["s"] and not factors["q"]*X
    data = json.loads(json.dumps(structural_section_dict(**factors, X=X, method="verified-seed")))
    assert verify_structural_section(data, expected=factors).valid
    bad = deepcopy(data)
    changed = X.copy()
    changed[0, 0] += ring.variable(dimension-1)**-1
    bad["solution"] = _matrix_dict(changed)
    assert not verify_structural_section(bad, expected=factors).valid
    bad = deepcopy(data)
    changed = d.copy()
    changed[0, 0] += 1
    bad["factors"]["d"] = _matrix_dict(changed)
    assert not verify_structural_section(bad, expected=factors).valid
