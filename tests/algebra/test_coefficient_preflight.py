"""Independent B checks: exact preflights, F2 ordering, and bounded shifts."""
import builtins
import json
import random
import sys

import pytest

from css_decoupling.algebra.laurent import Matrix, Ring, eye, zeros
import css_decoupling.algebra.structural_c4 as structural
import css_decoupling.algebra.sparse_linear as sparse
from css_decoupling.certificates.comparison import _matrix_dict
from css_decoupling.certificates.structural_c4 import verify_structural_section
from css_decoupling.errors import ResourceLimit


def caps(**changes):
    result = dict(max_unknowns=10000, max_equations=20000, max_terms=100000,
                  max_steps=100000, max_coefficient_bits=10000000, max_product_work=100000)
    result.update(changes)
    return result


def forbidden_rows(*args, **kwargs):
    raise AssertionError("Coefficient rows were allocated before a decisive preflight")


@pytest.mark.parametrize("D", [2, 3, 4, 5])
@pytest.mark.parametrize("p", [2, 3, 5])
def test_left_exact_contribution_guard_precedes_rows_and_accepts_the_exact_limit(monkeypatch, D, p):
    R = Ring(p, D)
    x = R.variable(0)
    A, B = Matrix.from_rows(R, [[x+1, x*x+x+1]]), eye(R, 1)
    support = [(0,)*D, (1,)+(0,)*(D-1)]
    # Five operator terms, two shifts, one RHS term: exactly eleven adds.
    with monkeypatch.context() as patch:
        patch.setattr(structural, "_Rows", forbidden_rows)
        with pytest.raises(ResourceLimit, match="contributions"):
            structural._left_coefficients(A, B, support, caps(max_terms=10))
    X, info = structural._left_coefficients(A, B, support, caps(max_terms=11))
    assert A*X == B and info["contributions"] == 11


@pytest.mark.parametrize("p", [2, 3, 5])
def test_joint_exact_contribution_count_includes_all_four_factors_before_rows(monkeypatch, p):
    R = Ring(p, 3)
    x = R.variable(0)
    d = Matrix.from_rows(R, [[1, x], [0, 1]])
    hnext = Matrix.from_rows(R, [[1], [x]])
    s = Matrix.from_rows(R, [[x+1], [0]])
    q = Matrix.from_rows(R, [[0, x**-1]])
    variables = [(0, 0, (0, 0, 0)), (0, 1, (1, 0, 0)), (1, 0, (-1, 0, 0))]
    # Contributions by variable: 4, 2, 6, plus two RHS entries.
    with monkeypatch.context() as patch:
        patch.setattr(structural, "_Rows", forbidden_rows)
        with pytest.raises(ResourceLimit, match="contributions"):
            structural._joint_coefficients(d, eye(R, 2), hnext, s, q, variables, caps(max_terms=13))
    _, info = structural._joint_coefficients(d, eye(R, 2), hnext, s, q, variables, caps(max_terms=14))
    assert info["contributions"] == 14


@pytest.mark.parametrize("p", [2, 3, 5])
def test_joint_rhs_has_distinct_ij_exponent_keys_but_shared_rhs_does_not(monkeypatch, p):
    R = Ring(p, 4)
    x = R.variable(0)
    B = Matrix.from_rows(R, [[x+1, x+1], [x+1, x+1]])
    d = zeros(R, 2, 0)
    with monkeypatch.context() as patch:
        patch.setattr(structural, "_Rows", forbidden_rows)
        with pytest.raises(ResourceLimit, match="equations"):
            structural._joint_coefficients(d, B, zeros(R, 2, 0), zeros(R, 2, 0),
                                             zeros(R, 0, 0), [], caps(max_equations=7))
    # Same eight RHS terms: four shared (row,exponent) keys versus eight
    # joint (row,column,exponent) keys. Both zero-operator systems are only
    # reported as inconsistent on their explicitly supplied finite support.
    answer, info = structural._left_coefficients(d, B, [(0,)*4], caps(max_equations=4))
    assert answer is None and info["equations"] == 4 and info["contributions"] == 8
    answer, info = structural._joint_coefficients(d, B, zeros(R, 2, 0), zeros(R, 2, 0),
                                                  zeros(R, 0, 0), [], caps(max_equations=8))
    assert answer is None and info["equations"] == 8


@pytest.mark.parametrize("p", [2, 3, 5])
def test_many_operator_contributions_colliding_in_one_equation_are_not_rejected(p):
    R = Ring(p, 2)
    A = Matrix(R, 1, 32, {(0, j): 1 for j in range(32)})
    B = Matrix(R, 1, 16, {(0, j): 1 for j in range(16)})
    X, info = structural._left_coefficients(A, B, [(0, 0)], caps(max_equations=1, max_terms=48))
    assert A*X == B and info["equations"] == 1 and info["contributions"] == 48
    variables = [(i, 0, (0, 0)) for i in range(32)]
    X, info = structural._joint_coefficients(A, eye(R, 1), zeros(R, 1, 0), zeros(R, 1, 0),
                                             zeros(R, 0, 32), variables, caps(max_equations=1, max_terms=33))
    assert A*X == eye(R, 1) and info["equations"] == 1 and info["contributions"] == 33


@pytest.mark.parametrize("with_seed", [False, True])
def test_plan_preflight_precedes_cartesian_variable_allocation(monkeypatch, with_seed):
    R = Ring(3, 2)
    d = Matrix(R, 3, 4, {(0, i): 1 for i in range(4)})
    projector = Matrix(R, 3, 3, {(0, 0): 1})
    hnext = Matrix.from_rows(R, [[0, 0], [1, 0], [0, 1]])
    seed = None
    if with_seed:
        x = R.variable(0)
        seed = Matrix(R, 4, 3, {(0, 0): 1, (0, 1): x, (1, 1): -x})
        assert d*seed == projector and seed*hnext
    monkeypatch.setattr(structural, "right_annihilator_by_units", lambda *a, **k: None)
    monkeypatch.setattr(structural, "_support", lambda *a, **k: [(0, 0)])
    monkeypatch.setattr(structural, "_Rows", forbidden_rows)

    def checked_range(*args):
        if sys._getframe(1).f_code.co_name == "plans" and args in ((3,), (4,)):
            raise AssertionError("Cartesian coefficient variables were allocated before preflight")
        return builtins.range(*args)

    monkeypatch.setattr(structural, "range", checked_range, raising=False)
    events = []
    # Constant full plan has 21 contributions; a nonconstant seed plan has
    # 25. The mathematical problem has a constant solution, despite this cap.
    with pytest.raises(ResourceLimit) as caught:
        structural.solve_structural_section(d, projector, hnext, zeros(R, 3, 0), zeros(R, 0, 4),
                                             zeros(R, 4, 4), seed=seed, max_radius=0,
                                             max_terms=20, observer=events.append)
    assert any(e.get("strategy") == "finite-radius-0" and "resource" in e for e in events)
    if with_seed:
        assert any(e.get("strategy") == "seed-support" and "resource" in e for e in events)
    assert caught.value.structural_attempts == events
    assert caught.value.structural_degree_bound["qualification"] == "conditional"
    assert not hasattr(caught.value, "structural_quotient")


def reference_binary(rows, width, right_width):
    """A's former mask-based Gaussian rule, independently kept for B tests."""
    mask, basis, steps = (1 << width)-1, {}, 0
    for left, right in rows:
        value = sum(1 << k for k in left) | sum(1 << (width+k) for k in right)
        while value & mask:
            pivot = (value & -value).bit_length()-1
            if pivot not in basis:
                basis[pivot] = value
                break
            value ^= basis[pivot]
            steps += 1
        else:
            if value:
                return None, dict(rank=len(basis), steps=steps, pivots=tuple(basis)), None
    columns = [0]*right_width
    steps += len(basis)*right_width
    for pivot in sorted(basis, reverse=True):
        value = basis[pivot]
        tail, rhs = (value & mask) ^ (1 << pivot), value >> width
        for j in range(right_width):
            if ((rhs >> j)&1) ^ ((tail & columns[j]).bit_count()&1):
                columns[j] |= 1 << pivot
    answer = {(i, j): 1 for j, column in enumerate(columns) for i in range(width) if (column >> i)&1}
    return answer, dict(rank=len(basis), steps=steps, pivots=tuple(basis)), columns


def compare_binary(rows, width, right_width):
    expected, reference, columns = reference_binary(rows, width, right_width)
    actual, info = structural._eliminate(rows, width, right_width, 2, max_steps=100000,
                                         max_coefficient_bits=10000000)
    assert actual == expected
    assert (info["rank"], info["steps"]) == (reference["rank"], reference["steps"])
    encoded = [sum(1 << k for k in left) | sum(1 << (width+k) for k in right) for left, right in rows]
    assert sparse._binary_solution(encoded, width, right_width, 100000) == columns
    if actual is not None:
        for left, rhs in rows:
            for j in range(right_width):
                assert sum(actual.get((i, j), 0) for i in left) % 2 == rhs.get(j, 0)
        assert sparse._binary_solution(encoded, width, right_width, reference["steps"]) == columns
        if reference["steps"]:
            with pytest.raises(ResourceLimit):
                structural._eliminate(rows, width, right_width, 2,
                                      max_steps=reference["steps"]-1, max_coefficient_bits=10000000)
            assert sparse._binary_solution(encoded, width, right_width, reference["steps"]-1) is None
    return actual, reference


@pytest.mark.parametrize("width,right_width,rows", [
    (0, 0, []), (0, 3, [({}, {}), ({}, {})]), (0, 2, [({}, {1: 1})]),
    (5, 0, [({0: 1, 4: 1}, {}), ({4: 1}, {}), ({}, {})]),
    (8, 3, [({7: 1}, {2: 1}), ({7: 1}, {2: 1})]),
    (8, 3, [({7: 1}, {2: 1}), ({7: 1}, {1: 1})]),
])
def test_lowest_bit_guard_preserves_zero_rows_zero_width_and_rhs_only_semantics(width, right_width, rows):
    compare_binary(rows, width, right_width)


@pytest.mark.parametrize("seed", range(6))
def test_lowest_bit_pivots_solutions_and_steps_match_old_gaussian_on_each_prefix(seed):
    rng = random.Random(6001+seed)
    width, count = 9, 6
    lefts = []
    for i in range(count):
        row = {j: 1 for j in range(3) if rng.randrange(2)}
        row[3+i] = 1  # Independent rows; every prefix admits an actual right inverse.
        lefts.append(row)
    previous = set()
    for length in range(1, count+1):
        rows = [(lefts[i], {i: 1}) for i in range(length)]
        actual, reference = compare_binary(rows, width, length)
        observed_pivots = {i for i, _ in actual}
        assert observed_pivots == set(reference["pivots"])
        # The right inverse exposes every selected variable; prefix changes
        # therefore independently expose the actual lowest-bit pivot order.
        assert observed_pivots-previous == {reference["pivots"][-1]}
        previous = observed_pivots


@pytest.mark.parametrize("D", [2, 3, 4, 5])
def test_shift_cache_is_bounded_and_never_reuses_a_previous_shift(D):
    cache = sparse._ShiftedExponents(max_entries=2)
    exponents = [(0,)*D, (1,)+(0,)*(D-1), (-2,)+(1,)*(D-1)]
    for shift in ((-1,)*D, (2,)+(0,)*(D-1), (-1,)*D):
        cache.set_shift(shift)
        assert not cache.cache
        for exponent in exponents:
            assert cache(exponent) == tuple(a+b for a, b in zip(exponent, shift))
        assert len(cache.cache) == 2
        assert exponents[-1] not in cache.cache
        assert cache(exponents[-1]) == tuple(a+b for a, b in zip(exponents[-1], shift))


@pytest.mark.parametrize("D", [2, 3, 4, 5])
@pytest.mark.parametrize("p", [2, 3, 5])
def test_per_shift_cache_and_uncached_assembly_give_identical_multi_rhs_equations(monkeypatch, D, p):
    R = Ring(p, D)
    x, y = R.variable(0), R.variable(D-1)
    A = Matrix.from_rows(R, [[-1, x-y**-1], [0, 1]])
    wanted = Matrix.from_rows(R, [[x**-1+y, y**-1], [1, x**-1]])
    B = A*wanted
    support = sorted({e for f in wanted.entries.values() for e in f.terms})
    first, info = structural._left_coefficients(A, B, support, caps())
    base = sparse._ShiftedExponents
    class NoCache(base):
        def __init__(self):
            super().__init__(max_entries=0)
    monkeypatch.setattr(structural, "_ShiftedExponents", NoCache)
    second, other = structural._left_coefficients(A, B, support, caps())
    assert first == second == wanted and A*second == B
    assert info == other


@pytest.mark.parametrize("D", [2, 3, 4, 5])
def test_sparse_binary_cached_and_uncached_supports_preserve_the_original_laurent_equation(monkeypatch, D):
    R = Ring(2, D)
    x, y = R.variable(0), R.variable(D-1)
    A = Matrix.from_rows(R, [[x, 0], [0, y]])
    wanted = Matrix.from_rows(R, [[x**-2+y, y**-1], [x**-1, 1]])
    B = A*wanted
    first = sparse.sparse_binary_solution(A, B)
    base = sparse._ShiftedExponents
    class NoCache(base):
        def __init__(self):
            super().__init__(max_entries=0)
    monkeypatch.setattr(sparse, "_ShiftedExponents", NoCache)
    second = sparse.sparse_binary_solution(A, B)
    assert first == second == wanted and A*second == B


def valid_factors(D, p, location):
    R = Ring(p, D)
    v = R.variable(0)
    u = R.variable(1)**-1 + R.variable(D-1)
    if location == "previous":
        d = Matrix.from_rows(R, [[v, 0, 0], [0, 0, 0]])
        P = Matrix(R, 3, 3, {(1, 1): 1})
        return dict(d=d, R=Matrix.from_rows(R, [[1, -u], [0, 0]]),
                    hnext=Matrix.from_rows(R, [[u], [1]]), s=zeros(R, 2, 2),
                    q=Matrix.from_rows(R, [[0, 1, 0], [0, 0, 0]]), Pprev=P)
    w = v**-1 + 2*R.variable(D-1)
    return dict(d=Matrix.from_rows(R, [[v, 0], [0, 0], [0, 0]]),
                R=Matrix.from_rows(R, [[1, -u, -w], [0, 0, 0], [0, 0, 0]]),
                hnext=Matrix.from_rows(R, [[u], [1], [0]]),
                s=Matrix.from_rows(R, [[w, w], [0, 0], [1, 1]]),
                q=zeros(R, 0, 2), Pprev=zeros(R, 2, 2))


@pytest.mark.parametrize("D", [2, 3, 4, 5])
@pytest.mark.parametrize("p", [2, 3, 5])
@pytest.mark.parametrize("location", ["current", "previous"])
@pytest.mark.parametrize("route", ["shared", "joint"])
def test_new_assembly_still_satisfies_all_original_c4_factors_and_rejects_tampering(monkeypatch, D, p, location, route):
    factors = valid_factors(D, p, location)
    before = {key: A.key() for key, A in factors.items()}
    if route == "joint":
        monkeypatch.setattr(structural, "right_annihilator_by_units", lambda *a, **k: None)
    X, certificate = structural.solve_structural_section(**factors, max_radius=2, return_certificate=True)
    assert factors["d"]*X == factors["R"]
    assert not X*factors["hnext"] and not X*factors["s"] and not factors["q"]*X
    assert before == {key: A.key() for key, A in factors.items()}
    data = json.loads(json.dumps(certificate))
    assert verify_structural_section(data, expected=factors).valid
    wrong = X.copy()
    wrong[0, 0] += 1
    data["solution"] = _matrix_dict(wrong)
    exponents = [e for f in wrong.entries.values() for e in f.terms]
    data["resources"] = dict(monomials=len(exponents),
        max_l1_radius=max(map(lambda e: sum(map(abs, e)), exponents), default=0),
        max_canonical_weight=max((sum(e)+(D+1)*max(0, -min(e)) for e in exponents), default=0))
    assert not verify_structural_section(data, expected=factors).valid


@pytest.mark.parametrize("p", [2, 3, 5])
@pytest.mark.parametrize("rows,columns,rhs", [(0, 0, 0), (0, 3, 2), (2, 0, 3), (3, 2, 0)])
def test_empty_left_coefficient_shapes_remain_exact(p, rows, columns, rhs):
    R = Ring(p, 2)
    A, B = zeros(R, rows, columns), zeros(R, rows, rhs)
    X, info = structural._left_coefficients(A, B, [(0, 0)], caps())
    assert X == zeros(R, columns, rhs) and A*X == B
    assert info["equations"] == info["contributions"] == 0
    assert sparse.sparse_binary_solution(A, B) is None


def test_extremely_wide_operator_is_declined_before_factor_scanning_or_rows(monkeypatch):
    R = Ring(2, 2)
    A = Matrix(R, 1, 10**9, {(0, 10**9-1): 1})
    with monkeypatch.context() as patch:
        patch.setattr(structural, "_Rows", forbidden_rows)
        patch.setattr(structural, "terms", lambda *a: (_ for _ in ()).throw(AssertionError("wide input scanned")))
        with pytest.raises(ResourceLimit, match="unknowns"):
            structural._left_coefficients(A, eye(R, 1), [(0, 0)], caps())


def test_finite_support_failure_has_no_laurent_nonmembership_result():
    R = Ring(2, 2)
    x = R.variable(0)
    A, B = Matrix.from_rows(R, [[x+1]]), Matrix.from_rows(R, [[x**9+1]])
    assert sparse.sparse_binary_solution(A, B) is None
    # A polynomial solution sum(x^j,j=0..8) exists, but the one-shot
    # shortcut did not find it. Its None cannot be a negative certificate.
    known = Matrix.from_rows(R, [[sum((x**j for j in range(9)), R.zero())]])
    assert A*known == B
