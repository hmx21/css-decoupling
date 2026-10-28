"""Structural C4 sections checked on independent complete-complex fixtures."""
from copy import deepcopy

import pytest

from css_decoupling.algebra.laurent import Matrix, Ring, eye, zeros
from css_decoupling.algebra.structural_c4 import c4_degree_bound, solve_structural_section
from css_decoupling.errors import NoLaurentSolution, ResourceLimit
from css_decoupling.topology.complexes import CochainComplex, direct_sum_complexes, disk


def strong_step_fixture(dimension, prime, degree=0, with_Q=False):
    """Produce all step factors by a known exact full-complex basis change.

    The optional Q is degree zero and uses two redundant coordinates. For
    degree zero the shear makes X*s cancellation nontrivial; for degree one
    it makes q*X cancellation nontrivial. All Laurent powers stay explicit.
    """
    R = Ring(prime, dimension)
    pieces = [disk(R, j) for j in (-2, -1, 0, 1)]
    if with_Q:
        pieces.append(CochainComplex(R, {0: 1}, {}))
    raw = direct_sum_complexes(*pieces)
    raw_h = {n: zeros(R, raw.rank(n-1), raw.rank(n)) for n in raw.degrees}
    for index, lower in enumerate((-2, -1, 0, 1)):
        row = sum(piece.rank(lower) for piece in pieces[:index])
        col = sum(piece.rank(lower+1) for piece in pieces[:index])
        raw_h[lower+1][row, col] = 1
    changes = {n: eye(R, raw.rank(n)) for n in raw.degrees}
    inverses = {n: eye(R, raw.rank(n)) for n in raw.degrees}
    shear = R.variable(0)**-1
    if with_Q:
        position = (0, raw.rank(0)-1) if degree == 0 else (raw.rank(0)-1, 1)
    else:
        position = (0, 1)
    changes[0][position] = shear
    inverses[0][position] = -shear
    C = CochainComplex(R, raw.ranks, {
        n: changes.get(n+1, eye(R, 0))*raw.d(n)*inverses[n] for n in raw.degrees})
    h = {n: changes.get(n-1, eye(R, 0))*raw_h[n]*inverses[n] for n in raw.degrees}
    if with_Q:
        v, w = Matrix.from_rows(R, [[1], [0]]), Matrix.from_rows(R, [[1, 1]])
        e = v*w
        inclusion = Matrix(R, raw.rank(0), 1, {(raw.rank(0)-1, 0): 1})
        projection = Matrix(R, 1, raw.rank(0), {(0, raw.rank(0)-1): 1})
        s0, q0 = changes[0]*inclusion*w, v*projection*inverses[0]
    else:
        e = zeros(R, 0, 0)
        s0, q0 = zeros(R, C.rank(0), 0), zeros(R, 0, C.rank(0))
    for n in C.degrees:
        hn = h[n]
        hnext = h.get(n+1, zeros(R, C.rank(n), C.rank(n+1)))
        P = s0*q0 if n == 0 else zeros(R, C.rank(n), C.rank(n))
        assert C.d(n-1)*hn + hnext*C.d(n) == eye(R, C.rank(n))-P
        assert not hn*hnext
    assert q0*s0 == e
    assert e*e == e and s0*e == s0 and e*q0 == q0
    n = degree
    d = C.d(n-1)
    hnext = h.get(n+1, zeros(R, C.rank(n), C.rank(n+1)))
    P = s0*q0 if n == 0 else zeros(R, C.rank(n), C.rank(n))
    projector = eye(R, C.rank(n))-P-hnext*C.d(n)
    s = s0 if n == 0 else zeros(R, C.rank(n), e.nrows)
    q = q0 if n == 1 else zeros(R, e.nrows, C.rank(n-1))
    Pprev = s0*q0 if n == 1 else zeros(R, C.rank(n-1), C.rank(n-1))
    return dict(d=d, R=projector, hnext=hnext, s=s, q=q, Pprev=Pprev,
                expected=h[n], C=C, degree=n, s0=s0, q0=q0, e=e, all_h=h)


def nonunit_step_fixture(dimension, prime):
    R = Ring(prime, dimension)
    x = R.variable(0)
    a, b = x+1, x*x+x+1
    d = Matrix.from_rows(R, [[a, b]])
    return dict(d=d, R=eye(R, 1), hnext=zeros(R, 1, 0), s=zeros(R, 1, 0),
                q=zeros(R, 0, 2), Pprev=zeros(R, 2, 2),
                expected=Matrix.from_rows(R, [[-x], [1]]))


def actual_four_equations(factors, X):
    d, R = factors["d"], factors["R"]
    assert X.ring == d.ring and X.shape == (d.ncols, d.nrows)
    assert d*X == R
    assert X*factors["hnext"] == zeros(d.ring, X.nrows, factors["hnext"].ncols)
    assert X*factors["s"] == zeros(d.ring, X.nrows, factors["s"].ncols)
    assert factors["q"]*X == zeros(d.ring, factors["q"].nrows, X.ncols)


def step_arguments(factors):
    return {key: factors[key] for key in ("d", "R", "hnext", "s", "q", "Pprev")}


@pytest.mark.parametrize("dimension", [2, 3, 4, 5])
@pytest.mark.parametrize("prime", [2, 3, 5])
@pytest.mark.parametrize("degree,with_Q", [(0, False), (0, True), (1, True)])
def test_finite_field_sections_on_independent_full_complexes(dimension, prime, degree, with_Q):
    factors = strong_step_fixture(dimension, prime, degree, with_Q)
    before = {name: A.key() for name, A in step_arguments(factors).items()}
    actual_four_equations(factors, factors["expected"])
    X = solve_structural_section(**step_arguments(factors), max_radius=1)
    actual_four_equations(factors, X)
    assert before == {name: A.key() for name, A in step_arguments(factors).items()}


@pytest.mark.parametrize("dimension", [2, 3, 4, 5])
@pytest.mark.parametrize("prime", [2, 3, 5])
def test_nonunit_bezout_family_uses_actual_polynomial_solution(dimension, prime):
    factors = nonunit_step_fixture(dimension, prime)
    assert all(not f.is_unit for f in factors["d"].entries.values())
    actual_four_equations(factors, factors["expected"])
    X = solve_structural_section(**step_arguments(factors), max_radius=1)
    actual_four_equations(factors, X)


def test_insufficient_finite_support_is_resource_incomplete_not_nonmembership():
    factors = nonunit_step_fixture(4, 5)
    # A verified solution of radius one is already known, but no constant
    # combination of x+1 and x^2+x+1 equals one.
    actual_four_equations(factors, factors["expected"])
    with pytest.raises(ResourceLimit) as caught:
        solve_structural_section(**step_arguments(factors), max_radius=0)
    assert not isinstance(caught.value, NoLaurentSolution)


@pytest.mark.parametrize("kind", ["wrong-ring", "wrong-shape", "wrong-left-equation"])
def test_bad_boundary_lifts_are_not_used_as_structural_evidence(kind):
    factors = nonunit_step_fixture(3, 5)
    if kind == "wrong-ring":
        seed = zeros(Ring(3, 3), 2, 1)
    elif kind == "wrong-shape":
        seed = zeros(factors["d"].ring, 3, 1)
    else:
        seed = zeros(factors["d"].ring, 2, 1)
    with pytest.raises(ValueError):
        c4_degree_bound(**step_arguments(factors), boundary_lift=seed)
    with pytest.raises(ValueError):
        solve_structural_section(**step_arguments(factors), seed=seed)


@pytest.mark.parametrize("factor", ["R", "Pprev"])
def test_bad_projectors_are_rejected_before_a_positive_structural_claim(factor):
    factors = strong_step_fixture(3, 5, degree=1, with_Q=True)
    bad = step_arguments(factors)
    bad[factor] = 2*bad[factor]
    assert bad[factor]*bad[factor] != bad[factor]
    with pytest.raises(ValueError):
        solve_structural_section(**bad)


def test_an_ordinary_solved_module_equation_is_not_a_c4_projector_system():
    R = Ring(5, 2)
    x = R.variable(0)
    d, target, X = eye(R, 1), Matrix.from_rows(R, [[x]]), Matrix.from_rows(R, [[x]])
    assert d*X == target
    assert target*target != target
    factors = dict(d=d, R=target, hnext=zeros(R, 1, 0), s=zeros(R, 1, 0),
                   q=zeros(R, 0, 1), Pprev=zeros(R, 1, 1))
    with pytest.raises(ValueError):
        solve_structural_section(**factors, seed=X)


@pytest.mark.parametrize("missing_homology", [False, True])
def test_fraction_field_rank_or_local_projector_identities_do_not_supply_boundary_surjectivity(missing_homology):
    R = Ring(3, 3)
    d = zeros(R, 1, 1) if missing_homology else Matrix.from_rows(R, [[R.variable(0)-1]])
    factors = dict(d=d, R=eye(R, 1), hnext=zeros(R, 1, 0), s=zeros(R, 1, 0),
                   q=zeros(R, 0, 1), Pprev=zeros(R, 1, 1))
    assert factors["R"]*d == d
    assert factors["R"]*factors["R"] == factors["R"]
    try:
        solve_structural_section(**factors, max_radius=1)
    except ResourceLimit:
        pass
    except NoLaurentSolution:
        pytest.fail("Finite coefficient failure was incorrectly promoted to Laurent nonmembership")
    except ValueError:
        pass
    else:
        pytest.fail("A missing image/homology assumption became a positive C4 certificate")


def test_nonzero_Q_identifying_only_part_of_homology_cannot_hide_the_rest():
    ring = Ring(5, 4)
    factors = dict(d=zeros(ring, 2, 1), R=Matrix.from_rows(ring, [[0, 0], [0, 1]]),
                   hnext=zeros(ring, 2, 0), s=Matrix.from_rows(ring, [[1, 1], [0, 0]]),
                   q=zeros(ring, 0, 1), Pprev=zeros(ring, 1, 1))
    assert not factors["R"]*factors["s"]
    try:
        solve_structural_section(**factors, max_radius=0)
    except ResourceLimit:
        pass
    except NoLaurentSolution:
        pytest.fail("A support search failure was labelled exact Laurent nonmembership")
    except ValueError:
        pass
    else:
        pytest.fail("An unaccounted homology direction became a positive structural certificate")


@pytest.mark.parametrize("shape", [(0, 0), (0, 3), (2, 0)])
def test_zero_boundary_projectors_and_empty_modules(shape):
    R = Ring(5, 4)
    target, source = shape
    factors = dict(d=zeros(R, target, source), R=zeros(R, target, target),
                   hnext=zeros(R, target, 0), s=zeros(R, target, 0),
                   q=zeros(R, 0, source), Pprev=zeros(R, source, source))
    X = solve_structural_section(**factors, max_radius=0)
    actual_four_equations(factors, X)
    assert X == zeros(R, source, target)


@pytest.mark.parametrize("limit", ["max_radius", "max_unknowns", "max_equations", "max_terms", "max_steps", "max_coefficient_bits", "max_product_work"])
@pytest.mark.parametrize("value", [True, -1, 0.5, "2"])
def test_invalid_limits_are_not_silently_coerced(limit, value):
    factors = nonunit_step_fixture(2, 3)
    with pytest.raises(ValueError):
        solve_structural_section(**step_arguments(factors), **{limit: value})


@pytest.mark.parametrize("limit,value", [("max_unknowns", 1), ("max_equations", 1),
                                        ("max_terms", 1), ("max_steps", 0),
                                        ("max_coefficient_bits", 1), ("max_product_work", 0)])
def test_explicit_resource_caps_preserve_incomplete_semantics(limit, value):
    factors = nonunit_step_fixture(3, 5)
    with pytest.raises(ResourceLimit):
        solve_structural_section(**step_arguments(factors), max_radius=1, **{limit: value})


@pytest.mark.parametrize("strategy", ["shared", "joint"])
def test_fake_finite_field_elimination_answer_is_rejected_by_actual_laurent_products(monkeypatch, strategy):
    import css_decoupling.algebra.structural_c4 as implementation
    factors = nonunit_step_fixture(2, 5)
    if strategy == "joint":
        monkeypatch.setattr(implementation, "right_annihilator_by_units", lambda *a, **k: None)
    monkeypatch.setattr(implementation, "_eliminate", lambda *a, **k: ({}, {"rank": 0, "steps": 0}))
    with pytest.raises(ValueError):
        solve_structural_section(**step_arguments(factors), max_radius=1)


@pytest.mark.parametrize("dimension", [2, 3, 4, 5])
def test_support_includes_projected_single_z_degree_not_just_a_laurent_l1_ball(dimension):
    from css_decoupling.algebra.structural_c4 import _support, _weight
    zero = (0,)*dimension
    generators = [zero, (-1,)*dimension]
    generators += [tuple(int(i == j) for i in range(dimension)) for j in range(dimension)]
    support_one = set(_support(dimension, 1, 1000))
    assert (-1,)*dimension in support_one
    assert sum(abs(e) for e in (-1,)*dimension) > 1
    assert _weight((-1,)*dimension) == 1
    expected_degree_two = {tuple(a+b for a, b in zip(left, right))
                           for left in generators for right in generators}
    assert expected_degree_two <= set(_support(dimension, 2, 10000))
    assert all(_weight(exponent) <= 2 for exponent in expected_degree_two)


@pytest.mark.parametrize("dimension", [2, 3, 4, 5])
@pytest.mark.parametrize("prime", [2, 3, 5])
def test_actual_single_z_degree_one_solution_can_have_laurent_l1_radius_D(dimension, prime):
    from css_decoupling.certificates.structural_c4 import verify_structural_section
    ring = Ring(prime, dimension)
    d = Matrix.from_rows(ring, [[ring.monomial((1,)*dimension)]])
    factors = dict(d=d, R=eye(ring, 1), hnext=zeros(ring, 1, 0), s=zeros(ring, 1, 0),
                   q=zeros(ring, 0, 1), Pprev=zeros(ring, 1, 1))
    X, certificate = solve_structural_section(**factors, max_radius=1, return_certificate=True)
    actual_four_equations(factors, X)
    assert X == Matrix.from_rows(ring, [[ring.monomial((-1,)*dimension)]])
    assert certificate["resources"]["max_canonical_weight"] == 1
    assert certificate["resources"]["max_l1_radius"] == dimension
    assert verify_structural_section(certificate, expected=factors).valid


@pytest.mark.parametrize("degree", [0, 1])
def test_a_current_left_lift_is_not_frozen_before_its_homology_side_conditions(degree):
    factors = strong_step_fixture(3, 5, degree=degree, with_Q=True)
    weak = factors["expected"].copy()
    weak[0 if degree == 0 else 2, 0] += 1
    assert factors["d"]*weak == factors["R"]
    assert weak*factors["s"] if degree == 0 else factors["q"]*weak
    original = weak.key()
    X = solve_structural_section(**step_arguments(factors), seed=weak, max_radius=1)
    actual_four_equations(factors, X)
    assert weak.key() == original


def test_degree_bound_remains_conditional_without_a_verified_image_inclusion():
    factors = nonunit_step_fixture(3, 5)
    conditional = c4_degree_bound(**step_arguments(factors))
    assert conditional["qualification"] == "conditional"
    verified = c4_degree_bound(**step_arguments(factors), boundary_lift=factors["expected"])
    assert verified["qualification"] == "verified-boundary-lift"
    assert {k: v for k, v in conditional.items() if k != "qualification"} == {
        k: v for k, v in verified.items() if k != "qualification"}


@pytest.mark.parametrize("kind", ["nonprojector-module-target", "incompatible-hnext", "incompatible-section"])
def test_verified_boundary_lift_alone_does_not_qualify_missing_c4_structure(kind):
    ring = Ring(5, 2)
    factors = dict(d=eye(ring, 1), R=eye(ring, 1), hnext=zeros(ring, 1, 0),
                   s=zeros(ring, 1, 0), q=zeros(ring, 0, 1), Pprev=zeros(ring, 1, 1))
    if kind == "nonprojector-module-target":
        factors["R"] = Matrix.from_rows(ring, [[ring.variable(0)]])
    elif kind == "incompatible-hnext":
        factors["hnext"] = eye(ring, 1)
    else:
        factors["s"] = eye(ring, 1)
    seed = factors["R"]
    assert factors["d"]*seed == factors["R"]
    with pytest.raises(ValueError):
        c4_degree_bound(**factors, boundary_lift=seed)


def test_positive_seed_outside_the_claimed_degree_ceiling_is_not_returned_with_a_false_certificate():
    from css_decoupling.certificates.structural_c4 import verify_structural_section
    ring = Ring(5, 3)
    factors = dict(d=Matrix.from_rows(ring, [[1, 0]]), R=eye(ring, 1),
                   hnext=zeros(ring, 1, 0), s=zeros(ring, 1, 0),
                   q=zeros(ring, 0, 2), Pprev=zeros(ring, 2, 2))
    seed = Matrix.from_rows(ring, [[1], [ring.variable(0)**19]])
    actual_four_equations(factors, seed)
    # All factors are constant, so the structural ceiling is zero. A
    # positive high-degree seed may be declined and a constant one solved.
    X, certificate = solve_structural_section(**factors, seed=seed, max_radius=0, return_certificate=True)
    actual_four_equations(factors, X)
    assert all(f.is_constant for f in X.entries.values())
    assert verify_structural_section(certificate, expected=factors).valid


def test_constant_data_bound_does_not_claim_positive_degree_is_needed():
    from css_decoupling.algebra.structural_c4 import bound_contains
    ring = Ring(3, 2)
    factors = dict(d=eye(ring, 1), R=eye(ring, 1), hnext=zeros(ring, 1, 0),
                   s=zeros(ring, 1, 0), q=zeros(ring, 0, 1), Pprev=zeros(ring, 1, 1))
    bound = c4_degree_bound(**factors, boundary_lift=eye(ring, 1))
    assert bound["zero_degree"]
    assert bound_contains(bound, 0)
    assert not bound_contains(bound, 1)


def test_direct_contraction_outer_product_cap_cannot_be_overridden_by_structural_defaults():
    from css_decoupling.algebra.modules import solve
    from css_decoupling.topology.direct_splitting import solve_strong_contraction, structural_sections
    C = disk(Ring(3, 2), 0)
    events = []
    with structural_sections():
        with pytest.raises(ResourceLimit):
            solve_strong_contraction(C, solve, max_product_work=0, observer=events.append)
    assert any(event["event"] == "solve-start" and event["degree"] == 1 for event in events)
    assert any(event["event"] == "structural-incomplete" and event["degree"] == 1 for event in events)
    assert not any(event["event"] == "solve-complete" and event["degree"] == 1 for event in events)


def test_shared_left_equation_replay_checks_product_budget_before_actual_multiplication(monkeypatch):
    import css_decoupling.algebra.structural_c4 as implementation
    ring = Ring(5, 3)
    x = ring.variable(0)
    A = Matrix.from_rows(ring, [[1+x+x*x]])
    B = A.copy()
    # The fake elimination is mathematically correct: its one coefficient
    # describes X=[1], so A*X=B, but that replay requires three term pairs.
    monkeypatch.setattr(implementation, "_eliminate", lambda *a, **k: (
        {(0, 0): 1}, {"rank": 1, "steps": 0}))
    actual_multiplications = []
    original_multiply = Matrix.__mul__

    def guarded_multiply(self, other):
        if self is A and isinstance(other, Matrix):
            actual_multiplications.append(other)
            raise AssertionError("The over-budget left equation was multiplied before its guard")
        return original_multiply(self, other)

    monkeypatch.setattr(Matrix, "__mul__", guarded_multiply)
    caps = dict(max_unknowns=100, max_equations=100, max_terms=1000, max_steps=1000,
                max_coefficient_bits=10000, max_product_work=2)
    with pytest.raises(ResourceLimit, match="product budget"):
        implementation._left_coefficients(A, B, [(0, 0, 0)], caps)
    assert not actual_multiplications


@pytest.mark.parametrize("prime", [2, 3, 5])
def test_pivot_bit_budget_tracks_actual_low_rank_storage_instead_of_a_dense_full_rank_guess(prime):
    from css_decoupling.algebra.structural_c4 import _eliminate
    width, count, cap = 10000, 1000, 200000
    rows = [({0: 1}, {0: 1}) for _ in range(count)]
    # These thousand consistent equations have just one actual pivot. A
    # hypothetical dense thousand-pivot basis would exceed this cap.
    assert count*(width+1)*(prime-1).bit_length() > cap
    answer, info = _eliminate(rows, width, 1, prime, max_steps=10000,
                              max_coefficient_bits=cap, max_output_terms=1)
    assert answer == {(0, 0): 1}
    assert info["rank"] == 1
    assert 0 < info["coefficient_bit_estimate"] <= cap
    assert all(sum(value*answer.get((column, 0), 0) for column, value in left.items()) % prime == rhs[0]
               for left, rhs in rows)


@pytest.mark.parametrize("prime,cap", [(2, 600), (3, 710), (5, 995)])
def test_actual_pivot_growth_stops_when_storage_exceeds_the_budget(prime, cap):
    from css_decoupling.algebra.structural_c4 import _eliminate
    rows = [({index: 1}, {0: 1}) for index in range(8)]
    # Input and the first pivot fit these declared limits. New independent
    # pivots require additional storage and must be checked as they appear.
    with pytest.raises(ResourceLimit, match="actual pivot storage"):
        _eliminate(rows, 128, 1, prime, max_steps=10000, max_coefficient_bits=cap)


@pytest.mark.parametrize("prime", [2, 3, 5])
def test_decoded_coefficient_output_terms_are_capped_for_binary_and_odd_fields(prime):
    from css_decoupling.algebra.structural_c4 import _eliminate
    rows = [({index: 1}, {0: 1, 1: 1}) for index in range(3)]
    with pytest.raises(ResourceLimit, match="output term budget"):
        _eliminate(rows, 3, 2, prime, max_steps=1000, max_coefficient_bits=10000, max_output_terms=5)
    answer, _ = _eliminate(rows, 3, 2, prime, max_steps=1000, max_coefficient_bits=10000, max_output_terms=6)
    assert answer == {(i, j): 1 for i in range(3) for j in range(2)}


@pytest.mark.parametrize("prime,cap", [(2, 50), (3, 80), (5, 110)])
def test_decoded_outputs_also_consume_the_coefficient_bit_budget(prime, cap):
    from css_decoupling.algebra.structural_c4 import _eliminate
    rows = [({index: 1}, {0: 1, 1: 1}) for index in range(2)]
    # All input rows and pivots fit; retaining decoded output does not.
    with pytest.raises(ResourceLimit, match="actual pivot storage"):
        _eliminate(rows, 2, 2, prime, max_steps=1000, max_coefficient_bits=cap, max_output_terms=4)


@pytest.mark.parametrize("degree", [0, 1])
def test_sparse_seed_with_only_a_left_equation_is_not_accepted_before_homology_sides(monkeypatch, degree):
    import css_decoupling.algebra.sparse_linear as sparse
    factors = strong_step_fixture(3, 2, degree=degree, with_Q=True)
    weak = factors["expected"].copy()
    weak[0 if degree == 0 else 2, 0] += 1
    assert factors["d"]*weak == factors["R"]
    assert (weak*factors["s"] if degree == 0 else factors["q"]*weak)
    calls = []

    def candidate(A, B, **options):
        calls.append((A, B, options))
        return weak.copy()

    monkeypatch.setattr(sparse, "sparse_binary_solution", candidate)
    X, certificate = solve_structural_section(**step_arguments(factors), try_sparse_seed=True,
                                              max_radius=1, return_certificate=True)
    assert len(calls) == 1
    actual_four_equations(factors, X)
    assert certificate["search"]["method"] != "verified-seed"


def test_sparse_seed_with_nonzero_high_degree_composition_is_not_frozen(monkeypatch):
    import css_decoupling.algebra.sparse_linear as sparse
    factors = strong_step_fixture(3, 2)
    weak = factors["expected"].copy()
    weak[0, 1] += 1
    assert factors["d"]*weak == factors["R"]
    assert weak*factors["hnext"]
    monkeypatch.setattr(sparse, "sparse_binary_solution", lambda *a, **k: weak.copy())
    X, certificate = solve_structural_section(**step_arguments(factors), try_sparse_seed=True,
                                              max_radius=1, return_certificate=True)
    actual_four_equations(factors, X)
    assert certificate["search"]["method"] != "verified-seed"


def test_sparse_seed_receives_structural_product_and_all_coefficient_caps(monkeypatch):
    import css_decoupling.algebra.sparse_linear as sparse
    factors = nonunit_step_fixture(3, 2)
    calls = []

    def candidate(A, B, **options):
        calls.append(options)
        assert A == factors["d"] and B == factors["R"]
        return factors["expected"].copy()

    monkeypatch.setattr(sparse, "sparse_binary_solution", candidate)
    limits = dict(max_unknowns=67, max_equations=71, max_steps=73, max_coefficient_bits=10000,
                  max_terms=123, max_product_work=5)
    X = solve_structural_section(**step_arguments(factors), try_sparse_seed=True, **limits)
    actual_four_equations(factors, X)
    assert calls == [limits]


@pytest.mark.parametrize("limit", ["max_unknowns", "max_equations", "max_steps", "max_coefficient_bits"])
@pytest.mark.parametrize("value", [0, 1])
def test_sparse_seed_search_cannot_bypass_zero_or_insufficient_coefficient_caps(limit, value):
    ring = Ring(2, 2)
    factors = dict(d=eye(ring, 2), R=eye(ring, 2), hnext=zeros(ring, 2, 0),
                   s=zeros(ring, 2, 0), q=zeros(ring, 0, 2), Pprev=zeros(ring, 2, 2))
    with pytest.raises(ResourceLimit):
        solve_structural_section(**factors, try_sparse_seed=True, max_radius=0, **{limit: value})


def test_explicit_verified_seed_does_not_need_a_coefficient_search_budget(monkeypatch):
    import css_decoupling.algebra.sparse_linear as sparse
    ring = Ring(2, 2)
    factors = dict(d=eye(ring, 2), R=eye(ring, 2), hnext=zeros(ring, 2, 0),
                   s=zeros(ring, 2, 0), q=zeros(ring, 0, 2), Pprev=zeros(ring, 2, 2))

    def forbidden(*args, **kwargs):
        raise AssertionError("An explicit positive witness requested a sparse search")

    monkeypatch.setattr(sparse, "sparse_binary_solution", forbidden)
    X = solve_structural_section(**factors, seed=eye(ring, 2), try_sparse_seed=True,
                                 max_unknowns=0, max_equations=0, max_steps=0, max_coefficient_bits=0)
    actual_four_equations(factors, X)


@pytest.mark.parametrize("prime,enabled", [(2, False), (3, True), (5, True)])
def test_binary_sparse_seed_is_opt_in_and_does_not_replace_odd_prime_solves(monkeypatch, prime, enabled):
    import css_decoupling.algebra.sparse_linear as sparse
    factors = nonunit_step_fixture(2, prime)

    def forbidden(*args, **kwargs):
        raise AssertionError("Sparse binary candidate was called outside its selected characteristic/mode")

    monkeypatch.setattr(sparse, "sparse_binary_solution", forbidden)
    X = solve_structural_section(**step_arguments(factors), try_sparse_seed=enabled, max_radius=1)
    actual_four_equations(factors, X)


@pytest.mark.parametrize("value", [0, 1, None, "true"])
def test_sparse_seed_opt_in_requires_an_actual_boolean(value):
    with pytest.raises(ValueError):
        solve_structural_section(**step_arguments(nonunit_step_fixture(2, 2)), try_sparse_seed=value)
