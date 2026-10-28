"""Independent JSON C4 section checks and source-bound tamper rejection."""
from copy import deepcopy
import json

import pytest

from css_decoupling.algebra.laurent import Matrix, Ring, eye, zeros
from css_decoupling.algebra.structural_c4 import solve_structural_section
from css_decoupling.certificates.comparison import _matrix_dict, _matrix_from_dict
from css_decoupling.certificates.structural_c4 import verify_structural_section


def projective_factors(dimension=3, prime=5, current_Q=False):
    """Explicit degree n=0 or n=1 step, with redundant nonzero Q data."""
    ring = Ring(prime, dimension)
    u = ring.variable(0)**-1
    d = Matrix.from_rows(ring, [[u, 0], [0, 0]])
    R = Matrix.from_rows(ring, [[1, 0], [0, 0]])
    if current_Q:
        s = Matrix.from_rows(ring, [[0, 0], [1, 1]])
        q, Pprev, hnext = zeros(ring, 2, 2), zeros(ring, 2, 2), zeros(ring, 2, 0)
    else:
        s = zeros(ring, 2, 2)
        q = Matrix.from_rows(ring, [[0, 1], [0, 0]])
        Pprev = Matrix.from_rows(ring, [[0, 0], [0, 1]])
        hnext = Matrix.from_rows(ring, [[0], [1]])
    return dict(d=d, R=R, hnext=hnext, s=s, q=q, Pprev=Pprev)


def make_certificate(dimension=3, prime=5, current_Q=False):
    factors = projective_factors(dimension, prime, current_Q)
    X, certificate = solve_structural_section(**factors, max_radius=1, return_certificate=True)
    assert factors["d"]*X == factors["R"]
    assert not X*factors["hnext"]
    assert not X*factors["s"]
    assert not factors["q"]*X
    return factors, X, json.loads(json.dumps(certificate))


def update_actual_resource_fields(data, X):
    """Keep metadata honest so tests reject bad equations, not stale counts."""
    exponents = [e for f in X.entries.values() for e in f.terms]
    data["resources"] = dict(
        monomials=sum(len(f.terms) for f in X.entries.values()),
        max_l1_radius=max((sum(map(abs, e)) for e in exponents), default=0),
        max_canonical_weight=max((sum(e)+(X.ring.D+1)*max(0, -min(e)) for e in exponents), default=0))


@pytest.mark.parametrize("dimension", [2, 3, 4, 5])
@pytest.mark.parametrize("prime", [2, 3, 5])
@pytest.mark.parametrize("current_Q", [False, True])
def test_json_round_trip_retains_all_actual_factors_and_solution(dimension, prime, current_Q):
    factors, X, data = make_certificate(dimension, prime, current_Q)
    report = verify_structural_section(data, expected=factors)
    assert report.valid, report.errors
    assert _matrix_from_dict(data["solution"], factors["d"].ring, "solution") == X


@pytest.mark.parametrize("name", ["d", "R", "hnext", "s", "q", "Pprev"])
def test_every_saved_factor_is_bound_to_the_original_input(name):
    factors, _, data = make_certificate(current_Q=(name == "s"))
    matrix = _matrix_from_dict(data["factors"][name], factors["d"].ring, name)
    assert matrix.nrows and matrix.ncols
    matrix[0, 0] += 1
    data["factors"][name] = _matrix_dict(matrix)
    report = verify_structural_section(data, expected=factors)
    assert not report.valid


@pytest.mark.parametrize("current_Q", [False, True])
def test_solution_coefficient_and_laurent_shift_tampering_fail_actual_products(current_Q):
    factors, _, original = make_certificate(current_Q=current_Q)
    for addition in (factors["d"].ring.one(), factors["d"].ring.variable(1)**-2):
        data = deepcopy(original)
        changed = _matrix_from_dict(data["solution"], factors["d"].ring, "solution")
        changed[0, 0] += addition
        data["solution"] = _matrix_dict(changed)
        update_actual_resource_fields(data, changed)
        assert not verify_structural_section(data, expected=factors).valid


def test_left_inverse_alone_cannot_hide_a_projection_side_condition_violation():
    factors, X, data = make_certificate(current_Q=False)
    X[1, 0] = 1
    assert factors["d"]*X == factors["R"]
    assert factors["q"]*X
    data["solution"] = _matrix_dict(X)
    update_actual_resource_fields(data, X)
    assert not verify_structural_section(data).valid


def test_left_equation_alone_cannot_hide_the_current_homology_side_condition():
    factors, X, data = make_certificate(current_Q=True)
    X[1, 1] = 1
    assert factors["d"]*X == factors["R"]
    assert X*factors["s"]
    data["solution"] = _matrix_dict(X)
    update_actual_resource_fields(data, X)
    assert not verify_structural_section(data).valid


def test_left_equation_alone_cannot_hide_non_square_zero_high_degree_composition():
    factors, X, data = make_certificate(current_Q=False)
    X[1, 1] = 1
    assert factors["d"]*X == factors["R"]
    assert X*factors["hnext"]
    # Remove the previous-Q condition so this specifically needs X*hnext=0.
    data["factors"]["q"] = _matrix_dict(zeros(X.ring, 2, 2))
    data["factors"]["Pprev"] = _matrix_dict(zeros(X.ring, 2, 2))
    data["solution"] = _matrix_dict(X)
    update_actual_resource_fields(data, X)
    assert not verify_structural_section(data).valid


def test_valid_certificate_for_other_inputs_cannot_rebind_expected_source():
    factors, _, _ = make_certificate(current_Q=False)
    other = dict(factors)
    other["d"] = 2*factors["d"]
    _, data = solve_structural_section(**other, return_certificate=True)
    assert verify_structural_section(data, expected=other).valid
    assert not verify_structural_section(data, expected=factors).valid


def test_ordinary_module_solution_cannot_be_relabelled_as_structural_c4():
    factors, _, data = make_certificate()
    ring = factors["d"].ring
    x = ring.variable(0)
    d, R = eye(ring, 2), Matrix.from_rows(ring, [[x, 0], [0, 0]])
    wrong = dict(d=d, R=R, hnext=zeros(ring, 2, 0), s=zeros(ring, 2, 0),
                 q=zeros(ring, 0, 2), Pprev=zeros(ring, 2, 2))
    assert d*R == R
    assert R*R != R
    data["factors"] = {name: _matrix_dict(A) for name, A in wrong.items()}
    data["solution"] = _matrix_dict(R)
    assert not verify_structural_section(data).valid


@pytest.mark.parametrize("missing", ["factors", "solution"])
def test_missing_mathematical_evidence_is_not_recovered_from_success_metadata(missing):
    _, _, data = make_certificate()
    data.pop(missing)
    data["success"] = True
    assert not verify_structural_section(data).valid


@pytest.mark.parametrize("missing", ["d", "R", "hnext", "s", "q", "Pprev"])
def test_no_structural_factor_can_be_silently_defaulted_from_a_certificate(missing):
    _, _, data = make_certificate(current_Q=(missing == "s"))
    data["factors"].pop(missing)
    assert not verify_structural_section(data).valid


def test_wrong_solution_shape_is_rejected_even_if_only_zero_padding_was_added():
    _, _, data = make_certificate()
    data["solution"]["rows"] += 1
    assert not verify_structural_section(data).valid


@pytest.mark.parametrize("field", ["base", "power", "additive", "actual_half_width", "target_rank"])
def test_degree_bound_numbers_are_recomputed_from_actual_factors(field):
    _, _, data = make_certificate()
    data["degree_bound"][field] += 1
    assert not verify_structural_section(data).valid


def test_boundary_qualification_cannot_be_forged_or_replaced_by_conditional_metadata():
    _, _, data = make_certificate()
    data["degree_bound"]["qualification"] = "conditional"
    assert not verify_structural_section(data).valid


@pytest.mark.parametrize("field", ["monomials", "max_l1_radius", "max_canonical_weight"])
@pytest.mark.parametrize("replacement", [0, True])
def test_support_metadata_cannot_replace_actual_counts_or_use_booleans(field, replacement):
    _, _, data = make_certificate()
    assert data["resources"][field] == 1
    data["resources"][field] = replacement
    assert not verify_structural_section(data).valid


@pytest.mark.parametrize("claim", [True, 0, "false"])
def test_positive_support_search_never_becomes_a_complete_nonmembership_proof(claim):
    _, _, data = make_certificate()
    data["search"]["complete_search"] = claim
    assert not verify_structural_section(data).valid


def test_fraction_field_text_cannot_enter_a_finite_laurent_witness():
    _, _, data = make_certificate()
    data["solution"]["entries"][0][2][0][1] = "1/(x1-1)"
    assert not verify_structural_section(data).valid


def test_expected_source_binding_requires_every_factor():
    factors, _, data = make_certificate()
    assert not verify_structural_section(data, expected={"d": factors["d"]}).valid


def test_verification_does_not_rerun_the_construction_solver(monkeypatch):
    factors, _, data = make_certificate()
    import css_decoupling.algebra.structural_c4 as construction
    from css_decoupling.algebra.modules import GeneralBackend

    def forbidden(*args, **kwargs):
        raise AssertionError("Independent certificate verification requested a construction solve")

    monkeypatch.setattr(construction, "solve_structural_section", forbidden)
    monkeypatch.setattr(GeneralBackend, "solve", forbidden)
    report = verify_structural_section(data, expected=factors)
    assert report.valid, report.errors
