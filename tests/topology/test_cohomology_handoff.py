"""Complete-kernel transport, representative provenance and backend boundaries."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from css_decoupling import (Ring, Matrix, coarse_grain, decouple_general as decouple, decouple_complex,
    verify_coarse_graining, verify_construction, build_cohomology_handoff,
    verify_cohomology_handoff, handoff_from_coarse_certificate)
from css_decoupling.algebra.laurent import eye, zeros
from css_decoupling.algebra.modules import GeneralBackend
from css_decoupling.algebra.bounded_linear import BoundedLinearBackend
from css_decoupling.certificates.bounded_linear import verify_trivial_cohomology
from css_decoupling.certificates.comparison import _matrix_dict
from css_decoupling.decoupling.splitting import restrict_complex
from css_decoupling.lattice.retained import Lattice
from css_decoupling.models.examples import toric_css
from css_decoupling.topology.complexes import CochainComplex, toric_stack, disk, direct_sum_complexes
from css_decoupling.topology.cohomology import profile
from css_decoupling.topology.cohomology_handoff import consume_cohomology_handoff


def signed_lattice(D, index=1):
    basis = [[int(i == j) for j in range(D)] for i in range(D)]
    basis[0][0] = -index
    basis[0][1] = 1
    return Lattice(basis)


@pytest.mark.parametrize("D", [2, 3, 4, 5])
@pytest.mark.parametrize("p", [2, 3, 5])
def test_complete_transport_all_dimensions_and_primes(D, p):
    C = toric_stack(Ring(p, D), {1: 1})
    lattice = signed_lattice(D)
    target = restrict_complex(C, lattice)
    data = build_cohomology_handoff(C, lattice, profile(C), restricted=target)
    serialized = json.loads(json.dumps(data))
    proof = verify_cohomology_handoff(serialized, complex=target)
    assert proof.valid, proof.errors
    layers = consume_cohomology_handoff(serialized, target)
    assert {n: layer.dimension for n, layer in layers.items()} == {1: 1}


def test_nontrivial_original_action_is_not_copied_by_cell_index():
    R = Ring(3, 2)
    x, y = R.variables
    f, g = x + 1, y - 1
    C = CochainComplex(R, {-1: 1, 0: 2, 1: 1},
                       {-1: Matrix.from_rows(R, [[-g], [f]]),
                        0: Matrix.from_rows(R, [[f, g]])})
    layers = profile(C)
    assert layers[1].actions[0] != eye(R, 1)
    L = Lattice([[2, 1], [0, 1]])
    # The second vector (1,1) does not kill x=-1.
    with pytest.raises(ValueError):
        build_cohomology_handoff(C, L, layers)
    L = Lattice([[-2, 0], [0, 1]])
    data = build_cohomology_handoff(C, L, layers)
    assert verify_cohomology_handoff(data).valid
    assert data["layers"][0]["layer"]["dimension"] == 1
    assert dict(data["target"]["ranks"])[0] == 4


def redundant_fixture():
    R = Ring(5, 3)
    C = direct_sum_complexes(toric_stack(R, {1: 1}), disk(R, 2))
    layers = profile(C)
    old = layers[1]
    W = old.cycles.hstack(old.cycles[:, [0]])
    B = old.boundary_coordinates.vstack(zeros(R, 1, old.boundary_coordinates.ncols))
    layers[1] = SimpleNamespace(cycles=W, boundary_coordinates=B,
        cycle_relations=GeneralBackend(R).kernel(W), representatives=R.variable(0) * old.representatives)
    L = signed_lattice(3, 2)
    return C, restrict_complex(C, L), build_cohomology_handoff(C, L, layers)


def test_redundant_generators_and_all_zero_layers_are_certified():
    C, target, data = redundant_fixture()
    assert verify_cohomology_handoff(data, complex=target).valid
    assert [(r["degree"], r["layer"]["dimension"]) for r in data["layers"]] == [(1, 1), (2, 0), (3, 0)]
    row = data["layers"][0]
    assert row["boundary_lift"]["entries"]
    bad = deepcopy(data)
    bad["layers"].pop()
    assert not verify_cohomology_handoff(bad, complex=target).valid
    # Dropping kernel relations still satisfies W*K=0, but is incomplete.
    bad = deepcopy(data)
    source = bad["layers"][0]["layer"]["relation_kernel_certificate"]["source"]
    source["cycle_kernel"] = _matrix_dict(zeros(C.ring, source["cycles"]["cols"], 0))
    assert not verify_cohomology_handoff(bad, complex=target).valid


@pytest.mark.parametrize("field", ["basis_change", "basis_change_inverse", "boundary_lift", "source_coordinates"])
def test_representative_binding_tampering_is_rejected(field):
    C, target, data = redundant_fixture()
    bad = deepcopy(data)
    bad["layers"][0][field]["entries"] = []
    assert not verify_cohomology_handoff(bad, complex=target).valid


def test_operator_lattice_and_translation_tampering_is_rejected():
    C, target, data = redundant_fixture()
    for location in ("source", "target", "basis", "translation"):
        bad = deepcopy(data)
        if location in ("source", "target"):
            bad[location]["ranks"][0][1] += 1
        elif location == "basis":
            bad["basis"][0][0] = -1
        else:
            bad["layers"][0]["layer"]["triviality_witnesses"][0]["entries"] = []
        assert not verify_cohomology_handoff(bad, complex=target).valid


def test_handoff_cannot_silently_run_native_in_bounded_mode(monkeypatch):
    C, target, data = redundant_fixture()
    bounded = BoundedLinearBackend(C.ring)
    def forbidden(*args, **kwargs):
        raise AssertionError("native kernel was used")
    monkeypatch.setattr(GeneralBackend, "kernel", forbidden)
    assert not verify_cohomology_handoff(data, complex=target, backend=bounded).valid
    assert not verify_trivial_cohomology(data["layers"][0]["layer"], complex=target, backend=bounded).valid
    with pytest.raises(ValueError, match="handoff"):
        decouple_complex(target, linear_backend="bounded", cohomology_handoff=data)


def test_independent_verification_does_not_trust_construction_kernel_cache():
    C, target, data = redundant_fixture()
    construction_backend = GeneralBackend(C.ring)
    def forbidden(*args, **kwargs):
        raise AssertionError("construction backend/cache was trusted by verifier")
    construction_backend.kernel = forbidden
    construction_backend.solve = forbidden
    proof = verify_cohomology_handoff(data, complex=target, backend=construction_backend)
    assert proof.valid, proof.errors


def test_complete_kernels_are_checked_on_original_not_expanded_operators(monkeypatch):
    C, target, data = redundant_fixture()
    maximum_original_rank = max(C.ranks.values())
    assert max(target.ranks.values()) > maximum_original_rank
    original_kernel = GeneralBackend.kernel
    observed = []
    def checked_kernel(self, operator):
        observed.append(operator.shape)
        assert max(operator.shape) <= maximum_original_rank
        return original_kernel(self, operator)
    monkeypatch.setattr(GeneralBackend, "kernel", checked_kernel)
    proof = verify_cohomology_handoff(data, complex=target)
    assert proof.valid, proof.errors
    assert observed


def test_optional_coarse_proof_rejects_stale_lattice_without_changing_old_v1():
    from css_decoupling.certificates.analysis import complex_dict
    HX, HZ = toric_css(Ring(3, 2), {1: 1})
    coarse = coarse_grain(HX, HZ, backend="sympy")
    data = coarse.certificate(include_handoff=True)
    lattice = signed_lattice(2)
    data["basis"] = [list(row) for row in lattice.basis]
    data["restricted"] = complex_dict(restrict_complex(coarse.complex, lattice))
    assert not verify_coarse_graining(data).valid
    data.pop("cohomology_handoff")
    assert verify_coarse_graining(data).valid


def test_native_pipeline_and_old_v1_coarse_json_use_bound_handoff(monkeypatch):
    HX, HZ = toric_css(Ring(3, 2), {1: 1})
    coarse = coarse_grain(HX, HZ, backend="sympy")
    assert "cohomology_handoff" not in coarse.certificate()
    encoded = json.loads(json.dumps(coarse.certificate(include_handoff=True)))
    assert verify_coarse_graining(encoded).valid
    old = deepcopy(encoded)
    old.pop("cohomology_handoff")
    assert verify_coarse_graining(old).valid
    restored = handoff_from_coarse_certificate(old)
    assert verify_cohomology_handoff(restored, complex=coarse.restricted).valid
    def forbidden(*args, **kwargs):
        raise AssertionError("restricted cohomology was rediscovered")
    monkeypatch.setattr("css_decoupling.topology.trivial_cohomology.trivial_cohomology", forbidden)
    result = decouple_complex(coarse)
    certificate = json.loads(json.dumps(result.certificate()))
    assert certificate["cohomology_handoff"]
    assert verify_construction(certificate).valid
    pipeline = decouple(HX, HZ, backend="sympy", resolution_method="native")
    assert pipeline.trace["construction"].cohomology_handoff
    assert verify_construction(pipeline.trace["construction"]).valid


def test_construction_optional_evidence_remains_source_bound():
    HX, HZ = toric_css(Ring(2, 2), {1: 1})
    result = decouple_complex(coarse_grain(HX, HZ, backend="sympy"))
    data = json.loads(json.dumps(result.certificate()))
    data["cohomology"][0]["representatives"]["entries"] = []
    assert not verify_construction(data).valid


def test_composed_handoff_binds_the_actual_algorithm_one_representatives():
    HX, HZ = toric_css(Ring(3, 2), {1: 1})
    result = decouple_complex(coarse_grain(HX, HZ, backend="sympy"))
    data = json.loads(json.dumps(result.certificate()))
    coarse = data["coarse_graining"]
    # This is a valid different basis of the SAME original cohomology, so
    # checking each certificate separately must not hide the provenance swap.
    for row, column, terms in coarse["layers"][0]["representatives"]["entries"]:
        for entry in terms:
            entry[1] = 2 * entry[1] % 3
    assert verify_coarse_graining(coarse).valid
    report = verify_construction(data)
    assert not report.valid
    assert "Algorithm 1 representatives" in "; ".join(report.errors)


def test_standalone_handoff_rejects_huge_metadata_before_expansion(monkeypatch):
    from css_decoupling import ResourceLimit
    from css_decoupling.certificates.cohomology_handoff import checked_handoff
    C, target, data = redundant_fixture()
    def forbidden(*args, **kwargs):
        raise AssertionError("oversized metadata reached matrix decoding or coset expansion")
    monkeypatch.setattr("css_decoupling.certificates.cohomology_handoff._read_complex", forbidden)
    monkeypatch.setattr(Lattice, "restrict", forbidden)
    huge = deepcopy(data)
    huge["basis"] = [[10**12, 0, 0], [0, 1, 0], [0, 0, 1]]
    with pytest.raises(ResourceLimit, match="index"):
        checked_handoff(huge)
    with pytest.raises(ResourceLimit, match="total free rank"):
        checked_handoff(huge, max_cell_index=None)
    huge = deepcopy(data)
    huge["target"]["ranks"][0][1] = 10**12
    with pytest.raises(ResourceLimit, match="total free rank"):
        verify_cohomology_handoff(huge)


@pytest.mark.parametrize("kind", ["kernel", "cycle-boundary"])
def test_standalone_kernel_proof_budget_precedes_kernel_and_identity_allocation(kind, monkeypatch):
    from css_decoupling import ResourceLimit
    from css_decoupling.certificates.cohomology_handoff import check_restricted_kernel
    C, target, data = redundant_fixture()
    raw = data["layers"][0]["layer"]
    proof = deepcopy(raw["kernel_certificate" if kind == "kernel" else "relation_kernel_certificate"])
    if kind == "kernel":
        proof["source"]["operator"]["cols"] = 10**12
    else:
        proof["source"]["boundary"]["cols"] = 10**12
    def forbidden(*args, **kwargs):
        raise AssertionError("oversized shape reached kernel work or output allocation")
    monkeypatch.setattr(GeneralBackend, "kernel", forbidden)
    monkeypatch.setattr(Lattice, "restrict", forbidden)
    monkeypatch.setattr("css_decoupling.certificates.cohomology_handoff.eye", forbidden)
    with pytest.raises(ResourceLimit, match="total free rank"):
        check_restricted_kernel(proof)


def test_explicit_larger_or_none_index_budget_reaches_nested_kernel_proofs():
    from css_decoupling import ResourceLimit
    R = Ring(3, 2)
    C = CochainComplex(R, {-1: 0, 0: 0, 1: 0}, {})
    L = Lattice([[5000, 0], [0, 1]], max_index=None)
    with pytest.raises(ResourceLimit, match="index"):
        build_cohomology_handoff(C, L, {})
    data = build_cohomology_handoff(C, L, {}, max_cell_index=6000)
    assert verify_cohomology_handoff(data, max_cell_index=6000).valid
    assert verify_cohomology_handoff(data, max_cell_index=None, max_total_rank=None).valid
    assert consume_cohomology_handoff(data, C, max_cell_index=None, max_total_rank=None) == {}


def test_explicit_none_total_rank_does_not_reset_to_standalone_default():
    from css_decoupling import ResourceLimit
    from css_decoupling.certificates.cohomology_handoff import KERNEL_SCHEMA, check_restricted_kernel
    R = Ring(2, 2)
    # Redundant zero generators have large formal rank but no allocated
    # coefficients; this checks an opt-out without a large actual matrix.
    proof = {"schema": KERNEL_SCHEMA, "version": 1, "ring": {"prime": 2, "dimension": 2},
             "kind": "kernel", "basis": [[1, 0], [0, 1]],
             "source": {"operator": _matrix_dict(zeros(R, 0, 0)),
                        "generators": _matrix_dict(zeros(R, 0, 100001))}}
    with pytest.raises(ResourceLimit, match="total free rank"):
        check_restricted_kernel(proof)
    A, K = check_restricted_kernel(proof, max_total_rank=None)
    assert A.shape == (0, 0) and K.shape == (0, 100001)


@pytest.mark.parametrize("budget", ["max_cell_index", "max_total_rank"])
@pytest.mark.parametrize("value", [True, -1, 0, 1.5])
def test_standalone_resource_caps_require_positive_integers(budget, value):
    C, target, data = redundant_fixture()
    assert not verify_cohomology_handoff(data, **{budget: value}).valid
