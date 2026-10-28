"""Exact Fourier conjugation, swapped disk sectors and dual target certificates."""
from copy import deepcopy
import json

import pytest

from css_decoupling.algebra.laurent import Ring, eye
from css_decoupling.circuits.gates import Circuit, SUM, SWAP
from css_decoupling.topology.complexes import koszul
from css_decoupling.decoupling.dual import (
    DualDecouplingResult, dualize_circuit,
)
from css_decoupling.certificates.dual import verify_dual_certificate
from css_decoupling.certificates.formats import DUAL_TARGET_BASIS as TARGET_BASIS
from css_decoupling.decoupling.pipeline import decouple
from css_decoupling.certificates.decoupling import verify_decoupling
from css_decoupling.algebra.modules import GeneralBackend, get_backend, using_backend


def test_dual_gate_sign_displacement_and_order_are_exact():
    r = Ring(5, 4)
    circuit = Circuit(r, 3, (SUM(0, 2, 2, (1, -1, 0, 0)), SWAP(1, 2),
                             SUM(2, 1, 4, (-2, 0, 0, 3))))
    dual = dualize_circuit(circuit)
    assert dual.gates[0] == SUM(2, 0, 3, (-1, 1, 0, 0))
    assert dual.gates[1] == SWAP(1, 2)
    assert dual.gates[2] == SUM(1, 2, 1, (2, 0, 0, -3))
    assert dual.matrix("Z") == circuit.matrix("X")
    assert dual.matrix("X") == circuit.matrix("Z")
    assert dual.matrix("Z") == circuit.inverse().matrix("Z").dagger()
    assert dualize_circuit(dual) == circuit


def test_explicit_backend_remains_active_during_dual_verification(monkeypatch):
    import css_decoupling.decoupling.dual as dual_module
    r = Ring(3, 2)
    C = koszul(r)
    original_backend = get_backend(r)
    selected_backend = GeneralBackend(r)
    original_verifier = dual_module.verify_dual_certificate
    observed = []

    def recording_verifier(value):
        observed.append(get_backend(r) is selected_backend)
        return original_verifier(value)

    monkeypatch.setattr(dual_module, "verify_dual_certificate", recording_verifier)
    result = decouple(C.d(0), C.d(-1).dagger(), sector="Z", backend=selected_backend,
                     resolution_timeout=20)
    assert result.sector == "Z"
    assert observed == [True]
    assert get_backend(r) is original_backend


@pytest.fixture(scope="module", params=[(3, 3, 2), (5, 4, 3)])
def dual_result(request):
    p, D, degree = request.param
    C = koszul(Ring(p, D), degree=degree)
    with using_backend("sympy"):
        result = decouple(C.d(0), C.d(-1).dagger(), sector="Z", resolution_timeout=20)
        assert isinstance(result, DualDecouplingResult)
        assert result.multiplicities == {degree: 1}
        assert result.inner.multiplicities == {D - degree: 1}
        yield result


def test_dual_pipeline_original_checks_disks_and_json(dual_result):
    result = dual_result
    assert result.sector == "Z"
    assert result.target_basis == TARGET_BASIS
    assert not hasattr(result, "stabilization")
    assert result.identity_checks["sector"] == "Z"
    inner_disks = result.inner.stabilization.ancillas
    assert result.ancillas["source_X_fixed"] == inner_disks.source_degree_minus_one
    assert result.ancillas["source_Z_fixed"] == inner_disks.source_degree_zero
    assert result.ancillas["target_X_fixed"] == inner_disks.target_degree_minus_one
    assert result.ancillas["target_Z_fixed"] == inner_disks.target_degree_zero
    inner = result.inner.stabilization
    assert result.circuit.matrix("Z") == inner.Uinv.dagger()
    assert result.circuit.matrix("X") == inner.U
    for candidate in [result, json.loads(json.dumps(result.certificate()))]:
        report = verify_decoupling(candidate)
        assert report.valid, report.errors
        assert report.scope == "original-CSS-to-dual-toric-stack-circuit"
        assert "original X and Z stabilizer images match the dual target in both directions" in report.checks


@pytest.mark.parametrize("mutation", ["input", "target-degree", "target-basis", "SUM-sign",
                                       "displacement", "ancilla", "check-sector", "inner-matrix",
                                       "recursive-wrapper", "extra-success"])
def test_dual_certificate_tampering_is_rejected(dual_result, mutation):
    data = deepcopy(dual_result.certificate())
    if mutation == "input":
        data["input"]["HX"]["entries"].pop()
    elif mutation == "target-degree":
        data["target"]["multiplicities"][0][0] += 1
    elif mutation == "target-basis":
        data["target"]["basis"] = "canonical-primal"
    elif mutation == "SUM-sign":
        gate = next(g for g in data["circuit"]["gates"] if g["kind"] == "SUM")
        gate["weight"] = (gate["weight"] + 1) % dual_result.HX.ring.p
    elif mutation == "displacement":
        gate = next(g for g in data["circuit"]["gates"] if g["kind"] == "SUM")
        gate["displacement"][0] += 1
    elif mutation == "ancilla":
        data["ancillas"]["source_Z_fixed"] += 1
    elif mutation == "check-sector":
        data["identity_checks"]["sector"] = "X"
    elif mutation == "inner-matrix":
        data["inner"]["stabilization"]["stabilization"]["Uinv"]["entries"].pop()
    elif mutation == "recursive-wrapper":
        data["inner"]["schema"] = data["schema"]
    elif mutation == "extra-success":
        data["success"] = True
    assert not verify_dual_certificate(data).valid
    assert not verify_decoupling(data).valid
