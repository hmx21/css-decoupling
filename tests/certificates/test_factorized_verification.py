"""Check all saved blocks and actual gate products in the factorized verifier."""
from dataclasses import replace
from unittest.mock import patch

import pytest

from css_decoupling.algebra.laurent import Matrix, Ring, eye, zeros
from css_decoupling.certificates.comparison import (
    _adjoint_matches, certificate_dict, verify_certificate,
)
from css_decoupling.circuits.gates import Circuit, SUM
from css_decoupling.decoupling.stabilization import LiftData, stabilize


def _example(prime):
    R = Ring(prime, 3)
    x = R.variable(0)
    # Unequal physical ranks and nonzero inverse-defect witnesses. The
    # last physical coordinate in each complex is an actual unit disk.
    return stabilize(LiftData(
        Matrix.from_rows(R, [[x - 1, 0, 1]]), Matrix.from_rows(R, [[x - 1, 1]]),
        zeros(R, 3, 2), zeros(R, 2, 3), zeros(R, 1, 1), zeros(R, 1, 1),
        Matrix.from_rows(R, [[0], [0], [1]]), Matrix.from_rows(R, [[0], [1]]),
    ))


@pytest.mark.parametrize("prime", [2, 3, 5])
def test_block_proofs_agree_with_expanded_inverse_square_and_pauli_products(prime):
    result = _example(prime)
    assert verify_certificate(result).valid
    R, n = result.U.ring, result.U.nrows
    assert result.U * result.Uinv == result.Uinv * result.U == eye(R, n)
    assert result.V * result.Vinv == result.Vinv * result.V == eye(R, result.V.nrows)
    assert result.dThat * result.U == result.V * result.dChat
    Z, X = result.circuit.matrix("Z"), result.circuit.matrix("X")
    assert Z.dagger() * X == eye(R, n)
    assert Z == result.U and X == result.Uinv.dagger()
    assert result.circuit.inverse().matrix("Z") == result.Uinv
    assert verify_certificate(certificate_dict(result)).valid


def test_verification_does_not_reexpand_the_two_saved_square_inverses():
    result = _example(5)
    original = Matrix.__mul__
    forbidden = {(id(A), id(B)) for A, B in ((result.U, result.Uinv),
                 (result.Uinv, result.U), (result.V, result.Vinv), (result.Vinv, result.V))}

    def checked(left, right):
        if isinstance(right, Matrix):
            assert (id(left), id(right)) not in forbidden
        return original(left, right)

    with patch.object(Matrix, "__mul__", checked):
        assert verify_certificate(result).valid


@pytest.mark.parametrize("which", ["physical", "checks"])
def test_coherently_changed_inverse_pair_does_not_bypass_saved_theta_blocks(which):
    result = _example(3)
    R = result.U.ring
    A, B = (result.U, result.Uinv) if which == "physical" else (result.V, result.Vinv)
    shear = Matrix(R, A.nrows, A.nrows, {(0, 1): R.variable(1)**-1})
    changed, inverse = (eye(R, A.nrows) + shear) * A, B * (eye(R, A.nrows) - shear)
    assert changed * inverse == inverse * changed == eye(R, A.nrows)
    forged = replace(result, **({"U": changed, "Uinv": inverse} if which == "physical"
                                else {"V": changed, "Vinv": inverse}))
    report = verify_certificate(forged)
    assert not report.valid
    assert ("physical Theta formula" if which == "physical" else "check Theta formula") in report.errors


@pytest.mark.parametrize("field", ["phihat", "psihat", "dChat", "dThat", "U", "Uinv", "V", "Vinv"])
def test_every_saved_matrix_is_checked_after_live_object_mutation(field):
    result = _example(5)
    matrix = getattr(result, field)
    matrix[0, 0] += result.U.ring.variable(2)**-1
    assert not verify_certificate(result).valid


@pytest.mark.parametrize("action", ["Z", "X", "inverse-Z"])
def test_each_actual_gate_product_is_required_independently(action):
    result = _example(3)
    original = Circuit.matrix
    forward = result.circuit

    def changed(circuit, pauli="Z"):
        matrix = original(circuit, pauli)
        if ((action == "Z" and circuit is forward and pauli == "Z")
                or (action == "X" and circuit is forward and pauli == "X")
                or (action == "inverse-Z" and circuit is not forward and pauli == "Z")):
            matrix[0, 0] += 1
        return matrix

    with patch.object(Circuit, "matrix", changed):
        report = verify_certificate(result)
    assert not report.valid
    assert {"Z": "gate Z product", "X": "gate X product",
            "inverse-Z": "inverse gate Z product"}[action] in report.errors


def test_changed_recorded_gate_weight_and_displacement_are_not_trusted():
    result = _example(5)
    original = result.circuit.gates
    position = next(i for i, gate in enumerate(original) if isinstance(gate, SUM))
    for change in ("weight", "displacement"):
        gates = list(original)
        gate = gates[position]
        gates[position] = replace(gate, **({"weight": gate.weight + 1} if change == "weight" else
                                          {"displacement": (3, -2, 1)}))
        forged = replace(result, circuit=Circuit(result.U.ring, result.U.nrows, tuple(gates)))
        assert not verify_certificate(forged).valid


def test_adjoint_comparison_checks_every_exponent_and_field_coefficient():
    R = Ring(5, 3)
    x, y, z = R.variables
    A = Matrix.from_rows(R, [[2*x**-2 + 3*y, z], [0, 4*x*y**-1]])
    assert _adjoint_matches(A.dagger(), A)
    wrong = A.dagger()
    wrong[0, 0] += x**2
    assert not _adjoint_matches(wrong, A)
