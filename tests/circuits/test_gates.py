"""Independent signs, rectangular stabilization and adversarial certificate checks."""
from copy import deepcopy
from dataclasses import replace

import pytest

from css_decoupling.algebra.laurent import Matrix, Ring, eye, zeros
from css_decoupling.certificates.comparison import (
    CertificateFormatError, certificate_dict, certificate_from_dict,
    dumps_certificate, loads_certificate, verify_certificate,
)
from css_decoupling.circuits.gates import Circuit, SUM, SWAP, compile_theta
from css_decoupling.decoupling.stabilization import LiftData, stabilize, theta


def comparison_fixture(p=3, dimension=4):
    """Unequal ranks, an X disk, a Z disk, and nonconstant basis changes."""
    r = Ring(p, dimension)
    x, y = r.variable(0), r.variable(1)
    dT = Matrix(r, 1, 2, {(0, 0): x - 1})
    base = Matrix(r, 2, 4, {(0, 0): x - 1, (1, 2): 1})
    shear = Matrix(r, 4, 4, {(0, 3): 2 * x * y ** -1,
                            (1, 2): y + x ** -1})
    P, Pinv = eye(r, 4) + shear, eye(r, 4) - shear
    row_shear = Matrix(r, 2, 2, {(0, 1): x ** -1 + 2 * y})
    Q, Qinv = eye(r, 2) + row_shear, eye(r, 2) - row_shear
    inclusion0 = Matrix(r, 4, 2, {(0, 0): 1, (1, 1): 1})
    projection0 = Matrix(r, 2, 4, {(0, 0): 1, (1, 1): 1})
    inclusion1 = Matrix(r, 2, 1, {(0, 0): 1})
    projection1 = Matrix(r, 1, 2, {(0, 0): 1})
    eta = Matrix(r, 4, 2, {(2, 1): 1})
    return LiftData(
        dC=Q * base * Pinv, dT=dT,
        phi0=P * inclusion0, psi0=projection0 * Pinv,
        phi1=Q * inclusion1, psi1=projection1 * Qinv,
        etaC=P * eta * Qinv, etaT=zeros(r, 2, 1),
    )


@pytest.mark.parametrize("p,dimension", [(2, 3), (3, 4), (5, 5)])
def test_rectangular_stabilization_and_json_roundtrip(p, dimension):
    result = stabilize(comparison_fixture(p, dimension))
    assert result.U.shape == (9, 9)
    assert result.V.shape == (3, 3)
    assert result.ancillas.source_degree_zero == 1
    assert result.ancillas.source_degree_minus_one == 4
    report = verify_certificate(result)
    assert report.valid, report.errors
    encoded = dumps_certificate(result)
    parsed = loads_certificate(encoded)
    assert parsed.U == result.U
    assert parsed.circuit == result.circuit
    assert verify_certificate(encoded).valid
    assert "success" not in certificate_dict(result)


@pytest.mark.parametrize("a,b", [(1, 3), (4, 2), (0, 2), (3, 0), (0, 0)])
def test_theta_rectangular_permutation_and_inverse(a, b):
    r = Ring(5, 4)
    f = Matrix(r, b, a, {(i, j): r.variable((i + j) % 4) + (i + j + 1)
                          for i in range(b) for j in range(a)})
    g = Matrix(r, a, b, {(i, j): r.variable((i + 2 * j) % 4) ** -1
                          for i in range(a) for j in range(b)})
    circuit = compile_theta(f, g)
    assert circuit.matrix() == theta(f, g)
    assert circuit.inverse().matrix() == theta(g, f)
    assert theta(f, g) * theta(g, f) == eye(r, a + b)
    assert circuit.matrix("X") == theta(g, f).dagger()
    monomials = sum(len(poly.terms) for m in (f, g) for poly in m.entries.values())
    assert sum(isinstance(gate, SUM) for gate in circuit.gates) == monomials
    assert sum(isinstance(gate, SWAP) for gate in circuit.gates) <= a + b


def _periodic_matrix(matrix, cells):
    """Independent first-axis cyclic specialization; other axes have length 1."""
    p = matrix.ring.p
    out = [[0] * (matrix.ncols * cells) for _ in range(matrix.nrows * cells)]
    for (i, j), poly in matrix.entries.items():
        for exponent, coefficient in poly.terms.items():
            for cell in range(cells):
                row = ((cell + exponent[0]) % cells) * matrix.nrows + i
                col = cell * matrix.ncols + j
                out[row][col] = (out[row][col] + coefficient) % p
    return out


@pytest.mark.parametrize("p", [3, 5])
def test_sum_weight_displacement_and_physical_pauli_conjugation(p):
    r = Ring(p, 4)
    displacement = (1, 0, 0, 0)
    gate = SUM(0, 1, 2, displacement)
    circuit = Circuit(r, 2, (gate,))
    cells, sites = 3, 6
    z = _periodic_matrix(circuit.matrix("Z"), cells)
    x = _periodic_matrix(circuit.matrix("X"), cells)
    # Check the conjugated Z phase on computational basis coordinate vectors.
    # S Z(v) S^-1 |a> has phase v dot (S^-1 a); this tests sign and cell direction.
    for pauli_site in range(sites):
        for state_site in range(sites):
            state = [int(i == state_site) for i in range(sites)]
            inverse_state = state[:]
            for cell in range(cells):
                control = ((cell + 1) % cells) * 2
                target = cell * 2 + 1
                inverse_state[target] = (state[target] - gate.weight * state[control]) % p
            actual_phase = z[state_site][pauli_site]
            assert actual_phase == inverse_state[pauli_site]
    # Conjugation of a unit X shift is S(e_i), computed directly on ket labels.
    for pauli_site in range(sites):
        ket = [int(i == pauli_site) for i in range(sites)]
        output = ket[:]
        for cell in range(cells):
            control = ((cell + 1) % cells) * 2
            target = cell * 2 + 1
            output[target] = (ket[target] + gate.weight * ket[control]) % p
        assert [x[row][pauli_site] for row in range(sites)] == output
    assert circuit.inverse().matrix() * circuit.matrix() == eye(r, 2)


def test_lift_defect_rejected_before_construction():
    lifts = comparison_fixture()
    with pytest.raises(ValueError, match="C inverse defect"):
        stabilize(replace(lifts, etaC=zeros(lifts.dC.ring, 4, 2)))


def test_stabilization_with_both_inverse_defects_nonzero():
    r = Ring(5, 4)
    x, y = r.variable(0), r.variable(1)
    dC = Matrix(r, 2, 4, {(0, 0): x - 1, (1, 2): 1})
    dT = Matrix(r, 2, 3, {(0, 0): x - 1, (1, 2): 2 * y})
    phi0 = Matrix(r, 4, 3, {(0, 0): 1, (1, 1): 1})
    psi0 = Matrix(r, 3, 4, {(0, 0): 1, (1, 1): 1})
    projection = Matrix(r, 2, 2, {(0, 0): 1})
    etaC = Matrix(r, 4, 2, {(2, 1): 1})
    etaT = Matrix(r, 3, 2, {(2, 1): (2 * y) ** -1})
    result = stabilize(LiftData(dC, dT, phi0, psi0, projection, projection, etaC, etaT))
    report = verify_certificate(result)
    assert report.valid, report.errors
    assert result.U.shape == (11, 11)


@pytest.mark.parametrize("field", ["U", "Uinv", "V", "Vinv", "dChat", "phihat"])
def test_tampered_matrix_rejected(field):
    result = stabilize(comparison_fixture())
    data = certificate_dict(result)
    encoded = data["stabilization"][field]
    encoded["entries"].pop()
    assert not verify_certificate(data).valid


def test_tampered_gate_or_ancilla_rejected():
    result = stabilize(comparison_fixture())
    data = certificate_dict(result)
    tampered = deepcopy(data)
    gate = next(g for g in tampered["circuit"]["gates"] if g["kind"] == "SUM")
    gate["weight"] = (gate["weight"] + 1) % 3
    assert not verify_certificate(tampered).valid
    tampered = deepcopy(data)
    tampered["ancillas"]["source_degree_zero"] += 1
    assert not verify_certificate(tampered).valid
    tampered = deepcopy(data)
    tampered["circuit"]["gates"].reverse()
    assert not verify_certificate(tampered).valid


def test_malformed_ring_exponent_duplicate_and_schema_rejected():
    result = stabilize(comparison_fixture())
    data = certificate_dict(result)
    malformed = []
    item = deepcopy(data)
    item["ring"]["prime"] = 4
    malformed.append(item)
    item = deepcopy(data)
    item["lifts"]["dC"]["entries"][0][2][0][0].append(0)
    malformed.append(item)
    item = deepcopy(data)
    item["lifts"]["dC"]["entries"].append(item["lifts"]["dC"]["entries"][0])
    malformed.append(item)
    item = deepcopy(data)
    item["lifts"]["dC"]["entries"][0][2][0][1] = True
    malformed.append(item)
    item = deepcopy(data)
    item["version"] = True
    malformed.append(item)
    item = deepcopy(data)
    item["success"] = True
    malformed.append(item)
    for item in malformed:
        with pytest.raises(CertificateFormatError):
            certificate_from_dict(item)
        assert not verify_certificate(item).valid
    with pytest.raises(CertificateFormatError, match="Duplicate JSON key"):
        loads_certificate('{"schema": "a", "schema": "b"}')


def test_live_object_tampering_is_not_trusted():
    result = stabilize(comparison_fixture())
    result.U.entries.clear()
    report = verify_certificate(result)
    assert not report.valid
    assert "physical Theta formula" in report.errors
    assert "gate Z product" in report.errors


@pytest.mark.parametrize("action", ["Z", "X"])
def test_sparse_row_replay_equals_independent_elementary_matrix_products(action):
    result = stabilize(comparison_fixture(5, 4))
    circuit = result.circuit
    expected = eye(circuit.ring, circuit.qudits)
    for gate in circuit.gates:
        expected = gate.matrix(circuit.ring, circuit.qudits, action) * expected
    assert circuit.matrix(action) == expected
    assert circuit.inverse().matrix(action) * expected == eye(circuit.ring, circuit.qudits)


@pytest.mark.parametrize("action", ["Z", "X"])
def test_mutable_support_replay_preserves_cancellation_and_swap_isolation(action):
    r = Ring(5, 4)
    # Accumulate a long entry using one new monomial per layer, mix its row,
    # then exactly cancel the whole sequence. Every row/entry must own its
    # mutable support: aliasing across a SWAP would corrupt this identity.
    first = tuple(SUM(0, 1, (k % 4) + 1, (k - 24, -k, 0, 1)) for k in range(48))
    middle = (SWAP(0, 2), SUM(1, 2, 3, (-2, 1, 0, 0)), SWAP(1, 2))
    forward = Circuit(r, 3, first + middle)
    circuit = Circuit(r, 3, forward.gates + forward.inverse().gates)
    expected = eye(r, 3)
    for gate in circuit.gates:
        expected = gate.matrix(r, 3, action) * expected
    assert expected == eye(r, 3)
    assert circuit.matrix(action) == expected
    # Also check the nonidentity intermediate, before cancellation can mask
    # missing terms or an incorrect sign.
    intermediate = eye(r, 3)
    for gate in forward.gates:
        intermediate = gate.matrix(r, 3, action) * intermediate
    assert forward.matrix(action) == intermediate
