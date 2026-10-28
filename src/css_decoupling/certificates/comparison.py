"""Versioned exact witnesses for the comparison-lifts-to-circuit stage.

Verification recomputes every identity and both Pauli actions from actual
matrix/gate data.  These certificates do not claim that dT is a toric stack,
that the original CSS complex is exact, or that the comparison data were
obtained by a complete automatic decoupling algorithm.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import json
from typing import Any

from css_decoupling.algebra.laurent import Matrix, Poly, Ring, block, direct_sum, eye, zeros
from css_decoupling.circuits.gates import Circuit, SUM, SWAP
from css_decoupling.decoupling.stabilization import DiskAncillas, LiftData, Stabilization

SCHEMA = "css-decoupling.general.stabilization"
SCOPE = "verified-comparison-lifts-to-stabilized-circuit"
CONVENTIONS = {
    "vectors": "columns",
    "differential": "degree-zero-is-HX",
    "phi": "T-to-C",
    "psi": "C-to-T",
    "pauli": "Z=U;X=inverse-dagger(U)",
    "sum": "control-at-r+displacement;target-at-r;ket(a,b)->ket(a,b+weight*a)",
    "execution_order": "first-listed-gate-first",
}
_MATRIX_NAMES = ("phihat", "psihat", "dChat", "dThat", "U", "Uinv", "V", "Vinv")
_LIFT_NAMES = tuple(field.name for field in fields(LiftData))
_ANCILLA_NAMES = tuple(field.name for field in fields(DiskAncillas))


class CertificateFormatError(ValueError):
    """Malformed or incompatible certificate, distinct from a false identity."""


@dataclass(frozen=True)
class VerificationReport:
    valid: bool
    checks: tuple[str, ...]
    errors: tuple[str, ...]
    scope: str = SCOPE


def _keys(value: Any, names: tuple[str, ...] | set[str], context: str) -> dict:
    if not isinstance(value, dict) or set(value) != set(names):
        raise CertificateFormatError(f"{context} must contain exactly {sorted(names)}")
    return value


def _int(value: Any, context: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CertificateFormatError(f"{context} must be an integer")
    if minimum is not None and value < minimum:
        raise CertificateFormatError(f"{context} must be >= {minimum}")
    return value


def _list(value: Any, context: str, length: int | None = None) -> list:
    if not isinstance(value, list) or (length is not None and len(value) != length):
        suffix = "" if length is None else f" of length {length}"
        raise CertificateFormatError(f"{context} must be a list{suffix}")
    return value


def _matrix_dict(matrix: Matrix) -> dict:
    return {"rows": matrix.nrows, "cols": matrix.ncols,
            "entries": [[i, j, [[list(exp), coeff]
                                  for exp, coeff in sorted(poly.terms.items())]]
                        for (i, j), poly in sorted(matrix.entries.items())]}


def _matrix_from_dict(data: Any, ring: Ring, context: str) -> Matrix:
    data = _keys(data, ("rows", "cols", "entries"), context)
    rows, cols = _int(data["rows"], context + ".rows", 0), _int(data["cols"], context + ".cols", 0)
    entries = {}
    for index, encoded in enumerate(_list(data["entries"], context + ".entries")):
        i, j, monomials = _list(encoded, f"{context}.entries[{index}]", 3)
        i, j = _int(i, "row", 0), _int(j, "column", 0)
        if i >= rows or j >= cols or (i, j) in entries:
            raise CertificateFormatError(f"{context}: duplicate or out-of-bounds entry")
        terms = {}
        for item in _list(monomials, context + ".monomials"):
            exponent, coefficient = _list(item, context + ".monomial", 2)
            exponent = tuple(_int(a, "exponent") for a in _list(exponent, "exponent", ring.D))
            coefficient = _int(coefficient, "coefficient", 1)
            if coefficient >= ring.p or exponent in terms:
                raise CertificateFormatError(f"{context}: duplicate exponent or noncanonical coefficient")
            terms[exponent] = coefficient
        if not terms:
            raise CertificateFormatError(f"{context}: zero entries must be omitted")
        entries[i, j] = Poly(ring, terms)
    return Matrix(ring, rows, cols, entries)


def circuit_dict(circuit: Circuit) -> dict:
    """Serialize gates in execution order with canonical field weights."""
    gates = []
    for gate in circuit.gates:
        if isinstance(gate, SUM):
            gates.append({"kind": "SUM", "control": gate.control, "target": gate.target,
                          "weight": gate.weight % circuit.ring.p,
                          "displacement": list(gate.displacement)})
        else:
            gates.append({"kind": "SWAP", "left": gate.left, "right": gate.right})
    return {"qudits": circuit.qudits, "gates": gates}


def certificate_dict(result: Stabilization) -> dict:
    """Serialize all claims and witnesses; no saved success flag is emitted."""
    ring = result.lifts.dC.ring
    return {
        "schema": SCHEMA, "version": 1, "scope": SCOPE,
        "ring": {"prime": ring.p, "dimension": ring.D},
        "conventions": dict(CONVENTIONS),
        "lifts": {name: _matrix_dict(getattr(result.lifts, name)) for name in _LIFT_NAMES},
        "stabilization": {name: _matrix_dict(getattr(result, name)) for name in _MATRIX_NAMES},
        "ancillas": {name: getattr(result.ancillas, name) for name in _ANCILLA_NAMES},
        "circuit": circuit_dict(result.circuit),
    }


def certificate_from_dict(data: Any) -> Stabilization:
    """Parse strictly; algebraic validity is checked by verify_certificate."""
    data = _keys(data, ("schema", "version", "scope", "ring", "conventions", "lifts",
                        "stabilization", "ancillas", "circuit"), "certificate")
    if data["schema"] != SCHEMA or _int(data["version"], "version") != 1 or data["scope"] != SCOPE:
        raise CertificateFormatError("Unsupported certificate schema, version, or scope")
    if data["conventions"] != CONVENTIONS:
        raise CertificateFormatError("Unsupported matrix/gate conventions")
    ring_data = _keys(data["ring"], ("prime", "dimension"), "ring")
    try:
        ring = Ring(_int(ring_data["prime"], "prime", 2),
                    _int(ring_data["dimension"], "dimension", 2))
    except (TypeError, ValueError) as exc:
        raise CertificateFormatError(f"Invalid ring: {exc}") from exc
    lift_data = _keys(data["lifts"], _LIFT_NAMES, "lifts")
    lifts = LiftData(**{name: _matrix_from_dict(lift_data[name], ring, "lifts." + name)
                        for name in _LIFT_NAMES})
    matrices = _keys(data["stabilization"], _MATRIX_NAMES, "stabilization")
    matrices = {name: _matrix_from_dict(matrices[name], ring, name) for name in _MATRIX_NAMES}
    ancillas = _keys(data["ancillas"], _ANCILLA_NAMES, "ancillas")
    ancillas = DiskAncillas(**{name: _int(ancillas[name], name, 0) for name in _ANCILLA_NAMES})
    circuit = circuit_from_dict(data["circuit"], ring)
    return Stabilization(lifts=lifts, **matrices, circuit=circuit, ancillas=ancillas)


def circuit_from_dict(data: Any, ring: Ring) -> Circuit:
    """Parse strict SUM/SWAP gate data in an explicitly supplied ring."""
    circuit = _keys(data, ("qudits", "gates"), "circuit")
    qudits = _int(circuit["qudits"], "qudits", 0)
    gates = []
    try:
        for gate in _list(circuit["gates"], "gates"):
            if not isinstance(gate, dict):
                raise CertificateFormatError("gate must be an object")
            if gate.get("kind") == "SUM":
                _keys(gate, ("kind", "control", "target", "weight", "displacement"), "SUM")
                weight = _int(gate["weight"], "weight", 0)
                if weight >= ring.p:
                    raise CertificateFormatError("SUM weight must be a canonical field coefficient")
                displacement = tuple(_int(a, "displacement") for a in
                                     _list(gate["displacement"], "displacement", ring.D))
                gates.append(SUM(_int(gate["control"], "control", 0),
                                 _int(gate["target"], "target", 0), weight, displacement))
            elif gate.get("kind") == "SWAP":
                _keys(gate, ("kind", "left", "right"), "SWAP")
                gates.append(SWAP(_int(gate["left"], "left", 0), _int(gate["right"], "right", 0)))
            else:
                raise CertificateFormatError("Unsupported gate kind")
        circuit = Circuit(ring, qudits, tuple(gates))
    except (TypeError, ValueError) as exc:
        raise CertificateFormatError(f"Invalid circuit: {exc}") from exc
    return circuit


def dumps_certificate(result: Stabilization, *, indent: int | None = 2) -> str:
    return json.dumps(certificate_dict(result), sort_keys=True, indent=indent)


def loads_certificate(text: str) -> Stabilization:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict:
        result = {}
        for key, value in pairs:
            if key in result:
                raise CertificateFormatError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    try:
        data = json.loads(text, object_pairs_hook=unique_object)
    except (ValueError, TypeError) as exc:
        raise CertificateFormatError(f"Invalid JSON certificate: {exc}") from exc
    return certificate_from_dict(data)


def _adjoint_matches(left: Matrix, right: Matrix) -> bool:
    """Compare every saved coefficient with right^dagger without copying it."""
    if left.ring != right.ring or left.shape != (right.ncols, right.nrows) or (
            len(left.entries) != len(right.entries)):
        return False
    for (i, j), polynomial in left.entries.items():
        other = right.entries.get((j, i))
        if other is None or len(polynomial.terms) != len(other.terms):
            return False
        if any(other.terms.get(tuple(-a for a in exponent)) != coefficient
               for exponent, coefficient in polynomial.terms.items()):
            return False
    return True


def _theta_with_checked_lower_product(f: Matrix, g: Matrix, physical_rows: int) -> Matrix:
    """Form Theta using an already proved lower row block of f*g=[*,*;0,I].

    The caller must have checked the two enlarged lift matrices and all
    LiftData chain/inverse-defect identities. Those equations prove this
    lower product block exactly. Only the remaining physical rows need a
    new multiplication; all entries of the saved Theta are still compared.
    """
    ring = f.ring
    top_identity = Matrix(ring, physical_rows, f.nrows,
                          {(i, i): 1 for i in range(physical_rows)})
    defect = (top_identity - f[:physical_rows, :] * g).vstack(
        zeros(ring, f.nrows - physical_rows, f.nrows))
    return block([[f, defect], [eye(ring, f.ncols), -g]])


def verify_certificate(value: Stabilization | dict | str) -> VerificationReport:
    """Verify saved matrices, exact block identities, and actual gate actions.

    Success certifies the stored comparison data and stabilized X-check
    intertwining.  Mapping the full CSS code additionally requires the
    topological/exactness assumptions, which this stage does not test.

    Both saved Theta matrices are compared in full. The identities
    Theta(f,g)*Theta(g,f)=I in both orders then certify the two inverse
    equations by exact block algebra, avoiding expansion of cancelling
    cubic products. No saved success flag or construction cache is used.
    """
    checks: list[str] = []
    errors: list[str] = []

    def check(name: str, condition: bool) -> bool:
        (checks if condition else errors).append(name)
        return condition

    try:
        if isinstance(value, str):
            result = loads_certificate(value)
        elif isinstance(value, dict):
            result = certificate_from_dict(value)
        elif isinstance(value, Stabilization):
            result = value
        else:
            raise CertificateFormatError("Expected stabilization, certificate object or JSON text")
        w = result.lifts
        r = w.dC.ring
        c1, c0 = w.dC.nrows, w.dC.ncols
        t1, t0 = w.dT.nrows, w.dT.ncols
        n = c0 + t1 + t0 + c1
        w.validate()
        checks.append("comparison chain and inverse-defect equations")
        for name in _MATRIX_NAMES:
            check(name + " ring", getattr(result, name).ring == r)
        check("circuit ring and size", result.circuit.ring == r and result.circuit.qudits == n)
        check("disk ancilla ranks", result.ancillas == DiskAncillas(t1, t0 + c1, c1, c0 + t1))
        phi_ok = check("enlarged phi lift", result.phihat == block([[w.phi0, w.etaC], [w.dT, -w.psi1]]))
        psi_ok = check("enlarged psi lift", result.psihat == block([[w.psi0, w.etaT], [w.dC, -w.phi1]]))
        source_ok = check("source stabilized differential", result.dChat ==
                          direct_sum(w.dC, eye(r, t1)).hstack(zeros(r, c1 + t1, t0 + c1)))
        target_ok = check("target stabilized differential", result.dThat ==
                          direct_sum(w.dT, eye(r, c1)).hstack(zeros(r, t1 + c1, c0 + t1)))
        f, g = result.psihat, result.phihat
        hats_ok = phi_ok and psi_ok
        U_ok = check("physical Theta formula", hats_ok and
                     result.U == _theta_with_checked_lower_product(f, g, t0))
        Uinv_ok = check("physical inverse Theta formula", hats_ok and
                        result.Uinv == _theta_with_checked_lower_product(g, f, c0))
        # LiftData.validate has checked dT*etaT=I-psi1*phi1 and the
        # corresponding C equation. Reuse those equalities, not their
        # expensive cancelling check-map products.
        V_ok = check("check Theta formula", result.V == block([
            [w.psi1, w.dT * w.etaT], [eye(r, c1), -w.phi1]]))
        Vinv_ok = check("check inverse Theta formula", result.Vinv == block([
            [w.phi1, w.dC * w.etaC], [eye(r, t1), -w.psi1]]))
        check("U right inverse", U_ok and Uinv_ok)
        U_left = check("U left inverse", U_ok and Uinv_ok)
        V_right = check("V right inverse", V_ok and Vinv_ok)
        check("V left inverse", V_ok and Vinv_ok)
        # Let dCp=diag(dC,I), dTp=diag(dT,I). The four independently
        # checked lift equations give dTp*f=V*dCp and dCp*g=Vinv*dTp.
        # Thus dTp*(I-f*g)=0 by the verified V/Vinv block inverse, which
        # proves the entire saved square dThat*U=V*dChat in both blocks.
        check("check intertwining", hats_ok and source_ok and target_ok and U_ok and V_right)
        z_action = result.circuit.matrix("Z")
        z_ok = check("gate Z product", z_action == result.U)
        del z_action
        x_action = result.circuit.matrix("X")
        x_ok = check("gate X product", _adjoint_matches(x_action, result.Uinv))
        del x_action
        # For the independently replayed actions, Z^dagger*X equals
        # U^dagger*Uinv^dagger=(Uinv*U)^dagger, already proved to be I.
        check("Pauli pairing", z_ok and x_ok and U_left)
        check("inverse gate Z product", result.circuit.inverse().matrix("Z") == result.Uinv)
    except (ValueError, TypeError, AttributeError, KeyError, IndexError) as exc:
        errors.append(f"invalid certificate: {exc}")
    return VerificationReport(not errors, tuple(checks), tuple(errors))
