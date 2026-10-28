"""Independent input, stabilizer and circuit checks for the Z sector."""
from css_decoupling.algebra.laurent import direct_sum, eye, zeros
from css_decoupling.algebra.modules import solve
from css_decoupling.circuits.duality import dualize_circuit
from css_decoupling.topology.complexes import toric_window
from css_decoupling.lattice.retained import Lattice
from .comparison import (
    VerificationReport, certificate_from_dict, circuit_from_dict,
    _keys, _int, _list, _matrix_from_dict,
)
from .formats import (COMPLETE_SCHEMA as INNER_SCHEMA, DUAL_SCHEMA as SCHEMA,
                      DUAL_TARGET_BASIS as TARGET_BASIS, DUAL_SCOPE as SCOPE,
                      DUAL_ANCILLA_KEYS as _ANCILLA_KEYS, dual_ancillas)


def verify_dual_certificate(value):
    """Bind the original inputs to an independent inner certificate and gates.

    Both Pauli products and both stabilizer-image containments are recomputed.
    Inner certificates must use the complete X-sector schema, preventing
    recursive wrappers or an unverified lower-level stabilization as evidence.
    """
    from .decoupling import verify_decoupling

    checks, errors = [], []
    try:
        data = value if isinstance(value, dict) else value.certificate()
        _keys(data, ("schema", "version", "input", "target", "ancillas",
                     "identity_checks", "inner", "circuit"), "dual certificate")
        if data["schema"] != SCHEMA or _int(data["version"], "version") != 1:
            raise ValueError("Unsupported dual certificate schema")
        inner = data["inner"]
        if not isinstance(inner, dict) or inner.get("schema") != INNER_SCHEMA:
            raise ValueError("A dual certificate requires a complete X-sector inner certificate")
        stable = certificate_from_dict(inner["stabilization"])
        ring = stable.lifts.dC.ring
        _keys(data["input"], ("HX", "HZ"), "dual input")
        HX = _matrix_from_dict(data["input"]["HX"], ring, "input.HX")
        HZ = _matrix_from_dict(data["input"]["HZ"], ring, "input.HZ")
        inner_HX = _matrix_from_dict(inner["input"]["HX"], ring, "inner.input.HX")
        inner_HZ = _matrix_from_dict(inner["input"]["HZ"], ring, "inner.input.HZ")
        if HX != inner_HZ or HZ != inner_HX:
            raise ValueError("Original checks do not equal the swapped inner checks")
        checks.append("original input bound to swapped inner CSS checks")
        inner_profile = dict(inner["multiplicities"])
        expected_profile = dict(sorted((ring.D - j, t) for j, t in inner_profile.items()))
        _keys(data["target"], ("basis", "multiplicities"), "dual target")
        actual_profile = {}
        for pair in _list(data["target"]["multiplicities"], "dual multiplicities"):
            j, t = _list(pair, "dual multiplicity", 2)
            j, t = _int(j, "dual degree", 1), _int(t, "dual multiplicity", 1)
            if j in actual_profile:
                raise ValueError("Duplicate dual target degree")
            actual_profile[j] = t
        if data["target"]["basis"] != TARGET_BASIS or actual_profile != expected_profile:
            raise ValueError("Dual target basis or complementary degree labels disagree")
        checks.append("dual target basis and complementary degree labels")
        _keys(data["ancillas"], _ANCILLA_KEYS, "dual ancillas")
        ancillas = {key: _int(data["ancillas"][key], key, 0) for key in _ANCILLA_KEYS}
        if ancillas != dual_ancillas(stable.ancillas):
            raise ValueError("Dual disk ancillas do not exchange X-fixed and Z-fixed states")
        identity = _keys(data["identity_checks"], ("sector", "source", "target"), "dual identity checks")
        if identity["sector"] != "Z":
            raise ValueError("Dual redundant identity checks belong in the Z sector")
        for side in ("source", "target"):
            if _int(identity[side], "identity checks", 0) != inner["identity_checks"][side]:
                raise ValueError("Dual identity-check counts disagree with inner construction")
        checks.append("disk ancilla and redundant-check sectors exchanged")
        circuit = circuit_from_dict(data["circuit"], ring)
        if circuit != dualize_circuit(stable.circuit):
            raise ValueError("Dual gate sequence is not the Fourier conjugate of the inner circuit")
        # Reject outer schema/binding errors before doing expensive algebra.
        # Every successful result still requires the full independent inner
        # verification; there is no trusted success bit or cached report.
        inner_report = verify_decoupling(inner)
        checks.extend(inner_report.checks)
        if not inner_report.valid:
            errors.extend("inner: " + error for error in inner_report.errors)
            return VerificationReport(False, tuple(checks), tuple(errors), SCOPE)
        z_action, x_action = circuit.matrix("Z"), circuit.matrix("X")
        if z_action != stable.Uinv.dagger() or x_action != stable.U:
            raise ValueError("Dual circuit Pauli products disagree with inverse-dagger conjugation")
        if z_action.dagger() * x_action != eye(ring, circuit.qudits):
            raise ValueError("Dual circuit fails the Pauli pairing identity")
        checks.append("Fourier-conjugated gate sequence and both Pauli products")
        lattice = Lattice(inner["lattice"]["basis_columns"], max_index=None)
        target = toric_window(ring, inner_profile)
        c1, c0 = stable.lifts.dC.shape
        t1, t0 = stable.lifts.dT.shape
        # Original X stabilizers are inner Z stabilizers; original Z checks
        # are inner X checks. Their disk ordering is preserved by conjugation.
        XC = direct_sum(lattice.restrict(HX.dagger()), zeros(ring, t1, 0), eye(ring, t0 + c1))
        XT = direct_sum(target.d(-1), zeros(ring, c1, 0), eye(ring, c0 + t1))
        ZC, ZT = stable.dChat.dagger(), stable.dThat.dagger()
        solve(XT, x_action * XC)
        solve(XC, stable.Uinv * XT)
        solve(ZT, z_action * ZC)
        solve(ZC, stable.U.dagger() * ZT)
        checks.append("original X and Z stabilizer images match the dual target in both directions")
    except (ValueError, TypeError, KeyError, AttributeError, IndexError) as exc:
        errors.append(str(exc))
    return VerificationReport(not errors, tuple(checks), tuple(errors), SCOPE)


# Public name shared with certificates.decoupling.verify_decoupling's sector dispatcher.
verify_dual_decoupling = verify_dual_certificate
