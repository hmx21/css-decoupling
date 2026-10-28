"""Source-bound physical QCA and circuit outputs, without changing v1 sources.

These outputs certify the physical map of an independently verified complete,
dual, or Algorithm 2 certificate. They do not accept intermediate comparison
certificates or load a source from a filename. A QCA is represented in the
fixed source/target register order; its check-generator map V is not exported.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json

from css_decoupling.algebra.laurent import Ring
from .comparison import (
    CertificateFormatError, VerificationReport, certificate_from_dict,
    circuit_dict, circuit_from_dict, _int, _keys, _matrix_dict, _matrix_from_dict,
)
from .formats import COMPLETE_SCHEMA, DUAL_SCHEMA, DUAL_TARGET_BASIS

QCA_SCHEMA = "css-decoupling.qca-output"
CIRCUIT_SCHEMA = "css-decoupling.circuit-output"
SCOPE = "verified-source-physical-clifford-output"
CONVENTIONS = {
    "vectors": "columns",
    "direction": "source-to-target",
    "inverse_direction": "target-to-source",
    "forward_coordinates": "rows=target_registers;columns=source_registers",
    "inverse_coordinates": "rows=source_registers;columns=target_registers",
    "pauli_order": ["X", "Z"],
    "pauli_product": "X(a) Z(b)",
    "css_action": "block-diagonal",
    "single_generator_phases": 0,
    "execution_order": "first-listed-gate-first",
    "source_hash": "sha256:sorted-compact-utf8-json-with-final-LF",
}
_ACTIONS = ("Z", "Zinv", "X", "Xinv")
_COMMON_KEYS = ("schema", "version", "scope", "source", "ring", "qudits", "context", "conventions")
_FORMAT_ERRORS = (ValueError, TypeError, AttributeError, KeyError, IndexError)


def _canonical_chunks(value):
    # Streaming avoids allocating another complete encoded source document.
    encoder = json.JSONEncoder(sort_keys=True, separators=(",", ":"),
                               ensure_ascii=False, allow_nan=False)
    yield from encoder.iterencode(value)
    yield "\n"


def _digest(value):
    digest = hashlib.sha256()
    for chunk in _canonical_chunks(value):
        digest.update(chunk.encode("utf-8"))
    return digest.hexdigest()


def _same_json(left, right):
    # JSON equality is type-sensitive: False must not stand for integer zero.
    return "".join(_canonical_chunks(left)) == "".join(_canonical_chunks(right))


def _physical_source(source):
    if source["schema"] == "css-decoupling.algorithm-two":
        if source["physical_decoupling"] is not None:
            return source["physical_decoupling"], "physical_decoupling"
        return source["decoupling"], "decoupling"
    return source, "self"


def _source_ring(source):
    physical, _ = _physical_source(source)
    active = physical["inner"] if physical["schema"] == DUAL_SCHEMA else physical
    spec = _keys(active["stabilization"]["ring"], ("prime", "dimension"), "source ring")
    return Ring(_int(spec["prime"], "prime", 2), _int(spec["dimension"], "dimension", 2))


def _verify_source(source, *, linear_backend, bounded_options, max_total_rank):
    if not isinstance(source, dict):
        raise CertificateFormatError("source_certificate must be an explicit certificate object, not a path")
    schema = source.get("schema")
    if schema not in (COMPLETE_SCHEMA, DUAL_SCHEMA, "css-decoupling.algorithm-two"):
        raise CertificateFormatError("Output source must be complete, dual, or Algorithm 2")
    if _int(source.get("version"), "source version") != 1:
        raise CertificateFormatError("Unsupported output source version")
    before = _digest(source)
    if schema == "css-decoupling.algorithm-two":
        from .construction import verify_construction
        selected = linear_backend
        if selected is None and source.get("configuration", {}).get("linear_backend") == "bounded":
            selected = "bounded"
        if selected != "bounded" and bounded_options is not None:
            raise CertificateFormatError("bounded_options require an explicit or declared bounded verifier")
        report = verify_construction(source, linear_backend=selected,
                                     bounded_options=bounded_options, max_total_rank=max_total_rank)
    else:
        from .decoupling import verify_decoupling
        if linear_backend is None:
            if bounded_options is not None:
                raise CertificateFormatError("bounded_options require an explicit bounded verifier")
            report = verify_decoupling(source)
        else:
            from css_decoupling.algebra.modules import using_backend
            selected = linear_backend
            if linear_backend == "bounded":
                from css_decoupling.algebra.bounded_linear import BoundedLinearBackend
                selected = BoundedLinearBackend(_source_ring(source), **(bounded_options or {}))
            elif bounded_options is not None:
                raise CertificateFormatError("bounded_options require an explicit bounded verifier")
            with using_backend(selected):
                report = verify_decoupling(source)
    if not report.valid:
        raise CertificateFormatError("Source certificate failed full verification: " + "; ".join(report.errors))
    if _digest(source) != before:
        raise CertificateFormatError("Source certificate changed during verification")
    return before, report


def _registers(physical, first, second, dual):
    return [
        {"role": "physical", "offset": 0, "qudits": physical},
        {"role": "ancilla", "offset": physical, "qudits": first,
         "state": "Z-fixed" if dual else "X-fixed"},
        {"role": "ancilla", "offset": physical + first, "qudits": second,
         "state": "X-fixed" if dual else "Z-fixed"},
    ]


def _verified_data(source, *, linear_backend=None, bounded_options=None, max_total_rank=100000):
    digest, report = _verify_source(source, linear_backend=linear_backend,
                                    bounded_options=bounded_options, max_total_rank=max_total_rank)
    physical, selected = _physical_source(source)
    dual = physical["schema"] == DUAL_SCHEMA
    active = physical["inner"] if dual else physical
    stable = certificate_from_dict(active["stabilization"])
    ring = stable.lifts.dC.ring
    circuit = circuit_from_dict(physical["circuit"], ring) if dual else stable.circuit
    if dual:
        actions = dict(Z=stable.Uinv.dagger(), Zinv=stable.U.dagger(),
                       X=stable.U, Xinv=stable.Uinv)
        target = dict(basis=DUAL_TARGET_BASIS, multiplicities=deepcopy(physical["target"]["multiplicities"]))
        ancillas = deepcopy(physical["ancillas"])
    else:
        actions = dict(Z=stable.U, Zinv=stable.Uinv,
                       X=stable.Uinv.dagger(), Xinv=stable.U.dagger())
        target = dict(basis="canonical-toric-stack", multiplicities=deepcopy(physical["multiplicities"]))
        ancillas = dict(source_X_fixed=stable.ancillas.source_degree_zero,
                        source_Z_fixed=stable.ancillas.source_degree_minus_one,
                        target_X_fixed=stable.ancillas.target_degree_zero,
                        target_Z_fixed=stable.ancillas.target_degree_minus_one)
    c1, c0 = stable.lifts.dC.shape
    t1, t0 = stable.lifts.dT.shape
    context = dict(
        input_scope="Algorithm2-input-complex" if selected == "decoupling" else "original-CSS-input",
        sector="Z" if dual else "X",
        lattice=deepcopy(active["lattice"]),
        variables="translations by the columns of lattice.basis_columns",
        source_physical_order="index = species*lattice.index + coset; coset follows lattice.representatives",
        target=target, identity_checks=deepcopy(physical["identity_checks"]),
        ancillas=ancillas,
        source_registers=_registers(c0, t1, t0 + c1, dual),
        target_registers=_registers(t0, c1, c0 + t1, dual),
    )
    binding = dict(schema=source["schema"], version=1, canonical_sha256=digest,
                   physical_certificate=selected)
    common = dict(version=1, scope=SCOPE, source=binding,
                  ring=dict(prime=ring.p, dimension=ring.D), qudits=circuit.qudits,
                  context=context, conventions=deepcopy(CONVENTIONS))
    return common, actions, circuit, report


def export_outputs(source_certificate, *, linear_backend=None, bounded_options=None,
                   max_total_rank=100000):
    """Verify one source and export two independently checkable views.

    The complete source is identified by its canonical SHA, never embedded
    or loaded implicitly. Algorithm 2 chooses physical_decoupling when it
    is present (including its dual sector); otherwise it uses decoupling.
    Source verification options are forwarded to the existing verifiers.
    Without an override, a declared bounded Algorithm 2 source retains its
    bounded verifier; other sources use the current native backend context.
    ResourceLimit propagates and is never converted into valid=False.
    """
    common, actions, circuit, _ = _verified_data(
        source_certificate, linear_backend=linear_backend, bounded_options=bounded_options,
        max_total_rank=max_total_rank)
    return {
        "qca": dict(schema=QCA_SCHEMA, **deepcopy(common),
                    actions={name: _matrix_dict(actions[name]) for name in _ACTIONS}),
        "circuit": dict(schema=CIRCUIT_SCHEMA, **deepcopy(common), circuit=circuit_dict(circuit)),
    }


def _header(data, schema, common, payload):
    _keys(data, (*_COMMON_KEYS, payload), "physical output")
    if data["schema"] != schema or _int(data["version"], "output version") != 1:
        raise CertificateFormatError("Unsupported physical output schema or version")
    _int(data["qudits"], "output qudits", 0)
    for name, expected in common.items():
        if not _same_json(data[name], expected):
            raise CertificateFormatError("Physical output changed its source-bound " + name)


def _qca(data, common, actions, ring):
    _header(data, QCA_SCHEMA, common, "actions")
    encoded = _keys(data["actions"], _ACTIONS, "QCA actions")
    parsed = {name: _matrix_from_dict(encoded[name], ring, "actions." + name) for name in _ACTIONS}
    for name in _ACTIONS:
        if parsed[name] != actions[name]:
            raise CertificateFormatError("QCA " + name + " differs from the verified physical source")
    # The complete verifier proved the full Theta and inverse formulas.
    # Equality to those four matrices inherits both inverse equations and
    # the dagger pairing without expanding another cancelling U*Uinv.
    return parsed


def _circuit(data, common, expected):
    _header(data, CIRCUIT_SCHEMA, common, "circuit")
    parsed = circuit_from_dict(data["circuit"], expected.ring)
    if parsed != expected:
        raise CertificateFormatError("Circuit output differs from the verified source gate sequence")
    return parsed


def _replay(circuit, actions, checks):
    inverse = circuit.inverse()
    for name, selected, pauli in (("Z", circuit, "Z"), ("X", circuit, "X"),
                                  ("Zinv", inverse, "Z"), ("Xinv", inverse, "X")):
        actual = selected.matrix(pauli)
        if actual != actions[name]:
            raise CertificateFormatError("Exported gate " + name + " product differs from the QCA")
        checks.append("exported gate " + name + " product")


def _verify(qca, circuit, *, check_qca, check_circuit, source_certificate,
            linear_backend, bounded_options, max_total_rank):
    checks, errors = [], []
    try:
        common, expected_actions, expected_circuit, source_report = _verified_data(
            source_certificate, linear_backend=linear_backend, bounded_options=bounded_options,
            max_total_rank=max_total_rank)
        checks.extend("source: " + name for name in source_report.checks)
        checks.append("complete source verified and canonical SHA bound")
        actions = expected_actions
        if check_qca:
            actions = _qca(qca, common, expected_actions, expected_circuit.ring)
            checks.append("four physical Pauli matrices, inverses, pairing and register context")
        if check_circuit:
            parsed = _circuit(circuit, common, expected_circuit)
            checks.append("exact source gate sequence and register context")
            _replay(parsed, actions, checks)
    except _FORMAT_ERRORS as exc:
        errors.append(str(exc))
    return VerificationReport(not errors, tuple(checks), tuple(errors), SCOPE)


def verify_qca_output(data, *, source_certificate, linear_backend=None, bounded_options=None,
                      max_total_rank=100000):
    """Verify the physical QCA against an explicitly supplied full source."""
    return _verify(data, None, check_qca=True, check_circuit=False,
                   source_certificate=source_certificate, linear_backend=linear_backend,
                   bounded_options=bounded_options, max_total_rank=max_total_rank)


def verify_circuit_output(data, *, source_certificate, linear_backend=None, bounded_options=None,
                          max_total_rank=100000):
    """Verify the source-bound gate list, replaying both actions and inverses."""
    return _verify(None, data, check_qca=False, check_circuit=True,
                   source_certificate=source_certificate, linear_backend=linear_backend,
                   bounded_options=bounded_options, max_total_rank=max_total_rank)


def verify_output_pair(qca, circuit, *, source_certificate, linear_backend=None, bounded_options=None,
                       max_total_rank=100000):
    """Independently replay exported gates against all four exported QCA maps."""
    return _verify(qca, circuit, check_qca=True, check_circuit=True,
                   source_certificate=source_certificate, linear_backend=linear_backend,
                   bounded_options=bounded_options, max_total_rank=max_total_rank)
