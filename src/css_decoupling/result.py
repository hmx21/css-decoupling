"""Public result views and lossless adapters for the two native algorithms."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json

from .algebra.laurent import Ring, Matrix, eye
from .circuits.qca import QCA
from .certificates.comparison import _matrix_dict, _matrix_from_dict

RESULT_SCHEMA = "css-decoupling.result"
RESULT_VERSION = 1
_CONVENTIONS = {"vectors": "columns", "pauli_order": ["X", "Z"],
    "execution_order": "first-listed-layer-first", "single_generator_phases": 0,
    "SUM": "control at r+displacement; target at r; |a,b> -> |a,b+weight*a>",
    "MULTIPLY": "|a> -> |multiplier*a>; Z exponent /multiplier; X exponent *multiplier"}


def _json_copy(value):
    """Canonical JSON values; reject opaque objects, non-finite values and keys."""
    def check(v):
        if isinstance(v, dict):
            if any(type(k) is not str for k in v):
                raise TypeError("Unified JSON objects require string keys")
            for item in v.values():check(item)
        elif isinstance(v, (list, tuple)):
            for item in v:check(item)
        elif v is not None and type(v) not in (str, int, float, bool):
            raise TypeError("Unified options/metadata must contain only JSON values")
    check(value)
    return json.loads(json.dumps(value, allow_nan=False))


def _same_json(a, b):
    return json.dumps(a, sort_keys=True, allow_nan=False, separators=(",", ":")) == \
           json.dumps(b, sort_keys=True, allow_nan=False, separators=(",", ":"))


def to_two_d(matrix):
    """Copy public Laurent data for the two-dimensional algorithm, which shares these types."""
    if not isinstance(matrix, Matrix):
        raise TypeError("Use the public css_decoupling.Matrix type")
    return matrix.copy()


def to_public(matrix):
    if not isinstance(matrix, Matrix):
        raise TypeError("Expected an exact native Laurent matrix")
    return matrix.copy()


def qca_dict(qca):
    return {"qudits": qca.qudits,
            "actions": {"Z": _matrix_dict(qca.z_action), "Zinv": _matrix_dict(qca.z_inverse),
                        "X": _matrix_dict(qca.x_action), "Xinv": _matrix_dict(qca.x_inverse)}}


def qca_from_dict(data, ring):
    if not isinstance(data, dict) or set(data) != {"qudits", "actions"}:
        raise ValueError("Malformed unified QCA")
    if type(data["qudits"]) is not int or data["qudits"] < 0:
        raise ValueError("Invalid QCA register count")
    if not isinstance(data["actions"], dict) or set(data["actions"]) != {"Z", "Zinv", "X", "Xinv"}:
        raise ValueError("Unified QCA requires all four actions")
    actions = {k: _matrix_from_dict(v, ring, "qca." + k) for k, v in data["actions"].items()}
    qca = QCA(actions["Z"], actions["Zinv"])
    if qca.qudits != data["qudits"] or qca.x_action != actions["X"] or qca.x_inverse != actions["Xinv"]:
        raise ValueError("QCA inverse-dagger actions or register count disagree")
    return qca


@dataclass
class CircuitView:
    ring: Ring
    qudits: int
    layers: list
    action: QCA
    relation_to_qca: str
    input_ancillas: int

    @property
    def gates(self):
        return tuple(deepcopy(gate) for layer in self.layers for gate in layer)

    def matrix(self, action="Z"):
        """Replay actual gate families in their declared execution order."""
        if action not in ("X", "Z"):raise ValueError("Action must be X or Z")
        result = eye(self.ring, self.qudits)
        for gate in self.gates:
            step = eye(self.ring, self.qudits)
            kind = gate["kind"]
            if kind == "SUM":
                i, j, w = gate["control"], gate["target"], gate["weight"]
                exp = tuple(gate["displacement"])
                if i == j:raise ValueError("SUM needs distinct species")
                if action == "Z":step[i, j] = self.ring.monomial(exp, -w)
                else:step[j, i] = self.ring.monomial(tuple(-a for a in exp), w)
            elif kind == "SWAP":
                i, j = gate["left"], gate["right"]
                if i == j:raise ValueError("SWAP needs distinct species")
                step[i, i] = step[j, j] = 0
                step[i, j] = step[j, i] = 1
            elif kind == "MULTIPLY":
                i, multiplier = gate["qudit"], self.ring(gate["multiplier"])
                step[i, i] = multiplier**-1 if action == "Z" else multiplier
            else:raise ValueError("Unsupported unified gate family")
            result = step * result
        return result

    def to_dict(self):
        return {"qudits": self.qudits, "layers": _json_copy(self.layers),
                "action": qca_dict(self.action), "relation_to_qca": self.relation_to_qca,
                "input_ancillas": self.input_ancillas, "conventions": deepcopy(_CONVENTIONS)}

    @classmethod
    def from_dict(cls, data, ring):
        keys = {"qudits", "layers", "action", "relation_to_qca", "input_ancillas", "conventions"}
        if not isinstance(data, dict) or set(data) != keys or data["conventions"] != _CONVENTIONS:
            raise ValueError("Malformed unified circuit view")
        if data["relation_to_qca"] not in ("exact", "code_equivalent"):
            raise ValueError("Invalid circuit/QCA relation")
        if (type(data["qudits"]) is not int or data["qudits"] < 0 or
                type(data["input_ancillas"]) is not int or data["input_ancillas"] < 0):
            raise ValueError("Invalid circuit register counts")
        return cls(ring, data["qudits"], _json_copy(data["layers"]),
                   qca_from_dict(data["action"], ring), data["relation_to_qca"], data["input_ancillas"])


def _two_d_layers(layers):
    converted = []
    for layer in layers:
        gates = []
        for gate in layer:
            if gate["kind"] in ("CNOT", "SUM"):
                gates.append({"kind": "SUM", "control": gate["control"], "target": gate["target"],
                              "weight": gate.get("coefficient", 1), "displacement": list(gate["offset"])})
            elif gate["kind"] == "MULTIPLY":
                gates.append({"kind": "MULTIPLY", "qudit": gate["qudit"], "multiplier": gate["multiplier"]})
            else:raise ValueError("Unsupported two-dimensional circuit gate")
        converted.append(gates)
    return converted


def derive_views(algorithm, evidence, requested_output):
    """Pure source extraction, used only after independent backend verification."""
    if algorithm == "two_d":
        from .two_d.serialization import decode
        raw = decode(evidence["qca"])
        circuit_raw = None if evidence["circuit"] is None else decode(evidence["circuit"])
        Q = raw["qca"]
        qca = QCA(to_public(Q["psi"][1]), to_public(Q["psi_inverse"][1]))
        scope = {"algorithm": algorithm, "lattice": _json_copy(raw["lattice"]), "qca_registers": qca.qudits,
            "input_qudits_per_original_cell": raw["original"]["HX"].ncols,
            "input_physical_qudits": raw["coarse_original_HX"].ncols,
            "output_physical_qudits": Q["standard_HX"].ncols,
            "ancillas": {"kind": "none", "input_count": 0, "output_count": 0},
            "qca_input_ancillas": 0, "qca_register_scope": "coarse-original-physical-register",
            "general_constructed_circuit_internally": False,
            "target": {"basis": "two-dimensional-standard-toric-and-products",
                       "toric_multiplicities": [[1, Q["toric_copies"]]] if Q["toric_copies"] else [],
                       "product_X": Q["product_x"], "product_Z": Q["product_z"],
                       "product_states": {"X_fixed": Q["product_x"], "Z_fixed": Q["product_z"]}}}
        circuit = None
        if circuit_raw is not None:
            actual = circuit_raw["actual_circuit"]
            action = QCA(to_public(actual["z_action"]), to_public(actual["z_inverse"]))
            relation = "exact" if action.z_action == qca.z_action and action.z_inverse == qca.z_inverse else "code_equivalent"
            circuit = CircuitView(qca.ring, action.qudits, _two_d_layers(actual["layers"]), action, relation, 0)
        return qca, circuit, scope
    if algorithm != "general":raise ValueError("Unknown unified algorithm")
    from .certificates.comparison import certificate_from_dict, circuit_from_dict
    from .certificates.formats import DUAL_SCHEMA
    source = evidence["qca"]
    dual = source["schema"] == DUAL_SCHEMA
    inner = source["inner"] if dual else source
    stable = certificate_from_dict(inner["stabilization"])
    qca = QCA(stable.Uinv.dagger(), stable.U.dagger()) if dual else QCA(stable.U.copy(), stable.Uinv.copy())
    actual = circuit_from_dict(source["circuit"], qca.ring) if dual else stable.circuit
    source_physical = stable.lifts.dC.ncols
    ancillas = deepcopy(source["ancillas"]) if dual else {
        "source_X_fixed": stable.ancillas.source_degree_zero,
        "source_Z_fixed": stable.ancillas.source_degree_minus_one,
        "target_X_fixed": stable.ancillas.target_degree_zero,
        "target_Z_fixed": stable.ancillas.target_degree_minus_one}
    scope = {"algorithm": algorithm, "lattice": _json_copy(inner["lattice"]), "qca_registers": qca.qudits,
        "input_qudits_per_original_cell": source["input"]["HX"]["cols"],
        "input_physical_qudits": source_physical,
        "output_physical_qudits": stable.lifts.dT.ncols,
        "ancillas": {"kind": "C4-disk-stabilization", **ancillas},
        "qca_input_ancillas": qca.qudits - source_physical,
        "qca_register_scope": "C4-stabilized-physical-register",
        "general_constructed_circuit_internally": True,
        "target": {"basis": source["target"]["basis"] if dual else "canonical-toric-stack",
                   "toric_multiplicities": _json_copy(source["target"]["multiplicities"] if dual else inner["multiplicities"]),
                   "identity_checks": _json_copy(source["identity_checks"])}}
    circuit = None
    if requested_output == "circuit":
        from .circuits.gates import SUM
        layers = []
        for gate in actual.gates:
            if isinstance(gate, SUM):
                item = {"kind": "SUM", "control": gate.control, "target": gate.target,
                        "weight": gate.weight % qca.ring.p, "displacement": list(gate.displacement)}
            else:item = {"kind": "SWAP", "left": gate.left, "right": gate.right}
            layers.append([item])
        circuit = CircuitView(qca.ring, qca.qudits, layers, qca, "exact", scope["qca_input_ancillas"])
    return qca, circuit, scope


@dataclass
class UnifiedResult:
    algorithm: str
    status: str
    qca: QCA
    circuit: CircuitView | None
    diagnostics: dict
    backend_evidence: dict
    input: dict
    request: dict
    metadata: dict | None = None

    def __post_init__(self):
        if self.metadata is None:self.metadata = deepcopy(self.diagnostics["scope"])

    @property
    def lattice(self):return deepcopy(self.metadata["lattice"])

    @property
    def target(self):return deepcopy(self.metadata["target"])

    @property
    def ancillas(self):return deepcopy(self.metadata["ancillas"])

    def certificate(self):
        return {"schema": RESULT_SCHEMA, "version": RESULT_VERSION, "algorithm": self.algorithm,
                "status": self.status, "input": _json_copy(self.input), "request": _json_copy(self.request),
                "qca": qca_dict(self.qca), "circuit": None if self.circuit is None else self.circuit.to_dict(),
                "diagnostics": _json_copy(self.diagnostics), "metadata": _json_copy(self.metadata),
                "backend_evidence": _json_copy(self.backend_evidence)}

    @classmethod
    def from_certificate(cls, data):
        from .raw_input import parse_css_input
        parsed = parse_css_input(data["input"])
        ring = parsed["HX"].ring
        return cls(data["algorithm"], data["status"], qca_from_dict(data["qca"], ring),
            None if data["circuit"] is None else CircuitView.from_dict(data["circuit"], ring),
            deepcopy(data["diagnostics"]), deepcopy(data["backend_evidence"]),
            deepcopy(data["input"]), deepcopy(data["request"]), deepcopy(data["metadata"]))


def verify_result(result, *, expected_HX=None, expected_HZ=None):
    from .certificates.unified import verify_result as verify
    return verify(result, expected_HX=expected_HX, expected_HZ=expected_HZ)


def save_result(result, path):
    from .certificates.unified import save_result as save
    return save(result, path)


def load_result(path, *, verify=True):
    from .certificates.unified import load_result as load
    return load(path, verify=verify)
