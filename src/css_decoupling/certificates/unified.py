"""Independent verification and exclusive I/O for unified result envelopes."""
from copy import deepcopy

from .comparison import VerificationReport, _matrix_dict, _matrix_from_dict
from .output_io import write_new_json_files
from ..result import (RESULT_SCHEMA, RESULT_VERSION, UnifiedResult, derive_views,
                      qca_dict, qca_from_dict, _same_json, _json_copy, to_two_d)
from ..raw_input import parse_css_input, css_input_dict, read_json, INPUT_SCHEMA

_KEYS = {"schema", "version", "algorithm", "status", "input", "request", "qca", "circuit",
         "diagnostics", "metadata", "backend_evidence"}
_SCOPE = "unified-source-bound-CSS-decoupling"


def _require(test, message):
    if not test:raise ValueError(message)


def verify_result(value, *, expected_HX=None, expected_HZ=None):
    """Verify native backend evidence and every normalized physical claim.

    Constructors/factorizers are never called. A valid circuit_incomplete
    envelope certifies its QCA only; its error text is diagnostic metadata,
    never mathematical evidence that a circuit does not exist.
    """
    checks, errors = [], []
    try:
        data = value.certificate() if isinstance(value, UnifiedResult) else deepcopy(value)
        _require(isinstance(data, dict) and set(data) == _KEYS, "Unknown or missing unified result field")
        _require(data["schema"] == RESULT_SCHEMA and type(data["version"]) is int and data["version"] == RESULT_VERSION,
                 "Unsupported unified result schema/version")
        _json_copy(data)
        _require(data["input"].get("schema") == INPUT_SCHEMA, "Unified results require canonical input metadata")
        parsed = parse_css_input(data["input"])
        HX, HZ = parsed["HX"], parsed["HZ"]
        _require(_same_json(data["input"], css_input_dict(HX, HZ,
                    lattice_mode=parsed["lattice_mode"], lattice_basis=parsed["lattice_basis"])),
                 "Input encoding is not the canonical unified representation")
        if expected_HX is not None:_require(HX == expected_HX, "Result is for another HX")
        if expected_HZ is not None:_require(HZ == expected_HZ, "Result is for another HZ")
        request = data["request"]
        _require(isinstance(request, dict) and set(request) == {"algorithm", "output", "options"},
                 "Malformed request metadata")
        from ..api import normalize_request
        selected, _, normalized_request = normalize_request(HX, HZ,
            algorithm=request["algorithm"], output=request["output"], options=request["options"],
            lattice_mode=parsed["lattice_mode"], lattice_basis=parsed["lattice_basis"])
        _require(data["algorithm"] == selected and _same_json(request, normalized_request), "Algorithm dispatch metadata mismatch")
        evidence = data["backend_evidence"]
        _require(isinstance(evidence, dict) and set(evidence) == {"kind", "qca", "circuit", "construction", "lattice_selection"} and
                 evidence["kind"] == selected, "Malformed or mismatched backend evidence")
        opts = request["options"]
        if selected == "two_d":
            from ..two_d.serialization import decode, encode
            from ..two_d.verify import verify
            from ..two_d.code_circuits import verify_no_ancilla_result
            _require(evidence["construction"] is None and evidence["lattice_selection"] is None,
                     "two_d does not carry high-dimensional construction or profile certificates")
            raw = decode(evidence["qca"])
            expected_lattice_request = {"mode": parsed["lattice_mode"], "basis_columns": parsed["lattice_basis"],
                "kernel_scope": "complete" if parsed["lattice_mode"] == "auto" else "specified-translations"}
            _require(_same_json(raw.get("lattice_request"), expected_lattice_request),
                     "two_d lattice request differs from the canonical input policy")
            circuit = None if evidence["circuit"] is None else decode(evidence["circuit"])
            caps = opts.get("circuit_options", {})
            limits = dict(max_families=caps.get("max_families", 200000),
                          max_matrix_terms=caps.get("max_matrix_terms", 1000000))
            # A stopped compiler's tiny caps must not prevent reading the
            # already completed QCA. No circuit proof is checked in that case.
            if circuit is None:
                verify(raw, expected_HX=to_two_d(HX), expected_HZ=to_two_d(HZ))
            else:
                _require(_same_json(encode(circuit["qca_result"]), evidence["qca"]),
                         "Circuit is not bound to the identical QCA source")
                verify_no_ancilla_result(circuit, expected_HX=to_two_d(HX), expected_HZ=to_two_d(HZ),
                    max_factors=caps.get("max_factors", 200000), **limits)
            _require(raw["circuit"] is None, "Unified two_d QCA source must not embed the retired doubled-register circuit")
            checks.append("independent two_d raw input, full QCA and optional original-register circuit evidence")
        else:
            from .decoupling import verify_decoupling
            from .construction import verify_construction
            from ..algebra.modules import using_backend
            source = evidence["qca"]
            _require(evidence["circuit"] is None, "General gates belong to their unchanged complete source certificate")
            active = source["inner"] if source.get("schema") == "css-decoupling.general.dual" else source
            _require(source["input"] == {"HX": _matrix_dict(HX), "HZ": _matrix_dict(HZ)},
                     "General source certificate is for another input")
            actual_sector = "Z" if source is not active else "X"
            _require(opts.get("sector", "X") in ("auto", actual_sector), "General source is for another requested sector")
            with using_backend("sympy"):
                report = verify_decoupling(source)
                _require(report.valid, "Invalid general certificate: " + "; ".join(report.errors))
                selection = evidence["lattice_selection"]
                if opts.get("coarse_graining", "adaptive") == "adaptive":
                    from .analysis import verify_analysis
                    _require(isinstance(selection, dict), "Adaptive general mode requires its profile lattice evidence")
                    _require(_same_json(selection["input"], active["input"]) and selection["sector"] == "X" and
                             _same_json(selection["ring"], {"prime": HX.ring.p, "dimension": HX.ring.D}),
                             "Profile lattice evidence is for another active CSS source")
                    profile_report = verify_analysis(selection)
                    _require(profile_report.valid, "Invalid profile lattice evidence: " + "; ".join(profile_report.errors))
                    _require(_same_json(selection["final_basis"], active["lattice"]["basis_columns"]),
                             "Profile lattice evidence does not end at the returned lattice")
                    if parsed["lattice_mode"] == "auto":
                        _require(_same_json(selection["initial_basis"], selection["translation_kernel"]["basis_columns"]),
                                 "Automatic general selection must start at the complete translation kernel")
                    else:
                        _require(_same_json(selection["initial_basis"], parsed["lattice_basis"]) and
                                 _same_json(selection["final_basis"], parsed["lattice_basis"]),
                                 "Fixed general profile changed the requested lattice")
                else:
                    _require(selection is None, "Uniform general mode uses Algorithm 1 lattice evidence")
                construction = evidence["construction"]
                if opts.get("comparison_method", "schanuel") == "schanuel":
                    _require(construction is not None, "Schanuel route requires its Algorithm 2 source certificate")
                if construction is not None:
                    linear = opts.get("linear_backend") or "sympy"
                    _require(construction["configuration"]["linear_backend"] == linear,
                             "Algorithm 2 backend declaration conflicts with the requested native backend")
                    proof = verify_construction(construction, linear_backend=linear,
                                                bounded_options=opts.get("bounded_options"))
                    _require(proof.valid, "Invalid Algorithm 2 evidence: " + "; ".join(proof.errors))
                    _require(_same_json(construction["decoupling"]["stabilization"], active["stabilization"]),
                             "Algorithm 2 stabilization differs from the original-code source")
                    if opts.get("coarse_graining") == "uniform":
                        _require(construction["coarse_graining"] is not None,
                                 "Uniform construction requires its Algorithm 1 source")
                        _require(_same_json(construction["physical_decoupling"], active),
                                 "Uniform Algorithm 1/2 proof differs from the original-code output")
            checks.append("independent general original-input and optional bounded Algorithm 2 certificates")
        qca, circuit_view, scope = derive_views(selected, evidence, request["output"])
        claimed = qca_from_dict(data["qca"], HX.ring)
        _require(_same_json(qca_dict(claimed), qca_dict(qca)), "Unified physical QCA differs from the verified source")
        expected_circuit = None if circuit_view is None else circuit_view.to_dict()
        _require(_same_json(data["circuit"], expected_circuit), "Unified gates/action/relation differ from the verified source")
        if parsed["lattice_mode"] == "fixed":
            _require(scope["lattice"]["basis_columns"] == parsed["lattice_basis"], "Fixed lattice was changed")
        _require(_same_json(data["metadata"], scope), "Unified lattice/target/ancilla metadata changed")
        diagnostic = data["diagnostics"]
        _require(isinstance(diagnostic, dict) and set(diagnostic) == {"scope", "error", "configuration_scope"},
                 "Malformed unified diagnostics")
        _require(_same_json(diagnostic["scope"], scope) and
                 diagnostic["configuration_scope"] == "requested options; not runtime attestation",
                 "Diagnostic mathematical scope changed")
        error = diagnostic["error"]
        if request["output"] == "qca":
            _require(data["status"] == "qca_ready" and circuit_view is None and evidence["circuit"] is None and error is None,
                     "QCA-only status/output mismatch")
        elif circuit_view is not None:
            _require(data["status"] == "circuit_ready" and error is None, "Complete circuit status mismatch")
        else:
            _require(selected == "two_d" and data["status"] == "circuit_incomplete", "Missing claimed complete circuit")
            _require(isinstance(error, dict) and set(error) == {"type", "message", "reason", "phase", "details"} and
                     isinstance(error["type"], str) and isinstance(error["message"], str) and bool(error["message"]) and
                     isinstance(error["reason"], str) and bool(error["reason"]) and
                     (error["phase"] is None or isinstance(error["phase"], str) and bool(error["phase"])) and
                     (error["details"] is None or isinstance(error["details"], dict)),
                     "Incomplete circuit requires a diagnostic reason")
            checks.append("completed QCA retained; circuit error is diagnostic, not a nonexistence certificate")
        checks.extend(("canonical input and requested fixed lattice bound to backend evidence",
                       "physical X/Z/inverses, actual gate view, target and ancilla scope bound to verified source"))
    except (ValueError, TypeError, KeyError, AttributeError, IndexError, ArithmeticError) as exc:
        errors.append(str(exc))
    # Resource stops in exact verification are not invalid mathematical proofs.
    return VerificationReport(not errors, tuple(checks), tuple(errors), _SCOPE)


def save_result(result, path):
    data = result.certificate() if isinstance(result, UnifiedResult) else deepcopy(result)
    report = verify_result(data)
    if not report.valid:raise ValueError("Cannot save invalid unified result: " + "; ".join(report.errors))
    return write_new_json_files({"result": data}, {"result": path})["result"]


def load_result(path, *, verify=True):
    if type(verify) is not bool:raise TypeError("verify must be boolean")
    data = read_json(path)
    if verify:
        report = verify_result(data)
        if not report.valid:raise ValueError("Invalid unified result: " + "; ".join(report.errors))
    return UnifiedResult.from_certificate(data)
