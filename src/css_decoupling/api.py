"""Unified native CSS input dispatch; legacy high-dimensional APIs stay separate."""
from __future__ import annotations

from .algebra.laurent import Matrix
from .raw_input import css_input_dict
from .result import UnifiedResult, _json_copy, derive_views, to_two_d

GENERAL_OPTIONS = frozenset({"max_cell_index", "max_layer_dimension", "max_unknowns",
    "resolution_timeout", "resolution_method", "max_dlog_steps", "backend", "sector",
    "qs_options", "lattice_strategy", "max_exterior_entries", "comparison_method",
    "linear_backend", "bounded_options", "contraction_strategy", "structural_search",
    "coarse_graining"})
TWO_D_OPTIONS = frozenset({"max_cell_index", "max_layer_dimension", "max_matrix_size", "max_terms",
                           "circuit_options"})
CIRCUIT_OPTIONS = frozenset({"factorization", "timeout", "max_steps", "max_matrix_terms",
    "max_factors", "max_families", "max_products", "max_patches", "max_field_trials", "seed",
    "rank_two_options"})
RANK_TWO_OPTIONS = frozenset({"search_width", "search_depth", "max_nodes",
                             "max_quotient_steps", "max_quotient_terms"})


def normalize_request(HX, HZ, dimension=None, p=None, *, algorithm="auto", output="qca",
                      lattice_mode="auto", lattice_basis=None, options=None):
    """Validate controls before invoking any algebra constructor."""
    if not isinstance(HX, Matrix) or not isinstance(HZ, Matrix):
        raise TypeError("HX/HZ must use the public css_decoupling.Matrix type")
    if HX.ring != HZ.ring or HX.ncols != HZ.ncols or HX.ring.D < 2:
        raise ValueError("HX/HZ require the same prime Laurent ring, D>=2, and physical columns")
    for value, actual, name in ((dimension, HX.ring.D, "dimension"), (p, HX.ring.p, "p")):
        if value is not None and (type(value) is not int or value != actual):
            raise ValueError(name + " conflicts with the input matrix ring")
    if algorithm not in ("auto", "two_d", "general"):
        raise ValueError("algorithm must be auto, two_d or general")
    selected = ("two_d" if HX.ring.D == 2 else "general") if algorithm == "auto" else algorithm
    if selected == "two_d" and HX.ring.D != 2:
        raise ValueError("The two_d algorithm requires dimension=2")
    if output not in ("qca", "circuit"):
        raise ValueError("output must be qca or circuit")
    # Explicit mode: a supplied basis never silently changes automatic policy.
    if lattice_mode not in ("auto", "fixed"):
        raise ValueError("lattice_mode must be auto or fixed")
    inp = css_input_dict(HX, HZ, lattice_mode=lattice_mode, lattice_basis=lattice_basis)
    if options is None:options = {}
    if not isinstance(options, dict):raise TypeError("options must be a dict")
    options = _json_copy(options)
    allowed = TWO_D_OPTIONS if selected == "two_d" else GENERAL_OPTIONS
    unknown = set(options) - allowed
    if unknown:raise ValueError("Unknown " + selected + " option(s): " + ", ".join(sorted(unknown)))
    for name, value in options.items():
        if name.startswith("max_") and value is not None and (type(value) is not int or value < 0):
            raise ValueError(name + " must be a nonnegative integer or None")
    if selected == "two_d":
        for name in ("max_cell_index", "max_layer_dimension", "max_matrix_size", "max_terms"):
            if name in options and (type(options[name]) is not int or options[name] < 1):
                raise ValueError(name + " must be a positive integer for two_d")
        if "circuit_options" in options:
            if output != "circuit":raise ValueError("circuit_options require output='circuit'")
            caps = options["circuit_options"]
            if not isinstance(caps, dict) or set(caps) - CIRCUIT_OPTIONS:
                raise ValueError("Unknown or malformed circuit_options")
            if caps.get("factorization", "auto") not in ("auto", "suslin", "fast"):
                raise ValueError("factorization must be auto, suslin or fast")
            for name, value in caps.items():
                if name.startswith("max_") and (type(value) is not int or value <
                        (0 if name == "max_families" else 1)):
                    raise ValueError("Invalid circuit limit: " + name)
            if "seed" in caps and type(caps["seed"]) is not int:raise ValueError("seed must be an integer")
            if "timeout" in caps and (type(caps["timeout"]) not in (int, float) or caps["timeout"] <= 0):
                raise ValueError("Circuit timeout must be finite and positive")
            if "rank_two_options" in caps:
                rank = caps["rank_two_options"]
                if not isinstance(rank, dict) or set(rank) - RANK_TWO_OPTIONS:
                    raise ValueError("Unknown or malformed rank_two_options")
                if any(type(v) is not int or v < (0 if k == "search_depth" else 1) for k, v in rank.items()):
                    raise ValueError("Invalid rank_two_options resource cap")
    else:
        if options.get("backend", "sympy") != "sympy":raise ValueError("backend must be sympy")
        if options.get("resolution_method", "native") != "native":raise ValueError("resolution_method must be native")
        if options.get("linear_backend", "sympy") not in (None, "sympy", "bounded"):
            raise ValueError("linear_backend must be sympy or bounded")
        if options.get("sector", "X") not in ("X", "Z", "auto"):raise ValueError("Invalid sector")
        if options.get("lattice_strategy", "auto") not in ("auto", "coordinate", "optimal"):
            raise ValueError("Invalid lattice_strategy")
        if options.get("comparison_method", "schanuel") not in ("schanuel", "free-basis"):
            raise ValueError("Invalid comparison_method")
        if options.get("coarse_graining", "adaptive") not in ("adaptive", "uniform"):
            raise ValueError("Invalid coarse_graining")
        if options.get("coarse_graining") == "uniform" and (lattice_mode == "fixed" or
                options.get("lattice_strategy", "auto") != "auto" or options.get("comparison_method", "schanuel") != "schanuel"):
            raise ValueError("Uniform coarse graining chooses its own lattice and requires Schanuel comparison")
        nested = {
            "qs_options": {"timeout", "max_terms", "max_steps", "max_patches", "max_field_trials", "seed"},
            "bounded_options": {"max_degree", "max_unknowns", "max_equations", "max_matrix_entries"}}
        for name, keys in nested.items():
            if name in options and options[name] is not None:
                value = options[name]
                if not isinstance(value, dict) or set(value) - keys:
                    raise ValueError("Unknown or malformed " + name)
        comparison = options.get("comparison_method", "schanuel")
        if options.get("qs_options") is not None and comparison != "free-basis":
            raise ValueError("qs_options require comparison_method='free-basis'")
        if options.get("bounded_options") is not None and options.get("linear_backend") != "bounded":
            raise ValueError("bounded_options require linear_backend='bounded'")
        if comparison == "free-basis" and (options.get("linear_backend") == "bounded" or
                options.get("bounded_options") is not None or options.get("contraction_strategy", "auto") != "auto" or
                options.get("structural_search", False)):
            raise ValueError("free-basis is incompatible with bounded or explicit contraction policy")
        from .decoupling.construction import _construction_policy
        _construction_policy(options.get("linear_backend") or "sympy",
            options.get("contraction_strategy", "auto"), options.get("structural_search", False))
    return selected, inp, {"algorithm": algorithm, "output": output, "options": options}


def _error_record(exc):
    details = getattr(exc, "details", None)
    try:details = _json_copy(details)
    except (TypeError, ValueError):details = {"omitted_type": type(details).__name__}
    category = "resource_limit" if any("Resource" in cls.__name__ for cls in type(exc).__mro__) else \
               "factorization_incomplete" if any("Incomplete" in cls.__name__ for cls in type(exc).__mro__) else "backend_unavailable"
    return {"type": type(exc).__name__, "message": str(exc) or type(exc).__name__,
            "reason": str(exc.reason) if getattr(exc, "reason", None) is not None else category,
            "phase": str(exc.phase) if getattr(exc, "phase", None) is not None else None,
            "details": details}


def decouple(HX, HZ, dimension=None, p=None, *, algorithm="auto", output="qca",
             lattice_mode="auto", lattice_basis=None, options=None):
    """Compute a source-bound QCA, optionally a separately described circuit.

    D=2 auto uses the two-dimensional paper; higher dimensions use the general
    construction. Use decouple_general for the previous root-level Python API.
    General QCA requests still execute its existing stabilized circuit pipeline.
    Two-dimensional circuit stops retain a fully verified QCA in the result.
    """
    selected, inp, request = normalize_request(HX, HZ, dimension, p, algorithm=algorithm,
        output=output, lattice_mode=lattice_mode, lattice_basis=lattice_basis, options=options)
    opts = dict(request["options"])
    error = None
    if selected == "two_d":
        from .two_d.pipeline import decouple as build_qca
        from .two_d.serialization import encode
        from .errors import ResourceLimit, BackendUnavailable, InfiniteLayer, NoLaurentSolution
        from .two_d.unstable import UnstableFactorizationIncomplete
        from .two_d.elementary import FactorizationIncomplete
        construction = {name: opts[name] for name in ("max_matrix_size", "max_terms") if name in opts}
        for public, native in (("max_cell_index", "max_index"), ("max_layer_dimension", "max_dimension")):
            if public in opts:construction[native] = opts[public]
        X, Z = to_two_d(HX), to_two_d(HZ)
        try:
            raw = build_qca(X, Z, lattice_mode=lattice_mode, lattice_basis=inp["lattice_basis"],
                            compile_circuit=False, **construction)
        except (ResourceLimit, BackendUnavailable, InfiniteLayer, NoLaurentSolution) as exc:
            cls = ResourceLimit if isinstance(exc, ResourceLimit) else BackendUnavailable if isinstance(exc, BackendUnavailable) else \
                  InfiniteLayer if isinstance(exc, InfiniteLayer) else NoLaurentSolution
            translated = cls(str(exc));translated.backend = "two_d";translated.diagnostics = _error_record(exc)
            raise translated from exc
        circuit = None
        if output == "circuit":
            from .two_d.code_circuits import compile_qca_no_ancilla
            try:
                circuit = compile_qca_no_ancilla(raw, verify_source=False, **opts.get("circuit_options", {}))
            except (ResourceLimit, BackendUnavailable, UnstableFactorizationIncomplete, FactorizationIncomplete) as exc:
                error = _error_record(exc)
                if raw["qca"]["psi"][1].nrows == 2:
                    if not isinstance(error["details"], dict):error["details"] = {}
                    error["details"]["general_suslin_rank_requirement"] = "q >= 3"
                    error["details"]["non_elementarity_proved"] = False
        evidence = {"kind": "two_d", "qca": encode(raw), "circuit": None if circuit is None else encode(circuit),
                    "construction": None, "lattice_selection": None}
        status = "qca_ready" if output == "qca" else "circuit_ready" if circuit is not None else "circuit_incomplete"
    else:
        from .decoupling.pipeline import decouple as build_general
        opts.setdefault("backend", "sympy")
        raw = build_general(HX, HZ, lattice_basis=inp["lattice_basis"], verify=False, **opts)
        active = raw.inner if hasattr(raw, "inner") else raw
        construction = active.trace.get("construction")
        analysis = active.trace.get("profile_analysis")
        evidence = {"kind": "general", "qca": raw.certificate(), "circuit": None,
                    "construction": None if construction is None else construction.certificate(),
                    "lattice_selection": None if analysis is None else analysis.certificate()}
        status = "qca_ready" if output == "qca" else "circuit_ready"
    qca, circuit, scope = derive_views(selected, evidence, output)
    result = UnifiedResult(selected, status, qca, circuit,
        {"scope": scope, "error": error, "configuration_scope": "requested options; not runtime attestation"},
        evidence, inp, request)
    from .result import verify_result
    report = verify_result(result, expected_HX=HX, expected_HZ=HZ)
    if not report.valid:raise ArithmeticError("Unified result failed independent verification: " + "; ".join(report.errors))
    return result
