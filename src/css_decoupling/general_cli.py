"""Command-line entry point for CSS inputs and exact certificate verification."""
import argparse
from dataclasses import asdict
import json
import logging
import sys
from pathlib import Path

from .algebra.modules import using_backend
from .certificates.decoupling import verify_decoupling
from .certificates.io import parse_input, save_decoupling
from .decoupling.pipeline import decouple
from .decoupling.analysis import analyze_profile
from .certificates.analysis import save_analysis, verify_analysis
from .certificates.gluing import verify_gluing
from .errors import BackendUnavailable, ResourceLimit, InfiniteLayer
from .certificates.output_io import (
    output_paths, check_new_output_paths, write_new_json_files,
    save_decoupling_outputs, save_exported_outputs,
)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="css-decouple",
        description="Exact stable decoupling of prime-qudit CSS codes",
    )
    parser.add_argument("--verbose", action="store_true", help="Log exact algebra operations")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", aliases=["general"], help="Decouple CSS checks from JSON")
    analyze = sub.add_parser("analyze", help="Export exact Lambda0 and successive gluing classes")
    coarse = sub.add_parser("coarse", help="Algorithm 1: uniform sufficient lattice and complete restricted free model")
    for command in (run, analyze, coarse):
        command.add_argument("input", type=Path)
        command.add_argument("--output", type=Path, required=True, help="Exact certificate output")
        command.add_argument("--backend", choices=["sympy"], default="sympy",
                             help="Laurent module solver; sympy requires no external executable")
        command.add_argument("--resolution-method", choices=["native"], default="native")
        command.add_argument("--sector", choices=["X", "Z", "auto"], default="auto")
        command.add_argument("--max-cell-index", type=int, default=4096,
                             help="Cell index admission cap (default: 4096); not a direction-wise scale")
        command.add_argument("--max-layer-dimension", type=int, default=10000)
        if command is not coarse:
            command.add_argument("--max-unknowns", type=int, default=250000)
        command.add_argument("--max-dlog-steps", type=int, default=100000)
        if command is not coarse:
            command.add_argument("--lattice-strategy", choices=["auto", "coordinate", "optimal"], default="auto",
                                 help="auto: exact translation kernel plus adaptive gluing reduction; optimal: require supported stage optimum, not general global optimality")
            command.add_argument("--max-exterior-entries", type=int, default=2000000)
        command.add_argument("--timeout", type=float, default=120,
                             help="Cooperative seconds per native resolution or QS row completion; not a whole-run deadline")
    run.add_argument("--analysis-output", type=Path,
                     help="Stage-one evidence; defaults to OUTPUT_STEM.analysis.json or .coarse.json")
    run.add_argument("--coarse-graining", choices=["adaptive", "uniform"], default="adaptive",
                     help="adaptive: actual gluing; uniform: paper Algorithm 1 sufficient lattice")
    run.add_argument("--comparison-method", choices=["schanuel", "free-basis"], default="schanuel")
    run.add_argument("--linear-backend", choices=["sympy", "bounded"])
    construct = sub.add_parser("construct", help="Algorithm 2: circuit from a complete free model or Algorithm 1 certificate")
    construct.add_argument("input", type=Path)
    construct.add_argument("--output", type=Path, required=True)
    construct.add_argument("--linear-backend", choices=["sympy", "bounded"], default="sympy")
    construct.add_argument("--max-layer-dimension", type=int, default=10000)
    construct.add_argument("--max-unknowns", type=int, default=250000)
    construct.add_argument("--max-total-rank", type=int, default=100000)
    construct.add_argument("--bounded-max-degree", type=int, default=64)
    construct.add_argument("--bounded-max-unknowns", type=int, default=10000)
    construct.add_argument("--bounded-max-equations", type=int, default=20000)
    construct.add_argument("--bounded-max-entries", type=int, default=2000000)
    for command in (run, construct):
        command.add_argument("--qca-output", type=Path, help="Physical QCA sidecar; default OUTPUT_STEM.qca.json")
        command.add_argument("--circuit-output", type=Path, help="Gate-list sidecar; default OUTPUT_STEM.circuit.json")
        command.add_argument("--contraction-strategy", choices=["auto", "normalized", "sections"], default="auto",
                             help="sections: Appendix C4 sections degree by degree; "
                                  "auto: native normalized, other backends sections")
        command.add_argument("--structural-search", action="store_true",
                             help="Opt into finite-support probes; requires native --contraction-strategy sections")
    torus = sub.add_parser("torus", help="Exact logical-qudit count on an integer period lattice")
    torus.add_argument("input", type=Path)
    torus.add_argument("--basis", type=json.loads, required=True,
                       help='Integer rows as JSON; columns are periods, e.g. "[[1,0],[0,3]]"')
    torus.add_argument("--analysis", type=Path, help="Verified source profile for the fast count and decoupling bound")
    torus.add_argument("--finite", action="store_true", help="Force finite periodic matrices for cross-checks")
    torus.add_argument("--output", type=Path, required=True)
    torus.add_argument("--backend", choices=["sympy"], default="sympy")
    torus.add_argument("--max-qudits", type=int, default=100000)
    torus.add_argument("--max-matrix-entries", type=int, default=2000000)
    torus.add_argument("--max-rank-work", type=int, default=20000000)
    query = sub.add_parser("lattice-check", help="Decide a specified lattice from CSS input or an existing profile analysis")
    query.add_argument("input", type=Path, help="Original CSS JSON or a verified profile-analysis certificate")
    query.add_argument("--basis", type=json.loads, required=True)
    query.add_argument("--output", type=Path, required=True)
    query.add_argument("--backend", choices=["sympy"], default="sympy")
    query.add_argument("--sector", choices=["X", "Z", "auto"], default="auto")
    query.add_argument("--resolution-method", choices=["native"], default="native")
    query.add_argument("--max-cell-index", type=int, default=4096)
    query.add_argument("--max-layer-dimension", type=int, default=10000)
    query.add_argument("--max-unknowns", type=int, default=250000)
    query.add_argument("--timeout", type=float, default=120)
    query.add_argument("--max-exterior-entries", type=int, default=2000000)
    comparison = sub.add_parser("koszul-map", help="Construct the explicit lattice comparison chain map chi")
    comparison.add_argument("--prime", type=int, required=True)
    comparison.add_argument("--dimension", type=int, required=True)
    comparison.add_argument("--basis", type=json.loads, required=True, help="Integer matrix with new lattice basis in columns")
    comparison.add_argument("--degrees", type=int, nargs="+", default=[1], help="Requested exterior degrees; only these and their predecessors are materialized")
    comparison.add_argument("--max-terms", type=int, default=2000000)
    comparison.add_argument("--max-entries", type=int, default=2000000)
    comparison.add_argument("--max-products", type=int, default=20000000)
    comparison.add_argument("--output", type=Path, required=True)
    verify = sub.add_parser("verify", help="Recompute exact identities from a certificate")
    verify.add_argument("path", type=Path)
    verify.add_argument("--source-certificate", type=Path,
                        help="Required complete source for a QCA or circuit sidecar")
    verify.add_argument("--backend", choices=["sympy"], default="sympy")
    for name in ("max-cell-index", "max-layer-dimension", "max-unknowns", "max-qudits",
                 "max-exterior-entries", "max-matrix-entries", "max-rank-work",
                 "max-terms", "max-entries", "max-products", "max-degree", "max-equations", "max-total-rank"):
        verify.add_argument("--" + name, type=int, default=argparse.SUPPRESS,
                            help="Override this resource cap for a supporting certificate schema")
    exporter = sub.add_parser("export-outputs", help="Export QCA and circuit from a verified complete certificate")
    exporter.add_argument("input", type=Path)
    exporter.add_argument("--qca-output", type=Path)
    exporter.add_argument("--circuit-output", type=Path)
    pair = sub.add_parser("verify-outputs", help="Independently verify QCA and circuit against a complete source")
    pair.add_argument("input", type=Path, help="The complete source certificate")
    pair.add_argument("--qca", type=Path, required=True)
    pair.add_argument("--circuit", type=Path, required=True)
    for command in (exporter, pair):
        command.add_argument("--linear-backend", choices=["sympy", "bounded"])
        command.add_argument("--bounded-options", type=json.loads)
        command.add_argument("--max-total-rank", type=int, default=100000)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    planned_outputs = None
    if args.command in ("run", "general", "construct"):
        planned_outputs = output_paths(args.output, qca_path=args.qca_output, circuit_path=args.circuit_output)
        preflight = dict(planned_outputs)
        if args.command != "construct":
            suffix = ".coarse.json" if args.coarse_graining == "uniform" else ".analysis.json"
            preflight["stage_one"] = args.analysis_output or args.output.with_name(args.output.stem + suffix)
        try:
            check_new_output_paths(preflight, protected=(args.input,))
        except (OSError, ValueError) as exc:
            print(json.dumps({"error": type(exc).__name__, "message": str(exc)}), file=sys.stderr)
            return 2
    if args.command in ("export-outputs", "verify-outputs"):
        from .certificates.outputs import verify_output_pair
        limits = dict(linear_backend=args.linear_backend, bounded_options=args.bounded_options,
                      max_total_rank=args.max_total_rank)
        try:
            if args.command == "export-outputs":
                paths = output_paths(args.input, qca_path=args.qca_output, circuit_path=args.circuit_output)
                check_new_output_paths({key: paths[key] for key in ("qca", "circuit")}, protected=(args.input,))
            source = json.loads(args.input.read_text(encoding="utf-8"))
            if args.command == "export-outputs":
                written = save_exported_outputs(source, qca_path=paths["qca"], circuit_path=paths["circuit"],
                                                source_path=args.input, **limits)
                print(json.dumps({"source_certificate": str(args.input),
                                  "qca_output": str(written["qca"]), "circuit_output": str(written["circuit"])}, indent=2))
                return 0
            report = verify_output_pair(json.loads(args.qca.read_text(encoding="utf-8")),
                json.loads(args.circuit.read_text(encoding="utf-8")), source_certificate=source, **limits)
            print(json.dumps(asdict(report), indent=2))
            return 0 if report.valid else 1
        except (BackendUnavailable, ResourceLimit, OSError, ValueError, TypeError) as exc:
            print(json.dumps({"error": type(exc).__name__, "message": str(exc)}), file=sys.stderr)
            return 2
    if args.command == "verify":
        data = json.loads(args.path.read_text(encoding="utf-8"))
        from .certificates.outputs import QCA_SCHEMA, CIRCUIT_SCHEMA, verify_qca_output, verify_circuit_output
        output_source = None
        if data.get("schema") in (QCA_SCHEMA, CIRCUIT_SCHEMA):
            if args.source_certificate is None:
                print(json.dumps({"error": "MissingSource", "message": "QCA/circuit verification requires --source-certificate"}), file=sys.stderr)
                return 2
            output_source = json.loads(args.source_certificate.read_text(encoding="utf-8"))
        elif args.source_certificate is not None:
            raise ValueError("--source-certificate only applies to QCA/circuit output schemas")
        from .certificates.periodic import verify_periodic, verify_periodic_layer
        from .certificates.lattice_query import verify_lattice_decision
        from .certificates.criterion import verify_lattice_criterion
        from .certificates.koszul_comparison import verify_koszul_comparison
        from .certificates.cone_decoupling import verify_cone_decoupling
        from .certificates.coarse_graining import verify_coarse_graining
        from .certificates.construction import verify_construction
        from .certificates.schanuel import verify_schanuel
        from .certificates.bounded_linear import verify_bounded_linear, verify_trivial_cohomology
        verifier = {"css-decoupling.profile-analysis": verify_analysis,
                    QCA_SCHEMA: verify_qca_output, CIRCUIT_SCHEMA: verify_circuit_output,
                    "css-decoupling.gluing": verify_gluing,
                    "css-decoupling.periodic": verify_periodic,
                    "css-decoupling.periodic-layer": verify_periodic_layer,
                    "css-decoupling.lattice-decision": verify_lattice_decision,
                    "css-decoupling.lattice-criterion": verify_lattice_criterion,
                    "css-decoupling.koszul-comparison": verify_koszul_comparison,
                    "css-decoupling.coarse-graining": verify_coarse_graining,
                    "css-decoupling.algorithm-two": verify_construction,
                    "css-decoupling.schanuel-cokernel": verify_schanuel,
                    "css-decoupling.bounded-linear": verify_bounded_linear,
                    "css-decoupling.trivial-cohomology": verify_trivial_cohomology,
                    "css-decoupling.cone-direct": verify_cone_decoupling}.get(data.get("schema"), verify_decoupling)
        supported = {
            QCA_SCHEMA: {"max_total_rank"}, CIRCUIT_SCHEMA: {"max_total_rank"},
            "css-decoupling.lattice-criterion": {"max_cell_index", "max_layer_dimension", "max_unknowns"},
            "css-decoupling.periodic": {"max_qudits", "max_exterior_entries", "max_matrix_entries", "max_rank_work"},
            "css-decoupling.periodic-layer": {"max_exterior_entries", "max_matrix_entries", "max_rank_work"},
            "css-decoupling.koszul-comparison": {"max_terms", "max_entries", "max_products"},
            "css-decoupling.coarse-graining": {"max_cell_index", "max_layer_dimension"},
            "css-decoupling.bounded-linear": {"max_unknowns", "max_equations", "max_matrix_entries", "max_degree"},
            "css-decoupling.trivial-cohomology": {"max_unknowns", "max_equations", "max_matrix_entries", "max_degree"},
            "css-decoupling.algorithm-two": {"max_total_rank"},
        }.get(data.get("schema"), set())
        overrides = {name: value for name, value in vars(args).items() if name.startswith("max_")}
        if set(overrides) - supported:
            raise ValueError("Unsupported verification resource options for this schema: " +
                             ", ".join(sorted(set(overrides) - supported)))
        try:
            with using_backend(args.backend):
                report = (verifier(data, source_certificate=output_source, **overrides)
                          if data.get("schema") in (QCA_SCHEMA, CIRCUIT_SCHEMA) else verifier(data, **overrides))
        except (BackendUnavailable, ResourceLimit) as exc:
            print(json.dumps({"error": type(exc).__name__, "message": str(exc)}), file=sys.stderr)
            return 2
        print(json.dumps(asdict(report), indent=2))
        return 0 if report.valid else 1

    if args.command == "construct":
        from .algebra.laurent import Ring
        from .certificates.coarse_graining import verify_coarse_graining, _bounded_model
        from .certificates.construction import verify_construction, circuit_resources
        from .decoupling.construction import decouple_complex
        try:
            data = json.loads(args.input.read_text())
            origin = None
            if data.get("schema") == "css-decoupling.coarse-graining":
                report = verify_coarse_graining(data)
                if not report.valid or not data["decouplable"]:
                    raise ValueError("Algorithm 2 requires a verified successful Algorithm 1 output")
                origin, model = data, data["restricted"]
            elif data.get("schema") == "css-decoupling.free-complex" and type(data.get("version")) is int and data["version"] == 1:
                model = data["model"]
            else:
                raise ValueError("Construct input must be an Algorithm 1 certificate or free-complex v1")
            C = _bounded_model(model, Ring(data["ring"]["prime"], data["ring"]["dimension"]))
            limits = dict(max_degree=args.bounded_max_degree, max_unknowns=args.bounded_max_unknowns,
                          max_equations=args.bounded_max_equations, max_matrix_entries=args.bounded_max_entries)
            handoff = None
            if origin is not None and args.linear_backend == "sympy":
                from .topology.cohomology_handoff import handoff_from_coarse_certificate
                handoff = handoff_from_coarse_certificate(origin, max_cell_index=None,
                                                          max_total_rank=args.max_total_rank)
            result = decouple_complex(C, linear_backend=args.linear_backend,
                bounded_options=limits if args.linear_backend == "bounded" else None,
                max_layer_dimension=args.max_layer_dimension, max_unknowns=args.max_unknowns,
                max_total_rank=args.max_total_rank, verify=True, cohomology_handoff=handoff,
                contraction_strategy=args.contraction_strategy, structural_search=args.structural_search,
                restriction_basis=origin["basis"] if origin is not None and args.linear_backend == "sympy" else None)
            result.coarse_graining = origin
            if origin is not None:
                report = verify_construction(result, linear_backend=args.linear_backend,
                    bounded_options=limits if args.linear_backend == "bounded" else None,
                    max_total_rank=args.max_total_rank)
                if not report.valid:
                    raise ArithmeticError("Composed algorithm certificate failed: " + "; ".join(report.errors))
            written = save_decoupling_outputs(result, args.output,
                qca_path=planned_outputs["qca"], circuit_path=planned_outputs["circuit"],
                linear_backend=args.linear_backend,
                bounded_options=limits if args.linear_backend == "bounded" else None,
                max_total_rank=args.max_total_rank)
            print(json.dumps({"profile": result.multiplicities, "input_size": result.input_size,
                              "contraction_strategy": result.contraction_strategy,
                              "structural_search": result.structural_search,
                              "resources": circuit_resources(result.decoupling.stabilization),
                              "certificate": str(written["certificate"]),
                              "qca_output": str(written["qca"]), "circuit_output": str(written["circuit"])}, indent=2))
            return 0
        except (BackendUnavailable, ResourceLimit, OSError) as exc:
            print(json.dumps({"error": type(exc).__name__, "message": str(exc)}), file=sys.stderr)
            return 2

    if args.command == "koszul-map":
        from .algebra.laurent import Ring
        from .lattice.koszul_comparison import koszul_comparison
        try:
            result = koszul_comparison(Ring(args.prime, args.dimension), args.basis,
                degrees=args.degrees, max_terms=args.max_terms,
                max_entries=args.max_entries, max_products=args.max_products)
        except (BackendUnavailable, ResourceLimit) as exc:
            print(json.dumps({"error": type(exc).__name__, "message": str(exc)}), file=sys.stderr)
            return 2
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result.certificate(), indent=2) + "\n")
        print(json.dumps({"requested_degrees": list(result.requested_degrees),
                          "materialized_degrees": sorted(result.maps),
                          "certificate": str(args.output)}, indent=2))
        return 0

    if args.command in ("torus", "lattice-check"):
        try:
            with using_backend(args.backend):
                if args.command == "lattice-check":
                    from .decoupling.lattice_query import check_two_layer_lattice
                    from .decoupling.criterion import check_lattice
                    from .certificates.comparison import _matrix_from_dict
                    from .algebra.laurent import Ring
                    from .lattice.retained import Lattice
                    data = json.loads(args.input.read_text())
                    sector, fast = args.sector, False
                    if data.get("schema") == "css-decoupling.profile-analysis":
                        report = verify_analysis(data)
                        if not report.valid:
                            raise ValueError("Invalid source analysis: " + "; ".join(report.errors))
                        ring = Ring(data["ring"]["prime"], data["ring"]["dimension"])
                        HX, HZ = (_matrix_from_dict(data["input"][name], ring, name) for name in ("HX", "HZ"))
                        if sector == "auto":
                            sector = data.get("sector", "X")
                        # Retain the inexpensive legacy query when its recorded
                        # gluing coordinates cover the candidate. Otherwise the
                        # general criterion recomputes on the requested lattice.
                        fast = sector == data.get("sector", "X") and len(data["layers"]) <= 2
                        if fast and len(data["layers"]) == 2:
                            fast = Lattice(data["initial_basis"], max_index=None).contains(Lattice(args.basis, max_index=None))
                    else:
                        HX, HZ, _ = parse_input(data)
                    if fast:
                        result = check_two_layer_lattice(data, args.basis,
                            max_exterior_entries=args.max_exterior_entries)
                    else:
                        result = check_lattice(HX, HZ, args.basis, sector=sector,
                            max_cell_index=args.max_cell_index,
                            max_layer_dimension=args.max_layer_dimension,
                            max_unknowns=args.max_unknowns,
                            resolution_timeout=args.timeout,
                            resolution_method=args.resolution_method)
                    summary = {"decouplable": result.decouplable, "reason": result.reason,
                               "cell_index": result.lattice.index}
                else:
                    from .topology.periodic import periodic_code, periodic_profile
                    from .certificates.comparison import _matrix_from_dict
                    HX, HZ, _ = parse_input(json.loads(args.input.read_text()))
                    limits = {name: getattr(args, name) for name in
                              ("max_qudits", "max_matrix_entries", "max_rank_work")}
                    if args.analysis:
                        analysis = json.loads(args.analysis.read_text())
                        # Bind the optional evidence to the requested input,
                        # not merely to another valid code with the same k.
                        if analysis["ring"] != {"prime": HX.ring.p, "dimension": HX.ring.D} or any(
                            _matrix_from_dict(analysis["input"][name], HX.ring, name) != matrix
                            for name, matrix in (("HX", HX), ("HZ", HZ))
                        ):
                            raise ValueError("Profile analysis belongs to a different CSS input")
                        result = periodic_profile(analysis, args.basis, prefer_fast=not args.finite, **limits)
                    else:
                        result = periodic_code(HX, HZ, args.basis, **limits)
                    summary = {name: getattr(result, name) for name in
                               ("index", "logical_qudits", "method", "upper_bound", "decouplable",
                                "layer_contributions", "translation_loss", "gluing_loss")}
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result.certificate(), indent=2) + "\n")
            print(json.dumps({**summary, "certificate": str(args.output)}, indent=2))
            return 0
        except (BackendUnavailable, ResourceLimit) as exc:
            print(json.dumps({"error": type(exc).__name__, "message": str(exc)}), file=sys.stderr)
            return 2

    HX, HZ, basis = parse_input(json.loads(args.input.read_text()))
    if args.command == "coarse":
        from .decoupling.coarse_graining import coarse_grain
        if basis is not None:
            raise ValueError("Algorithm 1 chooses its sufficient lattice; use lattice-check/run for a supplied basis")
        try:
            result = coarse_grain(HX, HZ, backend=args.backend, sector=args.sector,
                max_cell_index=args.max_cell_index, max_layer_dimension=args.max_layer_dimension,
                max_dlog_steps=args.max_dlog_steps, resolution_timeout=args.timeout,
                resolution_method=args.resolution_method)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result.certificate(include_handoff=True), indent=2) + "\n",
                                   encoding="utf-8", newline="\n")
            print(json.dumps({"decouplable": result.decouplable, "reason": result.reason,
                "profile": result.multiplicities, "sufficient_index": None if result.lattice is None else result.lattice.index,
                "certificate": str(args.output)}, indent=2))
            return 0
        except (BackendUnavailable, ResourceLimit) as exc:
            print(json.dumps({"error": type(exc).__name__, "message": str(exc),
                              "partial": getattr(exc, "coarse_graining_partial", None)}, default=str), file=sys.stderr)
            return 2
    options = dict(lattice_basis=basis,
        max_cell_index=args.max_cell_index,
        max_layer_dimension=args.max_layer_dimension,
        max_unknowns=args.max_unknowns,
        resolution_timeout=args.timeout,
        resolution_method=args.resolution_method, max_dlog_steps=args.max_dlog_steps,
        sector=args.sector, backend=args.backend, lattice_strategy=args.lattice_strategy,
        max_exterior_entries=args.max_exterior_entries,
    )
    if args.command == "analyze":
        analysis = analyze_profile(HX, HZ, **options)
        save_analysis(analysis, args.output)
        print(json.dumps({"sector": analysis.sector, "profile": analysis.multiplicities,
                          "Lambda0_index": analysis.translation_kernel.index,
                          "final_cell_index": analysis.lattice.index,
                          "lattice_guarantee": analysis.lattice_guarantee,
                          "gluing_stages": len(analysis.gluing), "certificate": str(args.output)}, indent=2))
        return 0
    evidence_suffix = ".coarse.json" if args.coarse_graining == "uniform" else ".analysis.json"
    analysis_path = args.analysis_output or args.output.with_name(args.output.stem + evidence_suffix)

    def save_coarse_evidence(coarse):
        write_new_json_files({"stage": coarse.certificate(include_handoff=True)}, {"stage": analysis_path})

    try:
        result = decouple(HX, HZ, comparison_method=args.comparison_method,
                          linear_backend=args.linear_backend,
                          contraction_strategy=args.contraction_strategy,
                          structural_search=args.structural_search,
                          coarse_graining=args.coarse_graining, **options)
    except (BackendUnavailable, ResourceLimit, InfiniteLayer) as exc:
        has_analysis, has_coarse = hasattr(exc, "profile_analysis"), hasattr(exc, "coarse_graining_result")
        failure_status = "incomplete"
        if isinstance(exc, InfiniteLayer):
            failure_status = ("certified-obstruction" if has_coarse and not exc.coarse_graining_result.decouplable
                              else "obstruction-without-certificate")
        if has_coarse:
            save_coarse_evidence(exc.coarse_graining_result)
        if hasattr(exc, "profile_analysis"):
            write_new_json_files({"stage": exc.profile_analysis.certificate()}, {"stage": analysis_path})
        print(json.dumps({"error": type(exc).__name__, "message": str(exc),
                          "status": failure_status,
                          "analysis_certificate": str(analysis_path) if has_analysis else None,
                          "coarse_graining_certificate": str(analysis_path) if has_coarse else None,
                          "partial": getattr(exc, "coarse_graining_partial", None)}, default=str),
              file=sys.stderr)
        return 2
    coarse = result.coarse_graining_result
    if coarse is not None:
        save_coarse_evidence(coarse)
        kernel_index, guarantee = coarse.translation_kernel.index, "uniform-sufficient-lattice"
    else:
        analysis = result.profile_analysis
        write_new_json_files({"stage": analysis.certificate()}, {"stage": analysis_path})
        kernel_index, guarantee = analysis.translation_kernel.index, analysis.lattice_guarantee
    try:
        # A raw complete certificate uses the original module/physical
        # verifier backend, not the separate Algorithm 2 linear backend.
        with using_backend(args.backend, timeout=args.timeout):
            written = save_decoupling_outputs(result, args.output,
                qca_path=planned_outputs["qca"], circuit_path=planned_outputs["circuit"])
    except (BackendUnavailable, ResourceLimit, OSError, ValueError, TypeError) as exc:
        print(json.dumps({"error": type(exc).__name__, "message": str(exc),
                          "stage_one_certificate": str(analysis_path)}), file=sys.stderr)
        return 2
    construction_trace = result.trace.get("inner", result.trace)
    print(json.dumps({
        "dimension": HX.ring.D, "prime": HX.ring.p,
        "profile": result.multiplicities, "cell_index": result.lattice.index,
        "Lambda0_index": kernel_index,
        "lattice_guarantee": guarantee,
        "coarse_graining": args.coarse_graining,
        "contraction_strategy": construction_trace.get("contraction_strategy", "free-basis"),
        "structural_search": construction_trace.get("structural_search", False),
        "physical_qudits": result.circuit.qudits,
        "gate_translation_families": len(result.circuit.gates),
        "timings_seconds": result.timings, "certificate": str(written["certificate"]),
        "qca_output": str(written["qca"]), "circuit_output": str(written["circuit"]),
        "analysis_certificate": None if coarse is not None else str(analysis_path),
        "coarse_graining_certificate": str(analysis_path) if coarse is not None else None,
    }, indent=2))
    return 0
