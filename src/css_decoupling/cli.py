"""One public CLI for specialized 2D and dimension-independent decoupling."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from .raw_input import read_json, load_css_input, save_css_input, _pairs
from .certificates.output_io import check_new_output_paths
from .errors import ResourceLimit, BackendUnavailable


def _object(value):
    def invalid(number):
        raise argparse.ArgumentTypeError("Non-finite option value: " + number)
    result = json.loads(value, object_pairs_hook=_pairs, parse_constant=invalid)
    if not isinstance(result, dict):
        raise argparse.ArgumentTypeError("Options must be a JSON object")
    return result


def _run(args):
    from .api import decouple
    from .result import save_result
    check_new_output_paths({"result": args.output}, protected=(args.input,))
    source = load_css_input(args.input)
    for flag, key in ((args.dimension, "dimension"), (args.prime, "p")):
        if flag is not None and flag != source[key]:
            raise ValueError("Explicit prime/dimension disagrees with input matrix metadata")
    options = {} if args.options_file is None else read_json(args.options_file)
    if not isinstance(options, dict):
        raise ValueError("Options file must contain a JSON object")
    def put(key, value):
        if key in options:
            raise ValueError("Option supplied more than once: " + key)
        options[key] = value
    for key, value in (args.options or {}).items():
        put(key, value)
    if args.max_cell_index is not None:
        put("max_cell_index", args.max_cell_index)
    if args.circuit_options is not None:
        put("circuit_options", args.circuit_options)
    basis = args.lattice if args.lattice is not None else source["lattice_basis"]
    mode = args.lattice_mode
    if mode is None:
        mode = "fixed" if args.lattice is not None else source["lattice_mode"]
    result = decouple(source["HX"], source["HZ"], source["dimension"], source["p"],
        algorithm=args.algorithm, output=args.format, lattice_mode=mode,
        lattice_basis=basis, options=options)
    path = save_result(result, args.output)
    print(json.dumps({"status": result.status, "algorithm": result.algorithm,
        "dimension": result.qca.ring.D, "prime": result.qca.ring.p,
        "physical_qudits": result.qca.qudits, "output": str(path),
        "diagnostics": result.diagnostics}, ensure_ascii=True, indent=2))
    return 3 if result.status == "circuit_incomplete" else 0


def _verify(args):
    from .result import RESULT_SCHEMA, verify_result
    raw = read_json(args.path)
    if not isinstance(raw, dict):
        raise ValueError("Result must be a JSON object")
    if raw.get("schema") != RESULT_SCHEMA:
        schema = raw.get("schema", "")
        if not isinstance(schema, str):
            raise ValueError("Result schema must be a string")
        if schema.startswith("css-decoupling-2d") or "__css_matrix_table__" in raw:
            from .two_d.serialization import load
            from .two_d.verify import verify
            from .two_d.code_circuits import verify_no_ancilla_result, SCHEMA
            data = load(args.path)
            expected = {}
            if args.input:
                from .two_d.algebra import Ring, Matrix
                source = load_css_input(args.input)
                R = Ring(source["p"], source["dimension"])
                for key in ("HX", "HZ"):
                    A = source[key]
                    expected["expected_" + key] = Matrix(R, A.nrows, A.ncols,
                        {ij: R(dict(f.terms)) for ij, f in A.entries.items()})
            actual_schema = data.get("schema", "")
            if actual_schema == SCHEMA:
                report = verify_no_ancilla_result(data, **expected)
            elif actual_schema.startswith("css-decoupling-2d/direct-decoupling/"):
                from .two_d.direct_pipeline import verify_direct_result
                report = verify_direct_result(data, **expected)
            elif actual_schema.startswith("css-decoupling-2d-direct-circuit/"):
                if expected:
                    raise ValueError("A standalone matrix factorization does not bind original CSS checks")
                from .two_d.direct_circuits import verify_direct_circuit
                report = verify_direct_circuit(data)
            else:
                report = verify(data, **expected)
            print(json.dumps(report, indent=2))
            return 0
        if args.input:
            raise ValueError("Use the general low-level verifier for historical general source binding")
        from .general_cli import main as general_main
        return general_main(["verify", str(args.path)])
    expected = {}
    if args.input:
        source = load_css_input(args.input)
        expected = {"expected_HX": source["HX"], "expected_HZ": source["HZ"]}
    report = verify_result(raw, **expected)
    print(json.dumps(asdict(report), indent=2))
    return 0 if report.valid else 1


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "general":
        from .general_cli import main as general_main
        return general_main(argv[1:])
    if argv and argv[0] in {"analyze", "coarse", "construct", "torus", "lattice-check",
                            "koszul-map", "export-outputs", "verify-outputs"}:
        from .general_cli import main as general_main
        return general_main(argv)
    parser = argparse.ArgumentParser(prog="css-decouple",
        description="Certified prime-qudit CSS decoupling; 2D defaults to the specialized algorithm.",
        epilog="Advanced general API: css-decouple general --help. Exit 3 means resource/factorization incomplete.")
    parser.add_argument("--version", action="version", version=__import__("css_decoupling").__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Compute a QCA, optionally compile a circuit")
    run.add_argument("input", type=Path)
    run.add_argument("--output", type=Path, required=True, help="New JSON result file; never overwritten")
    run.add_argument("--algorithm", choices=("auto", "two_d", "general"), default="auto")
    run.add_argument("--format", "--output-kind", choices=("qca", "circuit"), default="qca",
                     help="Requested mathematical output (the file format is JSON)")
    run.add_argument("--dimension", type=int)
    run.add_argument("--prime", type=int)
    run.add_argument("--lattice-mode", choices=("auto", "fixed"))
    run.add_argument("--lattice", "--basis", type=json.loads, help="Integer basis matrix as JSON; vectors are columns")
    run.add_argument("--max-cell-index", type=int)
    run.add_argument("--options", type=_object, help="Algorithm options as a JSON object")
    run.add_argument("--options-file", type=Path)
    run.add_argument("--circuit-options", type=_object, help="2D circuit budgets and factorization options")
    check = sub.add_parser("verify", help="Independently verify a saved result")
    check.add_argument("path", type=Path)
    check.add_argument("--input", type=Path, help="Also bind the result to these original checks")
    example = sub.add_parser("example", help="Write a small portable input")
    example.add_argument("name", choices=("toric", "glued"),
                         help="toric: TC_1; glued: the glued code of the manuscript (D=3) or its D>3 analogue")
    example.add_argument("--dimension", type=int, help="Default 2 for toric and 3 for glued")
    example.add_argument("--prime", type=int, default=2)
    example.add_argument("--output", type=Path, required=True)
    sub.add_parser("environment", help="Show the native runtime and package versions")
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            return _run(args)
        if args.command == "verify":
            return _verify(args)
        if args.command == "example":
            from .algebra.laurent import Ring
            from .models.examples import glued_css, toric_css
            glued = args.name == "glued"
            dimension = (3 if glued else 2) if args.dimension is None else args.dimension
            if dimension < (3 if glued else 2):
                raise ValueError("dimension must be >= 3 for glued and >= 2 for toric")
            ring = Ring(args.prime, dimension)
            HX, HZ = glued_css(ring) if glued else toric_css(ring, {1: 1})
            path = save_css_input(HX, HZ, args.output)
            print(json.dumps({"input": str(path), "dimension": dimension, "prime": args.prime}))
            return 0
        import platform
        import sympy
        from . import __version__
        print(json.dumps({"package": "css-decoupling", "version": __version__,
            "python": platform.python_version(), "sympy": sympy.__version__,
            "runtime": "native CPython + SymPy", "algorithms": ["two_d", "general"]}, indent=2))
        return 0
    except ResourceLimit as exc:
        print(json.dumps({"status": "resource_incomplete", "error": type(exc).__name__,
            "reason": str(exc), "qca_available": False}), file=sys.stderr)
        return 3
    except (ValueError, TypeError, OSError, ArithmeticError, BackendUnavailable) as exc:
        print(json.dumps({"status": "failed", "error": type(exc).__name__, "reason": str(exc)},
            ensure_ascii=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
