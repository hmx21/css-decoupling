"""Raw-check examples in isolated CPython/SymPy workers, with honest time limits.

The parent launches only the current Python executable. Each algebra worker
forbids subprocesses and executable discovery. A timeout never means that a
code cannot be decoupled. Existing evidence directories are never overwritten.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import faulthandler
import gzip
import hashlib
import importlib.util
import json
import logging
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter
from unittest.mock import patch

import sympy
import css_decoupling

from css_decoupling import (Ring, Matrix, coarse_grain, decouple_general as decouple, check_lattice,
                           verify_coarse_graining, verify_decoupling,
                           verify_lattice_criterion, ResourceLimit, InfiniteLayer,
                           NoLaurentSolution, BackendUnavailable)
from css_decoupling.certificates.io import input_dict
from css_decoupling.certificates.analysis import verify_analysis
from css_decoupling.models.examples import (toric_css, glued_css, glued_toric_chain,
                                           css_checks, shear_css)

# Load by path as well as when invoked as a script; the test suite imports this
# runner with spec_from_file_location, without adding benchmarks to sys.path.
_evidence_spec = importlib.util.spec_from_file_location(
    "direct_splitting_evidence", Path(__file__).with_name("direct_splitting_evidence.py"))
_evidence = importlib.util.module_from_spec(_evidence_spec)
_evidence_spec.loader.exec_module(_evidence)


CASES = {
    "toric-d4-p3-degree2": "Four-dimensional two-form toric control over GF(3)",
    "toric-d5-p5-sheared": "Five-dimensional two-form toric code with a Laurent shear",
    "color-d3-bcc": "Three-dimensional Freudenthal/BCC color code from simplex incidence",
    "color-d3-fcc": "Three-dimensional FCC/chamfered-cubic color-code checks",
    "color-d4-degree1": "Four-dimensional color code, qubits on 4-simplices, X on vertices",
    "color-d4-degree2": "Four-dimensional self-dual color code, both checks on edge stars",
    "tt72": "Published TT72 polynomial input; finite period is not an infinite-lattice assumption",
    "tt180": "Published TT180 polynomial input",
    "tt432": "Published TT432 polynomial input",
    "glued-d3-p3": "Nonzero adjacent-layer gluing over GF(3)",
    "glued-d4-ext3-p3": "Nonadjacent layers 1 and 3 with an Ext^3 attachment over GF(3)",
    "glued-d4-rank4-p2": "One pair of layers coupled by epsilon_12 + epsilon_34",
    "glued-d4-rank4-p3": "Rank-four two-form attachment over GF(3)",
    "glued-d4-three-p2": "Three layers with successive nonzero attachments over GF(2)",
    "glued-d4-three-p3": "Three layers with successive nonzero attachments over GF(3)",
    "glued-d5-p5": "Five-dimensional adjacent layers over GF(5)",
    "haah": "Haah cubic-code infinite-layer obstruction",
    "x-cube": "X-cube infinite-layer obstruction",
}

COMPRESS_CERTIFICATE_BYTES = 5 * 1024 * 1024


def build_case(name):
    """Only build ordinary matrices; no constructor receives a solution witness."""
    if name == "toric-d4-p3-degree2":
        return toric_css(Ring(3, 4), {2: 1})
    if name == "toric-d5-p5-sheared":
        R = Ring(5, 5)
        HX, HZ = toric_css(R, {2: 1})
        return shear_css(HX, HZ, 2, 6, R.variable(4)**-2 + 3 * R.variable(0))
    if name.startswith("color-"):
        from css_decoupling.models.color_codes import freudenthal_color_css, fcc_color_css
        if name == "color-d3-fcc":
            return fcc_color_css()
        return freudenthal_color_css(3 if name == "color-d3-bcc" else 4,
                                     2 if name.endswith("degree2") else 1)
    if name.startswith("tt"):
        from css_decoupling.models.tricycle import tricycle_case
        return tricycle_case(name)
    if name == "glued-d3-p3":
        return glued_css(Ring(3, 3))
    if name == "glued-d4-ext3-p3":
        return glued_css(Ring(3, 4), 1, 3)
    if name in ("glued-d4-rank4-p2", "glued-d4-rank4-p3"):
        from css_decoupling.models.stress import rank_four_glued_css
        return rank_four_glued_css(Ring(int(name[-1]), 4))
    if name in ("glued-d4-three-p2", "glued-d4-three-p3"):
        return css_checks(glued_toric_chain(Ring(int(name[-1]), 4), (1, 2, 3)))
    if name == "glued-d5-p5":
        return glued_css(Ring(5, 5))
    R = Ring(2, 3)
    x, y, z = R.variables
    # Vijay--Haah--Fu, arXiv:1603.04442v1, Appendix A.2.
    if name == "haah":
        f, g = 1 + x*y + y*z + x*z, 1 + x + y + z
        return Matrix.from_rows(R, [[f, g]]), Matrix.from_rows(R, [[g], [f]]).dagger()
    if name == "x-cube":
        HX = Matrix.from_rows(R, [[(1+x)*(1+y), (1+y)*(1+z), (1+x)*(1+z)]])
        return HX, Matrix.from_rows(R, [[1+z, 0], [1+x, 1+x], [0, 1+y]]).dagger()
    raise ValueError("Unknown example: " + name)


@contextmanager
def native_only():
    """Fail loudly, including attempts caught inside the algorithm."""
    observed = {"subprocess_attempts": 0, "executable_discovery_attempts": 0}

    def process(*args, **kwargs):
        observed["subprocess_attempts"] += 1
        raise RuntimeError("The native example attempted to launch an external program")

    def discovery(*args, **kwargs):
        observed["executable_discovery_attempts"] += 1
        raise RuntimeError("The native example attempted executable discovery")

    with patch("subprocess.Popen", process), patch("os.system", process), \
            patch("os.popen", process), patch("shutil.which", discovery):
        yield observed
    if any(observed.values()):
        raise RuntimeError("An external-program attempt was suppressed by the computation")


def write_json(path, data):
    _evidence.write_json(path, data)


def checked_certificate(result, verifier, path):
    # A JSON round trip checks the deliverable, not just a cached Python object.
    tick = perf_counter()
    encoded = (json.dumps(result.certificate(), sort_keys=True, separators=(",", ":")) + "\n").encode()
    data = json.loads(encoded)
    encoding_seconds = perf_counter() - tick
    tick = perf_counter()
    report = verifier(data)
    verification_seconds = perf_counter() - tick
    if not report.valid:
        raise ArithmeticError("Certificate rejected: " + "; ".join(report.errors))
    path = Path(path)
    tick = perf_counter()
    compressed = len(encoded) > COMPRESS_CERTIFICATE_BYTES
    if compressed:
        path = path.with_suffix(path.suffix + ".gz")
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as stream:
                stream.write(encoded)
        temporary.replace(path)
        # Verify the persisted representation, including its decompression,
        # equals the exact JSON bytes passed to the independent checker.
        if gzip.decompress(path.read_bytes()) != encoded:
            raise ArithmeticError("Compressed certificate changed during persistence")
    else:
        write_json(path, data)
    return {"file": path.name, "scope": report.scope, "checks": list(report.checks),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "encoding": "gzip" if compressed else "utf-8",
            "canonical_json_sha256": hashlib.sha256(encoded).hexdigest(),
            "timings": {"json_round_trip": encoding_seconds,
                        "independent_verification": verification_seconds,
                        "persistence": perf_counter() - tick}}


def worker(name, phase, directory, max_cell_index, timeout, construction_certificate=False):
    directory = Path(directory)
    # The parent preopens worker.log. Direct/hidden worker invocations must
    # obey the same immutable-evidence rule as the public parent command.
    if directory.exists() and any(p.name != "worker.log" or not p.is_file()
                                  for p in directory.iterdir()):
        raise FileExistsError("Choose a fresh worker directory; evidence will not be overwritten")
    directory.mkdir(parents=True, exist_ok=True)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", newline="\n")
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    faulthandler.enable()
    faulthandler.dump_traceback_later(max(1, timeout * 0.8), repeat=False)
    started = perf_counter()
    row = {"case": name, "phase": phase, "description": CASES[name],
           "backend": "sympy", "resolution_method": "native"}
    observed = {}
    recorder = _evidence.EvidenceRecorder(directory)
    try:
        with native_only() as observed, recorder.observe():
            recorder.timeline("stage-start", name="raw-check-input")
            HX, HZ = build_case(name)
            write_json(directory / "input.json", input_dict(HX, HZ))
            row["input"] = {"dimension": HX.ring.D, "prime": HX.ring.p,
                            "HX": list(HX.shape), "HZ": list(HZ.shape),
                            "monomials": sum(len(f.terms) for A in (HX, HZ)
                                             for f in A.entries.values()),
                            "matrix_statistics": {"HX": _evidence.matrix_stats(HX),
                                                  "HZ": _evidence.matrix_stats(HZ)},
                            "sha256": hashlib.sha256((directory / "input.json").read_bytes()).hexdigest()}
            recorder.timeline("stage-complete", name="raw-check-input",
                              seconds=perf_counter() - started, statistics=row["input"])
            options = dict(backend="sympy", resolution_method="native",
                           resolution_timeout=timeout, max_cell_index=max_cell_index)
            if phase == "coarse":
                result = coarse_grain(HX, HZ, **options)
                row.update(status="verified" if result.decouplable else "certified-obstruction",
                           decouplable=result.decouplable, reason=result.reason,
                           multiplicities=result.multiplicities,
                           index=result.lattice.index if result.lattice else None,
                           translation_index=result.translation_kernel.index if result.translation_kernel else None,
                           timings=result.timings)
                verifier = verify_coarse_graining
            elif phase == "criterion":
                identity = [[int(i == j) for j in range(HX.ring.D)] for i in range(HX.ring.D)]
                result = check_lattice(HX, HZ, identity, **options)
                row.update(status="verified" if result.decouplable else "certified-obstruction",
                           decouplable=result.decouplable, reason=result.reason,
                           multiplicities=result.multiplicities, index=result.lattice.index,
                           timings=result.timings)
                verifier = verify_lattice_criterion
            elif phase == "decouple":
                result = decouple(HX, HZ, comparison_method="schanuel", **options)
                row.update(status="verified", multiplicities=result.multiplicities,
                           index=result.lattice.index,
                           translation_index=result.profile_analysis.translation_kernel.index,
                           physical_qudits=result.circuit.qudits,
                           input_qudits_per_retained_cell=HX.ncols * result.lattice.index,
                           ancillas=result.circuit.qudits - HX.ncols * result.lattice.index,
                           gate_families=len(result.circuit.gates), timings=result.timings)
                verifier = verify_decoupling
                recorder.record_result("decoupling", result)
                row["stage_statistics"] = "diagnostics/decoupling-metrics.json"
                with recorder.stage("serialized-analysis-verification"):
                    row["analysis_certificate"] = checked_certificate(
                        result.profile_analysis, lambda value: verify_analysis(value), directory / "analysis.json")
            else:
                raise ValueError("Unknown phase: " + phase)
            with recorder.stage("serialized-final-verification"):
                row["certificate"] = checked_certificate(result, verifier, directory / "certificate.json")
            if phase == "decouple" and construction_certificate:
                from css_decoupling.algebra.modules import GeneralBackend
                from css_decoupling.lattice.restriction_backend import RestrictionBackend
                from css_decoupling.certificates.construction import verify_construction
                selected = RestrictionBackend(GeneralBackend(HX.ring), result.lattice)
                logging.info("independently verifying the serialized Algorithm 2 source and Schanuel certificate")
                with recorder.stage("serialized-construction-verification"):
                    row["construction_certificate"] = checked_certificate(
                        result.trace["construction"],
                        lambda value: verify_construction(value, linear_backend=selected),
                        directory / "construction.json")
    except ResourceLimit as exc:
        row.update(status="incomplete", reason="resource-limit", error=str(exc))
        if hasattr(exc, "coarse_graining_partial"):
            write_json(directory / "partial.json", exc.coarse_graining_partial)
            row["partial"] = "partial.json"
    except InfiniteLayer as exc:
        # An exception alone is not an independently verified negative certificate.
        row.update(status="obstruction-reported", reason="infinite-layer",
                   error=str(exc), certificate=None)
    except NoLaurentSolution as exc:
        row.update(status="algebraic-nonmembership", error=str(exc), certificate=None)
    except BackendUnavailable as exc:
        row.update(status="incomplete", reason="backend-unavailable", error=str(exc))
    except Exception as exc:
        import traceback
        traceback.print_exc()
        row.update(status="error", error_type=type(exc).__name__, error=str(exc))
    finally:
        faulthandler.cancel_dump_traceback_later()
    row.update(seconds=perf_counter() - started, native_guard=observed,
               diagnostics=recorder.summary())
    if any(observed.values()):
        row.update(status="error", error="External program attempted")
    write_json(directory / "result.json", row)
    print(json.dumps({k: row[k] for k in ("case", "phase", "status", "seconds")}), flush=True)


def source_hashes():
    root = Path(__file__).resolve().parents[2]
    if Path(css_decoupling.__file__).resolve().parent != root / "src" / "css_decoupling":
        raise RuntimeError("This source benchmark requires the current checkout; install it with pip -e .")
    files = [*root.joinpath("src").rglob("*.py"), Path(__file__).resolve(),
             Path(__file__).with_name("direct_splitting_evidence.py").resolve(), root / "pyproject.toml"]
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}


def run(output, names, phases, timeout, max_cell_index, construction_certificate=False):
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose a fresh output directory; evidence will not be overwritten")
    output.mkdir(parents=True, exist_ok=True)
    before = source_hashes()
    manuscript_before = _evidence.manuscript_hashes(Path(__file__).resolve().parents[2])
    summary = {"schema": "css-decoupling.native-examples", "version": 1,
               "started_utc": datetime.now(timezone.utc).isoformat(),
               "environment": {"python": sys.version, "sympy": sympy.__version__,
                               "platform": platform.platform(),
                               "package_file": str(Path(css_decoupling.__file__).resolve())},
               "limits": {"seconds_per_phase": timeout, "max_cell_index": max_cell_index},
               "construction_certificate": construction_certificate,
               "source_hashes": before, "manuscript_hashes": manuscript_before, "cases": [],
               "scope": "Finite raw-check examples, not a general proof or runtime guarantee. "
                        "Coarse/criterion success does not certify a constructed circuit; "
                        "timeouts and resource limits are incomplete computations."}
    write_json(output / "summary.json", summary)
    for name in names:
        for phase in phases:
            directory = output / name / phase
            directory.mkdir(parents=True)
            command = [sys.executable, str(Path(__file__).resolve()), "--worker", name,
                       "--phase", phase, "--output-dir", str(directory),
                       "--timeout", str(timeout), "--max-cell-index", str(max_cell_index)]
            if construction_certificate:
                command.append("--construction-certificate")
            started = perf_counter()
            with (directory / "worker.log").open("w", encoding="utf-8", newline="\n") as log:
                try:
                    child = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
                except subprocess.TimeoutExpired:
                    # subprocess.run kills and waits for this Python worker.
                    row = {"case": name, "phase": phase, "status": "incomplete",
                           "reason": "wall-time-limit", "seconds": perf_counter() - started}
                else:
                    result = directory / "result.json"
                    row = json.loads(result.read_text(encoding="utf-8")) if child.returncode == 0 and result.exists() else {
                        "case": name, "phase": phase, "status": "error",
                        "reason": "worker-exit", "returncode": child.returncode}
            for key, filename in (("diagnostics", "index.json"), ("last_stage", "progress.json")):
                diagnostic = directory / "diagnostics" / filename
                if diagnostic.exists():
                    row[key] = json.loads(diagnostic.read_text(encoding="utf-8"))
            row["command"] = command
            row["directory"] = str(directory.relative_to(output))
            summary["cases"].append(row)
            write_json(output / "summary.json", summary)
            print(f"{name}/{phase}: {row['status']} ({row.get('seconds', 0):.2f}s)", flush=True)
    summary["source_hashes_after"] = source_hashes()
    summary["sources_unchanged"] = before == summary["source_hashes_after"]
    summary["manuscript_hashes_after"] = _evidence.manuscript_hashes(Path(__file__).resolve().parents[2])
    summary["manuscript_unchanged"] = manuscript_before == summary["manuscript_hashes_after"]
    summary["completed_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(output / "summary.json", summary)
    write_json(output / "manifest.json", {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
                                         for p in sorted(output.rglob("*")) if p.is_file()})
    if not summary["sources_unchanged"]:
        raise RuntimeError("Sources changed during execution; these are development observations")
    if not summary["manuscript_unchanged"]:
        raise RuntimeError("Manuscript files changed during execution; inspect the recorded hashes")
    if any(row["status"] == "error" for row in summary["cases"]):
        raise RuntimeError("An example failed unexpectedly; see summary.json")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--case", action="append", choices=CASES)
    parser.add_argument("--phase", action="append", choices=("coarse", "criterion", "decouple"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--max-cell-index", type=int, default=128)
    parser.add_argument("--worker", choices=CASES, help=argparse.SUPPRESS)
    parser.add_argument("--construction-certificate", action="store_true",
                        help="Also independently verify and save the complete Algorithm 2 witness")
    args = parser.parse_args()
    if args.list:
        print(json.dumps(CASES, indent=2))
    else:
        if args.output_dir is None or args.timeout <= 0 or args.max_cell_index < 1:
            parser.error("Supply --output-dir and positive limits")
        if args.worker:
            if not args.phase or len(args.phase) != 1:
                parser.error("A worker needs exactly one phase")
            worker(args.worker, args.phase[0], args.output_dir, args.max_cell_index, args.timeout,
                   args.construction_certificate)
        else:
            run(args.output_dir, args.case or list(CASES), args.phase or ["coarse", "decouple"],
            args.timeout, args.max_cell_index, args.construction_certificate)
