"""Portable diagnostics for direct C4 splitting; these are not certificates.

Matrices use the existing certificate representation, in content-addressed
JSON/gzip files.  Each solve-start event is persisted before the solver runs.
No Kronecker expansion, rank computation, or algebraic shortcut is performed.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
import gzip
import hashlib
import importlib
import json
import logging
from pathlib import Path
from time import perf_counter
from unittest.mock import patch
import weakref

from css_decoupling.algebra.laurent import Matrix, Ring
from css_decoupling.certificates.comparison import _matrix_dict, _matrix_from_dict
from css_decoupling.topology.complexes import CochainComplex, CochainMap


def write_json(path, data):
    """Atomic UTF-8/LF output, independent of the Windows newline default."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(path)


def matrix_stats(matrix):
    lower, upper = None, None
    monomials = radius_l1 = radius_linf = 0
    for polynomial in matrix.entries.values():
        monomials += len(polynomial.terms)
        for exponent in polynomial.terms:
            if lower is None:
                lower, upper = list(exponent), list(exponent)
            else:
                lower = [min(a, b) for a, b in zip(lower, exponent)]
                upper = [max(a, b) for a, b in zip(upper, exponent)]
            radius_l1 = max(radius_l1, sum(map(abs, exponent)))
            radius_linf = max(radius_linf, max(map(abs, exponent), default=0))
    return {"shape": list(matrix.shape), "domain_free_rank": matrix.ncols,
            "codomain_free_rank": matrix.nrows, "nonzero_entries": len(matrix.entries),
            "monomials": monomials, "support_min": lower, "support_max": upper,
            "support_radius_l1": radius_l1, "support_radius_linf": radius_linf,
            "rank_convention": "free module dimensions; Laurent matrix rank not computed"}


def object_stats(value):
    """Inspect stored support only; never multiply matrices for statistics."""
    if isinstance(value, Matrix):
        return matrix_stats(value)
    if isinstance(value, CochainComplex):
        return {"free_module_ranks": [[n, value.rank(n)] for n in value.degrees],
                "differentials": {str(n): matrix_stats(A)
                                  for n, A in sorted(value.differentials.items())}}
    if isinstance(value, CochainMap):
        return {"source_free_ranks": dict(value.source.ranks),
                "target_free_ranks": dict(value.target.ranks),
                "components": {str(n): matrix_stats(A)
                               for n, A in sorted(value.components.items())}}
    if isinstance(value, dict):
        return {str(key): object_stats(item) for key, item in value.items()
                if isinstance(item, (Matrix, CochainComplex, CochainMap, dict, tuple, list))
                or item is None or isinstance(item, (str, int, float, bool))}
    if isinstance(value, (tuple, list)):
        return [object_stats(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return {"type": type(value).__name__}


def result_stats(result):
    stats = {}
    for key in ("free_C", "free_T"):
        if hasattr(result, key):
            stats[key] = getattr(result, key)
    for key in ("complex", "restricted", "phi", "psi", "hC", "hT", "comparison"):
        if hasattr(result, key):
            stats[key] = object_stats(getattr(result, key))
    trace = getattr(result, "trace", {})
    for key in ("original_complex", "restricted", "phi", "psi", "hC", "hT", "comparison"):
        if key in trace:
            stats[key] = object_stats(trace[key])
    if "construction" in trace:
        construction = trace["construction"]
        stats["construction"] = {"timings": construction.timings,
                                 "input_size": construction.input_size,
                                 "matrices": result_stats(construction)}
    if hasattr(result, "timings"):
        stats["timings"] = result.timings
    if hasattr(result, "layers"):
        stats["layers"] = {str(n): {"dimension": layer.dimension,
                                   "representatives": matrix_stats(layer.representatives)}
                           for n, layer in result.layers.items()}
    if hasattr(result, "stabilization"):
        stable = result.stabilization
        stats["stabilization"] = {
            key: matrix_stats(getattr(stable, key))
            for key in ("phihat", "psihat", "dChat", "dThat", "U", "Uinv", "V", "Vinv")}
        stats["ancilla_disks"] = dict(vars(stable.ancillas))
    if hasattr(result, "circuit"):
        circuit = result.circuit
        sum_families = sum(hasattr(gate, "displacement") for gate in circuit.gates)
        displacements = {gate.displacement for gate in circuit.gates
                         if hasattr(gate, "displacement")}
        stats["circuit"] = {
            "qudits": circuit.qudits, "gate_families": len(circuit.gates),
            "sum_families": sum_families,
            "swap_families": len(circuit.gates) - sum_families,
            "max_displacement_l1": max((sum(map(abs, x)) for x in displacements), default=0),
            "max_displacement_linf": max((max(map(abs, x), default=0)
                                           for x in displacements), default=0),
            "scope": "translation gate families and distances in retained-cell coordinates"}
        if hasattr(result, "lattice"):
            basis = result.lattice.basis
            original = {tuple(sum(basis[i][j] * x[j] for j in range(circuit.ring.D))
                              for i in range(circuit.ring.D)) for x in displacements}
            stats["circuit"].update(
                original_lattice_max_displacement_l1=max((sum(map(abs, x)) for x in original), default=0),
                original_lattice_max_displacement_linf=max((max(map(abs, x), default=0)
                                                            for x in original), default=0))
    return stats


class EvidenceRecorder(logging.Handler):
    def __init__(self, directory):
        super().__init__(level=logging.INFO)
        self.directory = Path(directory) / "diagnostics"
        self.directory.mkdir()
        (self.directory / "objects").mkdir()
        self.started = perf_counter()
        self.sequence = 0
        self.matrix_cache = {}
        self.last_event = None
        self.last_equations = None
        self.io_seconds = 0.0
        self.structural_verifications = 0
        self.handoff_verifications = []

    def timeline(self, event, **values):
        row = {"event": event, "elapsed_seconds": perf_counter() - self.started,
               "utc": datetime.now(timezone.utc).isoformat(), **values}
        with (self.directory / "stages.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        write_json(self.directory / "progress.json", row)

    def emit(self, record):
        if record.name.startswith("css_decoupling"):
            self.timeline("log", logger=record.name, message=record.getMessage())

    def _matrix(self, value):
        # Poly entries are immutable; a shallow snapshot detects Matrix edits
        # without copying every monomial. Weak references release old right
        # sides and remove their cache entries before object ids are reused.
        cached = self.matrix_cache.get(id(value))
        if cached and cached[0]() is value and cached[1] == value.entries \
                and cached[2] == value.shape and cached[3] == value.ring:
            return cached[4]
        document = {"schema": "css-decoupling.diagnostic-matrix", "version": 1,
                    "ring": {"prime": value.ring.p, "dimension": value.ring.D},
                    "matrix": _matrix_dict(value)}
        digest = hashlib.sha256()
        temporary = self.directory / "objects" / "writing.json.gz.tmp"
        with temporary.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=1) as stream:
                encoder = json.JSONEncoder(sort_keys=True, separators=(",", ":"))
                parts, size = [], 0
                for part in encoder.iterencode(document):
                    data = part.encode("utf-8")
                    parts.append(data)
                    size += len(data)
                    if size >= 65536:
                        data = b"".join(parts)
                        digest.update(data)
                        stream.write(data)
                        parts, size = [], 0
                data = b"".join(parts) + b"\n"
                digest.update(data)
                stream.write(data)
        canonical_hash = digest.hexdigest()
        path = self.directory / "objects" / (canonical_hash + ".json.gz")
        if path.exists():
            temporary.unlink()
        else:
            temporary.replace(path)
        ref = {"kind": "matrix", "file": str(path.relative_to(self.directory)).replace("\\", "/"),
               "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
               "canonical_json_sha256": canonical_hash, "statistics": matrix_stats(value)}
        identity = id(value)
        reference = weakref.ref(value, lambda _value, key=identity: self.matrix_cache.pop(key, None))
        self.matrix_cache[identity] = (reference, value.entries.copy(), value.shape, value.ring, ref)
        return ref

    def encode(self, value):
        if isinstance(value, Matrix):
            return self._matrix(value)
        if isinstance(value, CochainComplex):
            return {"kind": "complex", "ring": {"prime": value.ring.p, "dimension": value.ring.D},
                    "ranks": [[n, value.rank(n)] for n in value.degrees],
                    "differentials": [[n, self._matrix(value.d(n))] for n in value.degrees]}
        if isinstance(value, dict):
            # Pair lists preserve signed integer degree keys in JSON.
            return {"kind": "mapping", "items": [[key, self.encode(item)] for key, item in value.items()]}
        if isinstance(value, (tuple, list)):
            return [self.encode(item) for item in value]
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        raise TypeError("Unsupported diagnostic value: " + type(value).__name__)

    def __call__(self, event):
        tick = perf_counter()
        self.sequence += 1
        self.timeline("checkpoint-writing", name=event.get("name"), degree=event.get("degree"),
                      solver_event=event.get("event"), sequence=self.sequence)
        document = {"schema": "css-decoupling.direct-splitting-diagnostic", "version": 1,
                    "scope": "intermediate equations, not a successful contraction or circuit certificate",
                    "equations": ["dprev * X = R", "X * hnext = 0", "X * s = 0", "q * X = 0"],
                    "conventions": {"X": "h^n: C^n -> C^(n-1)",
                                    "R": "I - P^n - h^(n+1) d^n", "P": "s q in degree zero; zero otherwise",
                                    "coefficients": "Laurent polynomials over the recorded prime field",
                                    "unknown_layout": "matrix X; the four factored equations are linear in X"},
                    "event": self.encode(event)}
        path = self.directory / f"equations-{self.sequence:05d}.json"
        if path.exists():
            raise FileExistsError("Diagnostic event already exists: " + str(path))
        write_json(path, document)
        if event.get("structural_certificate") is not None:
            # Restore the persisted bytes, then bind to the original complex
            # and recurrence. This is a section check, not a circuit result.
            from css_decoupling.algebra.laurent import eye, zeros
            from css_decoupling.certificates.structural_c4 import verify_structural_section
            restored = load_snapshot(path)
            C, n = restored["C"], restored["degree"]
            section, projection = restored["section"], restored["projection"]
            P = section*projection if n == 0 else zeros(C.ring, C.rank(n), C.rank(n))
            previous_P = section*projection if n == 1 else zeros(C.ring, C.rank(n-1), C.rank(n-1))
            following = restored["completed_h"].get(n+1, zeros(C.ring, C.rank(n), C.rank(n+1)))
            expected = dict(d=C.d(n-1), R=eye(C.ring, C.rank(n))-P-following*C.d(n),
                            hnext=following, s=restored["s"], q=restored["q"], Pprev=previous_P)
            report = verify_structural_section(restored["structural_certificate"], expected=expected)
            if not report.valid:
                raise ArithmeticError("Persisted structural section rejected: "+"; ".join(report.errors))
            self.structural_verifications += 1
            self.timeline("serialized-section-verified", name=event.get("name"), degree=n, file=path.name)
        self.last_event = path.name
        if event.get("event") == "solve-start":
            self.last_equations = path.name
        self.io_seconds += perf_counter() - tick
        self.timeline("checkpoint-saved", name=event.get("name"), degree=event.get("degree"),
                      solver_event=event.get("event"), file=path.name,
                      strategy=event.get("strategy"), solver_seconds=event.get("seconds"),
                      diagnostic_seconds=self.io_seconds)
        write_json(self.directory / "index.json", self.summary())

    def summary(self):
        return {"events": self.sequence, "last_event": self.last_event,
                "last_solve_start": self.last_equations, "diagnostic_seconds": self.io_seconds,
                "serialized_structural_sections_verified": self.structural_verifications,
                "serialized_handoffs_verified": self.handoff_verifications,
                "scope": "diagnostics only; completion requires independent final certificates",
                "timing_scope": "wall time includes support statistics, hashing and diagnostic persistence"}

    @contextmanager
    def stage(self, name):
        tick = perf_counter()
        self.timeline("stage-start", name=name)
        try:
            yield
        except BaseException as exc:
            self.timeline("stage-error", name=name, seconds=perf_counter() - tick,
                          error_type=type(exc).__name__, error=str(exc))
            raise
        else:
            self.timeline("stage-complete", name=name, seconds=perf_counter() - tick)

    def record_result(self, name, result):
        path = self.directory / (name + "-metrics.json")
        write_json(path, result_stats(result))
        self.timeline("metrics-saved", name=name, file=path.name)

    @contextmanager
    def observe(self):
        """Observe existing stage boundaries without changing their algebra."""
        from css_decoupling.topology.direct_splitting import observe_direct_splitting
        bindings = (
            ("css_decoupling.decoupling.pipeline", "analyze_profile", "native-analysis"),
            ("css_decoupling.decoupling.pipeline", "image_resolution", "complete-negative-resolution"),
            ("css_decoupling.decoupling.splitting", "restrict_complex", "restrict-complex"),
            ("css_decoupling.decoupling.construction", "splitting_map", "toric-comparison"),
            ("css_decoupling.topology.cohomology_handoff", "build_cohomology_handoff", "cohomology-handoff"),
            ("css_decoupling.decoupling.comparison", "paper_stable_cokernel_isomorphism", "c4-schanuel"),
            ("css_decoupling.decoupling.construction", "stabilize", "sum-swap-compilation"),
        )
        root_logger = logging.getLogger()
        root_logger.addHandler(self)
        try:
            with ExitStack() as stack:
                stack.enter_context(observe_direct_splitting(self))
                for module_name, attribute, stage_name in bindings:
                    module = importlib.import_module(module_name)
                    original = getattr(module, attribute)

                    def observed(*args, _original=original, _name=stage_name, **kwargs):
                        with self.stage(_name):
                            value = _original(*args, **kwargs)
                            statistics = (object_stats(value) if isinstance(value, (dict, list, tuple,
                                          Matrix, CochainComplex, CochainMap)) else result_stats(value))
                            # Repeated restriction calls must not replace earlier evidence.
                            sequence = sum(1 for _ in self.directory.glob(_name + "-*.json")) + 1
                            path = self.directory / f"{_name}-{sequence:03d}.json"
                            write_json(path, statistics)
                            if _name == "cohomology-handoff":
                                from css_decoupling.certificates.cohomology_handoff import verify_cohomology_handoff
                                # object_stats retains these already-JSON matrices,
                                # but explicitly persist the mathematical payload.
                                write_json(path, value)
                                report = verify_cohomology_handoff(json.loads(path.read_text(encoding="utf-8")))
                                if not report.valid:
                                    raise ArithmeticError("Persisted handoff rejected: "+"; ".join(report.errors))
                                self.handoff_verifications.append(dict(file=path.name,
                                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(), checks=list(report.checks)))
                                write_json(self.directory / "index.json", self.summary())
                            self.timeline("metrics-saved", name=_name, file=path.name)
                            return value

                    stack.enter_context(patch.object(module, attribute, observed))
                yield self
        finally:
            root_logger.removeHandler(self)


def load_snapshot(path):
    """Restore a diagnostic event and verify all referenced artifact hashes."""
    path = Path(path).resolve()
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != "css-decoupling.direct-splitting-diagnostic" or document.get("version") != 1:
        raise ValueError("Unsupported diagnostic snapshot")

    def decode(value):
        if isinstance(value, list):
            return [decode(item) for item in value]
        if not isinstance(value, dict):
            return value
        if value.get("kind") == "mapping":
            return {key: decode(item) for key, item in value["items"]}
        if value.get("kind") == "matrix":
            artifact = (path.parent / value["file"]).resolve()
            if not artifact.is_relative_to(path.parent):
                raise ValueError("Diagnostic reference escapes its directory")
            data = artifact.read_bytes()
            if hashlib.sha256(data).hexdigest() != value["sha256"]:
                raise ValueError("Diagnostic compressed matrix hash mismatch")
            decoded = gzip.decompress(data)
            if hashlib.sha256(decoded).hexdigest() != value["canonical_json_sha256"]:
                raise ValueError("Diagnostic matrix JSON hash mismatch")
            matrix = json.loads(decoded)
            ring = Ring(matrix["ring"]["prime"], matrix["ring"]["dimension"])
            return _matrix_from_dict(matrix["matrix"], ring, "diagnostic matrix")
        if value.get("kind") == "complex":
            ring = Ring(value["ring"]["prime"], value["ring"]["dimension"])
            return CochainComplex(ring, dict(value["ranks"]),
                                  {n: decode(A) for n, A in value["differentials"]})
        raise ValueError("Unknown diagnostic object")

    return decode(document["event"])


def manuscript_hashes(project_root):
    root = Path(project_root).resolve().parent
    manuscript = root / "drafts"
    return {str(path.relative_to(root)).replace("\\", "/"):
            hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(manuscript.rglob("*")) if path.is_file()}
