"""From arbitrary topological CSS checks to a certified decoupling circuit."""
from dataclasses import dataclass
from functools import partial
import logging
from time import perf_counter

from css_decoupling.algebra.laurent import Matrix
from css_decoupling.certificates.comparison import certificate_dict, _matrix_dict
from css_decoupling.certificates.decoupling import verify_decoupling
from css_decoupling.certificates.formats import COMPLETE_SCHEMA as SCHEMA
from css_decoupling.decoupling.comparison import stable_cokernel_isomorphism
from css_decoupling.topology.complexes import homotopy_inverse_window
from css_decoupling.lattice.retained import Lattice
from css_decoupling.algebra.modules import solve
from css_decoupling.errors import InfiniteLayer
from css_decoupling.topology.resolutions import image_resolution
from css_decoupling.decoupling.analysis import analyze_profile
from css_decoupling.decoupling.stabilization import LiftData, Stabilization, stabilize

log = logging.getLogger(__name__)


@dataclass
class DecouplingResult:
    HX: Matrix
    HZ: Matrix
    lattice: Lattice
    multiplicities: dict[int, int]
    free_C: int
    free_T: int
    stabilization: Stabilization
    timings: dict[str, float]
    trace: dict[str, object]

    @property
    def circuit(self):
        return self.stabilization.circuit

    @property
    def qca(self):
        """Physical Pauli automorphism on the same stabilized register as circuit."""
        from css_decoupling.circuits.qca import QCA
        return QCA(self.stabilization.U.copy(), self.stabilization.Uinv.copy())

    @property
    def profile_analysis(self):
        if "profile_analysis" not in self.trace and self.coarse_graining_result is not None:
            raise AttributeError("Uniform coarse graining has an Algorithm 1 result; use coarse_graining_result")
        return self.trace["profile_analysis"]

    @property
    def coarse_graining_result(self):
        """Algorithm 1 source evidence for uniform runs; None for adaptive runs."""
        return self.trace.get("coarse_graining")

    def certificate(self):
        return {
            "schema": SCHEMA, "version": 1,
            "input": {"HX": _matrix_dict(self.HX), "HZ": _matrix_dict(self.HZ)},
            "lattice": self.lattice.metadata(),
            "multiplicities": [[j, t] for j, t in sorted(self.multiplicities.items())],
            "identity_checks": {"source": self.free_C, "target": self.free_T},
            "stabilization": certificate_dict(self.stabilization),
        }


def decouple(HX, HZ, *, lattice_basis=None, max_cell_index=4096,
             max_layer_dimension=10000, max_unknowns=250000,
             resolution_timeout=120, resolution_method="native", max_dlog_steps=100000,
             verify=True, backend=None, sector="X", qs_options=None,
             lattice_strategy="auto", max_exterior_entries=2000000,
             comparison_method="schanuel", linear_backend=None, bounded_options=None,
             contraction_strategy="auto", structural_search=False,
             coarse_graining="adaptive"):
    """Construct a circuit, without requiring resolutions or comparison maps.

    Compute exact Lambda_0, then select killing lattices from successive kappa
    classes (optimal in three dimensions under the paper's hypotheses). A supplied
    lattice is used exactly. The current paper's default comparison retains
    Schanuel stable directions; ``comparison_method='free-basis'`` opts into
    the older actual-basis/QS route. Only the bounded linear backend follows
    the coefficient-linearization complexity argument for Algorithm 2.
    Native auto contraction uses complete normalization/transfer; strong
    sections (``sections``) remain explicit, with finite-support probes disabled
    unless structural_search=True. This choice does not change the lattice.
    coarse_graining='uniform' composes the paper's sufficient Algorithm 1
    with Algorithm 2 and exposes its source as coarse_graining_result. It
    does not compute an adaptive profile analysis first.
    """
    if sector not in ("X", "Z", "auto"):
        raise ValueError("sector must be X, Z or auto")
    if comparison_method not in ("schanuel", "free-basis"):
        raise ValueError("comparison_method must be schanuel or free-basis")
    if coarse_graining not in ("adaptive", "uniform"):
        raise ValueError("coarse_graining must be adaptive or uniform")
    if coarse_graining == "uniform" and (lattice_basis is not None or lattice_strategy != "auto"
                                         or comparison_method != "schanuel"):
        raise ValueError("Uniform coarse graining chooses its own sufficient lattice and requires Schanuel comparison")
    if qs_options is not None and comparison_method != "free-basis":
        raise ValueError("qs_options require comparison_method='free-basis'")
    if comparison_method == "free-basis" and (linear_backend == "bounded" or bounded_options is not None):
        raise ValueError("The bounded Algorithm 2 backend requires comparison_method='schanuel'")
    if linear_backend is None:
        # The Laurent module backend is native, including context-selected
        # native adapters. Algorithm 2 separately offers coefficient solving.
        linear_backend = backend if isinstance(backend, str) else "sympy"
    from .construction import _construction_policy
    _construction_policy(linear_backend, contraction_strategy, structural_search)
    if comparison_method != "schanuel" and (contraction_strategy != "auto" or structural_search):
        raise ValueError("Contraction policy options require comparison_method='schanuel'")
    options = dict(lattice_basis=lattice_basis, max_cell_index=max_cell_index,
                   max_layer_dimension=max_layer_dimension, max_unknowns=max_unknowns,
                   resolution_timeout=resolution_timeout, resolution_method=resolution_method,
                   max_dlog_steps=max_dlog_steps, verify=verify, backend=backend, qs_options=qs_options,
                   lattice_strategy=lattice_strategy, max_exterior_entries=max_exterior_entries,
                   comparison_method=comparison_method, linear_backend=linear_backend,
                   bounded_options=bounded_options, contraction_strategy=contraction_strategy,
                   structural_search=structural_search, coarse_graining=coarse_graining)
    if sector == "Z":
        from css_decoupling.decoupling.dual import dual_decouple
        return dual_decouple(HX, HZ, **options)
    if sector == "auto":
        if coarse_graining == "uniform":
            return decouple(HX, HZ, sector="X", **options)
        try:
            return decouple(HX, HZ, sector="X", **options)
        except InfiniteLayer:
            return decouple(HX, HZ, sector="Z", **options)
    if backend is not None:
        from css_decoupling.algebra.modules import using_backend
        with using_backend(backend, timeout=resolution_timeout):
            return decouple(HX, HZ, lattice_basis=lattice_basis, max_cell_index=max_cell_index,
                            max_layer_dimension=max_layer_dimension, max_unknowns=max_unknowns,
                            resolution_timeout=resolution_timeout, resolution_method=resolution_method,
                            max_dlog_steps=max_dlog_steps, verify=verify, qs_options=qs_options,
                            lattice_strategy=lattice_strategy, max_exterior_entries=max_exterior_entries,
                            comparison_method=comparison_method, linear_backend=linear_backend,
                            bounded_options=bounded_options, contraction_strategy=contraction_strategy,
                            structural_search=structural_search, coarse_graining=coarse_graining)
    if coarse_graining == "uniform":
        return _decouple_uniform(HX, HZ, max_cell_index=max_cell_index,
            max_layer_dimension=max_layer_dimension, max_unknowns=max_unknowns,
            resolution_timeout=resolution_timeout, resolution_method=resolution_method,
            max_dlog_steps=max_dlog_steps, verify=verify, linear_backend=linear_backend,
            bounded_options=bounded_options, contraction_strategy=contraction_strategy,
            structural_search=structural_search)
    started = perf_counter()
    log.info("analyzing the original CSS profile (HX=%s, HZ=%s)", HX.shape, HZ.shape)
    analysis = analyze_profile(
        HX, HZ, lattice_basis=lattice_basis, max_cell_index=max_cell_index,
        max_layer_dimension=max_layer_dimension, max_unknowns=max_unknowns,
        resolution_timeout=resolution_timeout, resolution_method=resolution_method,
        max_dlog_steps=max_dlog_steps, lattice_strategy=lattice_strategy,
        max_exterior_entries=max_exterior_entries)
    C, layers = analysis.complex, analysis.layers
    lattice, restricted, phi = analysis.lattice, analysis.restricted, analysis.phi
    multiplicities = analysis.multiplicities
    timings = dict(analysis.timings)
    if comparison_method == "schanuel":
        from css_decoupling.algebra.modules import kernel
        from css_decoupling.topology.complexes import CochainComplex
        from .splitting import restrict_complex
        from .construction import decouple_complex
        from css_decoupling.errors import BackendUnavailable, ResourceLimit
        tick = perf_counter()
        try:
            log.info("completing the negative free model before restriction (index=%s)", lattice.index)
            negative = image_resolution(kernel(HZ.dagger()), timeout=resolution_timeout,
                                        method=resolution_method)
            ranks, differentials = dict(C.ranks), dict(C.differentials)
            for k, differential in enumerate(negative, start=2):
                ranks[-k], differentials[-k] = differential.ncols, differential
            full = CochainComplex(C.ring, ranks, differentials)
            log.info("restricting the complete free model (ranks=%s)", full.ranks)
            full_restricted = restrict_complex(full, lattice)
            handoff = None
            if linear_backend == "sympy":
                from css_decoupling.topology.cohomology_handoff import build_cohomology_handoff
                log.info("transporting original complete cohomology witnesses to the retained lattice")
                handoff = build_cohomology_handoff(full, lattice, layers, restricted=full_restricted,
                                                  max_cell_index=max_cell_index)
            construction = decouple_complex(full_restricted, linear_backend=linear_backend,
                bounded_options=bounded_options, max_layer_dimension=max_layer_dimension,
                max_unknowns=max_unknowns, verify=False,
                restriction_basis=lattice.basis if linear_backend == "sympy" else None,
                cohomology_handoff=handoff, contraction_strategy=contraction_strategy,
                structural_search=structural_search)
        except (BackendUnavailable, ResourceLimit) as exc:
            exc.profile_analysis = analysis
            raise
        timings["circuit_construction"] = perf_counter() - tick
        built = construction.decoupling
        result = DecouplingResult(HX, HZ, lattice, built.multiplicities, built.free_C, built.free_T,
            built.stabilization, timings, {**built.trace, "restricted": full_restricted,
                "original_complex": full, "profile_analysis": analysis,
                "construction": construction, "comparison_method": "schanuel", "gluing": analysis.gluing,
                "coarse_graining_mode": "adaptive",
                "attempts": [{"basis": r["lattice"].basis, "split": r["step"].is_zero} for r in analysis.gluing],
                "translation_lattice_kind": "exact-common-kernel",
                "translation_kernel": analysis.translation_kernel.kernel_certificate})
        if verify:
            tick = perf_counter()
            log.info("independently verifying the raw-CSS-to-toric circuit (%s qudits, %s gate families)",
                     result.circuit.qudits, len(result.circuit.gates))
            report = verify_decoupling(result)
            if not report.valid:
                raise ArithmeticError("Decoupling failed verification: " + "; ".join(report.errors))
            timings["verification"] = perf_counter() - tick
        timings["total"] = perf_counter() - started
        return result
    tick = perf_counter()
    log.info("contracting comparison cone")
    psi, hC, hT = homotopy_inverse_window(phi, physical_boundary=restricted.d(-1), solve=solve)
    log.info("constructing stable check-cokernel isomorphism")
    from css_decoupling.errors import BackendUnavailable, ResourceLimit
    try:
        comparison = stable_cokernel_isomorphism(
            phi, psi, hC, hT, solve, partial(image_resolution, timeout=resolution_timeout, method=resolution_method),
            window=True, qs_options={"timeout": resolution_timeout, **(qs_options or {})})
    except (BackendUnavailable, ResourceLimit) as exc:
        exc.profile_analysis = analysis
        raise
    timings["comparison"] = perf_counter() - tick
    tick = perf_counter()
    lifts = LiftData(**{name: comparison[name] for name in
                      ("dC", "dT", "phi0", "psi0", "phi1", "psi1", "etaC", "etaT")})
    stable = stabilize(lifts)
    timings["circuit"] = perf_counter() - tick
    result = DecouplingResult(HX, HZ, lattice, multiplicities,
                             comparison["free_C"], comparison["free_T"], stable,
                             timings, {"complex": C, "layers": layers, "restricted": restricted,
                                       "phi": phi, "psi": psi, "hC": hC, "hT": hT,
                                       "comparison": comparison, "profile_analysis": analysis,
                                       "coarse_graining_mode": "adaptive",
                                       "gluing": analysis.gluing,
                                       "attempts": [{"basis": r["lattice"].basis, "split": r["step"].is_zero}
                                                    for r in analysis.gluing],
                                       "translation_lattice_kind": "exact-common-kernel",
                                       "translation_kernel": analysis.translation_kernel.kernel_certificate})
    if verify:
        tick = perf_counter()
        report = verify_decoupling(result)
        if not report.valid:
            raise ArithmeticError("Decoupling failed verification: " + "; ".join(report.errors))
        timings["verification"] = perf_counter() - tick
    timings["total"] = perf_counter() - started
    return result


def _decouple_uniform(HX, HZ, *, max_cell_index, max_layer_dimension, max_unknowns,
                      resolution_timeout, resolution_method, max_dlog_steps, verify,
                      linear_backend, bounded_options, contraction_strategy,
                      structural_search):
    """Compose the existing two algorithms without inventing adaptive evidence."""
    from dataclasses import replace
    from .coarse_graining import coarse_grain
    from .construction import decouple_complex
    from css_decoupling.certificates.construction import verify_construction
    from css_decoupling.errors import BackendUnavailable, ResourceLimit
    started = perf_counter()
    coarse = coarse_grain(HX, HZ, max_cell_index=max_cell_index,
        max_layer_dimension=max_layer_dimension, resolution_timeout=resolution_timeout,
        resolution_method=resolution_method, max_dlog_steps=max_dlog_steps, verify=True)
    if not coarse.decouplable:
        exc = InfiniteLayer("Algorithm 1 certified an infinite layer")
        exc.coarse_graining_result = coarse
        raise exc
    try:
        options = dict(linear_backend=linear_backend, bounded_options=bounded_options,
            max_layer_dimension=max_layer_dimension, max_unknowns=max_unknowns, verify=False,
            contraction_strategy=contraction_strategy, structural_search=structural_search)
        construction = decouple_complex(coarse if linear_backend == "sympy" else coarse.restricted, **options)
        construction.coarse_graining = coarse.certificate()
        if verify:
            report = verify_construction(construction, linear_backend=linear_backend,
                                         bounded_options=bounded_options)
            if not report.valid:
                raise ArithmeticError("Uniform Algorithm 1/2 verification failed: " + "; ".join(report.errors))
    except (BackendUnavailable, ResourceLimit) as exc:
        exc.coarse_graining_result = coarse
        raise
    built = construction.decoupling
    timings = {"coarse_graining": coarse.timings["total"], **built.timings}
    result = replace(built, HX=HX, HZ=HZ, lattice=coarse.lattice, timings=timings,
        trace={**built.trace, "coarse_graining_mode": "uniform", "coarse_graining": coarse,
               "construction": construction, "original_complex": coarse.complex,
               "restricted": coarse.restricted, "comparison_method": "schanuel",
               "translation_lattice_kind": "uniform-sufficient-lattice",
               "translation_kernel": coarse.translation_kernel.kernel_certificate})
    if verify:
        try:
            report = verify_decoupling(result)
        except (BackendUnavailable, ResourceLimit) as exc:
            exc.coarse_graining_result = coarse
            raise
        if not report.valid:
            raise ArithmeticError("Uniform raw circuit verification failed: " + "; ".join(report.errors))
    timings["total"] = perf_counter() - started
    return result
