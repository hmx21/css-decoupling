"""Algorithm 2: construct a circuit from an already coarse-grained free model.

No resolution or translation kernel is computed here.  Appendix C4 uses the
negative free resolutions already in the input and a Schanuel stable frame.
The optional bounded backend implements the C5 coefficient-linearization
route; the practical exact backends do not inherit its complexity guarantee.
"""
from dataclasses import dataclass
import logging
from time import perf_counter

from css_decoupling.algebra.laurent import Matrix, zeros
from css_decoupling.algebra.modules import get_backend, using_backend
from css_decoupling.errors import ResourceLimit
from css_decoupling.lattice.retained import Lattice
from css_decoupling.topology.complexes import CochainComplex, CochainMap, toric_stack, homotopy_inverse
from .splitting import splitting_map
from .stabilization import LiftData, stabilize

log = logging.getLogger(__name__)


def construction_size(C):
    """Dense common-box input size of C5; never allocate the coefficient box."""
    if not isinstance(C, CochainComplex):
        raise TypeError("Algorithm 2 requires a CochainComplex")
    width = max((abs(e) for A in C.differentials.values() for f in A.entries.values()
                 for exponent in f.terms for e in exponent), default=0)
    slots = sum(C.rank(n) * C.rank(n + 1) for n in C.degrees)
    return {"dimension": C.ring.D, "half_width": width,
            "matrix_entries": slots, "dense_coefficients": slots * (2 * width + 1) ** C.ring.D,
            "stored_monomials": sum(len(f.terms) for A in C.differentials.values() for f in A.entries.values()),
            "complex_length": len(C.degrees), "total_free_rank": sum(C.ranks.values())}


@dataclass
class CircuitConstructionResult:
    complex: CochainComplex
    layers: dict
    phi: CochainMap
    psi: CochainMap
    hC: dict
    hT: dict
    comparison: dict
    decoupling: object
    linear_backend: str
    input_size: dict
    timings: dict
    coarse_graining: dict | None = None
    cohomology_handoff: dict | None = None
    contraction_strategy: str = "sections"
    structural_search: bool = False

    @property
    def circuit(self):
        return self.decoupling.circuit

    @property
    def qca(self):
        """Pauli action on the Algorithm 2 input complex, matching circuit."""
        return self.decoupling.qca

    @property
    def physical_circuit(self):
        """Original-sector circuit when Algorithm 1 source evidence is attached."""
        if self.coarse_graining is not None and self.coarse_graining["sector"] == "Z":
            from css_decoupling.circuits.duality import dualize_circuit
            return dualize_circuit(self.circuit)
        return self.circuit

    @property
    def physical_qca(self):
        """Original-sector QCA, matching physical_circuit without changing order."""
        return (self.qca.fourier_dual() if self.coarse_graining is not None
                and self.coarse_graining["sector"] == "Z" else self.qca)

    @property
    def multiplicities(self):
        return self.decoupling.multiplicities

    def certificate(self):
        from css_decoupling.certificates.construction import construction_dict
        return construction_dict(self)


def _construction_policy(linear_backend, contraction_strategy, structural_search):
    """Resolve explicit algorithm choices independently of the input model."""
    if linear_backend not in ("sympy", "bounded"):
        raise ValueError("linear_backend must be sympy or bounded")
    if contraction_strategy == "paper":
        raise ValueError("The coupled 'paper' contraction strategy was removed; use 'sections' or 'normalized'")
    if contraction_strategy not in ("auto", "normalized", "sections"):
        raise ValueError("contraction_strategy must be auto, normalized or sections")
    if type(structural_search) is not bool:
        raise ValueError("structural_search must be a boolean")
    selected = ("normalized" if linear_backend == "sympy" else "sections") if contraction_strategy == "auto" else contraction_strategy
    if selected == "normalized" and linear_backend != "sympy":
        raise ValueError("normalized contraction currently requires the native sympy backend")
    if structural_search and (selected != "sections" or linear_backend != "sympy"):
        raise ValueError("structural_search requires native sections contraction")
    return selected


def decouple_complex(C, *, linear_backend="sympy", bounded_options=None,
                     max_layer_dimension=10000, max_unknowns=250000,
                     max_total_rank=100000, verify=True, restriction_basis=None,
                     cohomology_handoff=None, contraction_strategy="auto",
                     structural_search=False):
    """Implement Algorithm 2 with the input lattice treated as the unit lattice.

    The input must include its complete negative free resolution and have
    trivial finite cohomology in degrees 1..D-1. Gluing is tested by solving
    the actual comparison equations. Invalid hypotheses, exact obstruction,
    and resource failures remain different exceptions.

    Native ``auto`` uses a complete inverse followed by normalization and
    transfer; ``sections`` instead solves the strong
    sections of Appendix C4 degree by degree. Finite-support probes are
    opt-in for native sections construction. Bounded auto keeps sections.
    Normalization has no section solver's 2M/50M admission cap; callers
    needing a hard wall/memory limit must supervise the process. All
    strategies retain complete certificate verification.
    """
    from css_decoupling.algebra.bounded_linear import BoundedLinearBackend
    from css_decoupling.topology.trivial_cohomology import trivial_cohomology
    from css_decoupling.certificates.coarse_graining import _verify_source_model
    from .comparison import paper_stable_cokernel_isomorphism
    from .pipeline import DecouplingResult

    strategy = _construction_policy(linear_backend, contraction_strategy, structural_search)

    if not isinstance(C, CochainComplex):
        from .coarse_graining import CoarseGrainingResult
        if not isinstance(C, CoarseGrainingResult) or not C.decouplable or C.restricted is None:
            raise TypeError("Algorithm 2 requires a complete complex or a successful Algorithm 1 result")
        from css_decoupling.topology.cohomology_handoff import build_cohomology_handoff
        coarse = C
        if linear_backend != "sympy":
            raise ValueError("Algorithm 1 witness handoff requires sympy; pass the restricted complex for another backend")
        evidence = cohomology_handoff or coarse.cohomology_handoff or build_cohomology_handoff(
            coarse.complex, coarse.lattice, coarse.layers, restricted=coarse.restricted,
            max_cell_index=None, max_total_rank=max_total_rank)
        answer = decouple_complex(coarse.restricted, linear_backend=linear_backend,
            bounded_options=bounded_options, max_layer_dimension=max_layer_dimension,
            max_unknowns=max_unknowns, max_total_rank=max_total_rank, verify=False,
            restriction_basis=coarse.lattice.basis, cohomology_handoff=evidence,
            contraction_strategy=strategy, structural_search=structural_search)
        answer.coarse_graining = coarse.certificate()
        if verify:
            from css_decoupling.certificates.construction import verify_construction
            report = verify_construction(answer, linear_backend=linear_backend, max_total_rank=max_total_rank)
            if not report.valid:
                raise ArithmeticError("Composed Algorithm 1/2 certificate failed: " + "; ".join(report.errors))
        return answer

    if linear_backend not in ("sympy", "bounded"):
        raise ValueError("linear_backend must be sympy or bounded")
    if bounded_options and linear_backend != "bounded":
        raise ValueError("bounded_options require the bounded linear backend")
    if restriction_basis is not None and linear_backend != "sympy":
        raise ValueError("Restriction-aware algebra requires the native sympy backend")
    if cohomology_handoff is not None and linear_backend != "sympy":
        raise ValueError("Native cohomology handoff cannot replace the bounded backend")
    for value, name in ((max_layer_dimension, "max_layer_dimension"), (max_unknowns, "max_unknowns")):
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
            raise ValueError(name + " must be a nonnegative integer or None")
    if C.ring.D < 2:
        raise ValueError("Algorithm 2 requires spatial dimension at least two")
    size = construction_size(C)
    if C.degrees and max(-C.degrees[0], C.degrees[-1]) > C.ring.D + 2:
        raise ValueError("Input free model is nonzero outside degrees |n| <= D+2 (Lemma 4)")
    if max_total_rank is not None:
        if isinstance(max_total_rank, bool) or not isinstance(max_total_rank, int) or max_total_rank < 1:
            raise ValueError("max_total_rank must be positive or None")
        if size["total_free_rank"] > max_total_rank:
            raise ResourceLimit("Input free-model rank exceeds max_total_rank")
    if linear_backend != "bounded":
        with using_backend(linear_backend):
            if restriction_basis is not None:
                from css_decoupling.lattice.restriction_backend import RestrictionBackend
                selected = RestrictionBackend(get_backend(C.ring), Lattice(restriction_basis, max_index=None))
                with using_backend(selected):
                    return _construct(C, selected, linear_backend, size, max_layer_dimension,
                                      max_unknowns, max_total_rank, verify, trivial_cohomology,
                                      _verify_source_model, paper_stable_cokernel_isomorphism, DecouplingResult,
                                      cohomology_handoff, strategy, structural_search)
            return _construct(C, get_backend(C.ring), linear_backend, size, max_layer_dimension,
                              max_unknowns, max_total_rank, verify, trivial_cohomology,
                              _verify_source_model, paper_stable_cokernel_isomorphism, DecouplingResult,
                              cohomology_handoff, strategy, structural_search)
    solver = BoundedLinearBackend(C.ring, **(bounded_options or {}))
    return _construct(C, solver, linear_backend, size, max_layer_dimension,
                      max_unknowns, max_total_rank, verify, trivial_cohomology,
                      _verify_source_model, paper_stable_cokernel_isomorphism, DecouplingResult,
                      cohomology_handoff, strategy, structural_search)


def _construct(C, backend, backend_name, size, max_dimension, max_unknowns,
               max_total_rank, verify, cohomology, validate_source, compare, result_class,
               cohomology_handoff=None, contraction_strategy="sections", structural_search=False):
    start = tick = perf_counter()
    HX, HZ = C.d(0), C.d(-1).dagger()
    log.info("Algorithm 2: verifying the complete input complex (ranks=%s)", C.ranks)
    transported = None
    if cohomology_handoff is None:
        validate_source(C, HX, HZ, kernel=backend.kernel, solve=backend.solve)
    else:
        from css_decoupling.topology.cohomology_handoff import consume_cohomology_handoff
        transported = consume_cohomology_handoff(cohomology_handoff, C, backend=backend,
                                                 max_dimension=max_dimension,
                                                 max_cell_index=None, max_total_rank=max_total_rank)
    timings = {"validate_input": perf_counter() - tick}
    tick = perf_counter()
    log.info("Algorithm 2: computing trivial-translation cohomology")
    layers = (cohomology(C, backend=backend, max_dimension=max_dimension)
              if transported is None else transported)
    multiplicities = {j: l.dimension for j, l in layers.items()}
    if any(not 1 <= j < C.ring.D for j in multiplicities):
        raise ValueError("Cohomology outside the topological CSS degree range")
    timings["cohomology"] = perf_counter() - tick
    tick = perf_counter()
    log.info("Algorithm 2: lifting the toric comparison (profile=%s)", multiplicities)
    positive = splitting_map(C, {j: l.representatives for j, l in layers.items()},
                             max_unknowns=max_unknowns, window=True, solver=backend.solve,
                             max_total_rank=max_total_rank)
    T = toric_stack(C.ring, multiplicities, max_total_rank=max_total_rank)
    maps = dict(positive.components)
    for n in range(-1, min(C.degrees + T.degrees + (0,)) - 1, -1):
        rhs = maps.get(n + 1, zeros(C.ring, C.rank(n + 1), T.rank(n + 1))) * T.d(n)
        if max_unknowns is not None and C.rank(n) * T.rank(n) > max_unknowns:
            raise ResourceLimit("Negative comparison lift exceeds max_unknowns")
        maps[n] = backend.solve(C.d(n), rhs)
    phi = CochainMap(T, C, maps)
    log.info("Algorithm 2: contracting the comparison cone")
    from css_decoupling.topology.direct_splitting import structural_sections
    # Explicitly clear an enclosing probe context for disabled/bounded calls.
    with structural_sections(enabled=structural_search):
        if contraction_strategy == "normalized":
            from .toric_comparison import toric_comparison_inverse
            try:
                inverse = toric_comparison_inverse(phi, layers, backend.solve)
            except ResourceLimit:
                # This is a local shortcut budget, not a global caller budget.
                # The complete weak-cone algorithm remains available.
                inverse = None
            if inverse is None:
                psi, hC, hT, cone_h = homotopy_inverse(
                    phi, backend.solve, return_contraction=True, strategy="weak")
            else:
                psi, hC, hT, cone_h = (inverse.psi, inverse.hC, inverse.hT,
                                       inverse.cone_contraction)
        else:
            section_solver = backend.solve
            if backend_name == "sympy":
                from .toric_comparison import toric_cone_solver
                section_solver = toric_cone_solver(phi, layers, backend.solve) or backend.solve
            psi, hC, hT, cone_h = homotopy_inverse(phi, section_solver, return_contraction=True)
    timings["comparison_maps"] = perf_counter() - tick
    tick = perf_counter()
    log.info("Algorithm 2: constructing and splitting the Schanuel cone")
    target_preimage = None
    if backend_name == "sympy":
        from .toric_comparison import toric_physical_preimage
        target_preimage = lambda rhs: toric_physical_preimage(T, multiplicities, rhs)
    with structural_sections(enabled=structural_search):
        comparison = compare(phi, psi, hC, hT, backend.solve,
                             comparison_contraction=cone_h, target_preimage=target_preimage,
                             contraction_strategy="normalized" if contraction_strategy == "normalized" else "direct")
    timings["schanuel"] = perf_counter() - tick
    tick = perf_counter()
    log.info("Algorithm 2: stabilizing lifts and compiling SUM/SWAP gates")
    lifts = LiftData(**{key: comparison[key] for key in
                       ("dC", "dT", "phi0", "psi0", "phi1", "psi1", "etaC", "etaT")})
    stable = stabilize(lifts)
    timings["circuit"] = perf_counter() - tick
    identity = Lattice([[int(i == j) for j in range(C.ring.D)] for i in range(C.ring.D)])
    inner = result_class(HX, HZ, identity, multiplicities, comparison["free_C"], comparison["free_T"],
                         stable, timings, {"complex": C, "layers": layers, "phi": phi, "psi": psi,
                                           "hC": hC, "hT": hT, "comparison": comparison,
                                           "contraction_strategy": contraction_strategy,
                                           "structural_search": structural_search})
    answer = CircuitConstructionResult(C, layers, phi, psi, hC, hT, comparison, inner,
                                       backend_name, size, timings, cohomology_handoff=cohomology_handoff,
                                       contraction_strategy=contraction_strategy,
                                       structural_search=structural_search)
    if verify:
        from css_decoupling.certificates.construction import verify_construction
        tick = perf_counter()
        report = verify_construction(answer, linear_backend=backend_name,
                                     bounded_options=getattr(backend, "caps", None), max_total_rank=max_total_rank)
        if not report.valid:
            raise ArithmeticError("Algorithm 2 certificate failed: " + "; ".join(report.errors))
        timings["verification"] = perf_counter() - tick
    timings["total"] = perf_counter() - start
    return answer
