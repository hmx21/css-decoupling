"""Algorithm 1: a sufficient cell and its actual bounded free complex.

The uniform cover uses only the finite profile and its exact common
translation kernel.  It does not solve gluing, splitting, or circuit
equations.  Algorithm 1 includes general module computations and finite-field
discrete logarithms; no polynomial-time claim is made for this entry point.
"""
from dataclasses import dataclass, field
from numbers import Integral
from time import perf_counter

from css_decoupling.algebra.modules import using_backend
from css_decoupling.errors import InfiniteLayer, ResourceLimit
from css_decoupling.lattice.kernel import translation_kernel
from css_decoupling.lattice.retained import Lattice
from css_decoupling.topology.cohomology import cohomology_layer
from css_decoupling.topology.resolutions import css_complex
from .criterion import _infinite_witness
from .splitting import restrict_complex


@dataclass
class CoarseGrainingResult:
    """A sufficient-cell output, or an input-bound infinite-layer obstruction.

    A positive result contains the entire bounded complex, including its
    resolved negative end, before and after restriction.  ``multiplicities``
    are the dimensions of the original nonzero layers and remain unchanged
    under restriction of scalars.  The lattice is sufficient, not necessarily
    minimal.  A negative result has no claimed lattice or restricted complex.
    """
    HX: object
    HZ: object
    sector: str
    complex: object
    layers: dict
    translation_kernel: object
    lattice: object
    restricted: object
    decouplable: bool
    reason: str
    obstruction: object = None
    timings: dict = field(default_factory=dict)
    cohomology_handoff: dict | None = None

    @property
    def multiplicities(self):
        return {j: layer.dimension for j, layer in self.layers.items()}

    def certificate(self, *, include_handoff=False):
        if type(include_handoff) is not bool:
            raise ValueError("include_handoff must be boolean")
        from css_decoupling.certificates.coarse_graining import coarse_graining_dict
        return coarse_graining_dict(self, include_handoff=include_handoff)


def _cap(value, name, *, nullable=True):
    if value is None and nullable:
        return
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer" + (" or None" if nullable else ""))


def coarse_grain(HX, HZ, *, max_cell_index=4096, max_layer_dimension=10000,
                 resolution_timeout=120, resolution_method="native",
                 max_dlog_steps=100000, max_linear_entries=2000000,
                 factor_limit=100000, backend=None, sector="X", verify=True):
    """Implement the manuscript's sufficient uniform coarse-graining.

    With ``m`` nonzero finite layers, return
    ``Lambda_suff = p**max(m-1, 0) Lambda_triv`` and the actual restriction of
    the complete bounded CSS free model to that lattice.  The exact kernel
    is computed without enumerating the whole translation image.

    An infinite layer gives a certified ``decouplable=False`` result.
    Invalid inputs and resource/backend failures raise, never report ``no``.
    The cell cap is checked before expanding any cosets.  When only that cap
    fails after the profile and kernel have been computed, the exception's
    ``coarse_graining_partial`` contains diagnostics, explicitly without a
    restricted complex or a successful output.

    ``sector='auto'`` selects X; the criterion is equally decisive in either
    sector.  A Z result's restricted physical window contains HZ then HX,
    while ``HX`` and ``HZ`` retain the caller's original input order.
    """
    if sector not in ("X", "Z", "auto"):
        raise ValueError("sector must be X, Z or auto")
    if type(verify) is not bool:
        raise ValueError("verify must be a boolean")
    for value, name in ((max_cell_index, "max_cell_index"),
                        (max_layer_dimension, "max_layer_dimension")):
        _cap(value, name)
    for value, name in ((max_dlog_steps, "max_dlog_steps"),
                        (max_linear_entries, "max_linear_entries"),
                        (factor_limit, "factor_limit")):
        _cap(value, name, nullable=False)
    if HX.ring != HZ.ring or HX.ncols != HZ.ncols:
        raise ValueError("CSS checks must share their ring and physical columns")
    if HX.ring.D < 2:
        raise ValueError("The topological algorithm requires D >= 2")
    options = dict(max_cell_index=max_cell_index,
                   max_layer_dimension=max_layer_dimension,
                   resolution_timeout=resolution_timeout,
                   resolution_method=resolution_method,
                   max_dlog_steps=max_dlog_steps,
                   max_linear_entries=max_linear_entries,
                   factor_limit=factor_limit, sector=sector, verify=verify)
    if backend is not None:
        with using_backend(backend, timeout=resolution_timeout):
            return coarse_grain(HX, HZ, **options)
    chosen = "X" if sector == "auto" else sector
    active_X, active_Z = (HX, HZ) if chosen == "X" else (HZ, HX)
    started = tick = perf_counter()
    C = css_complex(active_X, active_Z, timeout=resolution_timeout,
                    complete_negative=True, resolution_method=resolution_method)
    timings = {"complex": perf_counter() - tick}
    layers = {}
    tick = perf_counter()

    def finish(result):
        result.timings["total"] = perf_counter() - started
        if verify:
            from css_decoupling.certificates.coarse_graining import verify_coarse_graining
            report = verify_coarse_graining(
                result.certificate(include_handoff=True), max_cell_index=max_cell_index,
                max_layer_dimension=max_layer_dimension,
                max_linear_entries=max_linear_entries)
            if not report.valid:
                raise ArithmeticError("Coarse-graining verification failed: " + "; ".join(report.errors))
        return result

    for degree in C.degrees:
        if degree <= 0:
            continue
        try:
            layer = cohomology_layer(C, degree, max_dimension=max_layer_dimension)
        except InfiniteLayer:
            witness = _infinite_witness(C, degree)
            timings["profile"] = perf_counter() - tick
            return finish(CoarseGrainingResult(
                HX, HZ, chosen, C, layers, None, None, None,
                False, "infinite-layer", witness, timings))
        if layer.dimension:
            if degree >= C.ring.D:
                raise ValueError("Nonzero layer outside the topological CSS degree bound")
            layers[degree] = layer
    timings["profile"] = perf_counter() - tick
    tick = perf_counter()
    # Computing a geometric lattice does not need its coset representatives.
    # Defer the output-size cap so a completed exact kernel remains available
    # when the sufficient uniform restriction is too large to materialize.
    exact_kernel = translation_kernel(
        layers, C.ring.D, max_index=None, max_dlog_steps=max_dlog_steps,
        max_linear_entries=max_linear_entries, factor_limit=factor_limit)
    timings["translation_kernel"] = perf_counter() - tick
    factor = C.ring.p ** max(len(layers) - 1, 0)
    lattice = Lattice([[factor * value for value in row] for row in exact_kernel.basis],
                      max_index=None)
    if max_cell_index is not None and lattice.index > max_cell_index:
        exc = ResourceLimit(f"Coarse unit cell has index {lattice.index}; cap is {max_cell_index}")
        exc.coarse_graining_partial = {
            "status": "resource-limited", "sector": chosen,
            "multiplicities": {j: layer.dimension for j, layer in layers.items()},
            "translation_kernel": exact_kernel.kernel_certificate,
            "sufficient_basis": [list(row) for row in lattice.basis],
            "sufficient_index": lattice.index, "restricted": None,
        }
        raise exc
    tick = perf_counter()
    restricted = restrict_complex(C, lattice)
    timings["restriction"] = perf_counter() - tick
    result = CoarseGrainingResult(
        HX, HZ, chosen, C, layers, exact_kernel, lattice, restricted,
        True, "uniform-sufficient-lattice", timings=timings)
    from css_decoupling.algebra.modules import get_backend, GeneralBackend
    selected = get_backend(C.ring)
    if type(selected) is GeneralBackend:
        from css_decoupling.topology.cohomology_handoff import build_cohomology_handoff
        tick = perf_counter()
        result.cohomology_handoff = build_cohomology_handoff(C, lattice, layers,
                                            restricted=restricted, backend=selected,
                                            max_cell_index=max_cell_index, max_total_rank=None)
        timings["cohomology_handoff"] = perf_counter() - tick
    return finish(result)
