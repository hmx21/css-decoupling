"""Exact translation kernel and successive, cocycle-guided profile splitting."""
from dataclasses import dataclass
from time import perf_counter

from css_decoupling.algebra.laurent import eye
from css_decoupling.algebra.modules import using_backend
from css_decoupling.errors import InfiniteLayer, NoLaurentSolution
from css_decoupling.lattice.retained import Lattice, matrix_power
from css_decoupling.lattice.kernel import translation_kernel
from css_decoupling.topology.cohomology import profile
from css_decoupling.topology.resolutions import css_complex
from .gluing import gluing_classes, killing_axes
from .lattice_selection import select_lattice
from .splitting import restrict_complex, splitting_map


@dataclass
class ProfileAnalysis:
    HX: object
    HZ: object
    complex: object
    layers: dict
    translation_kernel: object
    initial_lattice: object
    lattice: object
    restricted: object
    phi: object
    gluing: list
    timings: dict
    sector: str = "X"

    @property
    def lattice_guarantee(self):
        """Scope of the returned cell's optimality, never inferred from success."""
        if self.initial_lattice.basis != self.translation_kernel.basis:
            return "supplied-lattice"
        if len(self.layers) < 2:
            return "minimum-cell"
        nonzero = [r for r in self.gluing if not r["step"].is_zero]
        if not nonzero:
            return "minimum-cell"
        if len(self.layers) == 2 and all(
            r.get("selection") is not None and
            r["selection"].optimality == "family-optimal" for r in nonzero
        ):
            return "minimum-cell"
        return "sufficient-cell"

    @property
    def multiplicities(self):
        return {j: layer.dimension for j, layer in self.layers.items()}

    def certificate(self):
        from css_decoupling.certificates.analysis import analysis_dict
        return analysis_dict(self)


def analyze_profile(HX, HZ, *, lattice_basis=None, max_cell_index=4096,
                    max_layer_dimension=10000, max_unknowns=250000,
                    resolution_timeout=120, resolution_method="native",
                    max_dlog_steps=100000, backend=None, sector="X",
                    lattice_strategy="auto", max_exterior_entries=2000000):
    """Analyze and split a profile without requiring a projective free basis.

    On the exact common translation kernel, inspect layers in increasing
    degree. Use the three-dimensional optimal lattice when applicable, or
    a sufficient coordinate cover in general, then recompute the attachment.
    ``coordinate`` retains the previous strategy; ``optimal`` fails outside
    the implemented exact cases. Stage optimality is not tower optimality.
    """
    if sector not in ("X", "Z", "auto"):
        raise ValueError("sector must be X, Z or auto")
    if lattice_strategy not in ("auto", "coordinate", "optimal"):
        raise ValueError("lattice_strategy must be auto, coordinate or optimal")
    options = dict(lattice_basis=lattice_basis, max_cell_index=max_cell_index,
                   max_layer_dimension=max_layer_dimension, max_unknowns=max_unknowns,
                   resolution_timeout=resolution_timeout, resolution_method=resolution_method,
                   max_dlog_steps=max_dlog_steps, backend=backend,
                   lattice_strategy=lattice_strategy, max_exterior_entries=max_exterior_entries)
    if sector == "Z":
        result = analyze_profile(HZ, HX, sector="X", **options)
        result.HX, result.HZ, result.sector = HX, HZ, "Z"
        return result
    if sector == "auto":
        try:
            return analyze_profile(HX, HZ, sector="X", **options)
        except InfiniteLayer:
            return analyze_profile(HX, HZ, sector="Z", **options)
    if backend is not None:
        options["backend"] = None
        with using_backend(backend, timeout=resolution_timeout):
            return analyze_profile(HX, HZ, **options)
    if HX.ring.D < 2:
        raise ValueError("The topological algorithm requires D >= 2")
    timings = {}
    started = tick = perf_counter()
    C = css_complex(HX, HZ, timeout=resolution_timeout, complete_negative=False,
                    resolution_method=resolution_method)
    timings["complex"] = perf_counter() - tick
    tick = perf_counter()
    layers = profile(C, max_dimension=max_layer_dimension)
    timings["profile"] = perf_counter() - tick
    ring = HX.ring
    tick = perf_counter()
    exact_kernel = translation_kernel(layers, ring.D, max_index=max_cell_index,
                                      max_dlog_steps=max_dlog_steps)
    timings["translation_kernel"] = perf_counter() - tick
    lattice = exact_kernel if lattice_basis is None else Lattice(lattice_basis, max_index=max_cell_index)
    if lattice_basis is not None:
        for layer in layers.values():
            for column in zip(*lattice.basis):
                action = eye(ring, layer.dimension)
                for a, exponent in zip(layer.actions, column):
                    action = action * matrix_power(a, exponent)
                if action != eye(ring, layer.dimension):
                    raise ValueError("Supplied retained translations act nontrivially on a layer")
    records = []
    initial_lattice = lattice
    tick = perf_counter()

    def at(current):
        return restrict_complex(C, current), {
            j: current.pack(layer.representatives) for j, layer in layers.items()}

    restricted, reps = at(lattice)
    for j in sorted(layers)[1:]:
        step = gluing_classes(restricted, reps, j, max_unknowns=max_unknowns,
                              max_layer_dimension=max_layer_dimension)
        axes = killing_axes(step.classes)
        record = {"lattice": lattice, "step": step, "killing_axes": axes}
        records.append(record)
        if step.is_zero:
            continue
        if lattice_basis is not None:
            exc = NoLaurentSolution(f"Nonzero gluing kappa_({j},i) on the supplied retained lattice")
            exc.gluing_certificate = step.certificate()
            raise exc
        selection = select_lattice(step.classes, D=ring.D, p=ring.p,
                                   strategy=lattice_strategy,
                                   max_exterior_entries=max_exterior_entries)
        transition = selection.relative_basis
        diagonal = all(transition[i][j] == 0 for i in range(ring.D)
                       for j in range(ring.D) if i != j)
        record["killing_axes"] = tuple(i for i in range(ring.D) if transition[i][i] == ring.p) if diagonal else ()
        # Coordinate-only records remain readable in the original v1 format.
        if lattice_strategy != "coordinate":
            record["selection"] = selection
        lattice = Lattice([[sum(lattice.basis[i][k] * transition[k][j]
                                for k in range(ring.D)) for j in range(ring.D)]
                           for i in range(ring.D)], max_index=max_cell_index)
        restricted, reps = at(lattice)
        after = gluing_classes(restricted, reps, j, max_unknowns=max_unknowns,
                              max_layer_dimension=max_layer_dimension)
        records.append({"lattice": lattice, "step": after, "killing_axes": ()})
        if not after.is_zero:
            raise ArithmeticError("Cocycle-guided restriction failed to kill the recorded attachment")
    phi = splitting_map(restricted, reps, max_unknowns=max_unknowns, window=True)
    timings["splitting"] = perf_counter() - tick
    timings["total"] = perf_counter() - started
    return ProfileAnalysis(HX, HZ, C, layers, exact_kernel, initial_lattice,
                           lattice, restricted, phi, records, timings)
