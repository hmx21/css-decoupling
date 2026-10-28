"""Decide the profile splitting criterion on a specified translation lattice.

This interface never computes a common translation kernel or selects a new
lattice.  It first inspects the original finite modules and translation
matrices, and expands the candidate cell only when those actions are trivial.
A negative answer stops at a genuine obstruction; resource failures propagate.
"""
from dataclasses import dataclass, field
from numbers import Integral
from time import perf_counter

from css_decoupling.algebra.laurent import eye
from css_decoupling.algebra.modules import get_backend, kernel, solve, using_backend
from css_decoupling.errors import InfiniteLayer, ResourceLimit
from css_decoupling.lattice.retained import Lattice, matrix_power
from css_decoupling.topology.cohomology import cohomology_layer
from css_decoupling.topology.resolutions import css_complex
from .gluing import gluing_classes
from .splitting import restrict_complex, splitting_map


@dataclass
class LatticeCriterion:
    """A source-bound decision, with a splitting map or an exact obstruction.

    ``layers`` is complete except for an ``infinite-layer`` result, where it
    contains the finite nonzero layers below the first infinite layer.
    ``gluing`` ends at the first nonzero attachment; no independent classes
    are asserted above an unsplit lower tower. ``phi`` is populated exactly
    for positive results and maps the nonnegative toric window to the actual
    restricted source complex. It is not itself a circuit certificate.
    """
    HX: object
    HZ: object
    sector: str
    complex: object
    layers: dict
    lattice: Lattice
    decouplable: bool
    reason: str
    obstruction: object = None
    restricted: object = None
    phi: object = None
    gluing: list = field(default_factory=list)
    timings: dict = field(default_factory=dict)

    @property
    def multiplicities(self):
        return {j: layer.dimension for j, layer in self.layers.items()}

    def certificate(self):
        from css_decoupling.certificates.criterion import criterion_dict
        return criterion_dict(self)


def _positive_cap(value, name):
    if value is not None and (isinstance(value, bool) or not isinstance(value, Integral) or value < 1):
        raise ValueError(f"{name} must be a positive integer or None")


def _infinite_witness(C, degree):
    """An actual quotient presentation and a standard monomial ray.

    The missing pure power is in the polynomial localization's D+1 variables;
    its meaning is verified by recomputing a Groebner basis of this quotient,
    not by trusting an exception message or a cached finiteness flag.
    """
    cycles = kernel(C.d(degree))
    boundary = solve(cycles, C.d(degree - 1))
    relations = kernel(cycles)
    presentation = boundary.hstack(relations)
    leading = get_backend(C.ring).quotient_groebner(presentation)["leading_terms"]
    for component in range(presentation.nrows):
        terms = [tuple(t[1:]) for t in leading if t[0] == component]
        if any(not any(t) for t in terms):
            continue
        for axis in range(C.ring.D + 1):
            if not any(t[axis] and all(not e for k, e in enumerate(t) if k != axis) for t in terms):
                return {"degree": degree, "cycles": cycles, "boundary_coordinates": boundary,
                        "cycle_relations": relations, "presentation": presentation,
                        "ray": {"component": component, "axis": axis}}
    raise ArithmeticError("Infinite quotient has no standard-monomial ray")


def check_lattice(HX, HZ, lattice_basis, *, max_cell_index=4096,
                  max_layer_dimension=10000, max_unknowns=250000,
                  resolution_timeout=120, resolution_method="native",
                  backend=None, sector="X"):
    """Check an arbitrary specified full-rank integer lattice, in any D>=2.

    Returns reasons ``split-profile``, ``infinite-layer``,
    ``nontrivial-layer-translation``, or ``nonzero-gluing``. A negative result
    contains an independently checkable witness. Invalid CSS inputs and
    resource/backend failures raise exceptions and are never mathematical
    negative answers. ``sector='auto'`` uses X: a certified X obstruction is
    already decisive, so the same decision does not require a second sector.

    Original action matrices are powered by repeated squaring. A negative
    translation result is independent of the candidate's volume and can
    exceed ``max_cell_index``. For gluing and an explicit splitting, cell
    restriction has unavoidable output size proportional to the cell index,
    and that cap is enforced before any cosets are materialized.
    """
    if sector not in ("X", "Z", "auto"):
        raise ValueError("sector must be X, Z or auto")
    for cap, name in ((max_cell_index, "max_cell_index"),
                      (max_layer_dimension, "max_layer_dimension"),
                      (max_unknowns, "max_unknowns")):
        _positive_cap(cap, name)
    if HX.ring.D < 2:
        raise ValueError("The topological criterion requires D >= 2")
    lattice = Lattice(lattice_basis, max_index=None)
    if lattice.D != HX.ring.D:
        raise ValueError("Lattice and input ring dimensions differ")
    options = dict(max_cell_index=max_cell_index, max_layer_dimension=max_layer_dimension,
                   max_unknowns=max_unknowns, resolution_timeout=resolution_timeout,
                   resolution_method=resolution_method, backend=None, sector=sector)
    if backend is not None:
        with using_backend(backend, timeout=resolution_timeout):
            return check_lattice(HX, HZ, lattice_basis, **options)
    chosen = "X" if sector == "auto" else sector
    active_X, active_Z = (HX, HZ) if chosen == "X" else (HZ, HX)
    started = tick = perf_counter()
    C = css_complex(active_X, active_Z, timeout=resolution_timeout,
                    complete_negative=False, resolution_method=resolution_method)
    timings = {"complex": perf_counter() - tick}
    layers = {}
    tick = perf_counter()

    def finish(reason, obstruction=None, *, restricted=None, phi=None, gluing=None):
        timings["total"] = perf_counter() - started
        return LatticeCriterion(HX, HZ, chosen, C, layers, lattice,
                                reason == "split-profile", reason, obstruction,
                                restricted, phi, gluing or [], dict(timings))

    for degree in C.degrees:
        if degree <= 0:
            continue
        try:
            layer = cohomology_layer(C, degree, max_dimension=max_layer_dimension)
        except InfiniteLayer:
            witness = _infinite_witness(C, degree)
            timings["profile"] = perf_counter() - tick
            return finish("infinite-layer", witness)
        if layer.dimension:
            if degree >= C.ring.D:
                raise ValueError("Nonzero layer outside the topological CSS degree bound")
            layers[degree] = layer
    timings["profile"] = perf_counter() - tick
    tick = perf_counter()
    for degree, layer in sorted(layers.items()):
        identity = eye(C.ring, layer.dimension)
        for column, vector in enumerate(zip(*lattice.basis)):
            action = identity
            for A, exponent in zip(layer.actions, vector):
                action = action * matrix_power(A, exponent)
            if action != identity:
                timings["translation"] = perf_counter() - tick
                return finish("nontrivial-layer-translation",
                              {"degree": degree, "column": column, "action": action})
    timings["translation"] = perf_counter() - tick
    if max_cell_index is not None and lattice.index > max_cell_index:
        raise ResourceLimit(f"Coarse unit cell has index {lattice.index}; cap is {max_cell_index}")
    tick = perf_counter()
    restricted = restrict_complex(C, lattice)
    reps = {j: lattice.pack(layer.representatives) for j, layer in layers.items()}
    timings["restriction"] = perf_counter() - tick
    tick = perf_counter()
    attachments = []
    for degree in sorted(layers)[1:]:
        step = gluing_classes(restricted, reps, degree, max_unknowns=max_unknowns,
                              max_layer_dimension=max_layer_dimension)
        attachments.append(step)
        if not step.is_zero:
            timings["gluing"] = perf_counter() - tick
            return finish("nonzero-gluing", {"degree": degree},
                          restricted=restricted, gluing=attachments)
    timings["gluing"] = perf_counter() - tick
    tick = perf_counter()
    phi = splitting_map(restricted, reps, max_unknowns=max_unknowns, window=True)
    timings["splitting"] = perf_counter() - tick
    return finish("split-profile", restricted=restricted, phi=phi, gluing=attachments)
