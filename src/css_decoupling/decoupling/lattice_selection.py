"""Choose killing lattices, proving the optimum in the three-dimensional case.

The optimization scope is the supplied family, relative to its current
translation-trivial lattice and lower splitting. It is never a claim of a
globally optimal lattice for an initially unsplit higher Postnikov tower.
"""
from dataclasses import dataclass

from css_decoupling.algebra.laurent import Ring
from css_decoupling.lattice.finite_algebra import rref, nullspace
from css_decoupling.lattice.subspaces import subspace_preimage
from css_decoupling.errors import BackendUnavailable
from .gluing import GluingClass, killing_axes


@dataclass(frozen=True)
class LatticeSelection:
    classes: tuple
    D: int
    p: int
    relative_basis: tuple
    relative_index: int
    method: str
    optimality: str
    witness: dict

    def certificate(self):
        from css_decoupling.certificates.lattice_selection import selection_dict
        return selection_dict(self)


def select_lattice(classes, *, D, p, strategy="auto", max_exterior_entries=2_000_000):
    """Select a lattice in the current coordinates, without enumerating cells.

    auto: exact D=3 two-form optimum, otherwise greedy coordinate covering.
    coordinate: greedy covering, with no minimum-index claim.
    optimal: require an implemented optimum (zero family or D=3 two-forms).
    """
    if isinstance(D, bool) or isinstance(p, bool):
        raise ValueError("Dimension and characteristic must be integers")
    ring = Ring(p, D)
    D, p = ring.D, ring.p
    if strategy not in ("auto", "coordinate", "optimal"):
        raise ValueError("Lattice strategy must be auto, coordinate or optimal")
    classes = tuple(classes)
    for cls in classes:
        if not isinstance(cls, GluingClass) or cls.coefficients.ring != ring:
            raise ValueError("All gluing classes must use the declared ring")
        cls.__post_init__()
    identity = tuple(tuple(int(i == j) for j in range(D)) for i in range(D))
    if not any(not cls.is_zero for cls in classes):
        W = identity
        method, optimality = "zero-family", "family-optimal"
        witness = {"subspace_basis": [list(row) for row in W]}
    elif strategy != "coordinate" and D == 3 and all(cls.is_zero or cls.ext_degree == 2 for cls in classes):
        rows = []
        for cls in classes:
            if cls.ext_degree != 2:
                continue
            blocks = {}
            for (target, column), value in cls.coefficients.entries.items():
                exterior, source = divmod(column, cls.source_dimension)
                blocks.setdefault((target, source), [0, 0, 0])[exterior] = value.constant()
            rows.extend(blocks.values())
        rank = len(rref(rows, 3, p)[1])
        relation = []
        if rank < 3:
            relation = list(nullspace(rows, 3, p)[0])
            a, b, c = relation
            plane = nullspace([[c, -b, a]], 3, p)
            W = tuple(tuple(v[i] for v in plane) for i in range(3))
        else:
            W = ((1,), (0,), (0,))
        method, optimality = "d3-two-form", "family-optimal"
        witness = {"subspace_basis": [list(row) for row in W],
                   "coefficient_rank": rank, "kernel_vector": relation}
    else:
        if strategy == "optimal":
            raise BackendUnavailable("A minimum killing lattice is implemented for zero families and D=3 two-form families")
        axes = killing_axes(classes)
        kept = [j for j in range(D) if j not in axes]
        W = tuple(tuple(int(i == j) for j in kept) for i in range(D))
        method, optimality = "coordinate-cover", "sufficient"
        witness = {"subspace_basis": [list(row) for row in W], "axes": list(axes)}
    lattice = subspace_preimage(W, p=p, D=D)
    for cls in classes:
        if not cls.vanishes_on(W, max_exterior_entries=max_exterior_entries):
            raise ArithmeticError("Selected subspace does not kill the supplied class family")
    return LatticeSelection(classes, D, p, lattice.basis, lattice.index, method, optimality, witness)
