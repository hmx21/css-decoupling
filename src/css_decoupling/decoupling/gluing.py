"""Postnikov gluing classes relative to an explicit splitting of lower layers.

For Phi:T_lower -> C, lift the j-th Koszul resolution into Cone(Phi).
Projection to T_lower[1] is the connecting map. Its augmentation in each
lower top degree gives kappa_ji in Hom_F(E_j,E_i) tensor exterior^(j-i+1).
These are conditional on the recorded lower splitting, never simultaneous
pairwise invariants of an unsplit lower Postnikov tower.
"""
from dataclasses import dataclass
from itertools import combinations
from math import comb
from numbers import Integral

from css_decoupling.algebra.laurent import Matrix, eye, zeros
from css_decoupling.algebra.exterior import integer_rows, pullback_coefficients, subset_at
from css_decoupling.algebra.modules import solve
from css_decoupling.errors import ResourceLimit
from css_decoupling.topology.cohomology import cohomology_layer
from css_decoupling.topology.complexes import (
    CochainComplex, CochainMap, koszul_window, mapping_cone,
)
from .splitting import splitting_map


def augmentation(A):
    """Evaluate every Laurent variable at 1, with exact prime-field signs."""
    return Matrix(A.ring, A.nrows, A.ncols,
                  {ij: sum(f.terms.values()) % A.ring.p for ij, f in A.entries.items()})


@dataclass
class GluingClass:
    upper: int
    lower: int
    source_dimension: int
    target_dimension: int
    coefficients: Matrix

    def __post_init__(self):
        for name in ("upper", "lower", "source_dimension", "target_dimension"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError("Gluing degrees and dimensions must be integers")
        if self.lower < 1 or self.upper <= self.lower:
            raise ValueError("A gluing class requires 1 <= lower < upper")
        if self.source_dimension < 0 or self.target_dimension < 0:
            raise ValueError("Layer dimensions cannot be negative")
        if not isinstance(self.coefficients, Matrix):
            raise ValueError("Gluing coefficients must be a Matrix")
        expected = (self.target_dimension,
                    comb(self.coefficients.ring.D, self.ext_degree) * self.source_dimension)
        if self.coefficients.shape != expected or any(not f.is_constant for f in self.coefficients.entries.values()):
            raise ValueError("Gluing coefficients must be a constant table of the declared shape")

    def evaluate(self, vectors, *, max_exterior_entries=2_000_000):
        """Evaluate on a sequence of n D-dimensional vectors, in that order."""
        vectors = integer_rows(vectors, nrows=self.ext_degree,
                               ncols=self.coefficients.ring.D, name="evaluation vectors")
        columns = tuple(tuple(vector[i] for vector in vectors)
                        for i in range(self.coefficients.ring.D))
        return pullback_coefficients(self.coefficients, self.ext_degree, self.source_dimension,
                                     columns, max_exterior_entries=max_exterior_entries)

    def restrict(self, M, *, max_exterior_entries=2_000_000):
        """Restrict along D by D integer columns: new translations in old basis."""
        D = self.coefficients.ring.D
        M = integer_rows(M, nrows=D, ncols=D, name="restriction matrix")
        coefficients = pullback_coefficients(self.coefficients, self.ext_degree, self.source_dimension,
                                             M, max_exterior_entries=max_exterior_entries)
        return GluingClass(self.upper, self.lower, self.source_dimension,
                           self.target_dimension, coefficients)

    def vanishes_on(self, W, *, max_exterior_entries=2_000_000):
        """Test a subspace given by a D by r column spanning table."""
        return not bool(pullback_coefficients(self.coefficients, self.ext_degree, self.source_dimension,
                        W, max_exterior_entries=max_exterior_entries))

    @property
    def ext_degree(self):
        return self.upper - self.lower + 1

    @property
    def exterior_basis(self):
        return tuple(combinations(range(self.coefficients.ring.D), self.ext_degree))

    @property
    def is_zero(self):
        return not bool(self.coefficients)

    def terms(self):
        """Sparse (axes, target layer index, source layer index, coefficient)."""
        blocks = {column // self.source_dimension for _, column in self.coefficients.entries}
        basis = {block: subset_at(block, self.coefficients.ring.D, self.ext_degree) for block in blocks}
        return tuple((basis[column // self.source_dimension], row,
                      column % self.source_dimension, f.constant())
                     for (row, column), f in sorted(self.coefficients.entries.items()))


@dataclass
class GluingStep:
    upper: int
    representatives: dict
    lower_splitting: CochainMap
    cone_lift: CochainMap
    classes: tuple[GluingClass, ...]

    @property
    def is_zero(self):
        return all(k.is_zero for k in self.classes)

    def certificate(self):
        from css_decoupling.certificates.gluing import gluing_dict
        return gluing_dict(self)


def classes_from_lift(upper, representatives, lower_splitting, cone_lift):
    """Extract the augmentation coefficients from an already constructed lift."""
    C = lower_splitting.target
    degrees = sorted(i for i in representatives if i < upper)
    pieces = {i: koszul_window(C.ring, i, representatives[i].ncols, lowest_degree=0)
              for i in degrees}
    answer = []
    for i in degrees:
        offset = C.rank(i - 1) + sum(pieces[k].rank(i) for k in degrees if k < i)
        ti, tj = representatives[i].ncols, representatives[upper].ncols
        projected = cone_lift.at(i - 1)[offset:offset + ti, :]
        coefficients = augmentation(projected)
        if coefficients.shape != (ti, comb(C.ring.D, upper - i + 1) * tj):
            raise ArithmeticError("Connecting map has an incompatible exterior basis")
        answer.append(GluingClass(upper, i, tj, ti, coefficients))
    return tuple(answer)


def gluing_classes(C, representatives, upper, *, max_unknowns=250000, max_layer_dimension=10000):
    """Compute every kappa_(upper,i), requiring lower layers already split.

    ``representatives`` supplies cohomology bases in the current retained
    lattice. Failure to split the lower layers is propagated; it is not a
    value for a higher pairwise class. Only nonnegative Koszul terms are
    needed because i>=1 implies the cocycle is read in degree i-1>=0.
    """
    if upper not in representatives or upper < 1:
        raise ValueError("An upper layer and its basis must be supplied")
    representatives = {i: r for i, r in representatives.items() if i <= upper}
    lower = {i: r for i, r in representatives.items() if i < upper}
    if not lower:
        raise ValueError("The first nonzero layer has no lower gluing components")
    C = CochainComplex(C.ring, {n: r for n, r in C.ranks.items() if n >= 0},
                       {n: d for n, d in C.differentials.items() if n >= 0})
    # The independent API must not silently omit an unsplit intermediate
    # layer. Validate all positive layers through upper, not just those the
    # caller happened to supply. This also establishes the lifting premise.
    for degree in range(1, upper + 1):
        layer = cohomology_layer(C, degree, max_dimension=max_layer_dimension)
        if not layer.dimension:
            if degree in representatives:
                raise ValueError("A zero layer was supplied as nonzero")
            continue
        if degree not in representatives or representatives[degree].shape != (C.rank(degree), layer.dimension):
            raise ValueError("Representatives must include every nonzero layer through upper degree")
        reps = representatives[degree]
        if C.d(degree) * reps:
            raise ValueError("Representatives must be cycles")
        coordinates = layer.quotient.coordinates(solve(layer.cycles, reps))
        solve(coordinates, eye(C.ring, layer.dimension))
        if any(a != eye(C.ring, layer.dimension) for a in layer.actions):
            raise ValueError("Gluing coordinates require trivial translation action on all layers")
    phi = splitting_map(C, lower, max_unknowns=max_unknowns, window=True)
    cone = mapping_cone(phi)
    top = representatives[upper]
    source = koszul_window(C.ring, upper, top.ncols, lowest_degree=0)
    size = sum(cone.rank(n) * source.rank(n) for n in source.degrees)
    if max_unknowns is not None and size > max_unknowns:
        raise ResourceLimit(f"Gluing lift has {size} Laurent entries; cap is {max_unknowns}")
    if top.nrows != C.rank(upper) or C.d(upper) * top:
        raise ValueError("Upper representatives must be cycles")
    components = {upper: top.vstack(zeros(C.ring, phi.source.rank(upper + 1), top.ncols))}
    # H^n(Cone(phi)) vanishes for 1 <= n < upper, while the upper layer
    # is annihilated by x_a-1. Thus every RHS here is a boundary. Greedy
    # lifting is valid in this cone, unlike lifting into C with lower layers.
    for n in range(upper - 1, -1, -1):
        components[n] = solve(cone.d(n), components[n + 1] * source.d(n))
    lift = CochainMap(source, cone, components)
    classes = classes_from_lift(upper, representatives, phi, lift)
    return GluingStep(upper, representatives, phi, lift, classes)


def killing_axes(classes):
    """Greedy coordinate cover killing all displayed exterior coefficients.

    A p-blocking of axis a sets its Ext-one restriction to zero. Covering
    every nonzero exterior support therefore kills every class. The greedy
    cover is polynomial in the explicit cocycle output; no minimum-index
    claim and no exhaustive search over subsets or lattices are made.
    """
    supports = {axes for cls in classes for axes, _, _, _ in cls.terms()}
    selected = []
    while supports:
        counts = {}
        for support in supports:
            for axis in support:
                counts[axis] = counts.get(axis, 0) + 1
        axis = min(counts, key=lambda a: (-counts[a], a))
        selected.append(axis)
        supports = {s for s in supports if axis not in s}
    return tuple(sorted(selected))
