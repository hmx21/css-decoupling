"""Direct, ancilla-free decoupling of an explicitly blocked mapping cone.

This optional construction implements the direct decoupling of the glued code
in Sec. V of the manuscript, written as the mapping cone of Appendix A5.  It
retains both given summands (after restriction of scalars); it does not
replace them by smaller canonical toric complexes.  The general CSS-input
pipeline remains a separate construction.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from css_decoupling.algebra.laurent import Matrix, block, eye, zeros
from css_decoupling.algebra.modules import solve, using_backend
from css_decoupling.circuits.gates import Circuit, SUM
from css_decoupling.errors import ResourceLimit
from css_decoupling.lattice.retained import Lattice
from css_decoupling.topology.complexes import (
    CochainComplex, direct_sum_complexes, solve_matrix_equations,
)

TARGET_KIND = "restricted-direct-sum-of-input-complexes"


@dataclass
class ConePresentation:
    """C^n=A^n+B^n, d_C^n=[[d_A^n,c^n],[0,d_B^n]].

    ``coupling[n]`` maps B^n to A^(n+1).  The cone condition is
    d_A^(n+1)c^n+c^(n+1)d_B^n=0.  No toric shape is required.
    """
    lower: CochainComplex
    upper: CochainComplex
    coupling: Mapping[int, Matrix]

    def __post_init__(self):
        if self.lower.ring != self.upper.ring:
            raise ValueError("Cone summands must share a Laurent ring")
        self.coupling = dict(self.coupling)
        self.lower.validate()
        self.upper.validate()
        for n, value in self.coupling.items():
            if isinstance(n, bool) or not isinstance(n, int):
                raise ValueError("Coupling degrees must be integers")
            if value.ring != self.ring or value.shape != (self.lower.rank(n + 1), self.upper.rank(n)):
                raise ValueError(f"Coupling has the wrong ring or shape at degree {n}")
        for n in self.degrees:
            if self.lower.d(n + 1) * self.c(n) + self.c(n + 1) * self.upper.d(n):
                raise ValueError(f"Cone differential does not square to zero at degree {n}")

    @property
    def ring(self):
        return self.lower.ring

    @property
    def degrees(self):
        return tuple(sorted(set(self.lower.degrees) | set(self.upper.degrees)))

    def c(self, n):
        return self.coupling.get(n, zeros(self.ring, self.lower.rank(n + 1), self.upper.rank(n)))

    def complex(self):
        A, B = self.lower, self.upper
        return CochainComplex(self.ring, {n: A.rank(n) + B.rank(n) for n in self.degrees}, {
            n: block([[A.d(n), self.c(n)],
                      [zeros(self.ring, B.rank(n + 1), A.rank(n)), B.d(n)]]) for n in self.degrees})

    def restrict(self, lattice):
        if lattice.D != self.ring.D:
            raise ValueError("Cone and restriction lattice dimensions differ")
        def restricted(C):
            return CochainComplex(C.ring, {n: r * lattice.index for n, r in C.ranks.items()},
                                  {n: lattice.restrict(d) for n, d in C.differentials.items()})
        return ConePresentation(restricted(self.lower), restricted(self.upper),
                                {n: lattice.restrict(self.c(n)) for n in self.degrees})

    @classmethod
    def from_complex(cls, C, lower_ranks):
        """Read an existing block-upper-triangular complex without changing bases."""
        if set(lower_ranks) - set(C.degrees):
            raise ValueError("Block ranks contain degrees outside the complex")
        a, b, da, db, coupling = {}, {}, {}, {}, {}
        for n in C.degrees:
            value = lower_ranks.get(n, 0)
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= C.rank(n):
                raise ValueError("Lower block ranks must be integers between zero and the full rank")
            a[n], b[n] = value, C.rank(n) - value
        for n in C.degrees:
            rows, cols = a.get(n + 1, 0), a[n]
            differential = C.d(n)
            if differential[rows:, :cols]:
                raise ValueError(f"Nonzero lower-left differential block at degree {n}")
            da[n], db[n] = differential[:rows, :cols], differential[rows:, cols:]
            coupling[n] = differential[:rows, cols:]
        return cls(CochainComplex(C.ring, a, da), CochainComplex(C.ring, b, db), coupling)


def solve_cone_homotopy(cone, *, max_unknowns=250000, solver=solve):
    """Solve all c=d_A h-h_next d_B equations simultaneously over the Laurent ring.

    Unknowns are polynomial matrix entries, never coefficients inside an
    enumerated degree box.  A normal-form nonmembership exception means this
    cone does not split on the supplied lattice.  No depth minimization is
    performed; another valid homotopy can yield a much shorter circuit.
    """
    if max_unknowns is not None and (isinstance(max_unknowns, bool) or
                                      not isinstance(max_unknowns, int) or max_unknowns < 0):
        raise ValueError("The homotopy unknown cap must be a nonnegative integer or None")
    shapes = {n: (cone.lower.rank(n), cone.upper.rank(n)) for n in cone.degrees}
    count = sum(a * b for a, b in shapes.values())
    if max_unknowns is not None and count > max_unknowns:
        raise ResourceLimit(f"Cone null-homotopy has {count} Laurent unknowns; cap is {max_unknowns}")
    equations = []
    for n in cone.degrees:
        terms = [(cone.lower.d(n), n, eye(cone.ring, cone.upper.rank(n)))]
        if n + 1 in shapes:
            terms.append((-eye(cone.ring, cone.lower.rank(n + 1)), n + 1, cone.upper.d(n)))
        equations.append((terms, cone.c(n)))
    return solve_matrix_equations(cone.ring, shapes, equations, solver)


def _normalize_homotopy(cone, homotopy):
    if any(isinstance(n, bool) or not isinstance(n, int) or n not in cone.degrees for n in homotopy):
        raise ValueError("Homotopy degrees must belong to the cone")
    h = {n: homotopy.get(n, zeros(cone.ring, cone.lower.rank(n), cone.upper.rank(n)))
         for n in cone.degrees}
    for n, value in h.items():
        if value.ring != cone.ring or value.shape != (cone.lower.rank(n), cone.upper.rank(n)):
            raise ValueError(f"Homotopy has the wrong ring or shape at degree {n}")
    for n in cone.degrees:
        following = h.get(n + 1, zeros(cone.ring, cone.lower.rank(n + 1), cone.upper.rank(n + 1)))
        if cone.lower.d(n) * h[n] - following * cone.upper.d(n) != cone.c(n):
            raise ValueError(f"Invalid cone null-homotopy at degree {n}")
    return h


def _triangular(ring, a, b, h):
    return block([[eye(ring, a), h], [zeros(ring, b, a), eye(ring, b)]])


def _schedule_disjoint(gates):
    """Linear-time conservative scheduling of translation families.

    Families in one layer have disjoint qudit species, hence no physical
    qudit occurs twice, even with displaced gates.  No optimum is claimed.
    """
    used, layers = {}, []
    for index, gate in enumerate(gates):
        layer = max(used.get(gate.control, -1), used.get(gate.target, -1)) + 1
        while len(layers) <= layer:
            layers.append([])
        layers[layer].append(index)
        used[gate.control] = used[gate.target] = layer
    return tuple(tuple(layer) for layer in layers)


@dataclass
class ConeDecouplingResult:
    presentation: ConePresentation
    lattice: Lattice
    restricted: ConePresentation
    source: CochainComplex
    target: CochainComplex
    homotopy: dict
    maps: dict
    inverse_maps: dict
    circuit: Circuit
    disjoint_layers: tuple
    homotopy_origin: str

    @property
    def U(self):
        return self.maps.get(0, eye(self.source.ring, self.source.rank(0)))

    @property
    def Uinv(self):
        return self.inverse_maps.get(0, eye(self.source.ring, self.source.rank(0)))

    @property
    def V(self):
        return self.maps.get(1, eye(self.source.ring, self.source.rank(1)))

    @property
    def Vinv(self):
        return self.inverse_maps.get(1, eye(self.source.ring, self.source.rank(1)))

    @property
    def HX(self):
        return self.source.d(0)

    @property
    def HZ(self):
        return self.source.d(-1).dagger()

    @property
    def target_kind(self):
        return TARGET_KIND

    @property
    def ancillas(self):
        return {"source": 0, "target": 0}

    @property
    def depth(self):
        return len(self.disjoint_layers)

    def certificate(self):
        from css_decoupling.certificates.cone_decoupling import cone_certificate_dict
        return cone_certificate_dict(self)


def decouple_cone(cone, *, lattice_basis=None, homotopy=None, backend=None,
                  max_cell_index=4096, max_unknowns=250000, verify=True):
    """Decouple known cone blocks with no added physical qudits.

    Supplied ``homotopy`` matrices are expressed in the restricted summand
    bases, ordered by original generator and then lattice representative.
    The default exact module solve is complete but does not optimize gates.
    """
    if backend is not None:
        with using_backend(backend):
            return decouple_cone(cone, lattice_basis=lattice_basis, homotopy=homotopy,
                                 max_cell_index=max_cell_index, max_unknowns=max_unknowns, verify=verify)
    if not isinstance(cone, ConePresentation):
        raise TypeError("Expected an explicit ConePresentation")
    if cone.ring.D < 2:
        raise ValueError("CSS cone decoupling requires dimension D >= 2")
    # Recheck mutable matrices instead of trusting an earlier constructor.
    cone = ConePresentation(cone.lower, cone.upper, cone.coupling)
    basis = lattice_basis if lattice_basis is not None else [
        [int(i == j) for j in range(cone.ring.D)] for i in range(cone.ring.D)]
    lattice = Lattice(basis, max_index=max_cell_index)
    restricted = cone.restrict(lattice)
    origin = "supplied" if homotopy is not None else "exact-Laurent-module-solve"
    if homotopy is None:
        homotopy = solve_cone_homotopy(restricted, max_unknowns=max_unknowns)
    h = _normalize_homotopy(restricted, homotopy)
    A, B = restricted.lower, restricted.upper
    maps = {n: _triangular(cone.ring, A.rank(n), B.rank(n), h[n]) for n in restricted.degrees}
    inverses = {n: _triangular(cone.ring, A.rank(n), B.rank(n), -h[n]) for n in restricted.degrees}
    h0 = h.get(0, zeros(cone.ring, A.rank(0), B.rank(0)))
    gates = tuple(SUM(i, A.rank(0) + j, (-coefficient) % cone.ring.p, exponent)
                  for (i, j), polynomial in sorted(h0.entries.items())
                  for exponent, coefficient in sorted(polynomial.terms.items()))
    result = ConeDecouplingResult(
        cone, lattice, restricted, restricted.complex(), direct_sum_complexes(A, B), h,
        maps, inverses, Circuit(cone.ring, A.rank(0) + B.rank(0), gates),
        _schedule_disjoint(gates), origin)
    if verify:
        from css_decoupling.certificates.cone_decoupling import verify_cone_decoupling
        report = verify_cone_decoupling(result)
        if not report.valid:
            raise ArithmeticError("Direct cone certificate failed: " + "; ".join(report.errors))
    return result
