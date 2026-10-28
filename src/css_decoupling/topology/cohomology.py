"""Finite Laurent quotients and cohomology with exact representatives."""
from collections import deque
from dataclasses import dataclass

from css_decoupling.algebra.laurent import Matrix, eye, zeros
from css_decoupling.algebra.modules import get_backend, kernel, solve
from css_decoupling.errors import ResourceLimit
from css_decoupling.errors import InfiniteLayer


def divides(a, b):
    return a[0] == b[0] and all(x <= y for x, y in zip(a[1:], b[1:]))


class FiniteQuotient:
    """Basis of coker(A), obtained by its monomial staircase.

    Breadth-first traversal visits O((D+1)*dimension) neighbors, rather than
    all points of a potentially exponentially larger rectangular bounding box.
    A Groebner pure-power criterion decides finiteness *before* traversal.
    """

    def __init__(self, A, *, max_dimension=10000, backend=None):
        self.presentation = A
        self.ring = A.ring
        self.backend = backend or get_backend(A.ring)
        data = self.backend.quotient_groebner(A)
        self.leading = tuple(tuple(t) for t in data["leading_terms"])
        self.groebner_data = data
        D = A.ring.D
        seed = []
        for component in range(A.nrows):
            base = (component,) + (0,) * (D + 1)
            leads = [t for t in self.leading if t[0] == component]
            if any(divides(t, base) for t in leads):
                continue
            for axis in range(D + 1):
                if not any(t[axis + 1] and all(not t[k + 1] for k in range(D + 1) if k != axis)
                           for t in leads):
                    raise InfiniteLayer(f"component {component}, localization variable {axis}: no pure-power leading relation")
            seed.append(base)
        pending, seen, standard = deque(seed), set(seed), []
        while pending:
            term = pending.popleft()
            if any(divides(t, term) for t in self.leading):
                continue
            standard.append(term)
            if max_dimension is not None and len(standard) > max_dimension:
                raise ResourceLimit(f"Finite layer dimension exceeds {max_dimension}")
            for axis in range(D + 1):
                neighbor = list(term)
                neighbor[axis + 1] += 1
                neighbor = tuple(neighbor)
                if neighbor not in seen:
                    seen.add(neighbor)
                    pending.append(neighbor)
        self.standard = tuple(sorted(standard))
        self.lookup = {t: i for i, t in enumerate(self.standard)}
        self.dimension = len(self.standard)
        self.basis = zeros(A.ring, A.nrows, self.dimension)
        for j, term in enumerate(self.standard):
            self.basis[term[0], j] = A.ring.monomial(tuple(e - term[-1] for e in term[1:-1]))
        self.actions = tuple(self.coordinates(A.ring.variable(axis) * self.basis) for axis in range(D))
        if self.coordinates(self.basis) != eye(A.ring, self.dimension):
            raise ArithmeticError("Finite quotient basis failed its coordinate check")
        if self.coordinates(A) != zeros(A.ring, self.dimension, A.ncols):
            raise ArithmeticError("Presentation relations have nonzero quotient coordinates")
        for axis, action in enumerate(self.actions):
            inverse = self.coordinates(A.ring.variable(axis) ** -1 * self.basis)
            if action * inverse != eye(A.ring, self.dimension):
                raise ArithmeticError("Quotient translation is not invertible")

    def coordinates(self, vectors):
        out = zeros(self.ring, self.dimension, vectors.ncols)
        for j, normal in enumerate(self.backend.reduce_quotient(self.presentation, vectors)):
            for term, coefficient in normal:
                if tuple(term) not in self.lookup:
                    raise ArithmeticError("Normal form escaped the standard-monomial basis")
                out[self.lookup[tuple(term)], j] = coefficient
        return out


@dataclass
class Layer:
    degree: int
    cycles: Matrix
    presentation: Matrix
    quotient: FiniteQuotient
    representatives: Matrix
    boundary_coordinates: Matrix
    cycle_relations: Matrix

    @property
    def dimension(self):
        return self.quotient.dimension

    @property
    def actions(self):
        return self.quotient.actions


def cohomology_layer(C, degree, *, max_dimension=10000):
    """ker(d^j)/im(d^(j-1)), including relations among kernel generators."""
    cycles = kernel(C.d(degree))
    boundary_coordinates = solve(cycles, C.d(degree - 1))
    relations = kernel(cycles)
    presentation = boundary_coordinates.hstack(relations)
    quotient = FiniteQuotient(presentation, max_dimension=max_dimension)
    return Layer(degree, cycles, presentation, quotient, cycles * quotient.basis,
                 boundary_coordinates, relations)


def profile(C, *, max_dimension=10000):
    result = {}
    for degree in C.degrees:
        if degree <= 0:
            continue
        layer = cohomology_layer(C, degree, max_dimension=max_dimension)
        if layer.dimension:
            if degree >= C.ring.D:
                raise ValueError("Nonzero layer outside the topological CSS degree bound")
            result[degree] = layer
    return result
