"""Restriction of scalars for arbitrary full-rank integer lattices.

Coset enumeration is proportional to the *output* unit-cell index.  It is
never a search through a translation group for a relation.
"""
from itertools import product
from math import lcm
from numbers import Integral

from sympy import Matrix as IntegerMatrix, Poly as SymPoly, symbols
from sympy.matrices.normalforms import hermite_normal_form

from css_decoupling.algebra.laurent import Matrix, Poly, eye
from css_decoupling.errors import ResourceLimit


class Lattice:
    """Columns of ``basis`` are retained translations in the old coordinates."""

    def __init__(self, basis, *, max_index=4096, representatives=None):
        rows = tuple(tuple(row) for row in basis)
        if not rows or any(len(row) != len(rows) for row in rows):
            raise ValueError("Lattice basis must be square")
        if any(isinstance(x, bool) or not isinstance(x, Integral) for row in rows for x in row):
            raise ValueError("Lattice basis must have integer entries")
        if max_index is not None and (isinstance(max_index, bool) or
                                      not isinstance(max_index, Integral) or max_index < 1):
            raise ValueError("Lattice index cap must be a positive integer or None")
        self.D = len(rows)
        self.B = IntegerMatrix(rows)
        determinant = int(self.B.det())
        if not determinant:
            raise ValueError("Lattice must have full rank")
        self.index = abs(determinant)
        if max_index is not None and self.index > max_index:
            raise ResourceLimit(f"Coarse unit cell has index {self.index}; cap is {max_index}")
        self.basis = rows
        self.H = hermite_normal_form(self.B)
        self.inverse = self.B.inv()
        self._radices = tuple(int(self.H[i, i]) for i in range(self.D))
        self._representatives = None
        self._lookup = None
        self._explicit_indices = None
        self._explicit_shifts = None
        if representatives is not None:
            supplied, indices, shifts = [], {}, {}
            for position, representative in enumerate(representatives):
                if position >= self.index:
                    raise ValueError('Too many explicit lattice representatives')
                representative = tuple(representative)
                if (len(representative) != self.D or
                        any(isinstance(value, bool) or not isinstance(value, Integral)
                            for value in representative)):
                    raise ValueError('Explicit representatives must be integer coordinate vectors')
                representative = tuple(map(int, representative))
                hnf_index, shift = self._reduce_hnf(representative)
                if hnf_index in indices:
                    raise ValueError('Explicit representatives repeat a lattice coset')
                supplied.append(representative)
                indices[hnf_index], shifts[hnf_index] = position, shift
            if len(supplied) != self.index:
                raise ValueError('Explicit representatives must cover every lattice coset')
            self._representatives = tuple(supplied)
            self._explicit_indices, self._explicit_shifts = indices, shifts

    @property
    def representatives(self):
        """Materialize output-cell representatives only when they are requested."""
        if self._representatives is None:
            self._representatives = tuple(product(*(range(n) for n in self._radices)))
        return self._representatives

    @property
    def lookup(self):
        """Backward-compatible explicit coset table, allocated on demand."""
        if self._lookup is None:
            self._lookup = {rep: i for i, rep in enumerate(self.representatives)}
        return self._lookup

    def relative_basis(self, child):
        """Coordinates of a sublattice's columns; reject noncontainment."""
        child = child if isinstance(child, Lattice) else Lattice(child, max_index=None)
        if child.D != self.D:
            raise ValueError("Lattice dimensions differ")
        relative = self.inverse * child.B
        if any(x.q != 1 for x in relative):
            raise ValueError("The child lattice is not contained in this lattice")
        return tuple(tuple(int(relative[i, j]) for j in range(self.D)) for i in range(self.D))

    def contains(self, child):
        """Whether another full-rank lattice is contained, without coset work."""
        child = child if isinstance(child, Lattice) else Lattice(child, max_index=None)
        if child.D != self.D:
            raise ValueError("Lattice dimensions differ")
        return all(x.q == 1 for x in self.inverse * child.B)

    def _reduce_hnf(self, exponent):
        """Legacy HNF index and shift, independent of selected representatives."""
        if len(exponent) != self.D:
            raise ValueError("Exponent dimension differs from lattice")
        if any(isinstance(e, bool) or not isinstance(e, Integral) for e in exponent):
            raise ValueError("Lattice exponents must be integers")
        rem = list(map(int, exponent))
        for j in reversed(range(self.D)):
            quotient = rem[j] // int(self.H[j, j])
            for i in range(j + 1):
                rem[i] -= quotient * int(self.H[i, j])
        residual = IntegerMatrix([int(exponent[i]) - rem[i] for i in range(self.D)])
        shift = self.inverse * residual
        if any(s.q != 1 for s in shift):
            raise ArithmeticError("Coset reduction did not produce an integral lattice coordinate")
        index = 0
        for digit, radix in zip(rem, self._radices):
            index = index * radix + digit
        return index, tuple(map(int, shift))

    def reduce(self, exponent):
        """Selected coset index and exact retained-lattice coordinate shift."""
        index, shift = self._reduce_hnf(exponent)
        if self._explicit_indices is None:
            return index, shift
        representative_shift = self._explicit_shifts[index]
        return self._explicit_indices[index], tuple(a-b for a, b in zip(shift, representative_shift))

    def restrict(self, A):
        if A.ring.D != self.D:
            raise ValueError("Ring and lattice dimensions differ")
        entries, coordinates, expansions = {}, {}, {}
        p = A.ring.p
        for (i, j), f in A.entries.items():
            for exponent, coefficient in f.terms.items():
                # Repeated polynomial supports occur throughout check and
                # homotopy matrices. Geometry depends only on an exponent,
                # not on the matrix position or its field coefficient.
                expanded = expansions.get(exponent)
                if expanded is None:
                    expanded = []
                    for rep in self.representatives:
                        translated = tuple(a + b for a, b in zip(exponent, rep))
                        coordinate = coordinates.get(translated)
                        if coordinate is None:
                            coordinate = coordinates[translated] = self.reduce(translated)
                        expanded.append(coordinate)
                    expanded = expansions[exponent] = tuple(expanded)
                for col, (row, shift) in enumerate(expanded):
                    at = (i * self.index + row, j * self.index + col)
                    target = entries.setdefault(at, {})
                    target[shift] = (target.get(shift, 0) + coefficient) % p
        return Matrix(A.ring, A.nrows * self.index, A.ncols * self.index,
                      {at: Poly(A.ring, terms) for at, terms in entries.items()})

    def pack(self, A):
        """Coordinates of individual old-ring columns, without coset replication."""
        if A.ring.D != self.D:
            raise ValueError("Ring and lattice dimensions differ")
        entries, coordinates = {}, {}
        p = A.ring.p
        for (i, j), f in A.entries.items():
            for exponent, coefficient in f.terms.items():
                coordinate = coordinates.get(exponent)
                if coordinate is None:
                    coordinate = coordinates[exponent] = self.reduce(exponent)
                row, shift = coordinate
                at = (i * self.index + row, j)
                target = entries.setdefault(at, {})
                target[shift] = (target.get(shift, 0) + coefficient) % p
        # Construct each immutable polynomial once, rather than repeatedly
        # copying its accumulated support for every added monomial. These
        # caches are local to this call, so no lattice mutation is retained.
        return Matrix(A.ring, A.nrows * self.index, A.ncols,
                      {at: Poly(A.ring, terms) for at, terms in entries.items()})

    def metadata(self):
        answer = {"basis_columns": [list(r) for r in self.basis], "index": self.index,
                  "representatives": [list(r) for r in self.representatives]}
        if self._explicit_indices is not None:
            answer['representative_strategy'] = 'explicit-v1'
        return answer


def matrix_power(A, exponent):
    if isinstance(exponent, bool) or not isinstance(exponent, Integral):
        raise ValueError("Matrix exponent must be an integer")
    exponent = int(exponent)
    if A.nrows != A.ncols:
        raise ValueError("Matrix powers require a square matrix")
    if exponent < 0:
        from css_decoupling.algebra.modules import solve
        A = solve(A, eye(A.ring, A.nrows))
        exponent = -exponent
    out = eye(A.ring, A.nrows)
    while exponent:
        if exponent & 1:
            out = out * A
        exponent >>= 1
        if exponent:
            A = A * A
    return out


def action_order_multiple(A):
    """Certified order multiple from characteristic-polynomial factor degrees.

    Avoids both element enumeration and discrete logarithms.  The answer need
    not be the minimal order.  Exponentiation verifies it independently.
    """
    if A == eye(A.ring, A.nrows):
        return 1
    if A.nrows != A.ncols:
        raise ValueError("Translation action must be square")
    zero = (0,) * A.ring.D
    if any(any(e != zero for e in f.terms) for f in A.entries.values()):
        raise ValueError("Translation actions must be constant finite-field matrices")
    dense = IntegerMatrix([[int(A[i, j].terms.get(zero, 0)) for j in range(A.ncols)]
                           for i in range(A.nrows)])
    if int(dense.det()) % A.ring.p == 0:
        raise ValueError("Translation action is singular")
    variable = symbols("s")
    characteristic = SymPoly(dense.charpoly(variable).as_expr(), variable, modulus=A.ring.p)
    _, factors = characteristic.factor_list()
    semisimple = lcm(*(A.ring.p ** f.degree() - 1 for f, _ in factors))
    unipotent = 1
    largest = max(e for _, e in factors)
    while unipotent < largest:
        unipotent *= A.ring.p
    bound = semisimple * unipotent
    if matrix_power(A, bound) != eye(A.ring, A.nrows):
        raise ArithmeticError("Characteristic-polynomial order bound failed")
    return bound


def trivializing_lattice(layers, D, *, max_index=4096):
    """A certified rectangular sublattice of Lambda_0, not necessarily Lambda_0.

    Computing all mixed relations may require discrete logarithms.  We make
    no maximal-symmetry claim and do not enumerate the finite image group.
    """
    periods = [1] * D
    for layer in layers.values():
        for axis, action in enumerate(layer.actions):
            periods[axis] = lcm(periods[axis], action_order_multiple(action))
    return Lattice([[periods[i] if i == j else 0 for j in range(D)] for i in range(D)],
                   max_index=max_index)
