"""Explicit path comparison for a full-rank integer lattice (Appendix B2).

All coefficients live in the *original* Laurent ring: the new variable y_j
acts by x**basis[:,j].  This is not restriction-of-scalars coset expansion.
Only requested exterior degrees and their immediate predecessors are built.
Minors use division-free Berkowitz arithmetic, never permutations of a minor
or a search over cells.  Caps are conservative operation/storage bounds.
"""
from dataclasses import dataclass
from itertools import combinations
from math import comb
from numbers import Integral

from sympy import Matrix as IntegerMatrix

from css_decoupling.algebra.exterior import integer_rows
from css_decoupling.algebra.laurent import Matrix, Ring, eye
from css_decoupling.errors import ResourceLimit


def _cap(value, name):
    if value is not None and (isinstance(value, bool) or not isinstance(value, Integral) or value < 1):
        raise ValueError(f"{name} must be a positive integer or None")
    return value


class _Limits:
    """Shared bounded arithmetic only; no geometric/construction decisions."""

    def __init__(self, max_terms, max_entries, max_products):
        self.max_terms = _cap(max_terms, "max_terms")
        self.max_entries = _cap(max_entries, "max_entries")
        self.max_products = _cap(max_products, "max_products")
        self.products = 0

    def terms(self, count):
        if self.max_terms is not None and count > self.max_terms:
            raise ResourceLimit(f"Koszul comparison needs {count} terms; max_terms={self.max_terms}")

    def entries(self, count):
        if self.max_entries is not None and count > self.max_entries:
            raise ResourceLimit(f"Koszul comparison needs {count} entries; max_entries={self.max_entries}")

    def work(self, count):
        self.products += count
        if self.max_products is not None and self.products > self.max_products:
            raise ResourceLimit("Koszul comparison arithmetic exceeds max_products")

    def mul(self, a, b):
        count = len(a.terms) * len(b.terms)
        self.terms(count)
        self.work(count)
        return a * b

    def add(self, a, b):
        self.terms(len(a.terms) + len(b.terms))
        self.work(len(b.terms))
        return a + b


def _degrees(degrees, D):
    try:
        degrees = tuple(degrees)
    except TypeError as exc:
        raise ValueError("degrees must be an iterable of exterior degrees") from exc
    if any(isinstance(n, bool) or not isinstance(n, Integral) or not 0 <= n <= D for n in degrees):
        raise ValueError("Exterior degrees must be integers between 0 and D")
    return tuple(sorted(set(map(int, degrees))))


def _determinant(rows, ring, limits):
    """Berkowitz characteristic coefficients, iterating trailing submatrices.

    O(n^4) ring operations, no divisions and no factorial/subset expansion.
    The final coefficient is (-1)^n det.  Zero dot products remain sparse.
    """
    n = len(rows)
    limits.work(n ** 3)
    coefficients = [ring.one()]
    for start in reversed(range(n)):
        size = n - start
        toeplitz = [ring.one(), -rows[start][start]]
        vector = [rows[i][start] for i in range(start + 1, n)]
        for power in range(size - 1):
            # Sparse/zero vectors must not cause uncharged dense matrix scans.
            # Once M**power C is zero, every later Toeplitz entry is zero.
            limits.work(len(vector))
            if not any(vector):
                toeplitz.extend([ring.zero()] * (size - 1 - power))
                break
            dot = ring.zero()
            limits.work(len(vector))
            for j, value in enumerate(vector, start + 1):
                if rows[start][j] and value:
                    dot = limits.add(dot, limits.mul(rows[start][j], value))
            toeplitz.append(-dot)
            if power < size - 2:
                limits.work(len(vector) ** 2)
                following = []
                for i in range(start + 1, n):
                    value = ring.zero()
                    for j, factor in enumerate(vector, start + 1):
                        if rows[i][j] and factor:
                            value = limits.add(value, limits.mul(rows[i][j], factor))
                    following.append(value)
                vector = following
                limits.terms(sum(len(f.terms) for f in vector))
        following = []
        for i in range(size + 1):
            value = ring.zero()
            limits.work(min(i, len(coefficients) - 1) + 1)
            for j in range(min(i, len(coefficients) - 1) + 1):
                if toeplitz[i - j] and coefficients[j]:
                    value = limits.add(value, limits.mul(toeplitz[i - j], coefficients[j]))
            following.append(value)
        coefficients = following
        limits.terms(sum(len(f.terms) for f in coefficients + toeplitz))
    return (-1) ** n * coefficients[-1]


def _path_matrix(ring, basis, limits):
    # A straight coordinate segment has exactly abs(mu_i) distinct terms.
    limits.terms(sum(abs(x) for row in basis for x in row))
    entries = {}
    for j in range(ring.D):
        vertex = [0] * ring.D
        for i in range(ring.D):
            length = basis[i][j]
            terms = {}
            for step in range(abs(length)):
                exponent = vertex.copy()
                exponent[i] += step if length > 0 else -step - 1
                terms[tuple(exponent)] = 1 if length > 0 else -1
            if terms:
                entries[i, j] = ring(terms)
            vertex[i] += length
    return Matrix(ring, ring.D, ring.D, entries)


@dataclass
class KoszulComparison:
    """A finite selection of maps K'_n -> K_n over the original ring.

    ``path_map`` is chi_1. ``maps`` contains 0, each requested degree and its
    predecessor; missing degrees are never silently materialized.  Coefficient
    matrices use lexicographically ordered increasing exterior subsets.
    """

    ring: Ring
    basis: tuple
    requested_degrees: tuple
    path_map: Matrix
    maps: dict

    def chi(self, degree):
        if degree not in self.maps:
            raise ValueError("This exterior degree was not materialized; request it explicitly")
        return self.maps[degree]

    def augmentation(self, degree):
        matrix = self.chi(degree)
        return Matrix(self.ring, matrix.nrows, matrix.ncols,
                      {ij: sum(f.terms.values()) for ij, f in matrix.entries.items()})

    def certificate(self):
        from css_decoupling.certificates.koszul_comparison import koszul_comparison_dict
        return koszul_comparison_dict(self)


def koszul_comparison(ring, basis, *, degrees=(1,), max_terms=2_000_000,
                      max_entries=2_000_000, max_products=20_000_000):
    """Build the signed paths and requested exterior powers of Appendix B2.

    ``basis`` is D by D, with incoming lattice generators as columns.  It
    must be nonsingular over Z, but may be singular modulo p.  ``degrees``
    defaults to (1,), never to all 2**D exterior cells.  Immediate predecessor
    maps are included so every requested chain equation can be checked.

    Limits cover total explicit matrix shapes, stored monomials, conservative
    intermediate polynomial sizes, and monomial arithmetic work.  Long paths
    are rejected *before* expansion.  No lattice-index or coset loop occurs.
    """
    if not isinstance(ring, Ring):
        raise ValueError("ring must be a prime Laurent Ring")
    limits = _Limits(max_terms, max_entries, max_products)
    limits.entries(ring.D ** 2)
    basis = integer_rows(basis, nrows=ring.D, ncols=ring.D, name="lattice basis")
    limits.work(ring.D ** 3)
    if not IntegerMatrix(basis).det():
        raise ValueError("Lattice basis must have full integer rank")
    requested = _degrees(degrees, ring.D)
    needed = sorted({0, *requested, *(n - 1 for n in requested if n)})
    limits.entries(ring.D ** 2 + sum(comb(ring.D, n) ** 2 for n in needed))
    path = _path_matrix(ring, basis, limits)
    maps, stored = {}, sum(len(f.terms) for f in path.entries.values())
    for n in needed:
        if n == 0:
            matrix = eye(ring, 1)
        elif n == 1:
            matrix = path.copy()
        else:
            subsets = tuple(combinations(range(ring.D), n))
            entries, degree_terms = {}, 0
            for i, I in enumerate(subsets):
                for j, J in enumerate(subsets):
                    limits.work(n ** 2)
                    rows = [[path[a, b] for b in J] for a in I]
                    # A zero row or column is a structural zero minor.
                    if any(not any(row) for row in rows) or any(not any(col) for col in zip(*rows)):
                        continue
                    value = _determinant(rows, ring, limits)
                    if value:
                        entries[i, j] = value
                        degree_terms += len(value.terms)
                        limits.terms(stored + degree_terms)
            matrix = Matrix(ring, len(subsets), len(subsets), entries)
        maps[n] = matrix
        stored += sum(len(f.terms) for f in matrix.entries.values())
        limits.terms(stored)
    return KoszulComparison(ring, basis, requested, path, maps)
