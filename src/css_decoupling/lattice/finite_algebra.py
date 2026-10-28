"""Small exact finite algebras used for translation relations.

Matrices are flat tuples of residues, never floating point arrays.  The
algorithms operate on vector-space bases, not on all elements of a group or
field.  This module is internal; Laurent matrices remain the public format.
"""
from math import gcd

from sympy import Matrix as ZMatrix
from sympy.matrices.normalforms import hermite_normal_form

from css_decoupling.errors import ResourceLimit


def identity(n):
    return tuple(int(i == j) for i in range(n) for j in range(n))


def add(a, b, p):
    return tuple((x + y) % p for x, y in zip(a, b))


def sub(a, b, p):
    return tuple((x - y) % p for x, y in zip(a, b))


def scale(a, c, p):
    return tuple(x * c % p for x in a)


def multiply(a, b, n, p):
    out = [0] * (n * n)
    for i in range(n):
        for k in range(n):
            c = a[i * n + k]
            if c:
                for j in range(n):
                    out[i * n + j] += c * b[k * n + j]
    return tuple(x % p for x in out)


def power(a, exponent, n, p, one=None):
    if exponent < 0:
        raise ValueError("Internal matrix power needs a nonnegative exponent")
    result = identity(n) if one is None else one
    while exponent:
        if exponent & 1:
            result = multiply(result, a, n, p)
        exponent >>= 1
        if exponent:
            a = multiply(a, a, n, p)
    return result


class Span:
    """Incremental GF(p) echelon basis with coordinates in original vectors."""

    def __init__(self, p, length, *, max_entries=2_000_000):
        self.p, self.length, self.max_entries = p, length, max_entries
        self.vectors, self.rows = [], []
        if length > max_entries:
            raise ResourceLimit("Translation-algebra matrix representation exceeds entry cap")

    def reduce(self, vector):
        vector = list(vector)
        for pivot, row, _ in self.rows:
            c = vector[pivot]
            if c:
                vector = [(x - c * y) % self.p for x, y in zip(vector, row)]
        return tuple(vector)

    def coordinates(self, vector):
        vector, out = list(vector), [0] * len(self.vectors)
        for pivot, row, expression in self.rows:
            c = vector[pivot]
            if c:
                vector = [(x - c * y) % self.p for x, y in zip(vector, row)]
                for j, value in enumerate(expression):
                    out[j] = (out[j] + c * value) % self.p
        if any(vector):
            raise ArithmeticError("Vector is outside the certified finite algebra")
        return tuple(out)

    def include(self, vector):
        if len(vector) != self.length:
            raise ValueError("Vector-space dimension mismatch")
        residual, expression = list(vector), [0] * len(self.vectors) + [1]
        for pivot, row, relation in self.rows:
            c = residual[pivot]
            if c:
                residual = [(x - c * y) % self.p for x, y in zip(residual, row)]
                for j, value in enumerate(relation):
                    expression[j] = (expression[j] - c * value) % self.p
        pivot = next((j for j, value in enumerate(residual) if value), None)
        if pivot is None:
            return False
        if (len(self.vectors) + 1) * self.length > self.max_entries:
            raise ResourceLimit("Translation-algebra basis exceeds entry cap")
        inv = pow(residual[pivot], -1, self.p)
        self.vectors.append(tuple(x % self.p for x in vector))
        self.rows.append((pivot, scale(residual, inv, self.p),
                          scale(expression, inv, self.p)))
        self.rows.sort(key=lambda row: row[0])
        return True


def linear_combination(vectors, coefficients, p, length=None):
    if length is None:
        length = len(vectors[0])
    out = [0] * length
    for vector, c in zip(vectors, coefficients):
        if c:
            for j, value in enumerate(vector):
                out[j] += c * value
    return tuple(x % p for x in out)


def rref(rows, ncols, p):
    rows = [[x % p for x in row] for row in rows if any(x % p for x in row)]
    pivots, position = [], 0
    for col in range(ncols):
        at = next((j for j in range(position, len(rows)) if rows[j][col]), None)
        if at is None:
            continue
        rows[position], rows[at] = rows[at], rows[position]
        rows[position] = list(scale(rows[position], pow(rows[position][col], -1, p), p))
        for j in range(len(rows)):
            if j != position and rows[j][col]:
                rows[j] = list(sub(rows[j], scale(rows[position], rows[j][col], p), p))
        pivots.append(col)
        position += 1
        if position == len(rows):
            break
    return rows[:position], pivots


def nullspace(rows, ncols, p):
    rows, pivots = rref(rows, ncols, p)
    result = []
    for j in range(ncols):
        if j in pivots:
            continue
        vector = [0] * ncols
        vector[j] = 1
        for k, pivot in enumerate(pivots):
            vector[pivot] = -rows[k][j] % p
        result.append(tuple(vector))
    return result


def modular_kernel(rows, D, p):
    """Column basis of {v in Z^D : rows*v = 0 mod p}."""
    rows, pivots = rref(rows, D, p)
    result = ZMatrix.eye(D)
    for pivot in pivots:
        result[pivot, pivot] = p
    for j in range(D):
        if j not in pivots:
            for k, pivot in enumerate(pivots):
                result[pivot, j] = -rows[k][j]
    return hermite_normal_form(result)


def _xgcd(a, b):
    old_r, r, old_s, s, old_t, t = a, b, 1, 0, 0, 1
    while r:
        q = old_r // r
        old_r, r = r, old_r - q * r
        old_s, s = s, old_s - q * s
        old_t, t = t, old_t - q * t
    if old_r < 0:
        return -old_r, -old_s, -old_t
    return old_r, old_s, old_t


def congruence_kernel(row, modulus):
    """An exact Z-basis for a single arbitrary-modulus congruence."""
    D, row = len(row), list(map(int, row))
    Q = ZMatrix.eye(D)
    for j in range(1, D):
        a, b = row[0], row[j]
        if not b:
            continue
        g, s, t = _xgcd(a, b)
        old0, oldj = Q[:, 0], Q[:, j]
        Q[:, 0], Q[:, j] = s * old0 + t * oldj, (-b // g) * old0 + (a // g) * oldj
        row[0], row[j] = g, 0
    Q[:, 0] *= modulus // gcd(row[0], modulus)
    return hermite_normal_form(Q)


class FiniteFieldAlgebra:
    """A simple factor already represented by matrix basis and idempotent."""

    def __init__(self, span, one, n):
        self.span, self.p, self.n = span, span.p, n
        self.d = len(span.vectors)
        self.one = span.coordinates(one)
        self.zero = (0,) * self.d
        if self.d ** 3 > span.max_entries:
            raise ResourceLimit("Finite-field multiplication table exceeds entry cap")
        self.table = tuple(tuple(span.coordinates(multiply(a, b, n, self.p))
                                 for b in span.vectors) for a in span.vectors)

    def mul(self, a, b):
        out = [0] * self.d
        for i, c in enumerate(a):
            if c:
                for j, d in enumerate(b):
                    if d:
                        for k, e in enumerate(self.table[i][j]):
                            out[k] += c * d * e
        return tuple(x % self.p for x in out)

    def pow(self, a, exponent):
        if exponent < 0:
            exponent %= self.p ** self.d - 1
        out = self.one
        while exponent:
            if exponent & 1:
                out = self.mul(out, a)
            exponent >>= 1
            if exponent:
                a = self.mul(a, a)
        return out

    def matrix(self, a):
        return linear_combination(self.span.vectors, a, self.p)
