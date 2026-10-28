"""Exact Euclidean completion of rows over one Laurent variable.

This is the complete one-variable base case, not a multivariate
Quillen--Suslin algorithm. All coefficients use Python integers modulo the
prime; polynomial factorization, support search and subprocesses are absent.
"""
from __future__ import annotations

from numbers import Integral

from css_decoupling.algebra.laurent import Matrix, eye
from css_decoupling.errors import BackendUnavailable, NoLaurentSolution


class UnivariateCompletionUnavailable(BackendUnavailable):
    """An input depends on variables beyond the specified univariate axis."""


class NoPolynomialSolution(ValueError):
    """An exact nonconstant gcd obstructs completion over F_p[t]."""


def _add(a, b, p, scale=1):
    out = a.copy()
    for degree, value in b.items():
        coefficient = (out.get(degree, 0) + scale * value) % p
        if coefficient:
            out[degree] = coefficient
        else:
            out.pop(degree, None)
    return out


def _multiply(a, b, p):
    out = {}
    for degree_a, value_a in a.items():
        for degree_b, value_b in b.items():
            degree = degree_a + degree_b
            value = (out.get(degree, 0) + value_a * value_b) % p
            if value:
                out[degree] = value
            else:
                out.pop(degree, None)
    return out


def _scale(a, coefficient, p):
    coefficient %= p
    return {degree: value * coefficient % p for degree, value in a.items()} if coefficient else {}


def _divide(a, b, p):
    """Exact finite-field polynomial division, with strictly decreasing degree."""
    if not b:
        raise ZeroDivisionError("Polynomial divisor is zero")
    remainder, quotient = a.copy(), {}
    divisor_degree = max(b)
    inverse = pow(b[divisor_degree], -1, p)
    while remainder and max(remainder) >= divisor_degree:
        degree = max(remainder) - divisor_degree
        coefficient = remainder[max(remainder)] * inverse % p
        quotient[degree] = (quotient.get(degree, 0) + coefficient) % p
        remainder = _add(remainder, {degree + k: v for k, v in b.items()}, p, -coefficient)
    return {k: v for k, v in quotient.items() if v}, remainder


def _gcdex(a, b, p):
    """Return monic g,s,t with s*a+t*b=g using the Euclidean algorithm."""
    old_r, r = a.copy(), b.copy()
    old_s, s, old_t, t = {0: 1}, {}, {}, {0: 1}
    while r:
        quotient, remainder = _divide(old_r, r, p)
        old_r, r = r, remainder
        old_s, s = s, _add(old_s, _multiply(quotient, s, p), p, -1)
        old_t, t = t, _add(old_t, _multiply(quotient, t, p), p, -1)
    if not old_r:
        return {}, {}, {}
    inverse = pow(old_r[max(old_r)], -1, p)
    return tuple(_scale(f, inverse, p) for f in (old_r, old_s, old_t))


def _embed(ring, polynomial, axis):
    return ring({tuple(degree if i == axis else 0 for i in range(ring.D)): coefficient
                 for degree, coefficient in polynomial.items()})


def _column_pair(U, column, a, b, c, d):
    """Right multiply columns (0,column) by [[a,b],[c,d]]."""
    rows = {i for i, j in U.entries if j in (0, column)}
    for i in rows:
        left, right = U[i, 0], U[i, column]
        U[i, 0], U[i, column] = left * a + right * c, left * b + right * d


def _row_pair(Uinv, row, a, b, c, d):
    """Left multiply rows (0,row) by [[a,b],[c,d]]."""
    columns = {j for i, j in Uinv.entries if i in (0, row)}
    for j in columns:
        top, bottom = Uinv[0, j], Uinv[row, j]
        Uinv[0, j], Uinv[row, j] = a * top + b * bottom, c * top + d * bottom


def complete_unimodular_row(row, *, axis=0, polynomial=False):
    """Return U,Uinv with row*U=(1,0,...), U*Uinv=Uinv*U=I.

    ``row`` is a 1-by-n Laurent Matrix and n can be arbitrary. It may belong
    to a higher-dimensional ambient Ring, but every entry must depend only
    on ``axis``. A Laurent unit is first removed independently from each
    nonzero column. With ``polynomial=True``, negative exponents are rejected
    and no monomial factors are removed; both output matrices are over
    F_p[t] itself. Extended Euclid then combines the current first entry
    with each remaining entry using a determinant-one two-column matrix.

    A nonunit gcd proves that the row is not unimodular and raises
    NoLaurentSolution (or NoPolynomialSolution in polynomial mode).
    Dependence on other axes raises the distinct
    UnivariateCompletionUnavailable; neither outcome is presented as a
    multivariate nonexistence result. All returned matrix identities are
    checked in the original Laurent ring, including odd-characteristic signs.
    """
    if not isinstance(row, Matrix) or row.nrows != 1:
        raise ValueError("Univariate completion requires a one-row Laurent matrix")
    if not isinstance(polynomial, bool):
        raise TypeError("polynomial must be a boolean")
    ring, n = row.ring, row.ncols
    if isinstance(axis, bool) or not isinstance(axis, Integral) or not 0 <= axis < ring.D:
        raise ValueError("Completion axis must be an integer within the ring dimension")
    axis = int(axis)
    U, Uinv, polynomials = eye(ring, n), eye(ring, n), []
    for j in range(n):
        f = row[0, j]
        if any(any(e for i, e in enumerate(exponent) if i != axis) for exponent in f.terms):
            raise UnivariateCompletionUnavailable(
                f"Row entry {j} depends on another axis; only F_p[x_{axis + 1}^±1] completion is implemented")
        minimum = min((e[axis] for e in f.terms), default=0)
        if polynomial:
            if minimum < 0:
                raise ValueError("Polynomial completion does not accept negative exponents")
            minimum = 0
        polynomials.append({e[axis] - minimum: c for e, c in f.terms.items()})
        exponent = tuple(-minimum if i == axis else 0 for i in range(ring.D))
        U[j, j], Uinv[j, j] = ring.monomial(exponent), ring.monomial(tuple(-e for e in exponent))
    if not n or not any(polynomials):
        error = NoPolynomialSolution if polynomial else NoLaurentSolution
        raise error("The zero row is not unimodular")
    current = polynomials[0]
    for j in range(1, n):
        following = polynomials[j]
        if not following:
            continue
        gcd, s, t = _gcdex(current, following, ring.p)
        a_over_g, a_remainder = _divide(current, gcd, ring.p)
        b_over_g, b_remainder = _divide(following, gcd, ring.p)
        if a_remainder or b_remainder or _add(_multiply(s, current, ring.p),
                                            _multiply(t, following, ring.p), ring.p) != gcd:
            raise ArithmeticError("Extended Euclidean certificate failed")
        s, t, ag, bg = (_embed(ring, f, axis) for f in (s, t, a_over_g, b_over_g))
        # [a,b] [[s,-b/g],[t,a/g]] = [g,0], determinant exactly one.
        _column_pair(U, j, s, -bg, t, ag)
        _row_pair(Uinv, j, ag, bg, -t, s)
        current = gcd
    # In Laurent mode, column normalization removes every t factor. In
    # polynomial mode, a remaining t factor is a genuine obstruction.
    if not current or set(current) != {0}:
        if polynomial:
            raise NoPolynomialSolution("The row has a nonconstant polynomial gcd and is not unimodular over F_p[t]")
        raise NoLaurentSolution("The row has a nonunit Laurent gcd and is not unimodular")
    coefficient = current[0]
    if coefficient != 1:
        inverse = pow(coefficient, -1, ring.p)
        for i in {i for i, j in U.entries if j == 0}:
            U[i, 0] = inverse * U[i, 0]
        for j in {j for i, j in Uinv.entries if i == 0}:
            Uinv[0, j] = coefficient * Uinv[0, j]
    target = Matrix(ring, 1, n, {(0, 0): 1})
    identity = eye(ring, n)
    if row * U != target or U * Uinv != identity or Uinv * U != identity:
        raise ArithmeticError("Univariate row-completion identities failed")
    if polynomial and any(e < 0 for matrix in (U, Uinv) for f in matrix.entries.values()
                          for exponent in f.terms for e in exponent):
        raise ArithmeticError("Polynomial completion returned a negative exponent")
    return U, Uinv
