"""Finite Laurent-to-polynomial normalization for unimodular rows.

This is the Laurent input reduction for a constructive Quillen--Suslin
algorithm, not the missing polynomial completion algorithm itself. It uses
integer monomial automorphisms and column units only, with exponents fixed
directly from the given support. No candidate support/degree search occurs.
"""
from dataclasses import dataclass
from numbers import Integral

from .laurent import Matrix, Poly, eye, zeros
from .modules import solve
from css_decoupling.errors import ResourceLimit


def monomial_substitution(value, exponents):
    """Apply x^e -> x^(B e); columns of B are the variable images.

    This helper also supports noninvertible monomial maps. Invertibility of
    the two maps used by normalization is checked by its separate verifier.
    """
    D = value.ring.D
    if len(exponents) != D or any(len(row) != D for row in exponents):
        raise ValueError("The monomial map must have D by D exponents")
    if any(isinstance(a, bool) or not isinstance(a, Integral)
           for row in exponents for a in row):
        raise ValueError("Monomial substitution exponents must be integers")

    def convert(f):
        terms = {}
        for e, coefficient in f.terms.items():
            image = tuple(sum(int(exponents[i][j]) * e[j] for j in range(D)) for i in range(D))
            terms[image] = (terms.get(image, 0) + coefficient) % f.ring.p
        return Poly(f.ring, terms)

    if isinstance(value, Poly):
        return convert(value)
    if isinstance(value, Matrix):
        return Matrix(value.ring, value.nrows, value.ncols,
                      {ij: convert(f) for ij, f in value.entries.items()})
    raise TypeError("Expected a Laurent polynomial or matrix")


def _integer_product(A, B):
    return tuple(tuple(sum(a * b for a, b in zip(row, column)) for column in zip(*B)) for row in A)


def _checked_product(a, b, max_terms):
    # A conservative cap on actual convolution pairs, not an invented degree
    # bound. Callers may increase it; failure never means non-unimodularity.
    if max_terms is not None and len(a.terms) * len(b.terms) > max_terms:
        raise ResourceLimit("Polynomial Bezout expansion exceeds the term-product cap")
    return a * b


def _power_sum(value, n, max_terms):
    """Return value^n and sum(value^i, i<n), with logarithmic recursion."""
    if n == 0:
        return value.ring.one(), value.ring.zero()
    power, series = _power_sum(value, n // 2, max_terms)
    series = series + _checked_product(power, series, max_terms)
    power = _checked_product(power, power, max_terms)
    if n % 2:
        series = series + power
        power = _checked_product(power, value, max_terms)
    if max_terms is not None and len(series.terms) > max_terms:
        raise ResourceLimit("Polynomial Bezout expansion exceeds the term cap")
    return power, series


@dataclass
class NormalizedRow:
    original: Matrix
    row: Matrix
    change: Matrix
    change_inverse: Matrix
    exponents: tuple
    inverse_exponents: tuple
    saturation_power: int
    saturation_bezout: Matrix
    face_remainder: Poly

    def polynomial_bezout(self, *, max_terms=2000000):
        """Expand a polynomial right inverse from the compact exact witness.

        For X=prod(x_a), row[0]=1+X h and row*b=X^M, the right
        inverse is e_0 sum_{k<M}(-Xh)^k + (-h)^M b. Binary powering
        avoids an iteration through all integers up to the degree M.
        """
        r = self.row.ring
        if self.saturation_power == 0:
            return self.saturation_bezout.copy()
        X = r.monomial((1,) * r.D)
        _, series = _power_sum(-X * self.face_remainder, self.saturation_power, max_terms)
        factor, _ = _power_sum(-self.face_remainder, self.saturation_power, max_terms)
        result = Matrix(r, self.row.ncols, 1,
                        {ij: _checked_product(factor, f, max_terms)
                         for ij, f in self.saturation_bezout.entries.items()})
        result[0, 0] = result[0, 0] + series
        if self.row * result != eye(r, 1):
            raise ArithmeticError("Polynomial Bezout expansion failed its identity")
        return result

    def lift_completion(self, U, Uinv):
        """Pull a polynomial-row completion back to the original Laurent row."""
        r, n = self.row.ring, self.row.ncols
        target = eye(r, n)[:1, :]
        if U.shape != (n, n) or Uinv.shape != (n, n):
            raise ValueError("Row completion has incompatible dimensions")
        if U * Uinv != eye(r, n) or Uinv * U != eye(r, n) or self.row * U != target:
            raise ValueError("Invalid completion of the normalized row")
        V = monomial_substitution(self.change * U, self.inverse_exponents)
        Vinv = monomial_substitution(Uinv * self.change_inverse, self.inverse_exponents)
        if self.original * V != target or V * Vinv != eye(r, n) or Vinv * V != eye(r, n):
            raise ArithmeticError("Laurent pullback failed the completion identities")
        return V, Vinv


def normalize_unimodular_row(row, *, right_inverse=None, axis=0):
    """Produce a polynomial-unimodular row with an explicit Laurent pullback.

    The normalized row is (1+X*h, X*h_1,...), X=prod(x_a), with
    polynomial h,h_i. Together with row*b=X^M this is a finite certificate
    of polynomial unimodularity, over every prime field including GF(2).
    The GL column operations need not have determinant one.
    """
    if not isinstance(row, Matrix) or row.nrows != 1 or not row.ncols:
        raise ValueError("Expected a nonempty row matrix")
    r, n, D = row.ring, row.ncols, row.ring.D
    if isinstance(axis, bool) or not isinstance(axis, Integral) or not 0 <= axis < D:
        raise ValueError("Normalization axis is outside the Laurent ring")
    axis = int(axis)
    a = solve(row, eye(r, 1)) if right_inverse is None else right_inverse
    if not isinstance(a, Matrix) or a.ring != r or a.shape != (n, 1) or row * a != eye(r, 1):
        raise ValueError("The supplied Laurent Bezout column is invalid")
    first = min(j for i, j in row.entries)
    permutation = eye(r, n)
    if first:
        permutation[0, 0] = permutation[first, first] = 0
        permutation[0, first] = permutation[first, 0] = 1

    support = row[0, first].terms
    spans = [max(e[i] for e in support) - min(e[i] for e in support) for i in range(D)]
    # Mixed-radix uniqueness on the translated finite support box. Because
    # the main-axis weight is one, this is a unimodular integer shear.
    first_map = [[int(i == j) for j in range(D)] for i in range(D)]
    first_inverse = [line[:] for line in first_map]
    weight = spans[axis] + 1
    for i in range(D):
        if i != axis:
            first_map[axis][i] = weight
            first_inverse[axis][i] = -weight
            weight *= spans[i] + 1
    f = monomial_substitution(row, first_map) * permutation
    least = min(f[0, 0].terms, key=lambda e: e[axis])
    diagonal = eye(r, n)
    diagonal[0, 0] = r.monomial(least, f[0, 0].terms[least]) ** -1
    # Column units make every other nonzero entry strictly causal at once.
    for j in range(1, n):
        if f[0, j]:
            shift = 1 - min(e[axis] for e in f[0, j].terms)
            diagonal[j, j] = r.variable(axis) ** shift
    change = permutation * diagonal
    change_inverse = Matrix(r, n, n, {(j, j): diagonal[j, j] ** -1 for j in range(n)}) * permutation
    f = monomial_substitution(row, first_map) * change

    # Make all nonconstant monomials strictly positive in every coordinate.
    # Non-strict positivity would leave spurious common zeros on the faces.
    slope = 1
    for polynomial in f.entries.values():
        for exponent in polynomial.terms:
            if not any(exponent):
                continue
            if exponent[axis] <= 0:
                raise ArithmeticError("Causal normalization failed")
            for i in range(D):
                if i != axis:
                    slope = max(slope, (-exponent[i]) // exponent[axis] + 1)
    second_map = [[int(i == j) for j in range(D)] for i in range(D)]
    second_inverse = [line[:] for line in second_map]
    for i in range(D):
        if i != axis:
            second_map[i][axis] = slope
            second_inverse[i][axis] = -slope
    exponents = _integer_product(second_map, first_map)
    inverse_exponents = _integer_product(first_inverse, second_inverse)
    change = monomial_substitution(change, second_map)
    change_inverse = monomial_substitution(change_inverse, second_map)
    g = monomial_substitution(f, second_map)
    a = change_inverse * monomial_substitution(a, exponents)
    power = max([0] + [-e_i for f_i in a.entries.values() for e in f_i.terms for e_i in e])
    b = a * r.monomial((power,) * D)
    h = (g[0, 0] - 1) * r.monomial((-1,) * D)
    result = NormalizedRow(row.copy(), g, change, change_inverse, exponents,
                           inverse_exponents, power, b, h)
    from css_decoupling.certificates.normalization import verify_normalization
    verify_normalization(result)
    return result
