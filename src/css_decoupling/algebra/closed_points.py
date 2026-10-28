"""Closed points of prime-field affine ideals without point enumeration.

A fixed triangular monic shear and elimination reduce the variable count.
Lying-over ensures that each chosen lower-dimensional point has a nonempty
finite fiber. Univariate gcd and certified factorization select a residue
field extension in that fiber. All multivariate Groebner arithmetic stays
over F_p; extension fields occur only in univariate fiber computations.

The returned evaluation map suffices to test units in the corresponding
local ring. No generators of its maximal-ideal kernel are required.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

from css_decoupling.algebra.laurent import Matrix, Poly, Ring
from css_decoupling.algebra.residue_fields import (
    ExtensionField, PrimeField, irreducible_factor, poly_gcd,
)
from css_decoupling.algebra.schreyer import _Budget, _groebner
from css_decoupling.errors import ResourceLimit


def _axes(ring, axes):
    axes = tuple(range(ring.D)) if axes is None else tuple(axes)
    if any(isinstance(a, bool) or not isinstance(a, Integral) for a in axes):
        raise TypeError("Polynomial axes must be integer indices")
    axes = tuple(map(int, axes))
    if len(set(axes)) != len(axes) or any(not 0 <= a < ring.D for a in axes):
        raise ValueError("Polynomial axes must be distinct indices in this ring")
    return axes


def _ordinary(polynomial, axes=None):
    if any(power < 0 for exponent in polynomial.terms for power in exponent):
        raise ValueError("Closed-point computation requires ordinary polynomials")
    if axes is not None:
        inactive = set(range(polynomial.ring.D)) - set(axes)
        if any(exponent[i] for exponent in polynomial.terms for i in inactive):
            raise ValueError("Ideal generator uses an inactive variable")


def _term_limit(max_terms):
    if isinstance(max_terms, bool) or not isinstance(max_terms, Integral) or max_terms < 1:
        raise ValueError("Polynomial shear term limit must be a positive integer")
    return int(max_terms)


def _multiply(a, b, max_terms):
    if len(a.terms) * len(b.terms) > max_terms:
        raise ResourceLimit("Polynomial shear exceeds the term-product limit")
    result = a * b
    if len(result.terms) > max_terms:
        raise ResourceLimit("Polynomial shear exceeds the term limit")
    return result


def _power(base, exponent, max_terms):
    result = base.ring.one()
    while exponent:
        if exponent & 1:
            result = _multiply(result, base, max_terms)
        exponent >>= 1
        if exponent:
            base = _multiply(base, base, max_terms)
    return result


def _scalar_leading(polynomial, axis):
    degree = max(e[axis] for e in polynomial.terms)
    top = [(e, c) for e, c in polynomial.terms.items() if e[axis] == degree]
    if len(top) == 1 and all(power == 0 for i, power in enumerate(top[0][0]) if i != axis):
        return top[0][1]
    return None


def monic_shear(polynomial: Poly, axes=None):
    """Return ``(weights, scalar)`` for a fixed monic triangular shear.

    With ``axis=axes[-1]`` and ``weights={i: w_i}``, substitution is
    ``x_i -> x_i + x_axis**w_i``. Its inverse changes the plus to minus.
    The transformed nonzero polynomial has scalar leading coefficient
    ``scalar`` as a polynomial in ``x_axis``. Dividing by that scalar is
    the caller's operation. No candidate weights are tested.
    """
    if not isinstance(polynomial, Poly):
        raise TypeError("Monic shear requires a polynomial")
    axes = _axes(polynomial.ring, axes)
    _ordinary(polynomial, axes)
    if not polynomial:
        raise ValueError("The zero polynomial has no monic normalization")
    if not axes:
        return {}, polynomial.constant()
    axis = axes[-1]
    scalar = _scalar_leading(polynomial, axis)
    if scalar is not None:
        return {}, scalar
    radix = 2 + max((e[i] for e in polynomial.terms for i in axes), default=0)
    weights = {i: radix ** (k + 1) for k, i in enumerate(axes[:-1])}
    leading = max(polynomial.terms,
                  key=lambda e: e[axis] + sum(e[i] * w for i, w in weights.items()))
    return weights, polynomial.terms[leading]


def apply_shear(value, axis, weights, *, sign=1, max_terms=100000):
    """Apply an explicit triangular polynomial automorphism to Poly/Matrix.

    ``x_i -> x_i + sign*x_axis**weights[i]``, with the distinguished
    variable and all unspecified variables fixed. Each weight is positive.
    A resource cap is allowed to reject expansion; it never means no point
    or no completion exists.
    """
    if not isinstance(value, (Poly, Matrix)):
        raise TypeError("Polynomial shear requires a Poly or Matrix")
    ring = value.ring
    axis, = _axes(ring, (axis,))
    max_terms = _term_limit(max_terms)
    if isinstance(sign, bool) or not isinstance(sign, Integral) or sign not in (-1, 1):
        raise ValueError("Polynomial shear sign must be +1 or -1")
    weights = dict(weights)
    _axes(ring, weights)
    if axis in weights:
        raise ValueError("The distinguished variable must remain fixed")
    if any(isinstance(w, bool) or not isinstance(w, Integral) or w < 1
           for w in weights.values()):
        raise ValueError("Polynomial shear weights must be positive integers")
    powers = {}
    shifted = {i: ring.variable(i) + ring.variable(axis) ** int(w) * int(sign)
               for i, w in weights.items()}

    def transform(polynomial):
        _ordinary(polynomial)
        result = ring.zero()
        for exponent, coefficient in polynomial.terms.items():
            fixed = tuple(0 if i in weights else e for i, e in enumerate(exponent))
            term = ring.monomial(fixed, coefficient)
            for i, base in shifted.items():
                key = (i, exponent[i])
                if key not in powers:
                    powers[key] = _power(base, exponent[i], max_terms)
                term = _multiply(term, powers[key], max_terms)
            result = result + term
            if len(result.terms) > max_terms:
                raise ResourceLimit("Polynomial shear exceeds the term limit")
        return result

    if isinstance(value, Poly):
        return transform(value)
    result = Matrix(ring, value.nrows, value.ncols)
    count = 0
    for at, polynomial in value.entries.items():
        result[at] = transform(polynomial)
        count += len(result[at].terms)
        if count > max_terms:
            raise ResourceLimit("Polynomial matrix shear exceeds the term limit")
    return result


def _evaluate(polynomial, field, coordinates):
    result = field.zero
    powers = {}
    for exponent, coefficient in polynomial.terms.items():
        term = field.coerce(coefficient)
        for i, power in enumerate(exponent):
            if not power:
                continue
            key = (i, power)
            if key not in powers:
                powers[key] = field.pow(coordinates[i], power)
            term = field.mul(term, powers[key])
        result = field.add(result, term)
    return result


@dataclass(frozen=True)
class ClosedPoint:
    """A point with a certified finite-field tower and full ambient coordinates."""

    ring: Ring
    field: object
    coordinates: tuple

    def __post_init__(self):
        if not isinstance(self.field, (PrimeField, ExtensionField)):
            raise TypeError("Point field must be an explicit finite residue field")
        if self.field.characteristic != self.ring.p or len(self.coordinates) != self.ring.D:
            raise ValueError("Point field or coordinate count does not match the ring")
        object.__setattr__(self, "coordinates", tuple(self.field.coerce(c) for c in self.coordinates))

    def evaluate(self, polynomial):
        polynomial = self.ring(polynomial)
        _ordinary(polynomial)
        return _evaluate(polynomial, self.field, self.coordinates)


def closed_point(ring: Ring, generators, *, axes=None, seed=0, max_trials=128,
                 timeout=60.0, max_terms=100000, max_basis=10000, max_pairs=100000):
    """Find a closed point of a proper polynomial ideal over F_p.

    All generators must use only the selected ``axes``; other coordinates
    in the result are zero. The selected order sets the successive monic
    variables. ``ValueError`` rejects the unit ideal or malformed input;
    ``ResourceLimit`` preserves the distinction from mathematical failure.
    Randomness is confined to certified, bounded univariate factorization.
    """
    if not isinstance(ring, Ring):
        raise TypeError("Closed points require an explicit prime polynomial ring")
    axes = _axes(ring, axes)
    generators = tuple(ring(f) for f in generators)
    for f in generators:
        _ordinary(f, axes)
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise TypeError("Closed-point seed must be an integer")
    if isinstance(max_trials, bool) or not isinstance(max_trials, Integral) or max_trials < 0:
        raise ValueError("Factor trial limit must be a nonnegative integer")
    budget = _Budget(timeout, max_basis, max_terms, max_pairs)
    prime = PrimeField(ring.p)

    def recurse(ideal, current_axes):
        ideal = tuple(f for f in ideal if f)
        budget.check(*(f.terms for f in ideal), basis_size=len(ideal))
        if any(f.is_constant for f in ideal):
            raise ValueError("The unit ideal has no closed point")
        if not ideal:
            return ClosedPoint(ring, prime, (prime.zero,) * ring.D)
        if not current_axes:
            raise ArithmeticError("A nonconstant ideal survived zero active variables")
        axis = current_axes[-1]
        selected = min(ideal, key=lambda f: (_scalar_leading(f, axis) is None,
                                            max(sum(e) for e in f.terms), len(f.terms), f.key()))
        weights, _ = monic_shear(selected, current_axes)
        transformed = tuple(apply_shear(f, axis, weights, max_terms=max_terms) for f in ideal)
        # Lex with the eliminated variable first. Vector component is last;
        # all vectors represent the rank-one ideal module.
        order = lambda m: (m[axis + 1],) + tuple(m[i + 1] for i in current_axes[:-1]) + (m[0],)
        basis = _groebner([{(0,) + e: c for e, c in f.terms.items()} for f in transformed],
                          order, ring.p, budget)
        basis = tuple(Poly(ring, {m[1:]: c for m, c in f.items()}) for f in basis)
        if any(f.is_constant for f in basis):
            raise ValueError("The unit ideal has no closed point")
        lower_ideal = tuple(f for f in basis if all(e[axis] == 0 for e in f.terms))
        lower = recurse(lower_ideal, current_axes[:-1])
        budget.check()
        field, gcd = lower.field, ()
        for f in transformed:
            degree = max((e[axis] for e in f.terms), default=0)
            if degree + 1 > max_terms:
                raise ResourceLimit("Closed-point univariate fiber exceeds the coefficient limit")
            coefficients = [field.zero] * (degree + 1)
            for exponent, coefficient in f.terms.items():
                e = tuple(0 if i == axis else power for i, power in enumerate(exponent))
                value = lower.evaluate(ring.monomial(e, coefficient))
                coefficients[exponent[axis]] = field.add(coefficients[exponent[axis]], value)
            gcd = poly_gcd(field, gcd, coefficients)
            budget.check()
        if len(gcd) < 2:
            raise ArithmeticError("Monic integral fiber unexpectedly has zero or unit gcd")
        factor = irreducible_factor(field, gcd, seed=int(seed) + len(current_axes),
                                    max_trials=int(max_trials))
        budget.check()
        if len(factor) == 2:
            result_field = field
            root = field.neg(factor[0])  # monic linear factor
            coordinates = list(lower.coordinates)
        else:
            result_field = ExtensionField(field, factor)
            root = result_field.generator
            coordinates = [result_field.from_base(c) for c in lower.coordinates]
        coordinates[axis] = root
        for i, weight in weights.items():
            coordinates[i] = result_field.add(coordinates[i], result_field.pow(root, weight))
        point = ClosedPoint(ring, result_field, tuple(coordinates))
        if any(point.evaluate(f) != result_field.zero for f in ideal):
            raise ArithmeticError("Closed-point recursion failed its generator evaluation certificate")
        return point

    try:
        point = recurse(generators, axes)
        if any(point.evaluate(f) != point.field.zero for f in generators):
            raise ArithmeticError("Closed-point output failed its input ideal certificate")
        return point
    except ResourceLimit as exc:
        raise ResourceLimit(f"Closed-point construction: {exc}") from exc
