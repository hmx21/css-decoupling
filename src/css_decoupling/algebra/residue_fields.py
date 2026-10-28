"""Exact finite residue fields, towers, and certified univariate factors.

Prime-field elements are integers. Extension elements are fixed-length
tuples of elements of the base field in the power basis. Polynomials are
tuples of coefficients in increasing degree, with trailing zeros removed.
``field.from_base(a)`` explicitly embeds a base-field element into a tower;
an unadorned tuple always describes coefficients in the current generator.

Factoring returns one irreducible divisor using squarefree reduction,
distinct-degree Frobenius gcds, and bounded Las Vegas equal-degree splitting.
The odd-characteristic splitter is Cantor--Zassenhaus; characteristic two
uses the absolute trace to F_2, including for nonprime base fields. Output is
independently checked by Rabin's criterion and exact polynomial division.
There is no enumeration of a finite field or unbounded degree search, and
no deterministic worst-case polynomial running-time claim.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from random import Random

from sympy import isprime

from css_decoupling.errors import ResourceLimit


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    return int(value)


@dataclass(frozen=True)
class PrimeField:
    p: int

    def __post_init__(self):
        p = _integer(self.p, "Characteristic")
        if not isprime(p):
            raise ValueError("The coefficient characteristic must be prime")
        object.__setattr__(self, "p", p)

    @property
    def characteristic(self):
        return self.p

    @property
    def order(self):
        return self.p

    @property
    def absolute_degree(self):
        return 1

    @property
    def zero(self):
        return 0

    @property
    def one(self):
        return 1

    def coerce(self, a):
        return _integer(a, "Prime-field element") % self.p

    from_base = coerce

    def add(self, a, b):
        return (self.coerce(a) + self.coerce(b)) % self.p

    def neg(self, a):
        return -self.coerce(a) % self.p

    def sub(self, a, b):
        return (self.coerce(a) - self.coerce(b)) % self.p

    def mul(self, a, b):
        return self.coerce(a) * self.coerce(b) % self.p

    def inv(self, a):
        a = self.coerce(a)
        if not a:
            raise ZeroDivisionError("Cannot invert zero in a finite field")
        return pow(a, -1, self.p)

    def pow(self, a, exponent):
        exponent = _integer(exponent, "Field exponent")
        a = self.coerce(a)
        if exponent < 0:
            a, exponent = self.inv(a), -exponent
        return pow(a, exponent, self.p)

    def random(self, rng):
        return rng.randrange(self.p)


@dataclass(frozen=True)
class ExtensionField:
    base: object
    modulus: tuple

    def __post_init__(self):
        if not isinstance(self.base, (PrimeField, ExtensionField)):
            raise TypeError("Extension base must be an explicit finite field")
        modulus = poly_normalize(self.base, self.modulus)
        if len(modulus) < 2 or modulus[-1] != self.base.one:
            raise ValueError("Extension modulus must be monic and nonconstant")
        if not is_irreducible(self.base, modulus):
            raise ValueError("Extension modulus is reducible over its base field")
        object.__setattr__(self, "modulus", modulus)

    @property
    def degree(self):
        return len(self.modulus) - 1

    @property
    def characteristic(self):
        return self.base.characteristic

    @property
    def order(self):
        return self.base.order ** self.degree

    @property
    def absolute_degree(self):
        return self.base.absolute_degree * self.degree

    @property
    def zero(self):
        return (self.base.zero,) * self.degree

    @property
    def one(self):
        return (self.base.one,) + (self.base.zero,) * (self.degree - 1)

    @property
    def generator(self):
        return self.coerce((self.base.zero, self.base.one))

    def from_base(self, a):
        return (self.base.coerce(a),) + (self.base.zero,) * (self.degree - 1)

    def coerce(self, a):
        if isinstance(a, Integral) and not isinstance(a, bool):
            return self.from_base(self.base.coerce(a))
        if not isinstance(a, (tuple, list)):
            raise TypeError("Extension elements must be power-basis tuples or prime-field integers")
        coefficients = poly_normalize(self.base, a)
        if len(coefficients) > self.degree:
            coefficients = poly_divmod(self.base, coefficients, self.modulus)[1]
        return coefficients + (self.base.zero,) * (self.degree - len(coefficients))

    def add(self, a, b):
        a, b = self.coerce(a), self.coerce(b)
        return tuple(self.base.add(x, y) for x, y in zip(a, b))

    def neg(self, a):
        return tuple(self.base.neg(x) for x in self.coerce(a))

    def sub(self, a, b):
        a, b = self.coerce(a), self.coerce(b)
        return tuple(self.base.sub(x, y) for x, y in zip(a, b))

    def mul(self, a, b):
        return self.coerce(poly_mul(self.base, self.coerce(a), self.coerce(b)))

    def inv(self, a):
        a = self.coerce(a)
        if a == self.zero:
            raise ZeroDivisionError("Cannot invert zero in a finite field")
        old_r, r = self.modulus, poly_normalize(self.base, a)
        old_s, s = (), (self.base.one,)
        while r:
            quotient, remainder = poly_divmod(self.base, old_r, r)
            old_r, r = r, remainder
            old_s, s = s, poly_sub(self.base, old_s, poly_mul(self.base, quotient, s))
        if len(old_r) != 1:
            raise ArithmeticError("A checked irreducible modulus acquired a nonunit residue")
        result = self.coerce(poly_scale(self.base, old_s, self.base.inv(old_r[0])))
        if self.mul(a, result) != self.one:
            raise ArithmeticError("Extension inverse certificate failed")
        return result

    def pow(self, a, exponent):
        exponent = _integer(exponent, "Field exponent")
        a = self.coerce(a)
        if exponent < 0:
            a, exponent = self.inv(a), -exponent
        result = self.one
        while exponent:
            if exponent & 1:
                result = self.mul(result, a)
            exponent >>= 1
            if exponent:
                a = self.mul(a, a)
        return result

    def random(self, rng):
        return tuple(self.base.random(rng) for _ in range(self.degree))


def poly_normalize(field, polynomial):
    coefficients = [field.coerce(c) for c in polynomial]
    while coefficients and coefficients[-1] == field.zero:
        coefficients.pop()
    return tuple(coefficients)


def poly_add(field, a, b):
    a, b = poly_normalize(field, a), poly_normalize(field, b)
    return poly_normalize(field, [field.add(a[i] if i < len(a) else field.zero,
                                           b[i] if i < len(b) else field.zero)
                                  for i in range(max(len(a), len(b)))])


def poly_neg(field, a):
    return poly_normalize(field, [field.neg(c) for c in a])


def poly_sub(field, a, b):
    return poly_add(field, a, poly_neg(field, b))


def poly_scale(field, a, coefficient):
    coefficient = field.coerce(coefficient)
    return poly_normalize(field, [field.mul(coefficient, c) for c in a])


def poly_mul(field, a, b):
    a, b = poly_normalize(field, a), poly_normalize(field, b)
    if not a or not b:
        return ()
    result = [field.zero] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        if x == field.zero:
            continue
        for j, y in enumerate(b):
            if y != field.zero:
                result[i + j] = field.add(result[i + j], field.mul(x, y))
    return poly_normalize(field, result)


def poly_divmod(field, a, b):
    a, b = poly_normalize(field, a), poly_normalize(field, b)
    if not b:
        raise ZeroDivisionError("Polynomial divisor is zero")
    quotient = [field.zero] * max(0, len(a) - len(b) + 1)
    remainder = list(a)
    inverse = field.inv(b[-1])
    while remainder and len(remainder) >= len(b):
        degree = len(remainder) - len(b)
        coefficient = field.mul(remainder[-1], inverse)
        quotient[degree] = field.add(quotient[degree], coefficient)
        for i, value in enumerate(b):
            remainder[i + degree] = field.sub(remainder[i + degree], field.mul(coefficient, value))
        while remainder and remainder[-1] == field.zero:
            remainder.pop()
    return poly_normalize(field, quotient), tuple(remainder)


def poly_monic(field, polynomial):
    polynomial = poly_normalize(field, polynomial)
    return poly_scale(field, polynomial, field.inv(polynomial[-1])) if polynomial else ()


def poly_gcd(field, a, b):
    a, b = poly_normalize(field, a), poly_normalize(field, b)
    while b:
        a, b = b, poly_divmod(field, a, b)[1]
    return poly_monic(field, a)


def poly_powmod(field, a, exponent, modulus):
    exponent = _integer(exponent, "Polynomial exponent")
    if exponent < 0:
        raise ValueError("Polynomial modular exponent must be nonnegative")
    modulus = poly_normalize(field, modulus)
    if not modulus:
        raise ZeroDivisionError("Polynomial modulus is zero")
    a = poly_divmod(field, a, modulus)[1]
    result = poly_divmod(field, (field.one,), modulus)[1]
    while exponent:
        if exponent & 1:
            result = poly_divmod(field, poly_mul(field, result, a), modulus)[1]
        exponent >>= 1
        if exponent:
            a = poly_divmod(field, poly_mul(field, a, a), modulus)[1]
    return result


def poly_derivative(field, polynomial):
    polynomial = poly_normalize(field, polynomial)
    return poly_normalize(field, [field.mul(field.coerce(i), polynomial[i])
                                  for i in range(1, len(polynomial))])


def _prime_divisors(n):
    result, divisor = [], 2
    while divisor * divisor <= n:
        if n % divisor == 0:
            result.append(divisor)
            while n % divisor == 0:
                n //= divisor
        divisor += 1
    if n > 1:
        result.append(n)
    return result


def is_irreducible(field, polynomial):
    """Independent Rabin test: Frobenius closure and prime-degree gcd tests."""
    f = poly_monic(field, polynomial)
    n = len(f) - 1
    if n < 1:
        return False
    if n == 1:
        return True
    x = (field.zero, field.one)
    at, required = x, {n // prime for prime in _prime_divisors(n)}
    for i in range(1, n + 1):
        at = poly_powmod(field, at, field.order, f)
        if i in required and len(poly_gcd(field, f, poly_sub(field, at, x))) != 1:
            return False
    return at == x


def _squarefree_divisor(field, polynomial):
    """Return one nonconstant squarefree divisor, taking Frobenius roots if needed."""
    f = poly_monic(field, polynomial)
    while True:
        derivative = poly_derivative(field, f)
        if derivative:
            common = poly_gcd(field, f, derivative)
            result, remainder = poly_divmod(field, f, common)
            if remainder or len(result) < 2:
                raise ArithmeticError("Squarefree-divisor calculation failed")
            return poly_monic(field, result)
        p = field.characteristic
        if any(c != field.zero for i, c in enumerate(f) if i % p):
            raise ArithmeticError("Zero derivative has a non-Frobenius exponent")
        # The inverse of a -> a^p on F_q is a -> a^(q/p), also in towers.
        f = tuple(field.pow(f[i], field.order // p) for i in range(0, len(f), p))
        f = poly_monic(field, f)


def _equal_degree_factor(field, f, degree, rng, max_trials):
    trials = 0
    while len(f) - 1 > degree:
        if trials >= max_trials:
            raise ResourceLimit(f"Equal-degree factor splitting exhausted max_trials={max_trials}")
        trials += 1
        a = poly_normalize(field, [field.random(rng) for _ in range(len(f) - 1)])
        divisor = poly_gcd(field, f, a)
        if not 1 < len(divisor) < len(f):
            if field.characteristic == 2:
                term, trace = a, ()
                for _ in range(field.absolute_degree * degree):
                    trace = poly_add(field, trace, term)
                    term = poly_powmod(field, term, 2, f)
                divisor = poly_gcd(field, f, trace)
            else:
                power = poly_powmod(field, a, (field.order ** degree - 1) // 2, f)
                divisor = poly_gcd(field, f, poly_sub(field, power, (field.one,)))
        if 1 < len(divisor) < len(f):
            quotient, remainder = poly_divmod(field, f, divisor)
            if remainder:
                raise ArithmeticError("Equal-degree splitter produced a nondivisor")
            # Follow a smaller branch rather than constructing all factors.
            f = poly_monic(field, divisor if len(divisor) <= len(quotient) else quotient)
    if len(f) - 1 != degree:
        raise ArithmeticError("Equal-degree splitter left its prescribed degree")
    return f


def irreducible_factor(field, polynomial, *, seed=0, max_trials=128):
    """Return one certified monic irreducible divisor over an explicit finite field.

    Random equal-degree splits use a reproducible seed and a strict total
    trial cap. Exhaustion is ResourceLimit, not evidence of irreducibility or
    absence of a factor. Distinct-degree iteration is bounded by the input
    degree. No finite-field point set is enumerated, including for towers.
    """
    if not isinstance(field, (PrimeField, ExtensionField)):
        raise TypeError("Factor coefficients require an explicit finite field")
    seed, max_trials = _integer(seed, "Random seed"), _integer(max_trials, "Trial limit")
    if max_trials < 0:
        raise ValueError("Trial limit must be nonnegative")
    original = poly_normalize(field, polynomial)
    if len(original) < 2:
        raise ValueError("An irreducible factor requires a nonconstant nonzero polynomial")
    f = _squarefree_divisor(field, original)
    x, frobenius = (field.zero, field.one), (field.zero, field.one)
    candidate = f
    for degree in range(1, (len(f) - 1) // 2 + 1):
        frobenius = poly_powmod(field, frobenius, field.order, f)
        divisor = poly_gcd(field, f, poly_sub(field, frobenius, x))
        if len(divisor) > 1:
            candidate = _equal_degree_factor(field, divisor, degree, Random(seed), max_trials)
            break
    candidate = poly_monic(field, candidate)
    if not is_irreducible(field, candidate) or poly_divmod(field, original, candidate)[1]:
        raise ArithmeticError("Irreducible factor failed independent Rabin/divisibility verification")
    return candidate
