"""Constructive Park column normalization over Fp[x^+-1,y^+-1].

This returns an elementary trace and a polynomial unimodular column, not a
completion or a Suslin factorization. A supplied, checked Bezout row is part
of the input; no ideal-membership or completion oracle is called.

Source: H.-J. Park, UCB/ERL M95/39 (1995), sections6.2--6.3 and11.2.
The polynomial Bezout witness below is an explicit finite geometric-series
identity using the tracked input witness. No external CAS is used.
"""
from __future__ import annotations

from time import perf_counter

from .algebra import Matrix
from ..errors import ResourceLimit


_SCHEMA = 'park-laurent-column-polynomialization-v1'
_KEYS = {'schema', 'input_column', 'input_bezout', 'coordinate_map',
         'coordinate_inverse', 'factors', 'polynomial_column',
         'tracked_laurent_bezout', 'clearing_power', 'cleared_row',
         'polynomial_bezout', 'complete_suslin_factorization'}


class _Budget:
    def __init__(self, timeout, max_steps, max_terms, max_products,
                 max_factors, max_exponent):
        if type(timeout) not in (int, float) or not 0 < timeout < float('inf'):
            raise ValueError('timeout must be finite and positive')
        self.deadline = perf_counter() + timeout
        for name, value in [('max_steps', max_steps), ('max_terms', max_terms),
                            ('max_products', max_products), ('max_factors', max_factors),
                            ('max_exponent', max_exponent)]:
            if type(value) is not int or value < 1:
                raise ValueError(name + ' must be a positive integer')
            setattr(self, name, value)
        self.steps = self.products = 0

    def check(self, *polynomials):
        self.steps += 1
        if perf_counter() > self.deadline or self.steps > self.max_steps:
            raise ResourceLimit('Laurent polynomialization time/operation cap reached')
        if sum(len(p.terms) for p in polynomials) > self.max_terms:
            raise ResourceLimit('Laurent polynomialization term cap reached')
        if any(abs(e) > self.max_exponent for p in polynomials
               for exponent in p.terms for e in exponent):
            raise ResourceLimit('Laurent polynomialization exponent cap reached')

    def multiply(self, a, b):
        self.check(a, b)
        self.products += len(a.terms)*len(b.terms)
        if self.products > self.max_products:
            raise ResourceLimit('Laurent polynomialization scalar-product cap reached')
        result = a*b
        self.check(result)
        return result

    def power(self, value, exponent):
        result = value.ring.one()
        while exponent:
            self.check()
            if exponent & 1:
                result = self.multiply(result, value)
            exponent >>= 1
            if exponent:
                value = self.multiply(value, value)
        return result


def _coordinate_matrix(value):
    if (not isinstance(value, (tuple, list)) or len(value) != 2
            or any(not isinstance(row, (tuple, list)) or len(row) != 2
                   or any(type(x) is not int for x in row) for row in value)):
        raise ValueError('Coordinate substitution must be an integer2x2 matrix')
    a, b = value[0]
    c, d = value[1]
    if a*d-b*c != 1:
        raise ValueError('Coordinate substitution must have determinant1')
    return ((a, b), (c, d))


def _coordinate_inverse(value):
    (a, b), (c, d) = _coordinate_matrix(value)
    return ((d, -b), (-c, a))


def _substitute(poly, coordinates, budget):
    (a, b), (c, d) = coordinates
    budget.check(poly)
    result = poly.ring({(a*i+b*j, c*i+d*j): v for (i, j), v in poly.terms.items()})
    budget.check(result)
    return result


def _dot(row, column, budget):
    result = column[0].ring.zero()
    for a, b in zip(row, column):
        result += budget.multiply(a, b)
        budget.check(result)
    return result


def _inputs(column, bezout, budget):
    if (not isinstance(column, Matrix) or not isinstance(bezout, Matrix)
            or column.ring != bezout.ring or column.ncols != 1
            or column.nrows < 2 or bezout.shape != (1, column.nrows)):
        raise ValueError('Expected an n-by1 column and1-by-n Bezout row, n>=2')
    if column.ring.D != 2:
        raise ValueError('This constructive step requires a prime-field ring with D=2')
    if column.nrows > budget.max_steps:
        raise ResourceLimit('Column dimension exceeds the operation cap')
    v = [column[i, 0] for i in range(column.nrows)]
    h = [bezout[0, i] for i in range(column.nrows)]
    budget.check(*(v+h))
    if _dot(h, v, budget) != 1:
        raise ValueError('Supplied Bezout row does not multiply the column to1')
    return v, h


def _apply(v, h, i, j, coefficient, budget):
    # E_ij(c) acts on the column; the row receives its inverse on the right.
    v[i] += budget.multiply(coefficient, v[j])
    h[j] -= budget.multiply(coefficient, h[i])
    budget.check(*(v+h))


def _polynomial(values):
    return all(a >= 0 and b >= 0 for f in values for a, b in f.terms)


def _geometric_sum(value, count, budget):
    """Return1+value+...+value^(count-1), using exact binary blocks."""
    total = value.ring.zero()
    prefix_power = block_sum = value.ring.one()
    block_power = value
    while count:
        budget.check()
        if count & 1:
            total += budget.multiply(prefix_power, block_sum)
            prefix_power = budget.multiply(prefix_power, block_power)
            budget.check(total)
        count >>= 1
        if count:
            block_sum += budget.multiply(block_power, block_sum)
            block_power = budget.multiply(block_power, block_power)
            budget.check(block_sum)
    return total


def laurent_column_to_polynomial(column, bezout, *, timeout=120,
                                 max_steps=1000000, max_terms=1000000,
                                 max_products=5000000, max_factors=100000,
                                 max_exponent=1000000):
    """Return checked elementary/coordinate data and polynomial Bezout data.

    If the chronological elementary word has product T and integer exponent
    substitution is B, the returned column is w=phi_B(T*column). All recorded
    factors are in the ORIGINAL variables. The returned polynomial row a
    satisfies a*w=1. This does not reduce w to a basis vector and does not
    factor the original QCA. Every imposed budget stop is ResourceLimit.
    """
    budget = _Budget(timeout, max_steps, max_terms, max_products,
                     max_factors, max_exponent)
    return _laurent_column_to_polynomial(column, bezout, budget)


def _laurent_column_to_polynomial(column, bezout, budget):
    """The same construction with a caller-owned, compatible shared budget."""
    v, h = _inputs(column, bezout, budget)
    ring = column.ring
    coordinates = ((1, 0), (0, 1))
    factors = []
    factor_terms = 0

    def add(i, j, coefficient):
        nonlocal factor_terms
        if not coefficient:
            return
        if len(factors) >= budget.max_factors:
            raise ResourceLimit('Laurent polynomialization factor cap reached')
        original = _substitute(coefficient, _coordinate_inverse(coordinates), budget)
        factor_terms += len(original.terms)
        if factor_terms > budget.max_terms:
            raise ResourceLimit('Laurent polynomialization factor-support cap reached')
        factors.append({'row': i, 'column': j,
                        'coefficient': Matrix.from_rows(ring, [[original]])})
        _apply(v, h, i, j, coefficient, budget)

    first = next(i for i, entry in enumerate(v) if entry)
    if first:
        for i, j, c in ((0, first, 1), (first, 0, -1), (0, first, 1)):
            add(i, j, ring(c))

    # a+N*b distinguishes the finite support without any generic field choice.
    xs = [e[0] for e in v[0].terms]
    shear = max(xs)-min(xs)+1
    coordinates = ((1, shear), (0, 1))
    v = [_substitute(p, coordinates, budget) for p in v]
    h = [_substitute(p, coordinates, budget) for p in h]
    low = min(v[0].terms, key=lambda e: e[0])
    unit = ring.monomial((-low[0], -low[1]),pow(v[0].terms[low],-1,ring.p))
    # Written product E01(u-1) E10(1) E01(u^-1-1) E10(-u)
    # equals diag(u,u^-1). Record the reverse order for left multiplication.
    for i, j, c in ((1, 0, -unit), (0, 1, unit**-1-1),
                    (1, 0, ring.one()), (0, 1, unit-1)):
        add(i, j, c)
    if any(e[0] < 0 for e in v[0].terms) or ring(
            {e: c for e, c in v[0].terms.items() if e[0] == 0}) != 1:
        raise ArithmeticError('Monomial normalization did not give constant coefficient1')

    for j in range(1, len(v)):
        while v[j] and min(e[0] for e in v[j].terms) <= 0:
            budget.check()
            degree = min(e[0] for e in v[j].terms)
            coefficient = ring({e: c for e, c in v[j].terms.items() if e[0] == degree})
            add(j, 0, -coefficient)

    # x=s*t^L, y=t: all nonconstant terms now have positive x degree.
    power = max([1] + [(1-b+a-1)//a for f in v for a, b in f.terms if a > 0])
    second = ((1, 0), (power, 1))
    v = [_substitute(p, second, budget) for p in v]
    h = [_substitute(p, second, budget) for p in h]
    coordinates = ((1, shear), (power, power*shear+1))
    if not _polynomial(v):
        raise ArithmeticError('Coordinate substitution did not produce polynomials')
    clearing = max([0] + [-e for f in h for exponent in f.terms for e in exponent])
    cleared = [budget.multiply(ring.monomial((clearing, clearing)), f) for f in h]
    # Put t=1-v0=(xy)g. Then v0*(1+...+t^(c-1))+t^c=1.
    # Multiplying the cleared row by g^c supplies the second summand.
    g = budget.multiply(1-v[0], ring.monomial((-1, -1)))
    correction = budget.power(g, clearing)
    geometric = _geometric_sum(1-v[0], clearing, budget)
    polynomial_bezout = [budget.multiply(correction, f) for f in cleared]
    polynomial_bezout[0] += geometric
    budget.check(*polynomial_bezout)
    certificate = {'schema': _SCHEMA, 'input_column': column.copy(),
                   'input_bezout': bezout.copy(), 'coordinate_map': coordinates,
                   'coordinate_inverse': _coordinate_inverse(coordinates),
                   'factors': factors,
                   'polynomial_column': Matrix.from_rows(ring, [[f] for f in v]),
                   'tracked_laurent_bezout': Matrix.from_rows(ring, [h]),
                   'clearing_power': clearing,
                   'cleared_row': Matrix.from_rows(ring, [cleared]),
                   'polynomial_bezout': Matrix.from_rows(ring, [polynomial_bezout]),
                   'complete_suslin_factorization': False}
    _verify(column, bezout, certificate, budget)
    return certificate


def _verify(column, bezout, certificate, budget):
    v, h = _inputs(column, bezout, budget)
    ring, n = column.ring, column.nrows
    if (not isinstance(certificate, dict) or set(certificate) != _KEYS
            or certificate['schema'] != _SCHEMA
            or certificate['complete_suslin_factorization'] is not False
            or certificate['input_column'] != column
            or certificate['input_bezout'] != bezout):
        raise ValueError('Polynomialization certificate schema/input differs')
    coordinates = _coordinate_matrix(certificate['coordinate_map'])
    if _coordinate_matrix(certificate['coordinate_inverse']) != _coordinate_inverse(coordinates):
        raise ValueError('Coordinate inverse differs from the actual integer inverse')
    factors = certificate['factors']
    if not isinstance(factors, (tuple, list)):
        raise ValueError('Expected an elementary factor sequence')
    if len(factors) > budget.max_factors:
        raise ResourceLimit('Polynomialization verifier factor cap reached')
    factor_terms = 0
    for item in factors:
        if not isinstance(item, dict) or set(item) != {'row', 'column', 'coefficient'}:
            raise ValueError('Malformed elementary factor')
        i, j, coefficient = item['row'], item['column'], item['coefficient']
        if (type(i) is not int or type(j) is not int or not 0 <= i < n
                or not 0 <= j < n or i == j or not isinstance(coefficient, Matrix)
                or coefficient.ring != ring or coefficient.shape != (1, 1)
                or not coefficient[0, 0]):
            raise ValueError('Invalid elementary factor or coefficient')
        factor_terms += len(coefficient[0, 0].terms)
        if factor_terms > budget.max_terms:
            raise ResourceLimit('Polynomialization verifier factor-support cap reached')
        _apply(v, h, i, j, coefficient[0, 0], budget)
    v = [_substitute(f, coordinates, budget) for f in v]
    h = [_substitute(f, coordinates, budget) for f in h]
    for key, shape in [('polynomial_column', (n, 1)),
                       ('tracked_laurent_bezout', (1, n)),
                       ('cleared_row', (1, n)), ('polynomial_bezout', (1, n))]:
        value = certificate[key]
        if not isinstance(value, Matrix) or value.ring != ring or value.shape != shape:
            raise ValueError('Wrong certificate matrix type, ring or shape: '+key)
    if (certificate['polynomial_column'] != Matrix.from_rows(ring, [[f] for f in v])
            or certificate['tracked_laurent_bezout'] != Matrix.from_rows(ring, [h])):
        raise ValueError('Elementary/coordinate replay differs from certificate')
    if not _polynomial(v) or any(a < 1 or b < 1 for f in [v[0]-1]+v[1:] for a, b in f.terms):
        raise ValueError('Polynomial column lacks the claimed boundary normalization')
    clearing = certificate['clearing_power']
    if type(clearing) is not int or clearing < 0:
        raise ValueError('Clearing power must be a nonnegative integer')
    if clearing != max([0]+[-e for f in h for exponent in f.terms for e in exponent]):
        raise ValueError('Clearing power differs from the tracked row valuation')
    monomial = ring.monomial((clearing, clearing))
    cleared = [certificate['cleared_row'][0, i] for i in range(n)]
    actual = [budget.multiply(monomial, f) for f in h]
    if cleared != actual or not _polynomial(cleared):
        raise ValueError('Cleared polynomial row differs')
    answer = [certificate['polynomial_bezout'][0, i] for i in range(n)]
    budget.check(*answer)
    if not _polynomial(answer) or _dot(answer, v, budget) != 1:
        raise ValueError('Polynomial Bezout witness fails')
    if _dot(h, v, budget) != 1 or _dot(cleared, v, budget) != monomial:
        raise ValueError('Tracked Laurent/cleared Bezout identity fails')
    return True


def verify_laurent_column_to_polynomial(column, bezout, certificate, *, timeout=120,
                                        max_steps=1000000, max_terms=1000000,
                                        max_products=5000000, max_factors=100000,
                                        max_exponent=1000000):
    """Independently replay supplied factors and check exact witness equations.

    This verifier never invokes the constructor, searches for a completion,
    or interprets an exhausted cap as a nonexistence result.
    """
    return _verify(column, bezout, certificate,
                   _Budget(timeout, max_steps, max_terms, max_products,
                           max_factors, max_exponent))
