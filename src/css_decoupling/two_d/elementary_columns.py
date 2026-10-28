"""Fixed-rank elementary column reduction over prime bivariate Laurent rings.

Park--Woodburn, alg-geom/9405003, Section 4 supplies the resultant patching
argument. Closed points here are irreducible factors over F_p, not merely
F_p-rational points. Every operation is an E_ij(a) on the ORIGINAL n slots.
The third slot in a commutator is an existing coordinate, never an ancilla.
"""
from __future__ import annotations

from .algebra import Matrix, eye
from ..errors import ResourceLimit
from ..algebra.closed_points import apply_shear, monic_shear as _generic_monic_shear
from ..algebra.residue_fields import (
    PrimeField, ExtensionField, irreducible_factor, poly_normalize,
    poly_divmod, poly_sub, poly_mul, poly_neg,
)
from ..algebra.univariate import _divide, _gcdex
from .laurent_to_polynomial import (
    _Budget as _ParkBudget, _laurent_column_to_polynomial,
)


class _Budget(_ParkBudget):
    def __init__(self, timeout, max_steps, max_terms, max_products,
                 max_factors, max_exponent, max_patches, max_resultant_size,
                 max_field_trials):
        super().__init__(timeout, max_steps, max_terms, max_products,
                         max_factors, max_exponent)
        for key, value in (('max_patches', max_patches),
                           ('max_resultant_size', max_resultant_size),
                           ('max_field_trials', max_field_trials)):
            if type(value) is not int or value < 1:
                raise ValueError(key+' must be a positive integer')
            setattr(self, key, value)
        self.patches = []
        self.row_choices = []

    def word(self, factors):
        if len(factors) > self.max_factors:
            raise ResourceLimit('Fixed-rank column elementary factor budget exceeded')
        self.check(*(a for _, _, a in factors))


def _inverse(word):
    return [(i, j, -a) for i, j, a in reversed(word)]


def _swap(ring, i, j):
    # The signed swap sends (0,1) to (1,0), and has determinant one.
    return [] if i == j else [(i, j, ring.one()), (j, i, -ring.one()),
                              (i, j, ring.one())]


def _unit_pair(ring, i, j, unit):
    if unit == 1:
        return []
    written = [(i, j, unit), (j, i, -unit**-1), (i, j, unit),
               (i, j, -ring.one()), (j, i, ring.one()), (i, j, -ring.one())]
    return list(reversed(written))


def _apply(values, word, budget):
    out = list(values)
    for i, j, a in word:
        out[i] += budget.multiply(a, out[j])
        budget.check(*out)
    return out


def _records(ring, word):
    return [{'row': i, 'column': j,
             'coefficient': Matrix.from_rows(ring, [[a]])}
            for i, j, a in word if a]


def _atoms(records):
    return [(f['row'], f['column'], f['coefficient'][0, 0]) for f in records]


def _ordinary(values):
    return all(all(a >= 0 for a in e) for f in values for e in f.terms)


def _support_monic_shear(polynomial):
    """One exact support bound for x->x+y^w; no expansion or weight search.

    The rightmost, then uppermost support point (A,B) is the unique leading
    point when w(A-a)>b-B for every a<A. The displayed integer bound makes
    all those inequalities strict. Only uniqueness of the leading point is
    needed; distinguishing every pair of interior terms is unnecessary.
    """
    if not polynomial or not _ordinary((polynomial,)):
        raise ValueError('Monic shear requires a nonzero ordinary polynomial')
    highest = max(e[1] for e in polynomial.terms)
    leading = [(e, c) for e, c in polynomial.terms.items() if e[1] == highest]
    if len(leading) == 1 and leading[0][0][0] == 0:
        return {}, leading[0][1]
    A, B = max(polynomial.terms)
    weight = max([1]+[(b-B)//(A-a)+1 for a, b in polynomial.terms if a < A])
    return {0: weight}, polynomial.terms[A, B]


def _row_score(polynomial, *, laurent=False):
    """Support-only score of an existing row; no candidate polynomial search.

    A Laurent unit shift is used only to estimate the normalized support.
    The chosen physical permutation is recorded separately as elementary
    operations; this score itself changes no matrix or coordinate system.
    """
    if not polynomial:
        raise ValueError('The zero entry has no monic score')
    support = polynomial.terms
    if laurent:
        low = tuple(min(e[i] for e in support) for i in range(2))
        polynomial = polynomial.ring({(a-low[0], b-low[1]): c
                                       for (a, b), c in support.items()})
    # Before Laurent normalization, retain its conservative support-separation
    # score: a small degree after a later polynomial shear does not estimate
    # the intervening Laurent coordinate expansion. Once polynomialized, the
    # exact leading-support bound below is the degree actually used.
    weights, _ = (_generic_monic_shear(polynomial, axes=(0, 1)) if laurent
                   else _support_monic_shear(polynomial))
    degree = max(e[1]+sum(e[i]*w for i, w in weights.items())
                 for e in polynomial.terms)
    span = sum(max(e[i] for e in polynomial.terms)-min(e[i] for e in polynomial.terms)
               for i in range(2))
    return degree, len(polynomial.terms), span


def _choose_row(values, budget, *, laurent=False):
    score, row = min((_row_score(value, laurent=laurent), i)
                     for i, value in enumerate(values) if value)
    budget.row_choices.append({'stage': 'Laurent' if laurent else 'polynomial',
                               'row': row, 'score': list(score)})
    budget.check()
    return row


def _coefficient(f, degree, axis=1):
    return f.ring({tuple(0 if i == axis else a for i, a in enumerate(e)): c
                   for e, c in f.terms.items() if e[axis] == degree})


def _evaluate(f, value, budget):
    """Substitute the second variable; coefficients remain in F_p[x]."""
    ring = f.ring
    if not f:
        return ring.zero()
    degree = max(e[1] for e in f.terms)
    if min(e[1] for e in f.terms) < 0:
        raise ValueError('Polynomial substitution received a negative degree')
    result = ring.zero()
    for k in range(degree, -1, -1):
        result = budget.multiply(result, value)+_coefficient(f, k)
        budget.check(result)
    return result


def _x_dict(f):
    if any(e[1] or e[0] < 0 for e in f.terms):
        raise ValueError('Expected an ordinary polynomial in the coefficient variable')
    return {e[0]: c for e, c in f.terms.items()}


def _from_x(ring, data):
    return ring({(a, 0): c for a, c in data.items() if c})


def _divide_coefficient(f, divisor, budget):
    denominator = _x_dict(divisor)
    groups = {}
    for (a, b), c in f.terms.items():
        groups.setdefault(b, {})[a] = c
    entries = {}
    for b, numerator in groups.items():
        budget.check(f, divisor)
        quotient, remainder = _divide(numerator, denominator, f.ring.p)
        if remainder:
            raise ArithmeticError('Resultant patch difference is not divisible by its coefficient')
        entries.update({(a, b): c for a, c in quotient.items()})
    result = f.ring(entries)
    if budget.multiply(divisor, result) != f:
        raise ArithmeticError('Exact coefficient division failed')
    return result


def _dot(left, right, ring, budget):
    result = ring.zero()
    for a, b in zip(left, right):
        if a and b:
            result += budget.multiply(a, b)
            budget.check(result)
    return result


def _matvec(matrix, vector, ring, budget):
    result = [_dot(row, vector, ring, budget) for row in matrix]
    budget.check(*result)
    return result


def _characteristic(matrix, ring, budget):
    """Division-free Berkowitz coefficients, valid also when p divides n!."""
    n = len(matrix)
    if not n:
        return [ring.one()]
    budget.check(*(f for row in matrix for f in row))
    lower = [row[1:] for row in matrix[1:]]
    coefficients = _characteristic(lower, ring, budget)
    toeplitz = [ring.one(), -matrix[0][0]]
    vector = [row[0] for row in matrix[1:]]
    for k in range(n-1):
        toeplitz.append(-_dot(matrix[0][1:], vector, ring, budget))
        if k < n-2:
            vector = _matvec(lower, vector, ring, budget)
    answer = [_dot(toeplitz[k::-1][:len(coefficients)],
                   coefficients[:k+1], ring, budget) for k in range(n+1)]
    budget.check(*answer)
    return answer


def _resultant_relation(f, g, budget):
    """Return r,a,b in F_p[x],R,R with a*f+b*g=r via a Sylvester adjugate.

    All determinant work is division-free and counted by the same scalar
    multiplication budget. No fraction-field denominator is specialized.
    """
    ring = f.ring
    if not f or not g or not _ordinary((f, g)):
        raise ValueError('Resultant needs two nonzero ordinary polynomials')
    m, n = max(e[1] for e in f.terms), max(e[1] for e in g.terms)
    size = m+n
    if m == 0:
        return f, ring.one(), ring.zero()
    if size > budget.max_resultant_size:
        raise ResourceLimit('Fixed-rank column Sylvester dimension budget exceeded')
    if size*size > budget.max_terms:
        raise ResourceLimit('Fixed-rank column Sylvester allocation budget exceeded')
    S = [[ring.zero() for _ in range(size)] for _ in range(size)]
    for column in range(n):
        for degree in range(m+1):
            S[column+degree][column] = _coefficient(f, degree)
    for column in range(m):
        for degree in range(n+1):
            S[column+degree][n+column] = _coefficient(g, degree)
    coefficients = _characteristic(S, ring, budget)
    determinant = coefficients[-1]*((-1)**size)
    if not determinant:
        raise ArithmeticError('Chosen residue pair has a zero resultant')
    vector = [ring.one()]+[ring.zero()]*(size-1)
    for k in range(1, size):
        vector = _matvec(S, vector, ring, budget)
        vector[0] += coefficients[k]
        budget.check(*vector)
    vector = [f*((-1)**(size-1)) for f in vector]
    y = ring.variable(1)
    a = sum((budget.multiply(vector[k], y**k) for k in range(n)), ring.zero())
    b = sum((budget.multiply(vector[n+k], y**k) for k in range(m)), ring.zero())
    if budget.multiply(a, f)+budget.multiply(b, g) != determinant:
        raise ArithmeticError('Sylvester adjugate Bezout identity failed')
    return determinant, a, b


def _residue(f, field):
    values = {}
    for (a, b), c in f.terms.items():
        value = field.mul(field.coerce(c), field.pow(field.generator, a))
        values[b] = field.add(values.get(b, field.zero), value)
    return poly_normalize(field, [values.get(i, field.zero)
                                  for i in range(max(values, default=-1)+1)])


def _lift_residue(ring, coefficients):
    return ring({(i, j): c for j, element in enumerate(coefficients)
                 for i, c in enumerate(element) if c})


def _tail_euclid(values, field, budget):
    ring, n = values[0].ring, len(values)
    residual = [_residue(f, field) for f in values]
    word = []

    def add(i, j, coefficients):
        if not coefficients:
            return
        a = _lift_residue(ring, coefficients)
        word.append((i, j, a))
        residual[i] = poly_sub(field, residual[i],
                               poly_mul(field, poly_neg(field, coefficients), residual[j]))
        budget.word(word)

    def swap(i, j):
        for a, b, c in ((i, j, field.one), (j, i, field.neg(field.one)), (i, j, field.one)):
            add(a, b, (c,))

    for j in range(2, n):
        while residual[j]:
            budget.check()
            if not residual[1]:
                swap(1, j)
                continue
            quotient, remainder = poly_divmod(field, residual[j], residual[1])
            add(j, 1, poly_neg(field, quotient))
            if residual[j] != remainder:
                raise ArithmeticError('Residue Euclidean row operation failed')
            if residual[j]:
                swap(1, j)
    if not residual[1]:
        raise ValueError('Monic unimodular column cannot have zero residue tail')
    return word


def _univariate_reduce(values, axis, budget):
    """Euclidean column operations over F_p[t], ending at the LAST unit vector."""
    ring, n = values[0].ring, len(values)
    work, word = list(values), []

    def add(records):
        nonlocal work
        work = _apply(work, records, budget)
        word.extend(records)
        budget.word(word)

    for j in range(1, n):
        while work[j]:
            if not work[0]:
                add(_swap(ring, 0, j))
                continue
            numerator = {e[axis]: c for e, c in work[j].terms.items()}
            denominator = {e[axis]: c for e, c in work[0].terms.items()}
            quotient, _ = _divide(numerator, denominator, ring.p)
            coefficient = ring({tuple(k if i == axis else 0 for i in range(2)): -c
                                for k, c in quotient.items()})
            if coefficient:
                add([(j, 0, coefficient)])
            if work[j]:
                add(_swap(ring, 0, j))
    if not work[0].is_constant or not work[0]:
        raise ValueError('Column is not unimodular over the polynomial ring')
    add(_unit_pair(ring, 0, n-1, work[0]**-1))
    # Inverse signed swap sends e0 to e_last, with positive coefficient.
    add(_inverse(_swap(ring, 0, n-1)))
    if work != [ring.zero()]*(n-1)+[ring.one()]:
        raise ArithmeticError('Univariate elementary reduction failed')
    return word


def _conjugate_pair(B, Bi, factor, n, budget):
    """Expand B E_ij(a) B^-1, B acting on slots 0,1, at unchanged rank."""
    i, j, a = factor
    ring = a.ring
    if i < 2 and j >= 2:
        word = []
        for k in range(2):
            value = budget.multiply(B[k, i], a)
            if value:
                word.append((k, j, value))
        return word
    if i >= 2 and j < 2:
        word = []
        for k in range(2):
            value = budget.multiply(a, Bi[j, k])
            if value:
                word.append((i, k, value))
        return word
    if i >= 2 and j >= 2:
        return [factor]
    # E_ij(a)=[E_i2(a),E_2j(1)]. Reverse written commutator for chronology.
    P = _conjugate_pair(B, Bi, (i, 2, a), n, budget)
    Q = _conjugate_pair(B, Bi, (2, j, ring.one()), n, budget)
    word = _inverse(Q)+_inverse(P)+Q+P
    budget.word(word)
    return word


def _patch_matrix(f, g, a, b, r, previous, following, budget):
    ring = f.ring
    old = [_evaluate(h, previous, budget) for h in (f, g, a, b)]
    new = [_evaluate(h, following, budget) for h in (f, g, a, b)]
    s, t, u, v = [_divide_coefficient(n-o, r, budget) for o, n in zip(old, new)]
    f0, g0, a0, b0 = old
    B = Matrix.from_rows(ring, [
        [1+budget.multiply(s, a0)+budget.multiply(v, g0),
         budget.multiply(s, b0)-budget.multiply(v, f0)],
        [budget.multiply(t, a0)-budget.multiply(u, g0),
         1+budget.multiply(t, b0)+budget.multiply(u, f0)],
    ])
    Bi = Matrix.from_rows(ring, [[B[1, 1], -B[0, 1]], [-B[1, 0], B[0, 0]]])
    if (budget.multiply(B[0, 0], B[1, 1])-budget.multiply(B[0, 1], B[1, 0]) != 1 or
            B*Matrix.from_rows(ring, [[f0], [g0]]) != Matrix.from_rows(ring, [[new[0]], [new[1]]])):
        raise ArithmeticError('Resultant SL2 patch equation failed')
    return B, Bi


def _polynomial_reduce(values, budget):
    """Polynomial column -> last unit vector, with prime-field closed points.

    For a patch, W=E(next)^-1 B C E(previous) carries v(previous) to
    v(next); B only acts on slots 0,1. If Q*v(previous)=e_last, then
    Q_next=(B*(Q*E(previous)^-1*C^-1)*B^-1)*E(next) reduces v(next).
    Conjugation is elementary by the explicit existing-slot commutators.
    In particular no factorization of the arbitrary SL2 matrix B is assumed.
    """
    ring, n = values[0].ring, len(values)
    axes = {i for f in values for e in f.terms for i, k in enumerate(e) if k}
    if len(axes) <= 1:
        return _univariate_reduce(values, next(iter(axes), 0), budget)
    first = _choose_row(values, budget)
    prefix = _swap(ring, 0, first)
    current = _apply(values, prefix, budget)
    weights, leading = _support_monic_shear(current[0])
    current = [apply_shear(f, 1, weights, max_terms=budget.max_terms) for f in current]
    budget.check(*current)
    normalization = _unit_pair(ring, 0, n-1, ring(pow(leading, -1, ring.p)))
    current = _apply(current, normalization, budget)
    if _coefficient(current[0], max(e[1] for e in current[0].terms)) != 1:
        raise ArithmeticError('Polynomial coordinate shear did not give a monic first entry')
    if current[0].is_constant:
        word = normalization+_univariate_unit_reduce(current, budget)
    else:
        patches, cover, gcd = [], [], {}
        point = (0, 1)  # x=0, followed by factors over extension fields.
        prime = PrimeField(ring.p)
        while True:
            if len(patches) >= budget.max_patches:
                raise ResourceLimit('Fixed-rank column patch budget exceeded')
            field = ExtensionField(prime, point)
            local_word = _tail_euclid(current, field, budget)
            changed = _apply(current, local_word, budget)
            r, a, b = _resultant_relation(changed[0], changed[1], budget)
            if not _residue(r, field):
                raise ArithmeticError('Resultant vanishes at its selected closed point')
            new_gcd, s, t = _gcdex(gcd, _x_dict(r), ring.p)
            if gcd and max(new_gcd) >= max(gcd):
                raise ArithmeticError('Resultant cover failed strict gcd degree descent')
            s, t = _from_x(ring, s), _from_x(ring, t)
            cover = [budget.multiply(s, h) for h in cover]+[t]
            gcd = new_gcd
            patches.append((local_word, changed, r, a, b))
            budget.patches.append({'residue_modulus': list(point),
                                   'resultant': Matrix.from_rows(ring, [[r]]),
                                   'cover_gcd_degree': max(gcd)})
            if max(gcd) == 0:
                break
            coefficients = tuple(gcd.get(i, 0) for i in range(max(gcd)+1))
            point = irreducible_factor(prime, coefficients, max_trials=budget.max_field_trials)
            budget.check(*cover)
        if sum((budget.multiply(h, patch[2]) for h, patch in zip(cover, patches)), ring.zero()) != 1:
            raise ArithmeticError('Resultant coefficients do not generate the unit ideal')
        zero = ring.zero()
        word = _univariate_reduce([_evaluate(f, zero, budget) for f in current], 0, budget)
        previous = zero
        for (local, changed, r, a, b), h in zip(patches, cover):
            following = previous+budget.multiply(budget.multiply(r, h), ring.variable(1))
            old_a, old_b = _evaluate(a, previous, budget), _evaluate(b, previous, budget)
            adjustments = []
            for j in range(2, n):
                difference = _evaluate(changed[j], following, budget)-_evaluate(changed[j], previous, budget)
                quotient = _divide_coefficient(difference, r, budget)
                for k, coefficient in ((0, old_a), (1, old_b)):
                    value = budget.multiply(quotient, coefficient)
                    if value:
                        adjustments.append((j, k, value))
            B, Bi = _patch_matrix(changed[0], changed[1], a, b, r,
                                  previous, following, budget)
            at_previous = [(i, j, _evaluate(c, previous, budget)) for i, j, c in local]
            at_following = [(i, j, _evaluate(c, following, budget)) for i, j, c in local]
            inner = _inverse(adjustments)+_inverse(at_previous)+word
            word = list(at_following)
            for factor in inner:
                word.extend(_conjugate_pair(B, Bi, factor, n, budget))
                budget.word(word)
            actual = _apply([_evaluate(f, following, budget) for f in current], word, budget)
            if actual != [ring.zero()]*(n-1)+[ring.one()]:
                raise ArithmeticError('Elementary resultant patch does not reduce its column')
            previous = following
        if previous != ring.variable(1):
            raise ArithmeticError('Resultant patch path does not end at the original variable')
        word = normalization+word
    # Pull factors back through the polynomial shear. It is not a gate.
    word = prefix+[(i, j, apply_shear(c, 1, weights, sign=-1,
                                     max_terms=budget.max_terms)) for i, j, c in word]
    budget.word(word)
    return word


def _univariate_unit_reduce(values, budget):
    """A unit first entry needs only shears, even for multivariate tails."""
    ring, n = values[0].ring, len(values)
    word = _unit_pair(ring, 0, n-1, values[0]**-1)
    current = _apply(values, word, budget)
    for i in range(1, n):
        if current[i]:
            word.append((i, 0, -current[i]))
    word += _inverse(_swap(ring, 0, n-1))
    budget.word(word)
    return word


def _laurent_unit_reduce(values, budget):
    """Reduce an actual Laurent-unit entry without any coordinate expansion."""
    ring, n = values[0].ring, len(values)
    pivot = next(i for i, value in enumerate(values) if value.is_unit)
    word = _swap(ring, 0, pivot)
    current = _apply(values, word, budget)
    normalization = _unit_pair(ring, 0, 1, current[0]**-1)
    current = _apply(current, normalization, budget)
    word.extend(normalization)
    word.extend((i, 0, -current[i]) for i in range(1, n) if current[i])
    budget.word(word)
    return word


def _zero_slot_reduce(values, bezout, budget):
    """Use an already zero coordinate as workspace; never add a new slot."""
    ring, n = values[0].ring, len(values)
    pivot = next(i for i, value in enumerate(values) if not value)
    # None of these additions changes a source coordinate. The checked
    # Bezout identity therefore makes precisely this existing slot equal 1.
    word = [(pivot, i, bezout[i]) for i in range(n)
            if i != pivot and bezout[i]]
    word.extend((i, pivot, -values[i]) for i in range(n)
                if i != pivot and values[i])
    if pivot:
        # On the column alone, (0,1)->(1,0) takes two elementary additions;
        # no full permutation matrix is needed or claimed.
        word.extend(((0, pivot, ring.one()), (pivot, 0, -ring.one())))
    budget.word(word)
    if _apply(values, word, budget) != [ring.one()]+[ring.zero()]*(n-1):
        raise ArithmeticError('Existing zero-slot Bezout reduction failed')
    return word


def elementary_column_reduce(column, bezout, *, timeout=120,
                             max_steps=1000000, max_terms=1000000,
                             max_products=30000000, max_factors=200000,
                             max_exponent=1000000, max_patches=128,
                             max_resultant_size=64, max_field_trials=128):
    """Return an actual n-slot elementary word T with T*column=e0 (n>=3).

    A checked Bezout row is required. Laurent polynomialization, residue-field
    Euclidean algorithms and division-free resultant patches are constructive;
    a cap exhaustion is ResourceLimit, never absence of an elementary word.
    """
    if (not isinstance(column, Matrix) or not isinstance(bezout, Matrix) or
            column.ring != bezout.ring or column.ring.D != 2 or column.ncols != 1 or
            column.nrows < 3 or bezout.shape != (1, column.nrows)):
        raise ValueError('Expected n>=3 column and matching Bezout row over a prime bivariate ring')
    budget = _Budget(timeout, max_steps, max_terms, max_products, max_factors,
                     max_exponent, max_patches, max_resultant_size, max_field_trials)
    ring, n = column.ring, column.nrows
    if 2*n > budget.max_terms:
        raise ResourceLimit('Fixed-rank column inverse-pair identity allocation budget exceeded')
    values = [column[i, 0] for i in range(n)]
    row = [bezout[0, i] for i in range(n)]
    budget.check(*(values+row))
    if _dot(row, values, ring, budget) != 1:
        raise ValueError('Supplied Bezout row does not multiply the column to one')
    if any(f.is_unit for f in values):
        word = _laurent_unit_reduce(values, budget)
    elif any(not f for f in values):
        word = _zero_slot_reduce(values, row, budget)
    else:
        prefix, inverse_coordinates = [], ((1, 0), (0, 1))
        if not _ordinary(values+row):
            pivot = _choose_row(values, budget, laurent=True)
            prefix = _swap(ring, 0, pivot)
            prepared_values = _apply(values, prefix, budget)
            prepared_row = list(row)
            for i, j, coefficient in prefix:
                prepared_row[j] -= budget.multiply(prepared_row[i], coefficient)
                budget.check(*prepared_row)
            preparation = _laurent_column_to_polynomial(
                Matrix.from_rows(ring, [[f] for f in prepared_values]),
                Matrix.from_rows(ring, [prepared_row]), budget)
            prefix += _atoms(preparation['factors'])
            values = [preparation['polynomial_column'][i, 0] for i in range(n)]
            row = [preparation['polynomial_bezout'][0, i] for i in range(n)]
            inverse_coordinates = preparation['coordinate_inverse']
        # Exact normalization may expose a usable existing coordinate even
        # when the raw column had neither a unit nor a zero. Reuse it before
        # expanding any polynomial coordinate shear or resultant.
        if any(f.is_unit for f in values):
            word = _laurent_unit_reduce(values, budget)
        elif any(not f for f in values):
            word = _zero_slot_reduce(values, row, budget)
        else:
            word = _polynomial_reduce(values, budget)+_swap(ring, 0, n-1)
        (a, b), (c, d) = inverse_coordinates
        word = prefix+[(i, j, ring({(a*u+b*v, c*u+d*v): coefficient
                                    for (u, v), coefficient in f.terms.items()}))
                       for i, j, f in word]
    word = [(i, j, f) for i, j, f in word if f]
    budget.word(word)
    # Independent physical-size replay; no enlargement or completion oracle.
    T, Ti = eye(ring, n), eye(ring, n)
    for i, j, f in word:
        for k in range(n):
            T[i, k] += budget.multiply(f, T[j, k])
            Ti[k, j] -= budget.multiply(Ti[k, i], f)
        budget.check(*(list(T.entries.values())+list(Ti.entries.values())))
    target = Matrix(ring, n, 1, {(0, 0): 1})
    if T*column != target or T*Ti != eye(ring, n) or Ti*T != eye(ring, n):
        raise ArithmeticError('Fixed-rank elementary column word failed actual inverse identities')
    return {'factors': _records(ring, word), 'transform': T, 'inverse': Ti,
            'reduced_column': target, 'tracked_bezout': bezout*Ti,
            'added_ancillas': 0, 'method': 'park-woodburn-fixed-rank-column-fp-v1',
            'statistics': {'operations': budget.steps, 'convolution_pairs': budget.products,
                           'factors': len(word), 'patches': len(budget.patches),
                           'closed_points': budget.patches,
                           'row_choices': budget.row_choices}}
