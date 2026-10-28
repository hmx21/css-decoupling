"""Bounded positive exact division for binary two-variable Laurent columns.

For every coordinate, valuation(f*q)=valuation(f)+valuation(q), since its
leading coefficient is in an integral domain. Removing coordinate valuations
therefore reduces a possible Laurent quotient to polynomial exact division.

If a divisor term dominates every other term coordinatewise, cancellation
replaces a leading exponent e by e-d+f <= e. The initial coordinate box is
preserved, so mixed-radix integer indices are injective and additive throughout
division. Otherwise graded-lex indices (x+y)*w+x, w=max_degree+1, are safe:
reduction never increases total degree and x<w. Integer bits implement XOR
of coefficients, not ordinary integer multiplication or a finite quotient.

Only checked positive answers are returned. A remainder, unsupported field,
encoded-span cap or exhausted budget returns None for the complete backend.
Every accepted answer satisfies ALL rows of A*Q=B, checked in bounded column
blocks. No negative mathematical conclusion follows from a shortcut miss.
"""
from math import isfinite
from time import perf_counter

from ..algebra.laurent import Matrix, Poly, zeros


class _Limit(Exception):
    pass


def _normalize(polynomial):
    valuation = tuple(min(e[i] for e in polynomial.terms) for i in range(2))
    return valuation, {(e[0]-valuation[0], e[1]-valuation[1]): c
                       for e, c in polynomial.terms.items()}


def single_generator_solution(A, B, *, max_steps=10000000, max_terms=8000000,
                              max_encoded_bits=8000000, timeout=60,
                              verify_columns=32):
    """Return (checked quotient, statistics), or (None, fallback statistics).

    max_terms bounds normalized dictionary support, retained output and the
    current quotient term count, excluding caller-owned inputs. Encoded
    remainders are bounded by max_encoded_bits, not repeatedly bit-counted.
    Each encoded plane is bounded before allocation. These logical caps
    complement a process memory limit; Python object overhead is not counted.
    """
    if not isinstance(A, Matrix) or not isinstance(B, Matrix):
        raise TypeError('Single-generator division requires Laurent matrices')
    if A.ring != B.ring or A.nrows != B.nrows:
        raise ValueError('Single-generator division requires a common ring and ambient rows')
    for name, value in (('max_steps', max_steps), ('max_terms', max_terms),
                        ('max_encoded_bits', max_encoded_bits), ('verify_columns', verify_columns)):
        if type(value) is not int or value < 1:
            raise ValueError(name+' must be a positive integer')
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not isfinite(timeout) or timeout <= 0:
        raise ValueError('timeout must be positive and finite')
    start = perf_counter()
    stats = dict(steps=0, peak_terms=0, encoded_peak_bits=0, rectangular_columns=0,
                 graded_columns=0, columns_completed=0,
                 caps=dict(max_steps=max_steps, max_terms=max_terms,
                           max_encoded_bits=max_encoded_bits, timeout=timeout,
                           verify_columns=verify_columns))
    fixed_terms = 0

    def finish(status, answer=None, **details):
        stats.update(status=status, seconds=perf_counter()-start, **details)
        return answer, stats

    def check(amount=1, stored=0):
        stored += fixed_terms
        stats['steps'] += amount
        stats['peak_terms'] = max(stats['peak_terms'], stored)
        if stats['steps'] > max_steps:
            raise _Limit('step_cap')
        if stored > max_terms:
            raise _Limit('support_cap')
        if perf_counter()-start > timeout:
            raise _Limit('time_cap')

    if A.ncols != 1 or not A:
        return finish('unsupported_operator')
    if A.ring.p != 2 or A.ring.D != 2:
        return finish('unsupported_field_or_dimension')
    if not B:
        return finish('verified', zeros(A.ring, 1, B.ncols), solution_terms=0)
    selected = min(A.entries, key=lambda ij: (len(A.entries[ij].terms), ij))[0]
    divisor = A[selected, 0]
    if len(divisor.terms) > max_terms:
        stats['peak_terms'] = len(divisor.terms)
        return finish('resource_cap', reason='divisor_support_cap')
    valuation_f, F = _normalize(divisor)
    fixed_terms = len(F)
    leading = max(F, key=lambda e: (sum(e), e))
    dominant = tuple(max(e[i] for e in F) for i in range(2))
    use_rectangle = dominant in F
    answer, output_terms = {}, 0
    try:
        for column in range(B.ncols):
            check(stored=output_terms)
            polynomial = B[selected, column]
            if not polynomial:
                stats['columns_completed'] = column+1
                continue
            if len(polynomial.terms)+output_terms > max_terms:
                raise _Limit('support_cap')
            if len(F) == 1:
                # A monomial divisor needs no polynomial plane, even when
                # its Laurent exponents are enormous.
                terms = {(e[0]-valuation_f[0], e[1]-valuation_f[1]): 1
                         for e in polynomial.terms}
                check(len(terms), output_terms+len(terms))
            else:
                valuation_b, P = _normalize(polynomial)
                if use_rectangle:
                    leading = dominant
                    bounds = tuple(max(max(e[i] for e in P), dominant[i]) for i in range(2))
                    radices = tuple(value+1 for value in bounds)
                    bound = radices[0]*radices[1]
                    encode = lambda e: e[0]+radices[0]*e[1]
                    decode = lambda code: (code % radices[0], code // radices[0])
                    stats['rectangular_columns'] += 1
                else:
                    degree = max(max(map(sum, P)), sum(leading))
                    width = degree+1
                    bound = degree*width+degree+1
                    encode = lambda e: sum(e)*width+e[0]

                    def decode(code):
                        total, x = divmod(code, width)
                        return x, total-x

                    stats['graded_columns'] += 1
                stats['encoded_peak_bits'] = max(stats['encoded_peak_bits'], bound)
                if bound > max_encoded_bits:
                    return finish('encoded_span_cap', column=column)
                # All input exponents are distinct; encoded indices are
                # injective. OR therefore assembles the coefficient bits.
                fbits = bits = 0
                for exponent in F:
                    fbits |= 1 << encode(exponent)
                for exponent in P:
                    bits |= 1 << encode(exponent)
                qbits = qcount = 0
                while bits:
                    check(stored=output_terms+len(P)+qcount)
                    exponent = decode(bits.bit_length()-1)
                    shift = (exponent[0]-leading[0], exponent[1]-leading[1])
                    if min(shift) < 0:
                        return finish('nonexact_polynomial_division', column=column)
                    displacement = encode(shift)
                    qbits ^= 1 << displacement
                    qcount += 1
                    bits ^= fbits << displacement
                # Leading codes strictly decrease, so quotient monomials
                # cannot repeat; qcount is their exact count without rescans.
                restore = (valuation_b[0]-valuation_f[0], valuation_b[1]-valuation_f[1])
                terms = {}
                while qbits:
                    low = qbits & -qbits
                    exponent = decode(low.bit_length()-1)
                    qbits ^= low
                    terms[exponent[0]+restore[0], exponent[1]+restore[1]] = 1
                if len(terms) != qcount:
                    raise ArithmeticError('Binary quotient encoding produced a repeated monomial')
            output_terms += len(terms)
            check(stored=output_terms)
            if terms:
                # Internal arithmetic above has canonical validated integer
                # coordinates and GF(2) coefficients; take the fresh dict.
                answer[0, column] = Poly._from_canonical(A.ring, terms)
            stats['columns_completed'] = column+1
        quotient = Matrix._from_entries(A.ring, 1, B.ncols, answer)
        mark = perf_counter()
        for column in range(0, B.ncols, verify_columns):
            check()
            if A*quotient[:, column:column+verify_columns] != B[:, column:column+verify_columns]:
                return finish('other_row_mismatch', column=column)
        return finish('verified', quotient, solution_terms=output_terms, selected_row=selected,
                      verification_seconds=perf_counter()-mark)
    except _Limit as exc:
        return finish('resource_cap', reason=str(exc))
