"""Explicit prime-field-linear lifts in the standard Laurent Koszul complex.

K_k has the lexicographic exterior basis, with copy index varying fastest,
and differential contraction by (x_0-1,...,x_(D-1)-1).  The homotopy here is
F_p-linear, not Laurent-linear: it acts on polynomial columns and must not
be interpreted as one Laurent matrix representing a universal operator.
"""
from math import comb
from numbers import Integral

from css_decoupling.algebra.exterior import subset_at
from css_decoupling.algebra.laurent import Matrix, Poly, zeros
from css_decoupling.errors import NoLaurentSolution, ResourceLimit


DEFAULT_MAX_TERMS = 2_000_000


def _parameters(B, degree, copies, max_terms, *, allow_empty_top=False):
    if not isinstance(B, Matrix):
        raise TypeError("Koszul lifts require a Laurent Matrix")
    if (isinstance(degree, bool) or not isinstance(degree, Integral) or
            not 0 <= degree <= B.ring.D + int(allow_empty_top)):
        raise ValueError("Exterior degree is outside the Koszul complex")
    if isinstance(copies, bool) or not isinstance(copies, Integral) or copies < 0:
        raise ValueError("Koszul copy count must be a nonnegative integer")
    if max_terms is not None and (isinstance(max_terms, bool) or
            not isinstance(max_terms, Integral) or max_terms < 1):
        raise ValueError("Koszul term cap must be a positive integer or None")
    degree, copies = int(degree), int(copies)
    if B.nrows != comb(B.ring.D, degree) * copies:
        raise ValueError("Koszul column rows do not match exterior degree and copy count")
    return degree, copies


def _subset_index(indices, dimension):
    """Rank one increasing subset without enumerating its exterior layer."""
    index, start, remaining = 0, 0, len(indices)
    for axis in indices:
        index += comb(dimension-start, remaining) - comb(dimension-axis, remaining)
        start, remaining = axis+1, remaining-1
    return index


def _check_terms(count, maximum):
    if maximum is not None and count > maximum:
        raise ResourceLimit(f"Explicit Koszul lift needs more than {maximum} monomial contributions")


def _add(entries, row, column, exponent, coefficient, prime):
    polynomial = entries.setdefault((row, column), {})
    value = (polynomial.get(exponent, 0) + coefficient) % prime
    if value:
        polynomial[exponent] = value
    else:
        polynomial.pop(exponent, None)


def _matrix(ring, rows, columns, entries):
    return Matrix(ring, rows, columns,
                  {at: Poly(ring, terms) for at, terms in entries.items() if terms})


def koszul_boundary(B, exterior_degree, *, copies=1, max_terms=DEFAULT_MAX_TERMS):
    """Apply the oriented differential K_k -> K_(k-1) to sparse columns.

    Degrees zero and D+1 have zero differential; K_(D+1) has rank zero.
    The cap is checked before allocating the signed monomial contributions.
    """
    k, copies = _parameters(B, exterior_degree, copies, max_terms, allow_empty_top=True)
    D, p = B.ring.D, B.ring.p
    target_rows = comb(D, k-1)*copies if k else 0
    if not k or not B:
        return zeros(B.ring, target_rows, B.ncols)
    _check_terms(2*k*sum(len(f.terms) for f in B.entries.values()), max_terms)
    decoded, entries = {}, {}
    for (row, column), polynomial in B.entries.items():
        block, copy = divmod(row, copies)
        if block not in decoded:
            decoded[block] = subset_at(block, D, k)
        indices = decoded[block]
        for position, axis in enumerate(indices):
            target = _subset_index(indices[:position]+indices[position+1:], D)*copies + copy
            sign = -1 if position % 2 else 1
            for exponent, coefficient in polynomial.terms.items():
                shifted = list(exponent)
                shifted[axis] += 1
                _add(entries, target, column, tuple(shifted), sign*coefficient, p)
                _add(entries, target, column, exponent, -sign*coefficient, p)
    return _matrix(B.ring, target_rows, B.ncols, entries)


def koszul_homotopy(B, exterior_degree, *, copies=1, max_terms=DEFAULT_MAX_TERMS):
    """Apply the explicit F_p-linear homotopy H:K_k -> K_(k+1).

    On f e_I, sum over i<min(I), or all i when I is empty.  First evaluate
    x_0,...,x_(i-1) at one, then take the divided difference in x_i and
    prepend e_i to e_I.  Thus

        partial H + H partial = identity - augmentation_section.

    The augmentation term is present only on K_0.  For exponent a>0 the
    divided difference is sum_(t=0)^(a-1) x_i^t; for a<0 it is
    -sum_(t=a)^(-1) x_i^t.  There is no fraction-field division.  The sum of
    absolute exponents bounds the actual finite geometric work and is
    checked before constructing any of these terms.  Cancellation may make
    the retained result smaller than this conservative work bound.
    """
    k, copies = _parameters(B, exterior_degree, copies, max_terms)
    D, p = B.ring.D, B.ring.p
    target_rows = comb(D, k+1)*copies
    if not B or k == D:
        return zeros(B.ring, target_rows, B.ncols)
    decoded, work = {}, 0
    for (row, _), polynomial in B.entries.items():
        block, _ = divmod(row, copies)
        if block not in decoded:
            decoded[block] = subset_at(block, D, k)
        indices = decoded[block]
        stop = indices[0] if indices else D
        for exponent in polynomial.terms:
            work += sum(abs(exponent[i]) for i in range(stop))
            _check_terms(work, max_terms)
    entries = {}
    for (row, column), polynomial in B.entries.items():
        block, copy = divmod(row, copies)
        indices = decoded[block]
        stop = indices[0] if indices else D
        for axis in range(stop):
            target = _subset_index((axis,)+indices, D)*copies + copy
            for exponent, coefficient in polynomial.terms.items():
                power = exponent[axis]
                if not power:
                    continue
                start, end, sign = (0, power, 1) if power > 0 else (power, 0, -1)
                # Earlier variables have already been augmented, while later
                # variables retain all their positive and negative powers.
                prefix, suffix = (0,)*axis, exponent[axis+1:]
                for value in range(start, end):
                    _add(entries, target, column, prefix+(value,)+suffix, sign*coefficient, p)
    return _matrix(B.ring, target_rows, B.ncols, entries)


def koszul_preimage(B, exterior_degree, *, copies=1, max_terms=DEFAULT_MAX_TERMS):
    """Return Y with partial Y=B for a standard Koszul boundary column.

    A noncycle is rejected by its actual nonzero differential.  On K_0 the
    augmentation must vanish.  These are exact nonmembership obstructions,
    not resource failures.  The constructed preimage is always multiplied
    through the oriented differential and compared with the entire input.
    """
    k, copies = _parameters(B, exterior_degree, copies, max_terms)
    if k == 0:
        if any(sum(f.terms.values()) % B.ring.p for f in B.entries.values()):
            raise NoLaurentSolution("A degree-zero Koszul boundary must have zero augmentation")
    elif koszul_boundary(B, k, copies=copies, max_terms=max_terms):
        raise NoLaurentSolution("Koszul preimage input is not a cycle")
    result = koszul_homotopy(B, k, copies=copies, max_terms=max_terms)
    if koszul_boundary(result, k+1, copies=copies, max_terms=max_terms) != B:
        raise ArithmeticError("Explicit Koszul preimage failed its exact boundary identity")
    return result
