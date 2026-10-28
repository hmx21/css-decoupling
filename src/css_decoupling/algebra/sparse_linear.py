"""A bounded positive shortcut for binary Laurent module membership.

One finite coefficient system uses exponent differences already present in
the input.  Its failure says nothing about Laurent solvability: callers must
fall back to a complete module algorithm.  No kernel generators or negative
certificates are inferred from this restricted system.  Python integer bits
implement exact Gaussian elimination over F_2, without an external runtime.
"""
from css_decoupling.algebra.laurent import Matrix, Poly


class _ShiftedExponents:
    """Bounded exponent sums for the current shift only; never reorder work."""
    def __init__(self, max_entries=4096):
        self.maximum = max_entries
        self.shift, self.cache = None, {}

    def set_shift(self, shift):
        if shift != self.shift:
            self.shift = shift
            self.cache.clear()

    def __call__(self, exponent):
        found = self.cache.get(exponent)
        if found is not None:
            return found
        result = tuple(a+b for a, b in zip(exponent, self.shift))
        if len(self.cache) < self.maximum:
            self.cache[exponent] = result
        return result


def _binary_solution(rows, width, right_width, max_steps):
    """A particular solution, with free coefficients zero, or no shortcut."""
    mask = (1 << width) - 1
    basis, steps = {}, 0
    for original in rows:
        row = original
        while row and (pivot := (row & -row).bit_length()-1) < width:
            if pivot not in basis:
                basis[pivot] = row
                break
            steps += 1
            if steps > max_steps:
                return None
            row ^= basis[pivot]
        else:
            if row:
                return None
    # Do not allocate a possibly wide RHS result just to reject its work on
    # the first back-substitution row.
    if len(basis) * right_width > max_steps - steps:
        return None
    solutions = [0] * right_width
    for pivot in sorted(basis, reverse=True):
        steps += right_width
        if steps > max_steps:
            return None
        row = basis[pivot]
        right = row >> width
        tail = (row & mask) ^ (1 << pivot)
        for column in range(right_width):
            if ((right >> column) & 1) ^ ((tail & solutions[column]).bit_count() & 1):
                solutions[column] |= 1 << pivot
    return solutions


def sparse_binary_solution(A, B, *, max_unknowns=20000, max_equations=40000,
                           max_support_pairs=200000, max_terms=1000000,
                           max_coefficient_bits=640000000, max_steps=4000000,
                           max_rhs=4096, max_solution_bits=80000000,
                           max_product_work=None):
    """Return a checked solution of A X = B, or None to request exact fallback.

    Candidate exponents are differences between a RHS exponent and an
    operator exponent in the same equation, together with zero.  When the
    budget permits, their union is shared across all active unknowns, so
    cancellation can use unknowns absent from nonzero RHS rows.  There is
    exactly one bounded linearization, with no radius or degree search.
    Its size depends on stored support, never on an exponent bounding box.

    Odd primes, cap exhaustion, and an inconsistent restricted system all
    return None.  In particular None is not a nonmembership certificate.
    """
    if A.ring != B.ring or A.nrows != B.nrows:
        raise ValueError("Sparse coefficient solve requires matching rings and rows")
    if A.ring.p != 2 or not A or not B:
        return None
    if min(max_unknowns, max_equations, max_support_pairs, max_terms,
           max_coefficient_bits, max_steps, max_rhs, max_solution_bits) < 1:
        return None
    if B.ncols > max_rhs:
        return None
    zero = (0,) * A.ring.D
    rhs_support, input_terms, rhs_equations = {}, 0, 0
    for (row, _), value in B.entries.items():
        input_terms += len(value.terms)
        if input_terms > max_terms:
            return None
        support = rhs_support.setdefault(row, set())
        before = len(support)
        support.update(value.terms)
        rhs_equations += len(support) - before
        if rhs_equations > max_equations:
            return None
    # Bound the pair work before forming any support difference sets.
    pairs = sum(len(value.terms) * len(rhs_support.get(row, ()))
                for (row, _), value in A.entries.items())
    if pairs > max_support_pairs:
        return None
    columns, operator_terms = {}, 0
    for (row, column), value in A.entries.items():
        operator_terms += len(value.terms)
        if operator_terms > max_terms:
            return None
        if column not in columns and len(columns) >= max_unknowns:
            return None
        columns.setdefault(column, []).append((row, value))
    supports = {column: {zero} for column in columns}
    count = len(supports)
    if count > max_unknowns:
        return None
    for column, entries in columns.items():
        for row, value in entries:
            for right in rhs_support.get(row, ()):
                for exponent in value.terms:
                    shift = tuple(b - a for a, b in zip(exponent, right))
                    if shift not in supports[column]:
                        count += 1
                        if count > max_unknowns:
                            return None
                        supports[column].add(shift)
    common = set().union(*supports.values())
    # Enlarging every unknown's support is optional.  Do not discard an
    # affordable individual-support ansatz merely because this enlargement
    # would exceed a later allocation/term cap (wide inverse RHS expose it).
    common_width = len(common) * len(columns)
    if (common_width <= max_unknowns and common_width * B.ncols <= max_solution_bits
            and len(common) * operator_terms <= max_terms):
        supports = {column: common for column in columns}
    width = sum(map(len, supports.values()))
    if width * B.ncols > max_solution_bits:
        return None
    contributions = sum(len(supports[column]) * sum(len(f.terms) for _, f in entries)
                        for column, entries in columns.items())
    if contributions > max_terms:
        return None
    # Small displacement first is a coefficient pivot preference, not a
    # minimum-support claim.  It avoids gratuitously large comparison maps.
    variables = sorted(((column, exponent) for column, support in supports.items()
                        for exponent in support),
                       key=lambda item: (sum(abs(e) for e in item[1]), item[1], item[0]))
    row_keys = {(row, exponent) for row, support in rhs_support.items() for exponent in support}
    if len(row_keys) > max_equations:
        return None
    shifted = _ShiftedExponents()
    for column, shift in variables:
        shifted.set_shift(shift)
        for row, value in columns[column]:
            for exponent in value.terms:
                row_keys.add((row, shifted(exponent)))
                if len(row_keys) > max_equations:
                    return None
    # Preflight the full bit rows before allocating coefficient vectors.
    if len(row_keys) * (width + B.ncols) > max_coefficient_bits:
        return None
    indices = {key: index for index, key in enumerate(sorted(row_keys))}
    rows = [0] * len(indices)
    shifted = _ShiftedExponents()
    for index, (column, shift) in enumerate(variables):
        shifted.set_shift(shift)
        bit = 1 << index
        for row, value in columns[column]:
            for exponent in value.terms:
                key = row, shifted(exponent)
                rows[indices[key]] ^= bit
    for (row, column), value in B.entries.items():
        bit = 1 << (width + column)
        for exponent in value.terms:
            rows[indices[row, exponent]] ^= bit
    solution = _binary_solution(rows, width, B.ncols, max_steps)
    if solution is None or sum(column.bit_count() for column in solution) > max_terms:
        return None
    entries = {}
    for column, coefficients in enumerate(solution):
        while coefficients:
            bit = coefficients & -coefficients
            row, exponent = variables[bit.bit_length() - 1]
            entries.setdefault((row, column), {})[exponent] = 1
            coefficients ^= bit
    answer = Matrix(A.ring, A.ncols, B.ncols,
                    {key: Poly(A.ring, terms) for key, terms in entries.items()})
    if max_product_work is not None:
        right_rows = {}
        for (row, _), value in answer.entries.items():
            right_rows[row] = right_rows.get(row, 0)+len(value.terms)
        work = sum(len(value.terms)*right_rows.get(column, 0)
                   for (_, column), value in A.entries.items())
        if work > max_product_work:
            return None
    if A * answer != B:
        raise ArithmeticError("Sparse coefficient shortcut failed its exact Laurent identity")
    return answer
