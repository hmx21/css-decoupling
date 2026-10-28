"""Prime-field exterior pullbacks, with columns as the incoming vectors.

Only the requested exterior degree is visited. Sparse source coefficients
are decoded by combinatorial unranking, without building all source subsets.
The explicit target coefficient-table size has a separate resource cap.
"""
from itertools import combinations
from math import comb
from numbers import Integral

from .laurent import Matrix, zeros
from css_decoupling.errors import ResourceLimit


def integer_rows(values, *, nrows=None, ncols=None, name="matrix"):
    """Validate an integer row table; its columns represent vectors."""
    if hasattr(values, "tolist"):
        values = values.tolist()
    try:
        rows = tuple(tuple(row) for row in values)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer row table") from exc
    width = len(rows[0]) if rows else (0 if ncols is None else ncols)
    if ((nrows is not None and len(rows) != nrows) or
            (ncols is not None and width != ncols) or any(len(row) != width for row in rows)):
        raise ValueError(f"{name} has incompatible dimensions")
    if any(isinstance(x, bool) or not isinstance(x, Integral) for row in rows for x in row):
        raise ValueError(f"{name} must contain integers")
    return tuple(tuple(int(x) for x in row) for row in rows)


def subset_at(index, dimension, degree):
    """Unrank one lexicographic exterior subset, without a subset list."""
    if not 0 <= index < comb(dimension, degree):
        raise ValueError("Exterior coefficient index is out of range")
    result, start = [], 0
    for left in range(degree, 0, -1):
        for value in range(start, dimension - left + 1):
            count = comb(dimension - value - 1, left - 1)
            if index < count:
                result.append(value)
                start = value + 1
                break
            index -= count
    return tuple(result)


def determinant_mod(rows, p):
    """Exact determinant over F_p; elimination also covers odd-prime signs."""
    size = len(rows)
    if any(len(row) != size for row in rows):
        raise ValueError("A determinant requires a square matrix")
    work = [[int(x) % p for x in row] for row in rows]
    determinant = 1
    for j in range(size):
        pivot = next((i for i in range(j, size) if work[i][j]), None)
        if pivot is None:
            return 0
        if pivot != j:
            work[j], work[pivot] = work[pivot], work[j]
            determinant = -determinant
        value = work[j][j]
        determinant = determinant * value % p
        inverse = pow(value, -1, p)
        for i in range(j + 1, size):
            factor = work[i][j] * inverse % p
            for k in range(j + 1, size):
                work[i][k] = (work[i][k] - factor * work[j][k]) % p
    return determinant % p


def rank_mod(rows, p):
    """Rank of a rectangular prime-field row table, including empty tables."""
    work = [[x % p for x in row] for row in rows]
    width = len(work[0]) if work else 0
    if any(len(row) != width for row in work):
        raise ValueError("Rank input is not rectangular")
    rank = 0
    for j in range(width):
        pivot = next((i for i in range(rank, len(work)) if work[i][j]), None)
        if pivot is None:
            continue
        work[rank], work[pivot] = work[pivot], work[rank]
        inverse = pow(work[rank][j], -1, p)
        for i in range(rank + 1, len(work)):
            factor = work[i][j] * inverse % p
            for k in range(j, width):
                work[i][k] = (work[i][k] - factor * work[rank][k]) % p
        rank += 1
        if rank == len(work):
            break
    return rank


def pullback_coefficients(coefficients, degree, source_dimension, columns, *,
                          max_exterior_entries=2_000_000):
    """Pull a matrix-valued exterior form back along a D by r integer table.

    Source and target table columns are ordered by exterior subset, then
    source-copy index. The returned Matrix uses the original ambient Ring
    as its constant coefficient container, even when r differs from D.
    """
    if not isinstance(coefficients, Matrix):
        raise ValueError("Exterior coefficients must be a Matrix")
    if (isinstance(degree, bool) or not isinstance(degree, Integral) or degree < 0 or
            isinstance(source_dimension, bool) or not isinstance(source_dimension, Integral) or
            source_dimension < 0):
        raise ValueError("Exterior degree and source dimension must be nonnegative integers")
    if max_exterior_entries is not None and (isinstance(max_exterior_entries, bool) or
            not isinstance(max_exterior_entries, Integral) or max_exterior_entries < 1):
        raise ValueError("Exterior entry cap must be positive or None")
    ring = coefficients.ring
    if coefficients.ncols != comb(ring.D, degree) * source_dimension:
        raise ValueError("Coefficient table does not match its exterior degree")
    if any(not f.is_constant for f in coefficients.entries.values()):
        raise ValueError("Exterior coefficients must be constant prime-field values")
    columns = integer_rows(columns, nrows=ring.D, name="exterior pullback")
    width = len(columns[0])
    blocks = comb(width, degree)
    output = zeros(ring, coefficients.nrows, blocks * source_dimension)
    if not coefficients or not blocks:
        return output
    by_subset = {}
    for (i, j), value in coefficients.entries.items():
        block, source = divmod(j, source_dimension)
        by_subset.setdefault(block, []).append((i, source, value.constant()))
    decoded = []
    for block, entries in by_subset.items():
        axes = subset_at(block, ring.D, degree)
        # A killed coordinate support, and more generally any dependent row
        # restriction, contributes zero to every minor. Discard it without
        # visiting a potentially enormous target exterior basis.
        if rank_mod([columns[i] for i in axes], ring.p) == degree:
            decoded.append((axes, entries))
    if not decoded:
        return output
    size = coefficients.nrows * output.ncols
    if max_exterior_entries is not None and size > max_exterior_entries:
        raise ResourceLimit(f"Exterior target coefficient table has {size} entries; cap is {max_exterior_entries}")
    for block, target_axes in enumerate(combinations(range(width), degree)):
        for source_axes, entries in decoded:
            minor = determinant_mod([[columns[i][j] for j in target_axes] for i in source_axes], ring.p)
            if minor:
                for row, source, value in entries:
                    at = row, block * source_dimension + source
                    output[at] = output[at] + minor * value
    return output
