"""Independent chain, path, exterior-minor and augmentation verification.

The constructor uses Berkowitz; this verifier checks minors by fraction-free
Gaussian elimination with exact Laurent polynomial division.  It never calls
the constructor, path construction, or its determinant implementation.
Resource exhaustion propagates as ResourceLimit, never a false obstruction.
"""
from itertools import combinations
from math import comb

from sympy import Matrix as IntegerMatrix

from css_decoupling.algebra.exterior import determinant_mod, integer_rows, subset_at
from css_decoupling.algebra.laurent import Matrix, Ring, eye
from css_decoupling.lattice.koszul_comparison import _Limits
from .comparison import VerificationReport, _int, _keys, _list, _matrix_dict, _matrix_from_dict

SCHEMA = "css-decoupling.koszul-comparison"
SCOPE = "explicit-lattice-koszul-comparison"
CONVENTIONS = {"vectors": "columns", "exterior_basis": "increasing-subsets-lexicographic",
               "paths": "coordinate-axes-in-increasing-order",
               "coefficients": "original-Laurent-ring;new-y_j=x^basis_column_j",
               "differential": "homological;remove-position-r-with-sign-(-1)^r"}


def koszul_comparison_dict(result):
    """Serialize current public maps, never a cached validation snapshot."""
    return {"schema": SCHEMA, "version": 1, "scope": SCOPE,
            "ring": {"prime": result.ring.p, "dimension": result.ring.D},
            "conventions": dict(CONVENTIONS), "basis": [list(row) for row in result.basis],
            "requested_degrees": list(result.requested_degrees),
            "path_map": _matrix_dict(result.path_map),
            "maps": [{"degree": n, "matrix": _matrix_dict(matrix)}
                     for n, matrix in sorted(result.maps.items())]}


def _exact_quotient(a, b, limits):
    """Exact Laurent quotient by ordinary multivariate leading-term reduction.

    Coordinate valuations normalize both operands to ordinary polynomials.
    If a Laurent quotient exists its normalized form has nonnegative powers.
    Every cancellation is charged to the arithmetic budget.
    """
    ring = a.ring
    if not b:
        raise ValueError("Zero divisor in exterior-minor verification")
    if not a:
        return ring.zero()
    if b.is_unit:
        return limits.mul(a, b ** -1)
    shift_a = tuple(min(e[i] for e in a.terms) for i in range(ring.D))
    shift_b = tuple(min(e[i] for e in b.terms) for i in range(ring.D))
    rem = {tuple(x - y for x, y in zip(e, shift_a)): c for e, c in a.terms.items()}
    divisor = {tuple(x - y for x, y in zip(e, shift_b)): c for e, c in b.terms.items()}
    lead_b = max(divisor)
    inverse = pow(divisor[lead_b], -1, ring.p)
    quotient = {}
    while rem:
        lead = max(rem)
        exponent = tuple(x - y for x, y in zip(lead, lead_b))
        if any(x < 0 for x in exponent):
            raise ValueError("Nonexact division in exterior-minor verification")
        coefficient = rem[lead] * inverse % ring.p
        quotient[exponent] = coefficient
        limits.work(len(divisor))
        limits.terms(len(rem) + len(divisor) + len(quotient))
        for e, c in divisor.items():
            position = tuple(x + y for x, y in zip(e, exponent))
            value = (rem.get(position, 0) - coefficient * c) % ring.p
            if value:
                rem[position] = value
            else:
                rem.pop(position, None)
    shift = tuple(x - y for x, y in zip(shift_a, shift_b))
    return ring({tuple(x + y for x, y in zip(e, shift)): c for e, c in quotient.items()})


def _bareiss(rows, ring, limits):
    """Polynomial-time fraction-free elimination with independently kept sign."""
    n = len(rows)
    limits.work(n ** 3)
    if not n:
        return ring.one()
    work = [row.copy() for row in rows]
    previous, sign = ring.one(), 1
    for k in range(n - 1):
        at = next((i for i in range(k, n) if work[i][k]), None)
        if at is None:
            return ring.zero()
        if at != k:
            work[k], work[at] = work[at], work[k]
            sign = -sign
        pivot = work[k][k]
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                numerator = limits.add(limits.mul(pivot, work[i][j]),
                                       -limits.mul(work[i][k], work[k][j]))
                work[i][j] = _exact_quotient(numerator, previous, limits)
            work[i][k] = ring.zero()
        previous = pivot
        limits.terms(sum(len(f.terms) for row in work for f in row))
    return sign * work[-1][-1]


def _read_matrix(raw, ring, name, shape, limits):
    _keys(raw, ("rows", "cols", "entries"), name)
    if (_int(raw["rows"], "rows", 0), _int(raw["cols"], "cols", 0)) != shape:
        raise ValueError(f"Wrong exterior matrix shape for {name}")
    entries = _list(raw["entries"], name + ".entries")
    limits.entries(len(entries))
    total = 0
    for encoded in entries:
        _list(encoded, name + ".entry", 3)
        total += len(_list(encoded[2], name + ".terms"))
        limits.terms(total)
    return _matrix_from_dict(raw, ring, name)


def _path_check(path, basis, limits):
    D, ring = path.ring.D, path.ring
    for j in range(D):
        endpoint = ring.zero()
        for i in range(D):
            length, value = basis[i][j], path[i, j]
            if len(value.terms) != abs(length):
                raise ValueError("Path does not have the required number of oriented edges")
            for exponent, coefficient in value.terms.items():
                low, high = (0, length) if length > 0 else (length, 0)
                if (not low <= exponent[i] < high or
                        coefficient != (1 if length > 0 else ring.p - 1) or
                        any(exponent[k] != (basis[k][j] if k < i else 0)
                            for k in range(D) if k != i)):
                    raise ValueError("Path violates coordinate order or edge orientation")
            endpoint = limits.add(endpoint, limits.mul(value, ring.variable(i) - 1))
            if sum(value.terms.values()) % ring.p != length % ring.p:
                raise ValueError("Path augmentation differs from its displacement")
        expected = ring.monomial(tuple(basis[i][j] for i in range(D))) - 1
        if endpoint != expected:
            raise ValueError("Path boundary differs from the declared lattice endpoint")


def _chain_check(n, matrix, previous, basis, limits):
    """Compare actual sparse boundaries without constructing other degrees."""
    ring, D = matrix.ring, matrix.ring.D
    lower = tuple(combinations(range(D), n - 1))
    lookup = {I: i for i, I in enumerate(lower)}
    difference, total = {}, 0

    def add(at, value):
        nonlocal total
        old = difference.get(at, ring.zero())
        new = limits.add(old, value)
        total += len(new.terms) - len(old.terms)
        limits.terms(total)
        if new:
            difference[at] = new
        else:
            difference.pop(at, None)

    for (i, j), f in matrix.entries.items():
        I = subset_at(i, D, n)
        for position, axis in enumerate(I):
            coefficient = (-1) ** position * (ring.variable(axis) - 1)
            add((lookup[I[:position] + I[position + 1:]], j), limits.mul(f, coefficient))
    by_column = {}
    for (i, j), f in previous.entries.items():
        by_column.setdefault(j, []).append((i, f))
    for j, J in enumerate(combinations(range(D), n)):
        for position, axis in enumerate(J):
            coefficient = (-1) ** (position + 1) * (
                ring.monomial(tuple(basis[i][axis] for i in range(D))) - 1)
            col = lookup[J[:position] + J[position + 1:]]
            for i, f in by_column.get(col, []):
                add((i, j), limits.mul(f, coefficient))
    if difference:
        raise ValueError(f"Chain equation fails in requested degree {n}")


def verify_koszul_comparison(certificate, *, max_terms=2_000_000,
                             max_entries=2_000_000, max_products=20_000_000):
    """Validate paths, exterior powers, all requested chain equations/minors.

    The scope is the stated original ring, integer basis and explicitly listed
    degrees.  Resource caps also apply to untrusted inputs before decoding.
    """
    limits = _Limits(max_terms, max_entries, max_products)
    checks, errors = [], []
    try:
        if hasattr(certificate, "certificate"):
            certificate = certificate.certificate()
        data = _keys(certificate, ("schema", "version", "scope", "ring", "conventions", "basis",
                                   "requested_degrees", "path_map", "maps"), "Koszul comparison")
        if data["schema"] != SCHEMA or _int(data["version"], "version") != 1 or data["scope"] != SCOPE:
            raise ValueError("Unsupported Koszul-comparison certificate")
        if data["conventions"] != CONVENTIONS:
            raise ValueError("Unsupported path or Koszul convention")
        _keys(data["ring"], ("prime", "dimension"), "ring")
        ring = Ring(_int(data["ring"]["prime"], "prime", 2), _int(data["ring"]["dimension"], "dimension", 1))
        D = ring.D
        limits.entries(D ** 2)
        basis = integer_rows(data["basis"], nrows=D, ncols=D, name="lattice basis")
        limits.work(D ** 3)
        if not IntegerMatrix(basis).det():
            raise ValueError("Lattice basis must have full integer rank")
        requested = [_int(n, "requested degree", 0) for n in _list(data["requested_degrees"], "requested degrees")]
        if requested != sorted(set(requested)) or any(n > D for n in requested):
            raise ValueError("Requested exterior degrees must be distinct, increasing and at most D")
        needed = sorted({0, *requested, *(n - 1 for n in requested if n)})
        limits.entries(D ** 2 + sum(comb(D, n) ** 2 for n in needed))
        limits.terms(sum(abs(x) for row in basis for x in row))
        path = _read_matrix(data["path_map"], ring, "path_map", (D, D), limits)
        _path_check(path, basis, limits)
        checks.append("full-rank integer basis; exact directed paths, endpoints and displacements")
        raw_maps = _list(data["maps"], "maps", len(needed))
        maps, stored = {}, sum(len(f.terms) for f in path.entries.values())
        for n, raw in zip(needed, raw_maps):
            _keys(raw, ("degree", "matrix"), "degree map")
            if _int(raw["degree"], "degree", 0) != n:
                raise ValueError("Certificate omits, adds or reorders an exterior degree")
            size = comb(D, n)
            matrix = _read_matrix(raw["matrix"], ring, f"chi_{n}", (size, size), limits)
            stored += sum(len(f.terms) for f in matrix.entries.values())
            limits.terms(stored)
            maps[n] = matrix
            if n == 0 and matrix != eye(ring, 1):
                raise ValueError("chi_0 must be the inclusion lifting the augmentation identity")
            if n == 1 and matrix != path:
                raise ValueError("chi_1 differs from the supplied directed paths")
            if n < 2:
                continue
            subsets = tuple(combinations(range(D), n))
            for i, I in enumerate(subsets):
                for j, J in enumerate(subsets):
                    limits.work(n ** 2)
                    rows = [[path[a, b] for b in J] for a in I]
                    # Avoid elimination for structural zero minors.
                    if any(not any(row) for row in rows) or any(not any(col) for col in zip(*rows)):
                        expected = ring.zero()
                    else:
                        expected = _bareiss(rows, ring, limits)
                    value = matrix[i, j]
                    if value != expected:
                        raise ValueError(f"chi_{n} is not the exterior product of the paths")
                    limits.work(n ** 3)
                    minor = determinant_mod([[basis[a][b] for b in J] for a in I], ring.p)
                    if sum(value.terms.values()) % ring.p != minor:
                        raise ValueError(f"chi_{n} augmentation disagrees with the integer minor")
        checks.append("exterior products checked independently by fraction-free elimination")
        checks.append("chi_0 augmentation identity and prime-field minors in every supplied degree")
        for n in requested:
            if n:
                _chain_check(n, maps[n], maps[n - 1], basis, limits)
        checks.append("actual source and target Koszul chain equations in every requested degree")
    except (ValueError, TypeError, KeyError, AttributeError, IndexError, OverflowError) as exc:
        errors.append(str(exc))
    return VerificationReport(not errors, tuple(checks), tuple(errors), SCOPE)
