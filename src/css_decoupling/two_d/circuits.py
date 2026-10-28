"""Exact, stabilized CSS-QCA compilation into disjoint weighted SUM families.

The declared circuit is diag(U, U_inverse), not U direct-sum identity.
The extra q qubits are all Z-product states.  Their stabilizer module is
preserved, with explicitly checked generator changes U_inverse and U.
No Quillen--Suslin completion or Suslin-factorization oracle is used here.
See docs/circuit-route.md for the six-shear identity and physical convention.
"""
from __future__ import annotations

import heapq
from numbers import Integral

from sympy.polys.domains import GF
from sympy.polys.matrices import DomainMatrix

from .algebra import Matrix, Ring, ResourceLimit, block_diag, eye, zeros


class CircuitResourceLimit(ResourceLimit):
    """A compilation/verification budget was reached, not a nonexistence result."""


_VALUATION_MAX_PACKED_BITS = 256_000_000
_VALUATION_MAX_SLOT_BITS = 65_536


def _natural(value, label):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return int(value)


def _term_count(matrix):
    return sum(len(f.terms) for f in matrix.entries.values())


def _same_metadata(actual, expected):
    """Compare JSON metadata without Python's True == 1 == 1.0 coercion."""
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return (actual.keys() == expected.keys() and
                all(_same_metadata(actual[key], value)
                    for key, value in expected.items()))
    if isinstance(expected, list):
        return (len(actual) == len(expected) and
                all(_same_metadata(a, e) for a, e in zip(actual, expected)))
    return actual == expected


def _require_qubit_inverse(U, V):
    if not isinstance(U, Matrix) or not isinstance(V, Matrix):
        raise ValueError("QCA and inverse must be Laurent matrices")
    if U.ring != V.ring or U.ring.D != 2:
        raise ValueError("This compiler requires a common two-dimensional prime ring")
    if U.nrows != U.ncols or U.shape != V.shape:
        raise ValueError("QCA and inverse must be square and have equal shapes")
    identity = eye(U.ring, U.nrows)
    if U * V != identity or V * U != identity:
        raise ValueError("QCA inverse fails a two-sided Laurent identity")


def determinant(matrix):
    """Exact Laurent determinant via SymPy's polynomial DomainMatrix.

    Independent row monomial clearing is undone after the polynomial
    determinant.  Computation is over GF(p), never floating point or a
    finite periodic quotient.  The determinant of the empty matrix is one.
    """
    if matrix.nrows != matrix.ncols:
        raise ValueError("Determinant requires a square matrix")
    ring, q = matrix.ring, matrix.nrows
    if q == 0:
        return ring.one()
    domain = GF(ring.p).poly_ring(*(f"z{i}" for i in range(ring.D)))
    row_terms = [[] for _ in range(q)]
    for (i, _), f in matrix.entries.items():
        row_terms[i].extend(f.terms)
    if any(not terms for terms in row_terms):
        return ring.zero()
    minima = [tuple(min(e[a] for e in terms) for a in range(ring.D))
              for terms in row_terms]
    cleared = {}
    for (i, j), f in matrix.entries.items():
        terms = {tuple(e[a] - minima[i][a] for a in range(ring.D)):
                 domain.domain.convert(c) for e, c in f.terms.items()}
        cleared.setdefault(i, {})[j] = domain.ring.from_dict(terms)
    value = DomainMatrix(cleared, (q, q), domain).det()
    shift = tuple(sum(minimum[a] for minimum in minima) for a in range(ring.D))
    return ring({tuple(e[a] + shift[a] for a in range(ring.D)): int(c)
                 for e, c in value.terms()})


class _BinarySpanLimit(Exception):
    """Internal fallback to sparse polynomials; never a mathematical failure."""


class _BinaryLaurent:
    """Internal F_2[z^+-1] polynomial with a bounded-width integer support.

    The exponent shift can be any Python integer. Only the distance between
    retained exponents is bounded; wide sparse supports use generic Laurent
    arithmetic instead. No numerical or characteristic-zero determinant is
    used, and bit positions never encode coefficients in another field.
    """
    __slots__ = ("shift", "bits")
    MAX_BITS = 65536

    def __init__(self, shift=0, bits=0):
        if bits:
            low = (bits & -bits).bit_length() - 1
            self.shift, self.bits = shift + low, bits >> low
        else:
            self.shift, self.bits = 0, 0

    @classmethod
    def from_poly(cls, value):
        if not value:
            return cls()
        powers = [exponent[0] for exponent in value.terms]
        low, high = min(powers), max(powers)
        if high - low >= cls.MAX_BITS:
            raise _BinarySpanLimit
        return cls(low, sum(1 << (exponent - low) for exponent in powers))

    def to_poly(self, ring):
        terms, bits = {}, self.bits
        while bits:
            low = bits & -bits
            terms[(self.shift + low.bit_length() - 1,)] = 1
            bits ^= low
        return ring(terms)

    def __bool__(self):
        return bool(self.bits)

    @property
    def is_unit(self):
        return self.bits == 1

    def __sub__(self, other):
        if not self:
            return other
        if not other:
            return self
        low = min(self.shift, other.shift)
        high = max(self.shift + self.bits.bit_length(),
                   other.shift + other.bits.bit_length())
        if high - low > self.MAX_BITS:
            raise _BinarySpanLimit
        return _BinaryLaurent(low, (self.bits << (self.shift - low)) ^
                              (other.bits << (other.shift - low)))

    def __mul__(self, other):
        if not self or not other:
            return _BinaryLaurent()
        if self.is_unit:
            return _BinaryLaurent(self.shift + other.shift, other.bits)
        if other.is_unit:
            return _BinaryLaurent(self.shift + other.shift, self.bits)
        if self.bits.bit_length() + other.bits.bit_length() - 1 > self.MAX_BITS:
            raise _BinarySpanLimit
        a, b = self.bits, other.bits
        if a.bit_count() > b.bit_count():
            a, b = b, a
        product = 0
        while a:
            low = a & -a
            product ^= b << (low.bit_length() - 1)
            a ^= low
        return _BinaryLaurent(self.shift + other.shift, product)

    def __rmul__(self, scalar):
        # Only the exact determinant row/column permutation sign reaches
        # this internal scalar operation; -1=1 in F_2.
        return self if scalar % 2 else _BinaryLaurent()

    def __pow__(self, exponent):
        if exponent != -1 or not self.is_unit:
            raise ValueError("Internal binary determinant only inverts monomials")
        return _BinaryLaurent(-self.shift, 1)

    def divide(self, divisor):
        """Laurent Euclidean division using independently cleared monomials.

        If f=z^a F and g=z^b G with nonzero constant terms, ordinary binary
        polynomial division F=Q G+R gives f=(z^(a-b)Q)g+z^a R. A nonzero
        remainder has Laurent support span strictly smaller than g's span.
        """
        if not divisor:
            raise ZeroDivisionError("Laurent division by zero")
        quotient, remainder = 0, self.bits
        width = divisor.bits.bit_length()
        while remainder and remainder.bit_length() >= width:
            shift = remainder.bit_length() - width
            quotient ^= 1 << shift
            remainder ^= divisor.bits << shift
        return (_BinaryLaurent(self.shift - divisor.shift, quotient),
                _BinaryLaurent(self.shift, remainder))


def _univariate_divmod(dividend, divisor):
    """Exact Euclidean division in a one-variable prime Laurent ring.

    Only coefficients in the finite field are inverted. Normalizing each
    operand by its own minimum exponent gives polynomial long division;
    restoring those units gives dividend=quotient*divisor+remainder.
    """
    if dividend.ring != divisor.ring or dividend.ring.D != 1:
        raise ValueError("Euclidean division requires a common univariate Laurent ring")
    if not divisor:
        raise ZeroDivisionError("Laurent division by zero")
    ring, p = dividend.ring, dividend.ring.p
    if not dividend:
        return ring.zero(), ring.zero()
    a = min(e[0] for e in dividend.terms)
    b = min(e[0] for e in divisor.terms)
    remainder = {e[0] - a: c for e, c in dividend.terms.items()}
    denominator = {e[0] - b: c for e, c in divisor.terms.items()}
    degree = max(denominator)
    leading_inverse = pow(denominator[degree], -1, p)
    quotient = {}
    while remainder:
        leading = max(remainder)
        if leading < degree:
            break
        shift, scalar = leading - degree, remainder[leading] * leading_inverse % p
        quotient[shift] = (quotient.get(shift, 0) + scalar) % p
        for power, coefficient in denominator.items():
            target = power + shift
            value = (remainder.get(target, 0) - scalar * coefficient) % p
            if value:
                remainder[target] = value
            else:
                remainder.pop(target, None)
    return (ring({(e + a - b,): c for e, c in quotient.items()}),
            ring({(e + a,): c for e, c in remainder.items()}))


def _unit_pivot_determinant(matrix):
    """Use exact bounded-width F_2 bit arithmetic when possible, else sparse.

    Falling back restarts from the original unchanged matrix, so a span
    limit encountered during Schur updates cannot leave a partial residual.
    """
    if matrix.ring.p == 2 and matrix.ring.D == 1:
        try:
            return _indexed_unit_pivot_determinant(matrix, binary=True)
        except _BinarySpanLimit:
            pass
    return _indexed_unit_pivot_determinant(matrix, binary=False)


def _indexed_unit_pivot_determinant(matrix, *, binary):
    """Exact unit Schur complements and univariate Euclidean row operations.

    If univariate units run out, Euclidean row additions reduce one column
    to its Laurent gcd. A resulting unit permits another Schur step; a
    nonunit alone in its column permits a division-free Laplace step. Only
    monomials are inverted. Multivariate residuals keep the general fallback.
    The pivot score reduces sparse fill; it does not alter correctness.
    """
    if matrix.nrows != matrix.ncols:
        raise ValueError("Determinant requires a square matrix")
    rows = {i: {} for i in range(matrix.nrows)}
    columns = {j: set() for j in range(matrix.ncols)}
    row_weight, column_weight = {}, {}
    def weight(value):
        return value.bits.bit_count() if binary else len(value.terms)

    def span(value):
        return (value.bits.bit_length() - 1 if binary else
                max(e[0] for e in value.terms) - min(e[0] for e in value.terms))

    for (i, j), value in matrix.entries.items():
        if binary:
            value = _BinaryLaurent.from_poly(value)
        rows[i][j] = value
        columns[j].add(i)
        row_weight[i] = row_weight.get(i, 0) + weight(value)
        column_weight[j] = column_weight.get(j, 0) + weight(value)
    active_rows, active_columns = list(rows), list(columns)
    pivots = [(row_weight[i] * column_weight[j], i, j)
              for (i, j), value in matrix.entries.items() if value.is_unit]
    heapq.heapify(pivots)
    entry_count = len(matrix.entries)
    ring = matrix.ring
    factor = _BinaryLaurent(0, 1) if binary else ring.one()
    zero = _BinaryLaurent() if binary else ring.zero()

    def replace(i, j, value):
        nonlocal entry_count
        old = rows[i].get(j)
        delta = weight(value) - (weight(old) if old is not None else 0)
        row_weight[i] = row_weight.get(i, 0) + delta
        column_weight[j] = column_weight.get(j, 0) + delta
        if value:
            rows[i][j] = value
            columns[j].add(i)
            entry_count += old is None
            if value.is_unit:
                heapq.heappush(pivots, (row_weight[i] * column_weight[j], i, j))
        elif old is not None:
            del rows[i][j]
            columns[j].remove(i)
            entry_count -= 1

    while entry_count:
        # Scores are sparse fill heuristics, not algebraic premises. Lazy
        # refresh avoids scanning the entire residual for every pivot.
        while pivots:
            score, i, j = heapq.heappop(pivots)
            pivot = rows.get(i, {}).get(j)
            if pivot is None or not pivot.is_unit:
                continue
            current_score = row_weight[i] * column_weight[j]
            if score != current_score:
                heapq.heappush(pivots, (current_score, i, j))
                continue
            break
        else:
            pivot = None
        if pivot is None:
            if ring.D != 1:
                break
            # No unit exists anywhere. Pick a narrow polynomial, then use
            # determinant-preserving row additions within its actual column.
            # This is Euclid over F_p[z^+-1], not fraction-field elimination.
            _, _, i, j = min((span(value), row_weight[r] * column_weight[c], r, c)
                            for r, values in rows.items() for c, value in values.items())
            pivot = rows[i][j]
            while len(columns[j]) > 1 and not pivot.is_unit:
                r = min((r for r in columns[j] if r != i),
                        key=lambda r: (span(rows[r][j]), row_weight[r], r))
                dividend = rows[r][j]
                quotient, remainder = (dividend.divide(pivot) if binary else
                                       _univariate_divmod(dividend, pivot))
                # Replacing row_r by row_r - quotient*row_i preserves det.
                # Store the already computed exact remainder in column j.
                if quotient:
                    for c, right in rows[i].items():
                        if c != j:
                            replace(r, c, rows[r].get(c, zero) - quotient * right)
                replace(r, j, remainder)
                if remainder:
                    # Its support span is strictly smaller, even when a
                    # negative monomial was cleared before polynomial division.
                    if span(remainder) >= span(pivot):
                        raise ArithmeticError("Laurent Euclidean remainder did not decrease")
                    i, pivot = r, remainder
        sign = -1 if (active_rows.index(i) + active_columns.index(j)) % 2 else 1
        factor *= sign * pivot
        row = [(c, value) for c, value in rows[i].items() if c != j]
        if pivot.is_unit:
            pivot_inverse = pivot ** -1
            column = [(r, rows[r][j] * pivot_inverse) for r in sorted(columns[j]) if r != i]
        else:
            # Laplace expansion along a column with one nonzero entry.
            # No nonunit inverse, rational function, or exact division is used.
            if columns[j] != {i}:
                raise ArithmeticError("Nonunit determinant pivot is not isolated")
            column = []
        for c in list(rows[i]):
            replace(i, c, zero)
        for r in list(columns[j]):
            replace(r, j, zero)
        del rows[i], columns[j]
        active_rows.remove(i)
        active_columns.remove(j)
        for r, left in column:
            for c, right in row:
                replace(r, c, rows[r].get(c, zero) - left * right)
        # Discard obsolete heap records before they dominate retained memory.
        if len(pivots) > 4 * (entry_count + 1):
            pivots = [(row_weight[r] * column_weight[c], r, c)
                      for r, values in rows.items() for c, value in values.items()
                      if value.is_unit]
            heapq.heapify(pivots)
    row_index, column_index = ({old: i for i, old in enumerate(indices)}
                               for indices in (active_rows, active_columns))
    residual = Matrix(ring, len(rows), len(columns),
                      {(row_index[i], column_index[j]): value.to_poly(ring) if binary else value
                       for i, values in rows.items() for j, value in values.items()})
    if binary:
        factor = factor.to_poly(ring)
    return factor * determinant(residual)


def unit_determinant(U, V):
    """Recover the determinant monomial only AFTER proving U V = V U = I.

    Invertibility proves det(U)=c*x^a*y^b.  Specializing each other variable
    to one therefore recovers c and that axis's exponent exactly, without
    general bivariate determinant swell.  This shortcut is invalid for an
    arbitrary matrix, so the inverse is a required and checked argument.
    """
    _require_qubit_inverse(U, V)
    return _unit_determinant_verified(U)


def _univariate_determinant_valuation(matrix):
    """Exact lowest exponent of a nonzero F_2 Laurent determinant.

    Clear each original row's Laurent monomial, then use constant-field row
    addition and common row-monomial division only. Every row division by
    z contributes one to the determinant valuation. Once the constant-term
    matrix has full rank, the remaining determinant has valuation zero.
    This computes a valuation even for a NONUNIT determinant; it does not
    infer that the determinant is a monomial. Only the caller's separately
    checked Laurent inverse justifies reconstructing a unit from this value.

    Row additions never increase the largest polynomial degree. A known
    initial determinant degree bound also proves singularity if stripping
    exceeds that bound. Packed and sparse implementations perform the same
    elementary operations; wide sparse exponents never require a dense box.
    """
    ring, n = matrix.ring, matrix.nrows
    if ring.p != 2 or ring.D != 1 or matrix.ncols != n:
        raise ValueError("Valuation requires a square binary univariate Laurent matrix")
    if not n:
        return 0
    minima, maxima = [None] * n, [None] * n
    for (i, _), polynomial in matrix.entries.items():
        for (power,) in polynomial.terms:
            minima[i] = power if minima[i] is None else min(minima[i], power)
            maxima[i] = power if maxima[i] is None else max(maxima[i], power)
    if any(value is None for value in minima):
        raise ValueError("Singular univariate determinant has no finite valuation")
    spans = [high - low for low, high in zip(minima, maxima)]
    slot = max(spans) + 1
    degree_bound, stripped = sum(spans), 0
    initial_shift = sum(minima)
    if slot <= _VALUATION_MAX_SLOT_BITS and n * n * slot <= _VALUATION_MAX_PACKED_BITS:
        rows = [0] * n
        for (i, j), polynomial in matrix.entries.items():
            for (power,) in polynomial.terms:
                rows[i] |= 1 << (j * slot + power - minima[i])
        constant_mask = ((1 << (n * slot)) - 1) // ((1 << slot) - 1)
        basis = {}
        for value in rows:
            while value:
                constants = value & constant_mask
                if not constants:
                    # No column has a low bit: right shift therefore divides
                    # every entry by z, without spilling into another column.
                    value >>= 1
                    stripped += 1
                    if stripped > degree_bound:
                        raise ValueError("Singular univariate determinant has no finite valuation")
                    continue
                pivot = ((constants & -constants).bit_length() - 1) // slot
                if pivot not in basis:
                    basis[pivot] = value
                    break
                value ^= basis[pivot]
            else:
                raise ValueError("Singular univariate determinant has no finite valuation")
    else:
        # The same recurrence on sparse supports, with full common-monomial
        # jumps. In particular an exponent of size 10**1000 causes no dense
        # integer allocation and no loop through the intervening powers.
        rows = [{} for _ in range(n)]
        for (i, j), polynomial in matrix.entries.items():
            rows[i][j] = {power - minima[i] for (power,) in polynomial.terms}
        basis = {}
        for row in rows:
            while row:
                pivot = min((j for j, support in row.items() if 0 in support), default=None)
                if pivot is None:
                    common = min(min(support) for support in row.values())
                    stripped += common
                    if stripped > degree_bound:
                        raise ValueError("Singular univariate determinant has no finite valuation")
                    row = {j: {power - common for power in support} for j, support in row.items()}
                    continue
                if pivot not in basis:
                    basis[pivot] = row
                    break
                for j, support in basis[pivot].items():
                    value = row.get(j, set()) ^ support
                    if value:
                        row[j] = value
                    else:
                        row.pop(j, None)
            else:
                raise ValueError("Singular univariate determinant has no finite valuation")
    return initial_shift + stripped


def _unit_determinant_verified(U):
    """Specialize after the caller has checked the actual inverse equations.

    This private helper avoids checking identical products twice during a
    single verification. It has no certificate flag or reusable success
    cache; the public determinant API always checks its inverse argument.
    """
    univariate = Ring(U.ring.p, 1)
    exponents = []
    constant = None
    for axis in range(U.ring.D):
        entries = {}
        for position, polynomial in U.entries.items():
            terms = {}
            for exponent, coefficient in polynomial.terms.items():
                power = (exponent[axis],)
                value = (terms.get(power, 0) + coefficient) % U.ring.p
                if value:
                    terms[power] = value
                else:
                    terms.pop(power, None)
            if terms:
                entries[position] = univariate(terms)
        specialized = Matrix(univariate, U.nrows, U.ncols, entries)
        if U.ring.p == 2:
            exponents.append(_univariate_determinant_valuation(specialized))
            constant = 1
        else:
            value = _unit_pivot_determinant(specialized)
            if not value.is_unit:
                raise ArithmeticError("A verified inverse specialized to a nonunit determinant")
            (exponent, coefficient), = value.terms.items()
            if constant is not None and coefficient != constant:
                raise ArithmeticError("Axis determinants disagree on their constant unit")
            constant = coefficient
            exponents.append(exponent[0])
    # Invertibility, not specialization alone, proves that det is one unit.
    return U.ring.monomial(tuple(exponents), constant)


def _determinant_record_verified(U):
    value = _unit_determinant_verified(U)
    if not value.is_unit:
        raise ValueError("An invertible Laurent QCA must have a unit determinant")
    (exponent, coefficient), = value.terms.items()
    return {
        "constant_unit": coefficient,
        "shift_exponent": list(exponent),
        "handling": "cancelled_by_inverse_action_on_Z_product_ancillas",
    }


def gate_matrix(ring, qubits, gate, *, pauli="Z", reverse=False):
    """Return one physical translation SUM family's support action.

    ``offset=e`` means SUM(control at n+e, target at n), for every n,
    with physical weight w: |a,b> -> |a,b+w*a>. Its Z action is
    I-w*x^e E_control,target; its X action is I+w*x^-e E_target,control.
    Reverse inverts the weight. Species must differ, making the family a
    matching on the infinite lattice, with no additional coarse graining.
    """
    qubits = _natural(qubits, "qubits")
    _validate_gate(ring, qubits, gate)
    if type(reverse) is not bool:
        raise ValueError("reverse must be boolean")
    result = eye(ring, qubits)
    control, target = gate["control"], gate["target"]
    offset = tuple(gate["offset"])
    weight = -gate["coefficient"] if reverse else gate["coefficient"]
    if pauli == "Z":
        result[control, target] = ring.monomial(offset, -weight)
    elif pauli == "X":
        result[target, control] = ring.monomial(tuple(-a for a in offset), weight)
    else:
        raise ValueError("pauli must be X or Z")
    return result


def _validate_gate(ring, qubits, gate):
    if not isinstance(ring, Ring) or ring.D != 2:
        raise ValueError("SUM compiler requires a two-dimensional prime ring")
    if not isinstance(gate, dict) or set(gate) != {
            "kind", "control", "target", "offset", "coefficient"}:
        raise ValueError("Malformed CNOT family record")
    kind = "CNOT" if ring.p == 2 else "SUM"
    if (gate["kind"] != kind or type(gate["coefficient"]) is not int or
            not 0 < gate["coefficient"] < ring.p):
        raise ValueError("Gate kind or canonical nonzero physical SUM weight is invalid")
    control = _natural(gate["control"], "control species")
    target = _natural(gate["target"], "target species")
    if control == target or max(control, target) >= qubits:
        raise ValueError("CNOT families require distinct, in-range species")
    offset = gate["offset"]
    if not isinstance(offset, (list, tuple)) or len(offset) != ring.D:
        raise ValueError("CNOT offset dimension is incorrect")
    if any(isinstance(a, bool) or not isinstance(a, Integral) for a in offset):
        raise ValueError("CNOT offsets must be exact integers")


def _validate_layer(ring, qubits, layer):
    if not isinstance(layer, list) or not layer:
        raise ValueError("Circuit layers must be nonempty lists of CNOT families")
    occupied = set()
    for gate in layer:
        _validate_gate(ring, qubits, gate)
        species = {gate["control"], gate["target"]}
        if occupied.intersection(species):
            raise ValueError("Layer contains overlapping translation gate families")
        occupied.update(species)


def _apply_gate_rows(ring, rows, gate, pauli, *, reverse=False):
    """Left-multiply by a gate using exact sparse row operations."""
    control, target = gate["control"], gate["target"]
    exponent = tuple(gate["offset"])
    weight = -gate["coefficient"] if reverse else gate["coefficient"]
    coefficient = -weight
    if pauli == "X":
        control, target = target, control
        exponent = tuple(-a for a in exponent)
        coefficient = weight
    scalar = ring.monomial(exponent, coefficient)
    # Distinct species ensure the source row is not mutated by this update.
    destination, source = rows[control], rows[target]
    term_delta = 0
    for j, f in source.items():
        previous = destination.get(j, ring.zero())
        value = previous + scalar * f
        term_delta += len(value.terms) - len(previous.terms)
        if value:
            destination[j] = value
        else:
            destination.pop(j, None)
    return term_delta


def circuit_product(ring, qubits, layers, *, pauli="Z", reverse=False,
                    max_families=200_000, max_matrix_terms=1_000_000):
    """Reconstruct a gate product, independently of the six-shear recipe.

    Layers are in chronological order: the last layer left-multiplies the
    accumulated action. ``reverse`` reverses order and negates each SUM
    weight to compute the inverse. Consecutive supplied factors are aggregated
    only while their source and destination species sets are disjoint.
    Their off-diagonal products then vanish, making their product exactly
    I plus their sum. No compiler recipe or declared matrix is consulted.

    The term budget bounds each retained shear and reconstructed action,
    not hypothetical matrices from an unperformed monomial-by-monomial
    expansion. Limits signal resource exhaustion only.
    """
    qubits = _natural(qubits, "qubits")
    if pauli not in ("X", "Z"):
        raise ValueError("pauli must be X or Z")
    if type(reverse) is not bool:
        raise ValueError("reverse must be boolean")
    if not isinstance(layers, list):
        raise ValueError("layers must be a list")
    max_families = _natural(max_families, "max_families")
    max_matrix_terms = _natural(max_matrix_terms, "max_matrix_terms")
    count = 0
    for layer in layers:
        _validate_layer(ring, qubits, layer)
        count += len(layer)
        if count > max_families:
            raise CircuitResourceLimit("CNOT-family verification budget exceeded")
    current = eye(ring, qubits)
    if qubits > max_matrix_terms:
        raise CircuitResourceLimit("Circuit product monomial budget exceeded")
    destinations, sources, entries = set(), set(), {}
    shear_terms = 0

    def apply_shear():
        nonlocal current
        if not entries:
            return
        # Matrix multiplication accumulates all inner-index contributions
        # together, avoiding repeated copying of growing row polynomials.
        shear = Matrix(ring, qubits, qubits,
                       {position: ring(terms) for position, terms in entries.items()
                        if terms})
        current = current + shear * current
        if _term_count(current) > max_matrix_terms:
            raise CircuitResourceLimit("Circuit product monomial budget exceeded")

    ordered = reversed(layers) if reverse else layers
    for layer in ordered:
        for gate in reversed(layer) if reverse else layer:
            destination, source = gate["control"], gate["target"]
            exponent = tuple(gate["offset"])
            weight = -gate["coefficient"] if reverse else gate["coefficient"]
            coefficient = -weight
            if pauli == "X":
                destination, source = source, destination
                exponent = tuple(-a for a in exponent)
                coefficient = weight
            if destination in sources or source in destinations:
                apply_shear()
                destinations, sources, entries = set(), set(), {}
                shear_terms = 0
            destinations.add(destination)
            sources.add(source)
            terms = entries.setdefault((destination, source), {})
            previous = terms.get(exponent, 0)
            value = (previous + coefficient) % ring.p
            shear_terms += bool(value) - bool(previous)
            if value:
                terms[exponent] = value
            else:
                terms.pop(exponent, None)
            if shear_terms > max_matrix_terms:
                raise CircuitResourceLimit("Circuit shear monomial budget exceeded")
    apply_shear()
    return current


def _shear_families(matrix, q, upper):
    for (i, j), polynomial in sorted(matrix.entries.items()):
        for exponent, coefficient in sorted(polynomial.terms.items()):
            yield {
                "kind": "CNOT" if matrix.ring.p == 2 else "SUM",
                "control": i if upper else q + i,
                "target": q + j if upper else j,
                "offset": list(exponent),
                "coefficient": -coefficient % matrix.ring.p,
            }


def _resources(q, layers):
    gates = [gate for layer in layers for gate in layer]
    return {
        "original_qubits_per_cell": q,
        "added_Z_ancillas_per_cell": q,
        "added_X_ancillas_per_cell": 0,
        "circuit_qubits_per_cell": 2 * q,
        "depth": len(layers),
        "gate_family_count": len(gates),
        "gates_per_cell": len(gates),
        "range_l1_in_supercell_coordinates": max(
            (sum(abs(a) for a in gate["offset"]) for gate in gates), default=0),
        "range_linf_in_supercell_coordinates": max(
            (max(abs(a) for a in gate["offset"]) for gate in gates), default=0),
        "translation_subgroup_in_supercell_coordinates": [[1, 0], [0, 1]],
        "scheduling": "one_disjoint_translation_family_per_layer",
    }


def compile_qca(U, V, *, max_families=200_000, max_matrix_terms=1_000_000):
    """Compile verified U and U^-1 to an exact stabilized SUM circuit.

    All introduced ancillas start and finish in |0>, stabilized by Z.
    No claim is made that the circuit on 2q species equals U on q species.
    It equals diag(U,V), and its restrictions to the original subsystem
    are recorded by explicit injection and projection matrices.
    """
    _require_qubit_inverse(U, V)
    q, ring = U.nrows, U.ring
    max_families = _natural(max_families, "max_families")
    count = 2 * _term_count(U) + _term_count(V) + 3 * q
    if count > max_families:
        raise CircuitResourceLimit(
            f"Stabilized compiler needs {count} CNOT families; budget {max_families}")
    identity = eye(ring, q)
    # Reverse the written product A(U) B(-V) A(U) A(-I) B(I) A(-I).
    chronological = ((-identity, True), (identity, False), (-identity, True),
                     (U, True), (-V, False), (U, True))
    layers = [[gate] for matrix, upper in chronological
              for gate in _shear_families(matrix, q, upper)]
    K = zeros(ring, 2 * q, q)
    J = zeros(ring, 2 * q, q)
    for i in range(q):
        K[q + i, i] = 1
        J[i, i] = 1
    z_action, z_inverse = block_diag(U, V), block_diag(V, U)
    result = {
        "schema": "css-decoupling-2d-circuit-v1" if ring.p == 2 else "css-decoupling-2d-circuit-v2",
        "method": "six_shear_inverse_stabilization",
        "field_characteristic": ring.p,
        "dimension": 2,
        "original_qca": U.copy(),
        "original_inverse": V.copy(),
        "z_action": z_action,
        "z_inverse": z_inverse,
        "x_action": z_inverse.dagger(),
        "x_inverse": z_action.dagger(),
        "layers": layers,
        "determinant": _determinant_record_verified(U),
        "stabilized_determinant": {"constant_unit": 1, "shift_exponent": [0, 0]},
        "ancillas": {"type": "Z", "state": "zero", "count": q,
                     "z_generators": K, "forward_generator_change": V.copy(),
                     "backward_generator_change": U.copy()},
        "equivalence": {"kind": "exact_block_diagonal_with_product_ancillas",
                        "input_injection": J, "output_projection": J.transpose()},
        "resources": _resources(q, layers),
    }
    verify_circuit(result, max_families=max_families,
                   max_matrix_terms=max_matrix_terms)
    return result


def verify_circuit(bundle, *, max_families=200_000, max_matrix_terms=1_000_000):
    """Check gates, scheduling, inverses, Pauli pairing, ancillas and resources.

    This does not rerun the compiler. It multiplies the supplied Z gates and
    checks the declared inverse and Pauli equations. Every validated SUM
    has inverse weight -w and X action equal to its inverse Z dagger. Thus the
    reversed, weight-negated Z stream has action Z^-1, the forward X stream has action
    (Z^-1)^dagger, and reversed X has action Z^dagger. These exact algebraic
    implications replace three redundant full replays; no gate or matrix
    is accepted from a trusted cache. Root code certificates additionally
    bind the QCA to original/target stabilizer generation witnesses.
    """
    if not isinstance(bundle, dict) or bundle.get("schema") not in (
            "css-decoupling-2d-circuit-v1", "css-decoupling-2d-circuit-v2"):
        raise ValueError("Unsupported circuit schema")
    if bundle.get("method") != "six_shear_inverse_stabilization":
        raise ValueError("Unsupported circuit construction")
    U, V = bundle["original_qca"], bundle["original_inverse"]
    _require_qubit_inverse(U, V)
    q, ring = U.nrows, U.ring
    schema = "css-decoupling-2d-circuit-v1" if ring.p == 2 else "css-decoupling-2d-circuit-v2"
    if (bundle["schema"] != schema or
            not _same_metadata(bundle.get("field_characteristic"), ring.p) or
            not _same_metadata(bundle.get("dimension"), 2)):
        raise ValueError("Circuit ring declaration or schema is inconsistent")
    z_action, z_inverse = block_diag(U, V), block_diag(V, U)
    x_action, x_inverse = z_inverse.dagger(), z_action.dagger()
    for name, expected in (("z_action", z_action), ("z_inverse", z_inverse),
                           ("x_action", x_action), ("x_inverse", x_inverse)):
        if bundle[name] != expected:
            raise ValueError(f"Declared {name} does not equal the stabilized QCA action")
    # For the exact block matrices just compared, every stabilized inverse
    # and Pauli-pairing equation reduces to UV=VU=I, checked above:
    # diag(U,V)diag(V,U)=I and dagger reverses the two factors. Avoid
    # recomputing those identical products six times at doubled dimension.
    layers = bundle["layers"]
    options = {"max_families": max_families, "max_matrix_terms": max_matrix_terms}
    if circuit_product(ring, 2 * q, layers, pauli="Z", **options) != z_action:
        raise ValueError("CNOT product does not equal declared Z action")
    if ring.p != 2:
        for pauli, reverse, expected in (("Z",True,z_inverse), ("X",False,x_action),
                                         ("X",True,x_inverse)):
            if circuit_product(ring, 2*q, layers, pauli=pauli, reverse=reverse, **options) != expected:
                raise ValueError("Actual signed SUM stream differs from the physical Pauli action")
    # Each accepted physical gate G has inverse SUM weight -w and
    # X(G)=(G^-1)^dagger, verified by the explicit gate definition.
    # If the actual chronological Z product is P=G_n ... G_1, then
    # reversed_Z=P^-1, forward_X=(P^-1)^dagger, reversed_X=P^dagger.
    # The declared two-sided inverse and all Pauli equalities above make
    # these products unique, so replaying them adds no independent premise.
    if not _same_metadata(bundle["determinant"], _determinant_record_verified(U)):
        raise ValueError("Original determinant record is incorrect")
    if not _same_metadata(bundle["stabilized_determinant"],
                          {"constant_unit": 1, "shift_exponent": [0, 0]}):
        raise ValueError("Stabilized determinant record is incorrect")
    K, J = zeros(ring, 2 * q, q), zeros(ring, 2 * q, q)
    for i in range(q):
        K[q + i, i] = 1
        J[i, i] = 1
    ancillas = bundle["ancillas"]
    if (not isinstance(ancillas, dict) or ancillas.get("type") != "Z" or
            ancillas.get("state") != "zero" or not _same_metadata(ancillas.get("count"), q)):
        raise ValueError("Added ancilla type/state/count is inconsistent")
    if ancillas["z_generators"] != K:
        raise ValueError("Ancilla Z stabilizers are not the declared product states")
    if ancillas["forward_generator_change"] != V or ancillas["backward_generator_change"] != U:
        raise ValueError("Ancilla generator changes disagree with the QCA")
    if z_action * K != K * V or z_inverse * K != K * U:
        raise ValueError("Ancilla stabilizers fail bidirectional generation")
    equivalence = bundle["equivalence"]
    if equivalence.get("kind") != "exact_block_diagonal_with_product_ancillas":
        raise ValueError("Unsupported QCA/circuit equivalence claim")
    if equivalence["input_injection"] != J or equivalence["output_projection"] != J.transpose():
        raise ValueError("QCA/circuit projection or injection is incorrect")
    if z_action * J != J * U or z_inverse * J != J * V:
        raise ValueError("Z restriction does not equal the original QCA")
    if x_action * J != J * V.dagger() or x_inverse * J != J * U.dagger():
        raise ValueError("X restriction does not equal the original QCA")
    if J.transpose() * J != eye(ring, q):
        raise ValueError("QCA subsystem identification is not an injection/retraction")
    if not _same_metadata(bundle["resources"], _resources(q, layers)):
        raise ValueError("Declared circuit resources do not match actual gates")
    return True
