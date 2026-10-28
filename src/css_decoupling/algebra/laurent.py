"""Sparse exact arithmetic over arbitrary-dimensional prime Laurent rings.

The dimension and characteristic are part of every object's type.  No implicit
change of coefficient field, variable count, or fraction-field division occurs.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from types import MappingProxyType
from typing import Mapping, Sequence

from sympy import isprime


# Encoding is worthwhile only when input terms are reused in many scalar
# products.  The bit limit bounds integer-key overhead; it does not restrict
# the supported Laurent exponents, since the tuple algorithm remains exact.
_CONVOLUTION_MIN_PRODUCTS = 4096
_CONVOLUTION_REUSE_FACTOR = 4
_CONVOLUTION_MAX_BITS = 4096
_BINARY_CONVOLUTION_MAX_SPAN = 16384
_BINARY_CONVOLUTION_MAX_STORAGE_BITS = 256000000


def _integer(value, what="integer"):
    if not isinstance(value, Integral):
        raise TypeError(f"{what} must be an integer")
    return int(value)


@dataclass(frozen=True)
class Ring:
    """F_p[x_1^±1,...,x_D^±1]; variable indices in the Python API start at 0."""

    p: int
    D: int

    def __post_init__(self):
        p, D = _integer(self.p, "characteristic"), _integer(self.D, "dimension")
        if not isprime(p):
            raise ValueError("The characteristic must be prime")
        if D < 1:
            raise ValueError("The Laurent ring must have at least one variable")
        object.__setattr__(self, "p", p)
        object.__setattr__(self, "D", D)

    def __call__(self, value=0):
        if isinstance(value, Poly):
            if value.ring != self:
                raise ValueError("Cannot coerce between different Laurent rings")
            return value
        if isinstance(value, Mapping):
            return Poly(self, value)
        coefficient = _integer(value, "coefficient") % self.p
        return Poly._from_canonical(self, {(0,) * self.D: coefficient} if coefficient else {})

    def zero(self):
        return Poly._from_canonical(self, {})

    def one(self):
        return self(1)

    def monomial(self, exponent, coefficient=1):
        return Poly(self, {tuple(exponent): coefficient})

    def variable(self, index):
        index = _integer(index, "variable index")
        if not 0 <= index < self.D:
            raise IndexError("Variable index outside this ring")
        return Poly._from_canonical(self, {tuple(int(i == index) for i in range(self.D)): 1})

    @property
    def variables(self):
        return tuple(self.variable(i) for i in range(self.D))


class Poly:
    """Immutable sparse Laurent polynomial; coefficients are canonical residues."""

    __slots__ = ("ring", "terms", "_key")

    def __init__(self, ring: Ring, terms: Mapping):
        self.ring = ring
        normalized = {}
        for exponent, coefficient in terms.items():
            exponent = tuple(_integer(e, "exponent") for e in exponent)
            if len(exponent) != ring.D:
                raise ValueError("Exponent dimension does not match the Laurent ring")
            coefficient = _integer(coefficient, "coefficient") % ring.p
            if coefficient:
                normalized[exponent] = coefficient
        self.terms = MappingProxyType(normalized)
        self._key = None

    @classmethod
    def _from_canonical(cls, ring, terms):
        """Take ownership of a fresh, already canonical internal dictionary.

        Only arithmetic in this module uses this constructor. Keys must be
        D-tuples of Python integers and coefficients integers in 1,...,p-1.
        The caller must not retain a mutable alias. Public construction
        always copies and validates external data before storing it.
        """
        result = object.__new__(cls)
        result.ring = ring
        result.terms = MappingProxyType(terms)
        result._key = None
        return result

    def key(self):
        if self._key is None:
            self._key = tuple(sorted(self.terms.items()))
        return self._key

    def __hash__(self):
        return hash((self.ring, self.key()))

    def __bool__(self):
        return bool(self.terms)

    def __eq__(self, other):
        if isinstance(other, Integral):
            other = self.ring(other)
        return isinstance(other, Poly) and self.ring == other.ring and self.terms == other.terms

    def __neg__(self):
        if self.ring.p == 2 or not self:
            return self
        p = self.ring.p
        return Poly._from_canonical(self.ring, {e: p-c for e, c in self.terms.items()})

    def __add__(self, other):
        if isinstance(other, Matrix):
            return NotImplemented
        other = self.ring(other)
        if not self:
            return other
        if not other:
            return self
        out = dict(self.terms)
        p = self.ring.p
        for e, c in other.terms.items():
            value = out.get(e, 0) + c
            if value >= p:
                value -= p
            if value:
                out[e] = value
            else:
                out.pop(e, None)
        return Poly._from_canonical(self.ring, out)

    __radd__ = __add__

    def __sub__(self, other):
        other = self.ring(other)
        if not other:
            return self
        if not self:
            return -other
        out = dict(self.terms)
        p = self.ring.p
        for e, c in other.terms.items():
            value = out.get(e, 0) - c
            if value < 0:
                value += p
            if value:
                out[e] = value
            else:
                out.pop(e, None)
        return Poly._from_canonical(self.ring, out)

    def __rsub__(self, other):
        return self.ring(other) - self

    def __mul__(self, other):
        if isinstance(other, Matrix):
            return NotImplemented
        other = self.ring(other)
        if not self or not other:
            return self.ring.zero()
        constant_exponent = (0,) * self.ring.D
        if len(other.terms) == 1 and other.terms.get(constant_exponent) == 1:
            return self
        if len(self.terms) == 1 and self.terms.get(constant_exponent) == 1:
            return other
        if len(self.terms) == 1 or len(other.terms) == 1:
            unit, polynomial = (self, other) if len(self.terms) == 1 else (other, self)
            (shift, coefficient), = unit.terms.items()
            if self.ring.D == 2:
                a, b = shift
                return Poly._from_canonical(self.ring, {
                    (e[0]+a, e[1]+b): c*coefficient % self.ring.p
                    for e, c in polynomial.terms.items()})
            return Poly._from_canonical(self.ring, {
                tuple(a+b for a, b in zip(e, shift)): c*coefficient % self.ring.p
                for e, c in polynomial.terms.items()})
        out = {}
        p = self.ring.p
        if p == 2 and self.ring.D == 2:
            support = set()
            for a, b in self.terms:
                for c, d in other.terms:
                    exponent = a+c, b+d
                    if exponent in support:
                        support.remove(exponent)
                    else:
                        support.add(exponent)
            return Poly._from_canonical(self.ring, dict.fromkeys(support, 1))
        for a, c in self.terms.items():
            for b, d in other.terms.items():
                e = tuple(x + y for x, y in zip(a, b))
                value = (out.get(e, 0) + c * d) % p
                if value:
                    out[e] = value
                else:
                    out.pop(e, None)
        return Poly._from_canonical(self.ring, out)

    __rmul__ = __mul__

    def __pow__(self, exponent):
        exponent = _integer(exponent, "power")
        if exponent < 0:
            if len(self.terms) != 1:
                raise ValueError("Only Laurent units admit negative powers")
            (e, c), = self.terms.items()
            return Poly._from_canonical(self.ring, {
                tuple(exponent * a for a in e): pow(c, exponent, self.ring.p)})
        out, base = self.ring.one(), self
        while exponent:
            if exponent & 1:
                out = out * base
            exponent //= 2
            if exponent:
                base = base * base
        return out

    def involution(self):
        return Poly._from_canonical(self.ring, {tuple(-a for a in e): c for e, c in self.terms.items()})

    dagger = involution

    @property
    def is_unit(self):
        return len(self.terms) == 1

    @property
    def is_constant(self):
        return all(not any(e) for e in self.terms)

    def constant(self):
        if not self.is_constant:
            raise ValueError("Polynomial is not constant")
        return self.terms.get((0,) * self.ring.D, 0)

    def __repr__(self):
        if not self:
            return "0"
        terms = []
        for exponent, coefficient in self.key():
            factors = [f"x{i + 1}" if e == 1 else f"x{i + 1}^{e}" for i, e in enumerate(exponent) if e]
            if coefficient != 1 or not factors:
                factors.insert(0, str(coefficient))
            terms.append("*".join(factors))
        return " + ".join(terms)


class Matrix:
    """Mutable sparse matrix, with strict dimensions and immutable entries."""

    def __init__(self, ring: Ring, nrows: int, ncols: int, entries=None):
        self.ring = ring
        self.nrows = _integer(nrows, "row count")
        self.ncols = _integer(ncols, "column count")
        if self.nrows < 0 or self.ncols < 0:
            raise ValueError("Matrix dimensions cannot be negative")
        self.entries = {}
        if entries is not None:
            if isinstance(entries, Mapping):
                for key, value in entries.items():
                    self[key] = value
            else:
                if len(entries) != self.nrows or any(len(row) != self.ncols for row in entries):
                    raise ValueError("Dense matrix data has incorrect dimensions")
                for i, row in enumerate(entries):
                    for j, value in enumerate(row):
                        self[i, j] = value

    @classmethod
    def _from_entries(cls, ring, nrows, ncols, entries):
        """Own a fresh sparse dict derived from validated matrices internally.

        Dimensions and indices are already valid Python integers, and all
        values are nonzero immutable Poly objects over ring. This private
        path never accepts external matrix data or changes public checks.
        """
        result = object.__new__(cls)
        result.ring, result.nrows, result.ncols = ring, nrows, ncols
        result.entries = entries
        return result

    @classmethod
    def from_rows(cls, ring, rows, *, ncols=None):
        rows = list(rows)
        width = len(rows[0]) if rows else (0 if ncols is None else ncols)
        return cls(ring, len(rows), width, rows)

    @property
    def shape(self):
        return self.nrows, self.ncols

    def copy(self):
        return Matrix._from_entries(self.ring, self.nrows, self.ncols, self.entries.copy())

    def key(self):
        return self.ring, self.shape, tuple((i, j, f.key()) for (i, j), f in sorted(self.entries.items()))

    def __bool__(self):
        return bool(self.entries)

    def _index(self, index, size):
        index = _integer(index, "matrix index")
        if index < 0:
            index += size
        if not 0 <= index < size:
            raise IndexError("Matrix index out of range")
        return index

    def _indices(self, index, size):
        if isinstance(index, slice):
            return list(range(*index.indices(size)))
        if isinstance(index, (list, tuple, range)):
            return [self._index(i, size) for i in index]
        return [self._index(index, size)]

    def __getitem__(self, key):
        i, j = key
        if isinstance(i, Integral) and isinstance(j, Integral):
            value = self.entries.get((self._index(i, self.nrows), self._index(j, self.ncols)))
            return self.ring.zero() if value is None else value
        rows, cols = self._indices(i, self.nrows), self._indices(j, self.ncols)
        row_lookup, col_lookup = {}, {}
        for k, old in enumerate(rows):
            row_lookup.setdefault(old, []).append(k)
        for k, old in enumerate(cols):
            col_lookup.setdefault(old, []).append(k)
        entries = {}
        for (old_i, old_j), value in self.entries.items():
            for new_i in row_lookup.get(old_i, []):
                for new_j in col_lookup.get(old_j, []):
                    entries[new_i, new_j] = value
        return Matrix._from_entries(self.ring, len(rows), len(cols), entries)

    def __setitem__(self, key, value):
        i, j = key
        if isinstance(i, Integral) and isinstance(j, Integral):
            index = self._index(i, self.nrows), self._index(j, self.ncols)
            value = self.ring(value)
            if value:
                self.entries[index] = value
            else:
                self.entries.pop(index, None)
            return
        rows, cols = self._indices(i, self.nrows), self._indices(j, self.ncols)
        if not isinstance(value, Matrix) or value.ring != self.ring or value.shape != (len(rows), len(cols)):
            raise ValueError("Slice assignment requires a matrix with matching ring and dimensions")
        value = value.copy() if value is self else value
        selected_rows, selected_cols = set(rows), set(cols)
        for row, col in list(self.entries):
            if row in selected_rows and col in selected_cols:
                del self.entries[row, col]
        for (row, col), polynomial in value.entries.items():
            self.entries[rows[row], cols[col]] = polynomial

    def __eq__(self, other):
        if isinstance(other, Integral) and other == 0:
            return not self
        return isinstance(other, Matrix) and self.ring == other.ring and self.shape == other.shape and self.entries == other.entries

    def _compatible(self, other, same_shape=True):
        if not isinstance(other, Matrix) or self.ring != other.ring:
            raise ValueError("Matrix operations require a common Laurent ring")
        if same_shape and self.shape != other.shape:
            raise ValueError("Matrix dimensions do not agree")

    def __neg__(self):
        return Matrix._from_entries(self.ring, self.nrows, self.ncols,
                                    {ij: -f for ij, f in self.entries.items()})

    def __add__(self, other):
        if isinstance(other, Integral) and other == 0:
            return self.copy()
        self._compatible(other)
        entries = self.entries.copy()
        for ij, f in other.entries.items():
            value = entries[ij] + f if ij in entries else f
            if value:
                entries[ij] = value
            else:
                entries.pop(ij, None)
        return Matrix._from_entries(self.ring, self.nrows, self.ncols, entries)

    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, Integral) and other == 0:
            return self.copy()
        self._compatible(other)
        entries = self.entries.copy()
        for ij, f in other.entries.items():
            value = entries[ij] - f if ij in entries else -f
            if value:
                entries[ij] = value
            else:
                entries.pop(ij, None)
        return Matrix._from_entries(self.ring, self.nrows, self.ncols, entries)

    def __rsub__(self, other):
        if isinstance(other, Integral) and other == 0:
            return -self
        return NotImplemented

    def __mul__(self, other):
        if not isinstance(other, Matrix):
            scalar = self.ring(other)
            if not scalar:
                return zeros(self.ring, self.nrows, self.ncols)
            return Matrix._from_entries(self.ring, self.nrows, self.ncols,
                                        {ij: f * scalar for ij, f in self.entries.items()})
        self._compatible(other, same_shape=False)
        if self.ncols != other.nrows:
            raise ValueError("Matrix multiplication dimensions do not agree")
        right_rows, right_work, input_terms = {}, {}, 0
        for (k, j), f in other.entries.items():
            right_rows.setdefault(k, []).append((j, f))
            term_count = len(f.terms)
            right_work[k] = right_work.get(k, 0) + term_count
            input_terms += term_count
        products = 0
        for (_, k), f in self.entries.items():
            term_count = len(f.terms)
            input_terms += term_count
            products += term_count * right_work.get(k, 0)
        if products >= max(_CONVOLUTION_MIN_PRODUCTS, _CONVOLUTION_REUSE_FACTOR * input_terms):
            encoded = _encoded_matrix_product(self, other, right_rows)
            if encoded is not None:
                return encoded
        entries = {}
        p = self.ring.p
        for (i, k), a in self.entries.items():
            for j, b in right_rows.get(k, []):
                terms = entries.setdefault((i, j), {})
                for left_exponent, left_coefficient in a.terms.items():
                    for right_exponent, right_coefficient in b.terms.items():
                        exponent = tuple(x + y for x, y in zip(left_exponent, right_exponent))
                        value = (terms.get(exponent, 0) + left_coefficient * right_coefficient) % p
                        if value:
                            terms[exponent] = value
                        else:
                            terms.pop(exponent, None)
        # Accumulate all inner-index contributions before normalizing each
        # output entry. Rebuilding a growing immutable polynomial per summand
        # can otherwise make sparse certificate verification quadratic.
        return Matrix._from_entries(self.ring, self.nrows, other.ncols,
                      {ij: Poly._from_canonical(self.ring, terms)
                       for ij, terms in entries.items() if terms})

    __matmul__ = __mul__

    def __rmul__(self, scalar):
        return self * scalar

    def __pow__(self, exponent):
        exponent = _integer(exponent, "matrix power")
        if self.nrows != self.ncols:
            raise ValueError("Matrix powers require a square matrix")
        if exponent < 0:
            return self.inverse() ** (-exponent)
        out, base = eye(self.ring, self.nrows), self
        while exponent:
            if exponent & 1:
                out = out * base
            exponent //= 2
            if exponent:
                base = base * base
        return out

    def transpose(self):
        return Matrix._from_entries(self.ring, self.ncols, self.nrows,
                                    {(j, i): f for (i, j), f in self.entries.items()})

    @property
    def T(self):
        return self.transpose()

    def dagger(self):
        return Matrix._from_entries(self.ring, self.ncols, self.nrows,
                                    {(j, i): f.involution() for (i, j), f in self.entries.items()})

    def columns(self):
        for j in range(self.ncols):
            yield tuple(self[i, j] for i in range(self.nrows))

    def rows(self):
        for i in range(self.nrows):
            yield tuple(self[i, j] for j in range(self.ncols))

    def hstack(self, *others):
        return hstack(self, *others)

    augment = hstack

    def vstack(self, *others):
        return vstack(self, *others)

    stack = vstack

    def inverse(self):
        if self.nrows != self.ncols:
            raise ValueError("Inverse requires a square matrix")
        from css_decoupling.algebra.modules import solve
        inverse = solve(self, eye(self.ring, self.nrows))
        if inverse * self != eye(self.ring, self.nrows):
            raise ArithmeticError("Two-sided matrix inverse certificate failed")
        return inverse

    def __repr__(self):
        return f"Matrix({self.ring!r}, {self.nrows}, {self.ncols}, {self.entries!r})"


def _encoded_matrix_product(left, right, right_rows):
    """Sparse mixed-radix convolution, or None when integer keys are too wide.

    Write exponent e in the left support as e-min_left, and analogously
    for the right support.  Radix_i = span_left_i+span_right_i+1 makes every
    *sum* of two coordinate digits smaller than its radix.  Consequently
    addition of encoded integer keys is exactly addition of exponent vectors,
    without carries between axes.  No bounding box is allocated or traversed.
    """
    dimension, p = left.ring.D, left.ring.p
    bounds = []
    for operand in (left, right):
        iterator = (exponent for f in operand.entries.values() for exponent in f.terms)
        seed = next(iterator, None)
        if seed is None:
            return Matrix(left.ring, left.nrows, right.ncols)
        low, high = list(seed), list(seed)
        for exponent in iterator:
            for axis, value in enumerate(exponent):
                if value < low[axis]:
                    low[axis] = value
                if value > high[axis]:
                    high[axis] = value
        bounds.append((low, high))
    low_left, high_left = bounds[0]
    low_right, high_right = bounds[1]
    radices, strides, stride, bit_bound = [], [], 1, 0
    for axis in range(dimension):
        radix = high_left[axis] - low_left[axis] + high_right[axis] - low_right[axis] + 1
        bit_bound += (radix - 1).bit_length()
        if bit_bound > _CONVOLUTION_MAX_BITS:
            return None
        radices.append(radix)
        strides.append(stride)
        stride *= radix
    shift = tuple(a + b for a, b in zip(low_left, low_right))

    if (p == 2 and stride <= _BINARY_CONVOLUTION_MAX_SPAN and
            stride * right.ncols * (left.nrows + right.nrows) <=
            _BINARY_CONVOLUTION_MAX_STORAGE_BITS):
        return _binary_row_encoded_matrix_product(left, right, low_left, low_right,
                                                   radices, strides, stride, shift)
    if p == 2 and stride <= _BINARY_CONVOLUTION_MAX_SPAN:
        # Tile only the output-column coordinate.  Every tile retains the
        # full inner matrix dimension and all Laurent terms, so its answer
        # is an exact column block of the same product.  This preserves row
        # bit parallelism when the complete packed block exceeds the cap.
        tile_width = _BINARY_CONVOLUTION_MAX_STORAGE_BITS // (
            stride * max(1, left.nrows+right.nrows))
        if tile_width and (right.ncols+tile_width-1)//tile_width <= 64:
            entries = {}
            for start in range(0, right.ncols, tile_width):
                tile = _binary_row_encoded_matrix_product(
                    left, right, low_left, low_right, radices, strides, stride, shift,
                    column_start=start, column_stop=min(start+tile_width, right.ncols))
                entries.update(tile.entries)
            return Matrix._from_entries(left.ring, left.nrows, right.ncols, entries)
    if (p == 2 and stride <= _BINARY_CONVOLUTION_MAX_SPAN and
            stride * (len(left.entries) + len(right.entries)) <=
            _BINARY_CONVOLUTION_MAX_STORAGE_BITS):
        return _binary_encoded_matrix_product(left, right, right_rows,
                                              low_left, low_right, radices, strides, shift)

    def encode_operand(operand, low):
        exponent_cache = {}
        encoded = {}
        for position, polynomial in operand.entries.items():
            terms = []
            for exponent, coefficient in polynomial.terms.items():
                key = exponent_cache.get(exponent)
                if key is None:
                    key = sum((value - minimum) * weight
                              for value, minimum, weight in zip(exponent, low, strides))
                    exponent_cache[exponent] = key
                terms.append((key, coefficient))
            encoded[position] = terms
        return encoded

    encoded_left = encode_operand(left, low_left)
    encoded_right = encode_operand(right, low_right)
    entries = {}
    for (i, k), a in encoded_left.items():
        for j, _ in right_rows.get(k, ()):
            b = encoded_right[k, j]
            terms = entries.setdefault((i, j), {})
            for exponent_left, coefficient_left in a:
                for exponent_right, coefficient_right in b:
                    exponent = exponent_left + exponent_right
                    value = (terms.get(exponent, 0) + coefficient_left * coefficient_right) % p
                    if value:
                        terms[exponent] = value
                    else:
                        terms.pop(exponent, None)
    result = {}
    decoded_cache = {}
    for position, encoded_terms in entries.items():
        if not encoded_terms:
            continue
        terms = {}
        for key, coefficient in encoded_terms.items():
            exponent = decoded_cache.get(key)
            if exponent is None:
                value, powers = key, []
                for radix, minimum in zip(radices, shift):
                    value, digit = divmod(value, radix)
                    powers.append(digit + minimum)
                exponent = tuple(powers)
                decoded_cache[key] = exponent
            terms[exponent] = coefficient
        result[position] = Poly._from_canonical(left.ring, terms)
    return Matrix._from_entries(left.ring, left.nrows, right.ncols, result)


def _binary_row_encoded_matrix_product(left, right, low_left, low_right,
                                       radices, strides, stride, shift,
                                       column_start=0, column_stop=None):
    """Pack whole sparse right rows, including the output-column coordinate.

    The column stride exceeds every product exponent code.  Therefore XOR
    and shifts perform all columns of one row multiplication at once without
    cross-column carries.  Allocation is preflighted by the caller; support
    and matrix dimensions otherwise have the usual unrestricted fallback.
    """
    if column_stop is None:
        column_stop = right.ncols
    exponent_cache, right_bits = {}, {}
    for (k, j), polynomial in right.entries.items():
        if not column_start <= j < column_stop:
            continue
        bits = 0
        for exponent in polynomial.terms:
            key = exponent_cache.get(exponent)
            if key is None:
                key = sum((value-minimum)*weight
                          for value, minimum, weight in zip(exponent, low_right, strides))
                exponent_cache[exponent] = key
            bits |= 1 << ((j-column_start)*stride+key)
        right_bits[k] = right_bits.get(k, 0) | bits
    exponent_cache, result_bits = {}, {}
    for (i, k), polynomial in left.entries.items():
        right_row = right_bits.get(k, 0)
        if not right_row:
            continue
        value = result_bits.get(i, 0)
        for exponent in polynomial.terms:
            key = exponent_cache.get(exponent)
            if key is None:
                key = sum((power-minimum)*weight
                          for power, minimum, weight in zip(exponent, low_left, strides))
                exponent_cache[exponent] = key
            value ^= right_row << key
        result_bits[i] = value
    result, decoded_cache, polynomial_cache = {}, {}, {}
    plane_mask = (1 << stride)-1
    for i, bits in result_bits.items():
        while bits:
            # Skip empty columns, then decode this complete Laurent entry.
            j = ((bits & -bits).bit_length()-1)//stride
            value = (bits >> (j*stride)) & plane_mask
            polynomial = polynomial_cache.get(value)
            if polynomial is None:
                terms, remaining = {}, value
                while remaining:
                    low = remaining & -remaining
                    code = low.bit_length()-1
                    exponent = decoded_cache.get(code)
                    if exponent is None:
                        quotient, powers = code, []
                        for radix, minimum in zip(radices, shift):
                            quotient, digit = divmod(quotient, radix)
                            powers.append(digit+minimum)
                        exponent = tuple(powers)
                        decoded_cache[code] = exponent
                    terms[exponent] = 1
                    remaining ^= low
                polynomial = Poly._from_canonical(left.ring, terms)
                if len(polynomial_cache) < 4096:
                    polynomial_cache[value] = polynomial
            result[i, j+column_start] = polynomial
            bits ^= value << (j*stride)
    return Matrix._from_entries(left.ring, left.nrows, right.ncols, result)


def _binary_encoded_matrix_product(left, right, right_rows,
                                   low_left, low_right, radices, strides, shift):
    """Exact carryless convolution using bounded Python integer bit planes.

    Mixed-radix digits cannot carry between Laurent axes.  Multiplication of
    F2 coefficients is XOR of shifted bit planes, with one shift per term of
    the smaller factor.  This is *not* ordinary integer multiplication.  The
    caller bounds encoded span and input allocation; no exponent-box search
    or finite-period quotient is involved.
    """
    def encode(operand, low):
        exponent_cache, result = {}, {}
        for position, polynomial in operand.entries.items():
            powers, bits = [], 0
            for exponent in polynomial.terms:
                key = exponent_cache.get(exponent)
                if key is None:
                    key = sum((value-minimum)*weight
                              for value, minimum, weight in zip(exponent, low, strides))
                    exponent_cache[exponent] = key
                powers.append(key)
                bits |= 1 << key
            result[position] = bits, powers
        return result

    encoded_left = encode(left, low_left)
    encoded_right = encode(right, low_right)
    entries = {}
    for (i, k), (a, a_powers) in encoded_left.items():
        for j, _ in right_rows.get(k, ()):
            b, b_powers = encoded_right[k, j]
            product = 0
            if len(a_powers) <= len(b_powers):
                for power in a_powers:
                    product ^= b << power
            else:
                for power in b_powers:
                    product ^= a << power
            if product:
                position = i, j
                value = entries.get(position, 0) ^ product
                if value:
                    entries[position] = value
                else:
                    entries.pop(position, None)
    result, polynomial_cache, exponent_cache = {}, {}, {}
    for position, bits in entries.items():
        if not bits:
            continue
        polynomial = polynomial_cache.get(bits)
        if polynomial is None:
            remaining, terms = bits, {}
            while remaining:
                low_bit = remaining & -remaining
                key = low_bit.bit_length()-1
                exponent = exponent_cache.get(key)
                if exponent is None:
                    value, powers = key, []
                    for radix, minimum in zip(radices, shift):
                        value, digit = divmod(value, radix)
                        powers.append(digit+minimum)
                    exponent = tuple(powers)
                    exponent_cache[key] = exponent
                terms[exponent] = 1
                remaining ^= low_bit
            polynomial = Poly._from_canonical(left.ring, terms)
            if len(polynomial_cache) < 4096:
                polynomial_cache[bits] = polynomial
        result[position] = polynomial
    return Matrix._from_entries(left.ring, left.nrows, right.ncols, result)


def zeros(ring, nrows, ncols):
    return Matrix(ring, nrows, ncols)


def eye(ring, size):
    return Matrix(ring, size, size, {(i, i): 1 for i in range(size)})


def hstack(*matrices):
    if not matrices:
        raise ValueError("An empty hstack needs an explicit ring and shape; use zeros")
    first = matrices[0]
    for matrix in matrices:
        first._compatible(matrix, same_shape=False)
        if matrix.nrows != first.nrows:
            raise ValueError("hstack requires equal row counts")
    entries, offset = {}, 0
    for matrix in matrices:
        for (i, j), value in matrix.entries.items():
            entries[i, j + offset] = value
        offset += matrix.ncols
    return Matrix._from_entries(first.ring, first.nrows, offset, entries)


def vstack(*matrices):
    if not matrices:
        raise ValueError("An empty vstack needs an explicit ring and shape; use zeros")
    first = matrices[0]
    for matrix in matrices:
        first._compatible(matrix, same_shape=False)
        if matrix.ncols != first.ncols:
            raise ValueError("vstack requires equal column counts")
    entries, offset = {}, 0
    for matrix in matrices:
        for (i, j), value in matrix.entries.items():
            entries[i + offset, j] = value
        offset += matrix.nrows
    return Matrix._from_entries(first.ring, offset, first.ncols, entries)


def block(rows: Sequence[Sequence[Matrix]]):
    if not rows:
        raise ValueError("block requires nonempty rows of matrices")
    rows = tuple(tuple(row) for row in rows)
    if not rows or any(not row for row in rows):
        raise ValueError("block requires nonempty rows of matrices")
    first = rows[0][0]
    heights, widths = [], []
    for row in rows:
        height = row[0].nrows
        for matrix in row:
            first._compatible(matrix, same_shape=False)
            if matrix.nrows != height:
                raise ValueError("hstack requires equal row counts")
        heights.append(height)
        widths.append(sum(matrix.ncols for matrix in row))
    if any(width != widths[0] for width in widths):
        raise ValueError("vstack requires equal column counts")
    entries, row_offset = {}, 0
    for row, height in zip(rows, heights):
        col_offset = 0
        for matrix in row:
            for (i, j), value in matrix.entries.items():
                entries[i + row_offset, j + col_offset] = value
            col_offset += matrix.ncols
        row_offset += height
    return Matrix._from_entries(first.ring, row_offset, widths[0], entries)


def direct_sum(*matrices):
    if not matrices:
        raise ValueError("An empty direct sum needs an explicit ring and shape; use zeros")
    first = matrices[0]
    for matrix in matrices:
        first._compatible(matrix, same_shape=False)
    entries = {}
    row_offset = col_offset = 0
    for matrix in matrices:
        for (i, j), value in matrix.entries.items():
            entries[i + row_offset, j + col_offset] = value
        row_offset += matrix.nrows
        col_offset += matrix.ncols
    return Matrix._from_entries(first.ring, row_offset, col_offset, entries)
