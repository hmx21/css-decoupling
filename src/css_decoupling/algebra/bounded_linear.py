"""Degree-bounded Laurent linear algebra using only finite-field elimination.

The polynomial lift uses x_1,...,x_D,y_1,...,y_D and, in each
equation, unknown multiples of x_i*y_i-1.  The field-case bound used
in Appendix C5 is (2*m*d)**(2**(2*D)); d includes the right-hand side.
The monomials are *linear unknowns*, not candidate coefficient assignments.

Solves may stop after an exactly verified solution at a smaller degree.
General kernels must reach the full theorem bound.  In practice this bound
is very conservative: a resource exception is deliberately not a statement
of nonmembership or incompleteness of the underlying mathematical method.
"""
from __future__ import annotations

from copy import deepcopy
from math import comb

from css_decoupling.algebra.laurent import Matrix, Ring, eye, zeros
from css_decoupling.algebra.modules import GeneralBackend
from css_decoupling.errors import BackendUnavailable, NoLaurentSolution, ResourceLimit


SCHEMA = "css-decoupling.bounded-linear"


def augmentation(A):
    """Evaluation at all Laurent variables equal to one."""
    return Matrix(A.ring, A.nrows, A.ncols,
                  {ij: sum(f.terms.values()) for ij, f in A.entries.items()})


def _monomials(variables, degree):
    """All total-degree <= degree monomials, after a caller's size preflight."""
    if variables == 0:
        return [()]
    result = []

    def visit(prefix, remaining, count):
        if count == 1:
            result.extend(prefix + (last,) for last in range(remaining + 1))
        else:
            for first in range(remaining + 1):
                visit(prefix + (first,), remaining - first, count - 1)
    visit((), degree, variables)
    return result


def _lift(exponent):
    return tuple(max(e, 0) for e in exponent) + tuple(max(-e, 0) for e in exponent)


def degree_bound(A, B=None):
    """Return a symbolic, overflow-safe theorem bound; no huge integer is built."""
    if B is not None and (A.ring != B.ring or A.nrows != B.nrows):
        raise ValueError("Bound inputs must have the same ring and row count")
    degree = max([2] + [sum(abs(e) for e in exponent)
                        for M in (A, B) if M is not None
                        for f in M.entries.values() for exponent in f.terms])
    m = max(1, A.nrows)
    return {"variables": 2 * A.ring.D, "equations": A.nrows,
            "polynomial_degree": degree, "base": 2 * m * degree,
            "exponent_power_of_two": 2 * A.ring.D,
            "formula": "base ** (2 ** exponent_power_of_two)",
            "original_unknowns": A.ncols,
            "localization_unknowns": A.nrows * A.ring.D}


def _bound_within(bound, limit):
    """The actual bound if <= limit, otherwise None, without constructing it."""
    if limit < 1:
        return None
    N = bound["exponent_power_of_two"]
    if N >= limit.bit_length().bit_length():
        return None
    exponent = 1 << N
    if (bound["base"].bit_length() - 1) * exponent >= limit.bit_length():
        return None
    result = pow(bound["base"], exponent)
    return result if result <= limit else None


def _rref(rows, width, p, rhs=None):
    """Sparse finite-field row reduction; returns pivots, RREF and RHS.

    ``rhs`` has one sparse dictionary per equation, supporting several right
    sides.  This routine never divides by a Laurent polynomial.
    """
    basis = {}
    for index, original in enumerate(rows):
        row = {j: v % p for j, v in original.items() if v % p}
        out = {} if rhs is None else {j: v % p for j, v in rhs[index].items() if v % p}
        while row:
            pivot = min(row)
            if pivot not in basis:
                inverse = pow(row[pivot], -1, p)
                row = {j: v * inverse % p for j, v in row.items()}
                out = {j: v * inverse % p for j, v in out.items()}
                basis[pivot] = (row, out)
                break
            other, other_rhs = basis[pivot]
            scale = row[pivot]
            for j, value in other.items():
                new = (row.get(j, 0) - scale * value) % p
                if new:
                    row[j] = new
                else:
                    row.pop(j, None)
            for j, value in other_rhs.items():
                new = (out.get(j, 0) - scale * value) % p
                if new:
                    out[j] = new
                else:
                    out.pop(j, None)
        if not row and out:
            return None
    pivots = sorted(basis)
    for pivot in reversed(pivots):
        row, out = basis[pivot]
        for previous in pivots:
            if previous >= pivot:
                break
            other, other_rhs = basis[previous]
            scale = other.get(pivot, 0)
            if not scale:
                continue
            for j, value in row.items():
                new = (other.get(j, 0) - scale * value) % p
                if new:
                    other[j] = new
                else:
                    other.pop(j, None)
            for j, value in out.items():
                new = (other_rhs.get(j, 0) - scale * value) % p
                if new:
                    other_rhs[j] = new
                else:
                    other_rhs.pop(j, None)
    return pivots, basis


def constant_solve(A, B):
    """Solve a constant matrix equation; no polynomial backend is consulted."""
    if A.ring != B.ring or A.nrows != B.nrows or any(
            not f.is_constant for M in (A, B) for f in M.entries.values()):
        raise ValueError("Constant solve requires constant matrices of matching rows")
    rows = [dict() for _ in range(A.nrows)]
    rhs = [dict() for _ in range(B.nrows)]
    for (i, j), value in A.entries.items():
        rows[i][j] = value.constant()
    for (i, j), value in B.entries.items():
        rhs[i][j] = value.constant()
    reduced = _rref(rows, A.ncols, A.ring.p, rhs)
    if reduced is None:
        raise NoLaurentSolution("Inconsistent finite-field system")
    _, basis = reduced
    return Matrix(A.ring, A.ncols, B.ncols,
                  {(pivot, j): value for pivot, (_, out) in basis.items()
                   for j, value in out.items()})


def constant_kernel(A):
    if any(not f.is_constant for f in A.entries.values()):
        raise ValueError("Constant kernel requires a constant matrix")
    rows = [dict() for _ in range(A.nrows)]
    for (i, j), value in A.entries.items():
        rows[i][j] = value.constant()
    pivots, basis = _rref(rows, A.ncols, A.ring.p)
    free = [j for j in range(A.ncols) if j not in basis]
    result = zeros(A.ring, A.ncols, len(free))
    for k, column in enumerate(free):
        result[column, k] = 1
        for pivot in pivots:
            result[pivot, k] = -basis[pivot][0].get(column, 0)
    return result


def _polynomial_operator(A, B, degree, caps):
    """Coefficient matrix of the complete 2D-variable lifted equation."""
    D, m, n = A.ring.D, A.nrows, A.ncols
    bound = degree_bound(A, B)
    polynomial_degree = bound["polynomial_degree"]
    variables = 2 * D
    columns_per_unknown = comb(degree + variables, variables)
    width = (n + m * D) * columns_per_unknown
    height = m * comb(degree + polynomial_degree + variables, variables)
    for value, limit, label in (
            (width, caps["max_unknowns"], "coefficient unknowns"),
            (height, caps["max_equations"], "coefficient equations"),
            (height * (width + (0 if B is None else B.ncols)),
             caps["max_matrix_entries"], "dense elimination entries")):
        if limit is not None and value > limit:
            raise ResourceLimit(f"Bounded Laurent degree {degree} needs {value} {label}; cap is {limit}")
    monomials = _monomials(variables, degree)
    row_monomials = _monomials(variables, degree + polynomial_degree)
    lookup = {exponent: i for i, exponent in enumerate(row_monomials)}
    row_count = len(row_monomials)
    rows = [dict() for _ in range(height)]
    rhs = [dict() for _ in range(height)]
    p = A.ring.p
    for (i, j), value in A.entries.items():
        for exponent, coefficient in value.terms.items():
            lifted = _lift(exponent)
            for k, monomial in enumerate(monomials):
                row = i * row_count + lookup[tuple(a + b for a, b in zip(lifted, monomial))]
                column = j * columns_per_unknown + k
                rows[row][column] = (rows[row].get(column, 0) + coefficient) % p
    for i in range(m):
        for axis in range(D):
            j = n + i * D + axis
            for k, monomial in enumerate(monomials):
                positive = list(monomial)
                positive[axis] += 1
                positive[D + axis] += 1
                column = j * columns_per_unknown + k
                rows[i * row_count + lookup[tuple(positive)]][column] = 1
                rows[i * row_count + lookup[monomial]][column] = p - 1
    if B is not None:
        for (i, j), value in B.entries.items():
            for exponent, coefficient in value.terms.items():
                rhs[i * row_count + lookup[_lift(exponent)]][j] = coefficient
    return rows, rhs, monomials, width


def _project_vectors(ring, original_unknowns, monomials, vectors):
    """Map polynomial coefficient vectors to their original Laurent components."""
    result = zeros(ring, original_unknowns, len(vectors))
    size, D = len(monomials), ring.D
    for j, vector in enumerate(vectors):
        for index, coefficient in vector.items():
            component, offset = divmod(index, size)
            if component < original_unknowns:
                exponent = monomials[offset]
                result[component, j] += ring.monomial(
                    tuple(exponent[i] - exponent[D + i] for i in range(D)), coefficient)
    return result


class BoundedLinearBackend(GeneralBackend):
    """Finite Gaussian backend for the fixed-dimensional polynomial cost route.

    Inherits only elementary Laurent-unit reduction and backend-context
    compatibility.  No Groebner routine is used, including for certificates.
    """

    name = "bounded-gaussian-prime-laurent"

    def __init__(self, ring: Ring, *, max_degree=64, max_unknowns=10000,
                 max_equations=20000, max_matrix_entries=2_000_000):
        super().__init__(ring)
        self.caps = dict(max_degree=max_degree, max_unknowns=max_unknowns,
                         max_equations=max_equations, max_matrix_entries=max_matrix_entries)
        for key, value in self.caps.items():
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                raise ValueError(f"{key} must be a nonnegative integer or None")
        if all(value is None for value in self.caps.values()):
            raise ValueError("At least one finite coefficient or degree resource limit is required")
        self.last_bound = None
        self.last_certificate = None

    def _record(self, operation, A, result=None, *, B=None, method, evidence=None):
        from css_decoupling.certificates.comparison import _matrix_dict
        self.last_bound = degree_bound(A, B)
        self.last_certificate = {
            "schema": SCHEMA, "version": 1, "operation": operation,
            "ring": {"prime": self.ring.p, "dimension": self.ring.D},
            "A": _matrix_dict(A), "B": None if B is None else _matrix_dict(B),
            "result": None if result is None else _matrix_dict(result),
            "bound": deepcopy(self.last_bound), "method": method,
            "evidence": {} if evidence is None else evidence}
        return result

    def _start(self, A, B=None):
        self._check(A, *(() if B is None else (B,)))
        if B is not None and A.nrows != B.nrows:
            raise ValueError("Solve requires matching row counts")
        self.last_bound = degree_bound(A, B)
        self.last_certificate = None

    def _finite_preflight(self, A, B):
        for count, limit, label in (
                (A.ncols, self.caps["max_unknowns"], "specialization unknowns"),
                (A.nrows, self.caps["max_equations"], "specialization equations"),
                (A.nrows * (A.ncols + B.ncols), self.caps["max_matrix_entries"],
                 "specialization elimination entries")):
            if limit is not None and count > limit:
                raise ResourceLimit(f"Bounded Laurent solve needs {count} {label}; cap is {limit}")

    def _inverse_unit_matrix(self, A):
        r, residual, L, V = self.reduce_units(A)
        if A.nrows != A.ncols or r != A.nrows or residual:
            raise ArithmeticError("A product of elementary unit operations was not invertible")
        result = V * L
        if A * result != eye(self.ring, A.nrows) or result * A != eye(self.ring, A.nrows):
            raise ArithmeticError("Unit-operation inverse failed")
        return result

    def _degree_cap(self, A):
        limits = [value for value in (self.caps["max_degree"], self.caps["max_unknowns"],
                                      self.caps["max_equations"], self.caps["max_matrix_entries"])
                  if value is not None]
        # Each nonzero lifted coefficient problem has at least degree+1
        # scalar unknowns/equations, so this is a safe preliminary ceiling.
        return min(limits)

    def solve(self, A, B):
        self._start(A, B)
        if not B:
            return self._record("solve", A, zeros(self.ring, A.ncols, B.ncols),
                                B=B, method="zero-rhs")
        if not A:
            self._record("no-solution", A, B=B, method="zero-operator")
            raise NoLaurentSolution("A nonzero RHS is outside the zero image")
        # A specialization obstruction is an exact necessary condition over R.
        self._finite_preflight(A, B)
        try:
            constant_solve(augmentation(A), augmentation(B))
        except NoLaurentSolution:
            self._record("no-solution", A, B=B, method="augmentation-obstruction")
            raise NoLaurentSolution("The system is inconsistent after evaluation at x_i=1")
        r, C, L, V = self.reduce_units(A)
        if r:
            transformed = L * B
            try:
                lower = self.solve(C, transformed[r:, :])
            except NoLaurentSolution:
                child = deepcopy(self.last_certificate)
                from css_decoupling.certificates.comparison import _matrix_dict
                self._record("no-solution", A, B=B, method="unit-reduction",
                             evidence={"rank": r, "L": _matrix_dict(L), "V": _matrix_dict(V),
                                       "Linv": _matrix_dict(self._inverse_unit_matrix(L)),
                                       "Vinv": _matrix_dict(self._inverse_unit_matrix(V)), "child": child})
                raise
            answer = V * transformed[:r, :].vstack(lower)
            if A * answer != B:
                raise ArithmeticError("Unit solve failed")
            return self._record("solve", A, answer, B=B, method="verified-solution")
        bound = degree_bound(A, B)
        cap = self._degree_cap(A)
        complete = _bound_within(bound, cap)
        last = cap if complete is None else complete
        for degree in range(last + 1):
            rows, rhs, monomials, width = _polynomial_operator(A, B, degree, self.caps)
            reduced = _rref(rows, width, self.ring.p, rhs)
            if reduced is None:
                continue
            _, basis = reduced
            vectors = [dict() for _ in range(B.ncols)]
            for pivot, (_, out) in basis.items():
                for j, value in out.items():
                    vectors[j][pivot] = value
            answer = _project_vectors(self.ring, A.ncols, monomials, vectors)
            if A * answer != B:
                raise ArithmeticError("Lifted polynomial solution failed its Laurent identity")
            return self._record("solve", A, answer, B=B, method="bounded-coefficients",
                                evidence={"degree": degree, "coefficient_unknowns": width,
                                          "coefficient_equations": len(rows)})
        if complete is not None:
            self._record("no-solution", A, B=B, method="exhausted-bound",
                         evidence={"degree": complete})
            raise NoLaurentSolution("The complete proven coefficient bound is inconsistent")
        raise ResourceLimit(f"No solution found through degree {last}; the proven degree bound is larger")

    def kernel(self, A):
        self._start(A)
        if not A:
            return self._record("kernel", A, eye(self.ring, A.ncols), method="zero-operator")
        if A.ncols == 1:
            return self._record("kernel", A, zeros(self.ring, 1, 0), method="domain-column")
        r, C, L, V = self.reduce_units(A)
        if r:
            lower = self.kernel(C)
            child = deepcopy(self.last_certificate)
            answer = V * zeros(self.ring, r, lower.ncols).vstack(lower)
            from css_decoupling.certificates.comparison import _matrix_dict
            return self._record("kernel", A, answer, method="unit-reduction",
                                evidence={"rank": r, "L": _matrix_dict(L), "V": _matrix_dict(V),
                                          "Linv": _matrix_dict(self._inverse_unit_matrix(L)),
                                          "Vinv": _matrix_dict(self._inverse_unit_matrix(V)),
                                          "child": child})
        bound = degree_bound(A)
        full_degree = _bound_within(bound, self._degree_cap(A))
        if full_degree is None:
            raise ResourceLimit("Complete kernel needs the full proven degree bound, beyond the resource ceiling")
        rows, _, monomials, width = _polynomial_operator(A, None, full_degree, self.caps)
        pivots, basis = _rref(rows, width, self.ring.p)
        free = [j for j in range(width) if j not in basis]
        if self.caps["max_matrix_entries"] is not None and width * len(free) > self.caps["max_matrix_entries"]:
            raise ResourceLimit("The complete coefficient nullspace exceeds the output entry budget")
        vectors = [{j: 1, **{pivot: -basis[pivot][0][j] % self.ring.p
                             for pivot in pivots if j in basis[pivot][0]}}
                   for j in free]
        answer = _project_vectors(self.ring, A.ncols, monomials, vectors)
        # Nullspace vectors projecting only to localization relations are zero.
        nonzero = sorted({j for _, j in answer.entries})
        answer = answer[:, nonzero]
        if A * answer:
            raise ArithmeticError("Complete lifted kernel failed its Laurent identity")
        return self._record("kernel", A, answer, method="exhausted-bound",
                            evidence={"degree": full_degree})

    kernel_generators = kernel

    def quotient_groebner(self, A):
        raise BackendUnavailable("Bounded linear backend has no Groebner quotient API; use trivial_cohomology")

    def reduce_quotient(self, A, B):
        raise BackendUnavailable("Bounded linear backend has no Groebner normal-form API")

    def normal_form(self, A, B):
        raise BackendUnavailable("Bounded linear backend has no Groebner normal-form API")
