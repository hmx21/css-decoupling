"""Constructive Laurent module membership and complete syzygy generation.

Membership uses the localization presentation
``F_p[x_1,...,x_D,t]/(t*x_1*...*x_D-1)`` and extended module Groebner bases.
Kernels use tracked polynomial Gröbner/Schreyer generators after clearing
column denominators; all S-pairs and transfer identities verify completeness.
Localization is flat, so the resulting generators span the entire Laurent kernel. There
is no degree-bound search and no fraction-field substitution for membership.

Groebner computations can be intrinsically expensive.  These routines make no
polynomial-complexity promise and do not convert resource failures to no-solution
answers.  SymPy's distributed-module API is isolated in this file.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
import logging
from time import perf_counter

from sympy import GF
from sympy.polys.agca.modules import ModuleOrder
from sympy.polys.distributedmodules import sdm_groebner, sdm_nf_buchberger
from sympy.polys.orderings import monomial_key

from css_decoupling.algebra.laurent import Matrix, Ring, eye, zeros
from css_decoupling.algebra.laurent_membership import membership_columns, _NonzeroRemainder
from css_decoupling.errors import NoLaurentSolution, ResourceLimit


_logger = logging.getLogger(__name__)


class _UnitReductionMatrix:
    """Private sparse row/column index for elementary unit elimination.

    The public Matrix remains an ordinary sparse dictionary.  During a
    reduction, a row operation touches only its source row and a column
    operation only its source column.  Weights and candidate unit positions
    are updated exactly, preserving the original Markowitz pivot ordering.
    """
    def __init__(self, matrix):
        self.ring = matrix.ring
        self.rows = [{} for _ in range(matrix.nrows)]
        self.columns = [set() for _ in range(matrix.ncols)]
        self.row_weight = [0] * matrix.nrows
        self.col_weight = [0] * matrix.ncols
        self.row_units = [set() for _ in range(matrix.nrows)]
        self.col_units = [set() for _ in range(matrix.ncols)]
        for (i, j), value in matrix.entries.items():
            self.set(i, j, value)

    def set(self, i, j, value):
        old = self.rows[i].get(j)
        if old is not None:
            self.row_weight[i] -= len(old.terms)
            self.col_weight[j] -= len(old.terms)
            if old.is_unit:
                self.row_units[i].discard(j)
                self.col_units[j].discard(i)
        if value:
            self.rows[i][j] = value
            self.columns[j].add(i)
            self.row_weight[i] += len(value.terms)
            self.col_weight[j] += len(value.terms)
            if value.is_unit:
                self.row_units[i].add(j)
                self.col_units[j].add(i)
        else:
            self.rows[i].pop(j, None)
            self.columns[j].discard(i)

    def choose_unit(self, start):
        """Exact Markowitz minimum without scanning every unit at every pivot.

        Completed earlier pivots form an isolated identity block.  In the
        remaining block every row/column weight is at least one.  A weight
        one row or column supplies a zero-score candidate.  Otherwise the
        least column weight is a lower bound for every row's best score;
        sorted row bounds permit early stopping without changing tie breaks.
        """
        zero_candidates = [(i, next(iter(self.row_units[i])))
                           for i in range(start, len(self.rows)) if self.row_weight[i] == 1]
        zero_candidates.extend((next(iter(self.col_units[j])), j)
                               for j in range(start, len(self.columns)) if self.col_weight[j] == 1)
        if zero_candidates:
            return min(zero_candidates)
        candidates = [i for i in range(start, len(self.rows)) if self.row_units[i]]
        if not candidates:
            return None
        minimum_column = min(self.col_weight[j] for j in range(start, len(self.columns))
                             if self.col_units[j])
        best = None
        for i in sorted(candidates, key=lambda i: (self.row_weight[i], i)):
            lower = (self.row_weight[i]-1)*(minimum_column-1)
            if best is not None and lower > best[0]:
                break
            if best is not None and lower == best[0] and i > best[1]:
                continue
            j = min(self.row_units[i], key=lambda j: (self.col_weight[j], j))
            candidate = ((self.row_weight[i]-1)*(self.col_weight[j]-1), i, j)
            if best is None or candidate < best:
                best = candidate
        return best[1], best[2]

    def swap_rows(self, a, b):
        left, right = self.rows[a].copy(), self.rows[b].copy()
        zero = self.ring.zero()
        for j in left:
            self.set(a, j, zero)
        for j in right:
            self.set(b, j, zero)
        for j, value in left.items():
            self.set(b, j, value)
        for j, value in right.items():
            self.set(a, j, value)

    def swap_columns(self, a, b):
        left = {i: self.rows[i][a] for i in self.columns[a]}
        right = {i: self.rows[i][b] for i in self.columns[b]}
        zero = self.ring.zero()
        for i in left:
            self.set(i, a, zero)
        for i in right:
            self.set(i, b, zero)
        for i, value in left.items():
            self.set(i, b, value)
        for i, value in right.items():
            self.set(i, a, value)

    def scale_row(self, i, coefficient):
        for j, value in list(self.rows[i].items()):
            self.set(i, j, coefficient * value)

    def add_row(self, target, source, coefficient):
        zero = self.ring.zero()
        for j, value in self.rows[source].items():
            self.set(target, j, self.rows[target].get(j, zero) + coefficient * value)

    def add_column(self, target, source, coefficient):
        zero = self.ring.zero()
        for i in self.columns[source]:
            self.set(i, target, self.rows[i].get(target, zero) + coefficient * self.rows[i][source])

    def matrix(self):
        return Matrix._from_entries(self.ring, len(self.rows), len(self.columns),
                                    {(i, j): value for i, row in enumerate(self.rows)
                                     for j, value in row.items()})


def _add_sparse_vector(target, source, coefficient):
    """Elementary row/column update on private dictionaries of polynomials."""
    for i, value in source.items():
        product = coefficient * value
        answer = target[i] + product if i in target else product
        if answer:
            target[i] = answer
        else:
            target.pop(i, None)


class GeneralBackend:
    name = "sympy-exact-prime-laurent"

    def __init__(self, ring: Ring):
        self.ring = ring
        self.field = GF(ring.p)
        self.order = ModuleOrder(monomial_key("lex"), monomial_key("grevlex"), True)
        self.elimination_order = ModuleOrder(monomial_key("ilex"), monomial_key("grevlex"), False)
        self._bases = {}
        self._quotient_bases = {}
        self._polynomial_bases = {}
        self._kernels = {}
        self._reductions = {}

    def _check(self, *matrices):
        if any(not isinstance(matrix, Matrix) or matrix.ring != self.ring for matrix in matrices):
            raise ValueError("Backend and matrices must have identical Laurent rings")

    def lift_exponent(self, exponent):
        shift = max(0, -min(exponent))
        return tuple(e + shift for e in exponent) + (shift,)

    def vector_sdm(self, column, *, localized=True, order=None):
        order = self.order if order is None else order
        terms = []
        for i, polynomial in enumerate(column):
            polynomial = self.ring(polynomial)
            for exponent, coefficient in polynomial.terms.items():
                if not localized and any(e < 0 for e in exponent):
                    raise ValueError("Polynomial graph contains an uncleared denominator")
                monomial = self.lift_exponent(exponent) if localized else exponent
                terms.append(((i,) + monomial, self.field(coefficient)))
        return sorted(terms, key=lambda term: order(term[0]), reverse=True)

    def generators(self, A):
        """All source columns (including zeros), then one localization relation per row."""
        self._check(A)
        generators = [self.vector_sdm(column) for column in A.columns()]
        for i in range(A.nrows):
            generators.append(sorted([
                ((i,) + (1,) * (self.ring.D + 1), self.field.one),
                ((i,) + (0,) * (self.ring.D + 1), -self.field.one),
            ], key=lambda term: self.order(term[0]), reverse=True))
        return generators

    def _basis(self, A):
        self._check(A)
        key = A.key()
        if key not in self._bases:
            active = [(i, generator) for i, generator in enumerate(self.generators(A)) if generator]
            basis, transition = sdm_groebner(
                [g for _, g in active], sdm_nf_buchberger, self.order, self.field, extended=True)
            self._bases[key] = basis, transition, [i for i, _ in active]
        return self._bases[key]

    def quotient_groebner(self, A):
        """A coefficient-preserving standard basis in the polynomial localization.

        ``basis`` is a list of SDM lists ``[(monomial, coefficient), ...]``;
        monomials are tuples ``(component, x_1_power, ..., x_D_power, t_power)``.
        The first monomial of each list is its leading monomial.  Coefficients
        are integers in ``range(p)`` and all powers are nonnegative.
        """
        basis = self._quotient_basis(A)
        return {
            "basis": [[(tuple(m), int(c) % self.ring.p) for m, c in g] for g in basis],
            "leading_terms": [tuple(g[0][0]) for g in basis],
            "variables": [f"x{i + 1}" for i in range(self.ring.D)] + ["(product x_i)^-1"],
            "arithmetic": f"GF({self.ring.p})",
        }

    def reduce_quotient(self, A, B):
        """Full normal forms of B's columns in coker(A), as coefficient SDM lists."""
        self._check(A, B)
        if A.nrows != B.nrows:
            raise ValueError("Quotient normal forms require matching ambient row counts")
        basis = self._quotient_basis(A)
        p = self.ring.p
        answer = []
        for column in B.columns():
            remaining = {m: int(c) % p for m, c in self.vector_sdm(column)}
            normal = {}
            while remaining:
                leading = max(remaining, key=self.order)
                coefficient = remaining[leading]
                reducer = next((g for g in basis if divides(g[0][0], leading)), None)
                if reducer is None:
                    normal[leading] = remaining.pop(leading)
                    continue
                base_monomial, base_coefficient = reducer[0]
                shift = tuple(b - a for a, b in zip(base_monomial[1:], leading[1:]))
                factor = coefficient * pow(int(base_coefficient), -1, p) % p
                for monomial, value in reducer:
                    translated = (monomial[0],) + tuple(a + b for a, b in zip(monomial[1:], shift))
                    value = (remaining.get(translated, 0) - factor * int(value)) % p
                    if value:
                        remaining[translated] = value
                    else:
                        remaining.pop(translated, None)
            answer.append(sorted(normal.items(), key=lambda term: self.order(term[0]), reverse=True))
        return answer

    def _quotient_basis(self, A):
        """Normal forms do not require extended generator expressions."""
        self._check(A)
        key = A.key()
        if key in self._bases:
            return self._bases[key][0]
        if key not in self._quotient_bases:
            rank, _, L, V = self.reduce_units(A)
            if rank == A.nrows:
                # This proves coker(A)=0 over the Laurent ring, rather than
                # inferring it from rank over its fraction field.  The unit
                # operations supply an actual finite-support right inverse.
                right_inverse = V[:, :rank] * L
                if A * right_inverse != eye(self.ring, A.nrows):
                    raise ArithmeticError("Zero quotient failed its explicit right inverse")
                self._quotient_bases[key] = [
                    [((i,) + (0,) * (self.ring.D + 1), self.field.one)]
                    for i in range(A.nrows)]
            else:
                self._quotient_bases[key] = sdm_groebner(
                    [g for g in self.generators(A) if g], sdm_nf_buchberger,
                    self.order, self.field)
        return self._quotient_bases[key]

    def normal_form(self, A, B):
        """Represent quotient normal forms by Laurent vectors in the original ambient module."""
        result = zeros(self.ring, B.nrows, B.ncols)
        for j, normal in enumerate(self.reduce_quotient(A, B)):
            for monomial, coefficient in normal:
                i, *powers = monomial
                exponent = tuple(a - powers[-1] for a in powers[:-1])
                result[i, j] += self.ring.monomial(exponent, coefficient)
        return result

    def reduce_units(self, A):
        """Return r,C,L,V with L*A*V=diag(I_r,C), using Laurent-unit pivots only."""
        self._check(A)
        key = A.key()
        if key in self._reductions:
            r, C, L, V = self._reductions[key]
            return r, C.copy(), L.copy(), V.copy()
        D = _UnitReductionMatrix(A)
        one = self.ring.one()
        left_rows = [{i: one} for i in range(A.nrows)]
        right_columns = [{j: one} for j in range(A.ncols)]
        r = 0
        while r < min(A.shape):
            position = D.choose_unit(r)
            if position is None:
                break
            i, j = position
            if i != r:
                D.swap_rows(i, r)
                left_rows[i], left_rows[r] = left_rows[r], left_rows[i]
            if j != r:
                D.swap_columns(j, r)
                right_columns[j], right_columns[r] = right_columns[r], right_columns[j]
            unit_inverse = D.rows[r][r] ** -1
            D.scale_row(r, unit_inverse)
            left_rows[r] = {j: unit_inverse * value for j, value in left_rows[r].items()}
            for i in sorted(i for i in D.columns[r] if i > r):
                coefficient = -D.rows[i][r]
                D.add_row(i, r, coefficient)
                _add_sparse_vector(left_rows[i], left_rows[r], coefficient)
            for j in sorted(j for j in D.rows[r] if j > r):
                coefficient = -D.rows[r][j]
                D.add_column(j, r, coefficient)
                _add_sparse_vector(right_columns[j], right_columns[r], coefficient)
            r += 1
        D = D.matrix()
        L = Matrix._from_entries(self.ring, A.nrows, A.nrows,
                                 {(i, j): value for i, row in enumerate(left_rows)
                                  for j, value in row.items()})
        V = Matrix._from_entries(self.ring, A.ncols, A.ncols,
                                 {(i, j): value for j, column in enumerate(right_columns)
                                  for i, value in column.items()})
        if L * A * V != D:
            raise ArithmeticError("Laurent unit elimination certificate failed")
        result = r, D[r:, r:], L, V
        self._reductions[key] = (r, result[1].copy(), L.copy(), V.copy())
        return result

    def solve(self, A, B):
        """Solve A*X=B over the Laurent ring, or prove nonmembership by a standard basis."""
        self._check(A, B)
        if A.nrows != B.nrows:
            raise ValueError("Solve requires matching row counts")
        if not B:
            return zeros(self.ring, A.ncols, B.ncols)
        if not A:
            raise NoLaurentSolution("A nonzero right-hand side lies outside the zero image")
        # Preserve the original sparse presentation for the positive bounded
        # solve before constructing potentially dense row/column witnesses.
        # A miss has no mathematical meaning and follows the full path below.
        from css_decoupling.algebra.sparse_linear import sparse_binary_solution
        candidate = sparse_binary_solution(A, B)
        if candidate is not None:
            return candidate
        r, C, L, V = self.reduce_units(A)
        use_reduced = r and sum(len(f.terms) for f in C.entries.values()) <= sum(
            len(f.terms) for f in A.entries.values())
        if use_reduced:
            transformed = L * B
            bottom = self._solve_reduced(C, transformed[r:, :])
            answer = V * transformed[:r, :].vstack(bottom)
        else:
            answer = self._solve_reduced(A, B)
        if A * answer != B:
            raise ArithmeticError("Laurent solution failed its exact certificate")
        return answer

    def _solve_reduced(self, A, B):
        answer = zeros(self.ring, A.ncols, B.ncols)
        if not B:
            return answer
        if not A:
            raise NoLaurentSolution("A nonzero right-hand side lies outside the zero image")
        # A bounded coefficient solve can avoid severe localization-normal-
        # form growth.  Only an exactly checked positive answer is accepted;
        # unsupported fields, budget caps and failed ansatzes use full GB.
        from css_decoupling.algebra.sparse_linear import sparse_binary_solution
        candidate = sparse_binary_solution(A, B)
        if candidate is not None:
            return candidate
        candidate = self._solve_polynomial(A, B)
        if candidate is not None:
            return candidate
        basis, transition, active = self._basis(A)
        next_column = 0
        self._last_membership_stats = {}
        def observe(event):
            self._last_membership_stats = dict(event['stats'], event=event['event'])
        try:
            for j, column in membership_columns(A, B, (basis, transition, active),
                    order=self.order, vector_sdm=self.vector_sdm, observer=observe):
                if (type(j) is not int or j != next_column or j >= B.ncols
                        or not isinstance(column, Matrix) or column.ring != self.ring
                        or column.shape != (A.ncols, 1)):
                    raise ArithmeticError("Membership replay yielded a malformed or out-of-order column")
                for (i, _), value in column.entries.items():
                    answer[i, j] = value
                next_column += 1
        except _NonzeroRemainder as exc:
            if type(exc.column) is not int or exc.column != next_column or exc.column >= B.ncols:
                raise ArithmeticError("Membership replay reported an out-of-order remainder") from exc
            original = self.vector_sdm([B[i, next_column] for i in range(B.nrows)])
            self._last_membership_stats.update(comparison_status='checking-original-remainder', column=next_column)
            _logger.info("membership_replay %s", self._last_membership_stats)
            reference_start = perf_counter()
            try:
                reference = sdm_nf_buchberger(original, basis, self.order, self.field)
            finally:
                self._last_membership_stats['reference_seconds'] = perf_counter()-reference_start
            canonical = [(tuple(m), int(c) % self.ring.p) for m, c in reference]
            replayed = [(tuple(m), int(c) % self.ring.p) for m, c in exc.remainder]
            if not replayed or canonical != replayed:
                self._last_membership_stats.update(comparison_status='remainder-mismatch',
                    fallback_column=next_column, replayed_terms=len(replayed), reference_terms=len(canonical))
                _logger.warning("membership_replay %s", self._last_membership_stats)
                error = ArithmeticError("Sparse and original polynomial weak remainders disagree")
                error.membership_mismatch = dict(column=next_column, original=original,
                    replayed=replayed, reference=canonical, basis=basis, order=repr(self.order))
                raise error from exc
            self._last_membership_stats.update(comparison_status='confirmed-nonzero-remainder',
                                              column=next_column, remainder_terms=len(canonical))
            _logger.info("membership_replay %s", self._last_membership_stats)
            raise NoLaurentSolution(f"Right-hand column {next_column} is not in the Laurent image") from exc
        except ResourceLimit as exc:
            self._last_membership_stats.update(fallback_column=next_column, fallback_reason=str(exc))
            _logger.info("membership_replay %s", self._last_membership_stats)
        else:
            if next_column != B.ncols:
                raise ArithmeticError("Membership replay ended before all RHS columns were verified")
            _logger.info("membership_replay %s", self._last_membership_stats)
        for j in range(next_column, B.ncols):
            column = [B[i, j] for i in range(B.nrows)]
            remainder, phantom = sdm_nf_buchberger(
                self.vector_sdm(column), basis, self.order, self.field, phantom=([], transition))
            if remainder:
                raise NoLaurentSolution(f"Right-hand column {j} is not in the Laurent image")
            # The phantom starts at zero and undergoes subtraction alongside B.
            # Thus B + A*phantom = 0: the solution is MINUS the phantom, an
            # essential sign that characteristic-two implementations can hide.
            for monomial, coefficient in phantom:
                generator, *powers = monomial
                original = active[generator]
                if original < A.ncols:
                    exponent = tuple(a - powers[-1] for a in powers[:-1])
                    answer[original, j] += self.ring.monomial(exponent, -int(coefficient))
        return answer

    def _solve_polynomial(self, A, B):
        """Try exact polynomial membership before adjoining inverse variables.

        Column-unit clearing preserves Laurent solutions.  Polynomial
        membership is only a sufficient condition after this clearing;
        a nonzero remainder or an exhausted native budget requests the full
        Laurent fallback, never establishes nonmembership.  Successful
        answers use the tracked G=A*T identity and are checked against A,B.
        """
        from css_decoupling.algebra.schreyer import (
            polynomial_groebner, _Budget, _Order, _vectors, _divide, _matrix)
        from css_decoupling.errors import ResourceLimit
        if (A.ncols > 20000 or B.ncols > 1024 or
                sum(len(f.terms) for M in (A, B) for f in M.entries.values()) > 2000000):
            return None
        P, input_units = self._polynomial_columns(A)
        Q, rhs_units = self._polynomial_columns(B)
        key = P.key()
        caps = dict(timeout=30, max_basis=20000, max_terms=2000000, max_pairs=200000)
        try:
            if key not in self._polynomial_bases:
                self._polynomial_bases[key] = polynomial_groebner(P, **caps)
            witness = self._polynomial_bases[key]
            if witness is None:
                return None
            G, T = witness['groebner_generators'], witness['basis_in_input']
            budget = _Budget(**caps)
            basis, columns = _vectors(G, budget), []
            for vector in _vectors(Q, budget):
                remainder, quotient = _divide(
                    vector, basis, _Order(), self.ring.p, budget, track=True)
                if remainder:
                    return None
                columns.append(quotient)
                budget.check(*columns)
            Y = T * _matrix(self.ring, G.ncols, columns)
        except ResourceLimit:
            # Do not repeatedly attempt a costly basis for the same input.
            if key not in self._polynomial_bases:
                self._polynomial_bases[key] = None
            return None
        one = self.ring.one()
        inverses = {j: value ** -1 for j, value in rhs_units.items()}
        answer = Matrix(self.ring, A.ncols, B.ncols,
                        {(i, j): value * input_units.get(i, one) * inverses.get(j, one)
                         for (i, j), value in Y.entries.items()})
        if A * answer != B:
            raise ArithmeticError("Polynomial membership failed its exact Laurent identity")
        return answer

    def kernel(self, A):
        """A finite generating matrix for the complete Laurent kernel of A."""
        self._check(A)
        key = A.key()
        if key in self._kernels:
            return self._kernels[key].copy()
        if not A:
            return eye(self.ring, A.ncols)
        r, C, _, V = self.reduce_units(A)
        # Unit elimination can lower matrix ranks while making polynomial
        # support much denser.  Kernel computations are often dominated by
        # that fill, so retain the original presentation when it is sparser.
        use_reduced = r and sum(len(f.terms) for f in C.entries.values()) <= sum(
            len(f.terms) for f in A.entries.values())
        reduced = self._kernel_reduced(C if use_reduced else A)
        answer = V * zeros(self.ring, r, reduced.ncols).vstack(reduced) if use_reduced else reduced
        if A * answer:
            raise ArithmeticError("Laurent kernel generators failed their exact certificate")
        self._kernels[key] = answer.copy()
        return answer

    kernel_generators = kernel

    def _polynomial_columns(self, A):
        """Clear column denominators by diagonal Laurent units, sparsely."""
        minima = {}
        for (_, j), value in A.entries.items():
            for exponent in value.terms:
                minima[j] = tuple(min(a, b) for a, b in zip(
                    minima.get(j, (0,) * self.ring.D), exponent))
        units = {j: self.ring.monomial(tuple(-a for a in low)) for j, low in minima.items()}
        polynomial = A.copy()
        for (i, j), value in A.entries.items():
            polynomial[i, j] = value * units[j]
        return polynomial, units

    def _kernel_reduced(self, A):
        if not A:
            return eye(self.ring, A.ncols)
        # AU is polynomial for diagonal Laurent-unit U. Flat localization gives
        # ker_R(A) = U (ker_P(AU) tensor_P R), including all Laurent syzygies.
        # The native routine verifies all S-pairs, the complete first Schreyer
        # module, and both generator-transfer identities; A*K=0 is not its
        # completeness argument. None retains this backend's existing lack
        # of a wall-clock bound; native basis/term/pair caps still apply.
        from css_decoupling.algebra.schreyer import polynomial_kernel
        polynomial, units = self._polynomial_columns(A)
        basis = polynomial_kernel(polynomial, timeout=None)
        one = self.ring.one()
        return Matrix(self.ring, A.ncols, basis.ncols,
                      {(i, j): value * units.get(i, one)
                       for (i, j), value in basis.entries.items()})

    def _kernel_graph(self, A):
        """Independent complete graph-elimination path for cross-checks."""
        if not A:
            return eye(self.ring, A.ncols)
        polynomial, units = self._polynomial_columns(A)
        graph = []
        for j, column in enumerate(polynomial.columns()):
            vector = self.vector_sdm(column, localized=False, order=self.elimination_order)
            vector.append(((A.nrows + j,) + (0,) * self.ring.D, self.field.one))
            graph.append(sorted(vector, key=lambda term: self.elimination_order(term[0]), reverse=True))
        basis = sdm_groebner(graph, sdm_nf_buchberger, self.elimination_order, self.field)
        columns = []
        seen = set()
        for vector in basis:
            if any(m[0] < A.nrows for m, _ in vector):
                continue
            column = zeros(self.ring, A.ncols, 1)
            for monomial, coefficient in vector:
                index = monomial[0] - A.nrows
                column[index, 0] += self.ring.monomial(monomial[1:], int(coefficient)) * units.get(
                    index, self.ring.one())
            if column and column.key() not in seen:
                seen.add(column.key())
                columns.append(column)
        return zeros(self.ring, A.ncols, 0).hstack(*columns)

    def close(self):
        self._bases.clear()
        self._quotient_bases.clear()
        self._polynomial_bases.clear()
        self._kernels.clear()
        self._reductions.clear()


def divides(a, b):
    """Divisibility of SDM monomials, including equality of their module component."""
    return a[0] == b[0] and all(x <= y for x, y in zip(a[1:], b[1:]))


def swap_rows(matrix, a, b):
    entries = [(ij, f) for ij, f in matrix.entries.items() if ij[0] in (a, b)]
    for ij, _ in entries:
        del matrix.entries[ij]
    for (i, j), f in entries:
        matrix[b if i == a else a, j] = f


def swap_columns(matrix, a, b):
    entries = [(ij, f) for ij, f in matrix.entries.items() if ij[1] in (a, b)]
    for ij, _ in entries:
        del matrix.entries[ij]
    for (i, j), f in entries:
        matrix[i, b if j == a else a] = f


def add_row(matrix, target, source, coefficient):
    for (i, j), f in list(matrix.entries.items()):
        if i == source:
            matrix[target, j] += coefficient * f


def add_column(matrix, target, source, coefficient):
    for (i, j), f in list(matrix.entries.items()):
        if j == source:
            matrix[i, target] += coefficient * f


_backend_selection = ContextVar("general_laurent_backend", default="sympy")


@lru_cache(maxsize=16)
def _cached_backend(ring, name):
    if name == "sympy":
        return GeneralBackend(ring)
    raise ValueError(f"Unknown Laurent module backend {name!r}")


def get_backend(ring):
    """Return the context-selected backend, bound to the explicit input ring."""
    choice = _backend_selection.get()
    if isinstance(choice, str):
        return _cached_backend(ring, choice)
    if not isinstance(choice, GeneralBackend) or choice.ring != ring:
        raise ValueError("The selected backend instance has a different Laurent ring")
    return choice


@contextmanager
def using_backend(backend="sympy", *, timeout=120):
    """Temporarily select SymPy or an explicit native backend instance.

    Selection is context-local and restored even after exceptions. ``timeout``
    is retained for caller compatibility; it does not impose a wall-clock bound
    on SymPy or alter a supplied instance's own resource limits. Resolution and
    bounded-linear limits are configured on their respective native APIs.
    """
    if isinstance(backend, str):
        if backend != "sympy":
            raise ValueError(f"Unknown Laurent module backend {backend!r}")
    elif not isinstance(backend, GeneralBackend):
        raise TypeError("Backend must be a known name or a GeneralBackend instance")
    token = _backend_selection.set(backend)
    try:
        yield backend
    finally:
        _backend_selection.reset(token)


def solve(A, B, *, backend=None):
    return (backend or get_backend(A.ring)).solve(A, B)


def kernel(A, *, backend=None):
    return (backend or get_backend(A.ring)).kernel(A)
