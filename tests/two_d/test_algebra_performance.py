"""Differential exactness checks for native sparse/bit-parallel algorithms."""
from random import Random
import unittest
from unittest.mock import patch

from css_decoupling.two_d.algebra import Ring, Matrix, eye, zeros, GeneralBackend
from css_decoupling.algebra import laurent
from css_decoupling.algebra.sparse_linear import (
    _binary_solution, sparse_binary_solution)


def reference_product(left, right):
    """Literal coefficient convolution independent of optimized Poly/Matrix mul."""
    result, p = {}, left.ring.p
    for (i, k), a in left.entries.items():
        for (kk, j), b in right.entries.items():
            if kk != k:
                continue
            terms = result.setdefault((i, j), {})
            for e, c in a.terms.items():
                for f, d in b.terms.items():
                    key = tuple(x+y for x, y in zip(e, f))
                    terms[key] = (terms.get(key, 0)+c*d) % p
    return Matrix(left.ring, left.nrows, right.ncols,
                  {ij: left.ring(terms) for ij, terms in result.items()})


def reference_unit_reduction(matrix):
    """Original full-scan elementary algorithm, retained only as a test oracle."""
    from css_decoupling.algebra.modules import (
        swap_rows, swap_columns, add_row, add_column)
    D, L, V = matrix.copy(), eye(matrix.ring, matrix.nrows), eye(matrix.ring, matrix.ncols)
    r = 0
    while r < min(matrix.shape):
        units = [(i, j) for (i, j), f in D.entries.items()
                 if i >= r and j >= r and f.is_unit]
        if not units:
            break
        rw, cw = {}, {}
        for (i, j), f in D.entries.items():
            if i >= r and j >= r:
                rw[i], cw[j] = rw.get(i, 0)+len(f.terms), cw.get(j, 0)+len(f.terms)
        i, j = min(units, key=lambda ij: ((rw[ij[0]]-1)*(cw[ij[1]]-1), ij))
        for M in (D, L):
            swap_rows(M, i, r)
        for M in (D, V):
            swap_columns(M, j, r)
        unit = D[r, r]**-1
        for M in (D, L):
            for (i, j), f in list(M.entries.items()):
                if i == r:
                    M[i, j] = unit*f
        for i in sorted({i for i, j in D.entries if j == r and i > r}):
            coefficient = -D[i, r]
            add_row(D, i, r, coefficient)
            add_row(L, i, r, coefficient)
        for j in sorted({j for i, j in D.entries if i == r and j > r}):
            coefficient = -D[r, j]
            add_column(D, j, r, coefficient)
            add_column(V, j, r, coefficient)
        r += 1
    return r, D[r:, r:], L, V


class AlgebraPerformanceTests(unittest.TestCase):
    def random_matrix(self, ring, rows, cols, seed, span=3):
        rng = Random(seed)
        return Matrix(ring, rows, cols, {
            (i, j): ring({tuple(rng.randrange(-span, span+1) for _ in range(ring.D)):
                         rng.randrange(1, ring.p) for _ in range(rng.randrange(1, 5))})
            for i in range(rows) for j in range(cols) if rng.randrange(3)})

    def test_packed_rows_and_per_entry_convolution_match_literal(self):
        for dimension in (1, 2, 3):
            ring = Ring(2, dimension)
            A = self.random_matrix(ring, 5, 7, 100+dimension)
            B = self.random_matrix(ring, 7, 6, 200+dimension)
            expected = reference_product(A, B)
            with patch.object(laurent, '_CONVOLUTION_MIN_PRODUCTS', 0), \
                 patch.object(laurent, '_CONVOLUTION_REUSE_FACTOR', 0), \
                 patch.object(laurent, '_binary_row_encoded_matrix_product',
                              wraps=laurent._binary_row_encoded_matrix_product) as packed:
                self.assertEqual(A*B, expected)
                self.assertTrue(packed.called)
            # Forbid packed-row allocation while permitting per-entry planes.
            with patch.object(laurent, '_CONVOLUTION_MIN_PRODUCTS', 0), \
                 patch.object(laurent, '_CONVOLUTION_REUSE_FACTOR', 0), \
                 patch.object(laurent, '_binary_row_encoded_matrix_product', return_value=None):
                # Force via the private entry engine with the same verified encoding.
                right_rows = {}
                for (i, j), f in B.entries.items():
                    right_rows.setdefault(i, []).append((j, f))
                # Large empty row dimension makes row storage preflight fail.
                C = Matrix(ring, 10**9, A.ncols, A.entries)
                actual = laurent._encoded_matrix_product(C, B, right_rows)
                self.assertEqual(actual[:5, :], expected)

    def test_tiled_binary_products_match_literal_under_small_storage_cap(self):
        ring = Ring(2, 2)
        A = self.random_matrix(ring, 5, 7, 851)
        B = self.random_matrix(ring, 7, 6, 852)
        with patch.object(laurent, '_CONVOLUTION_MIN_PRODUCTS', 0), \
             patch.object(laurent, '_CONVOLUTION_REUSE_FACTOR', 0), \
             patch.object(laurent, '_BINARY_CONVOLUTION_MAX_STORAGE_BITS', 8000), \
             patch.object(laurent, '_binary_row_encoded_matrix_product',
                          wraps=laurent._binary_row_encoded_matrix_product) as packed:
            self.assertEqual(A*B, reference_product(A, B))
            self.assertGreater(packed.call_count, 1)

    def test_fallback_odd_characteristic_and_huge_negative_exponents(self):
        for p in (2, 3, 5):
            ring = Ring(p, 2)
            A = self.random_matrix(ring, 3, 4, p)
            B = self.random_matrix(ring, 4, 3, p+10)
            A[0, 0] += ring.monomial((-10**30, 10**30), p-1)
            with patch.object(laurent, '_CONVOLUTION_MIN_PRODUCTS', 0), \
                 patch.object(laurent, '_CONVOLUTION_REUSE_FACTOR', 0):
                self.assertEqual(A*B, reference_product(A, B))

    def test_poly_monomials_and_binary_products_match_literal(self):
        for p in (2, 3, 7):
            for dimension in (1, 2, 3):
                ring = Ring(p, dimension)
                A = self.random_matrix(ring, 1, 1, p*10+dimension)
                if not A:
                    A[0, 0] = 1
                values = [ring.zero(), ring.one(), ring.monomial((-3,)*dimension, p-1), A[0, 0]]
                for a in values:
                    for b in values:
                        self.assertEqual(a*b, reference_product(Matrix.from_rows(ring, [[a]]),
                                                              Matrix.from_rows(ring, [[b]]))[0, 0])

    def test_indexed_unit_reduction_preserves_pivot_order_and_witnesses(self):
        for p in (2, 3, 5):
            ring = Ring(p, 2)
            x, y = ring.variables
            for seed in range(8):
                A = self.random_matrix(ring, 4, 5, p*100+seed, span=1)
                A[1, 2], A[2, 4] = x**-2, (p-1)*y
                actual = GeneralBackend(ring).reduce_units(A)
                expected = reference_unit_reduction(A)
                self.assertEqual(actual, expected)
                r, C, L, V = actual
                D = L*A*V
                self.assertEqual(D[:r, :r], eye(ring, r))
                self.assertFalse(D[:r, r:])
                self.assertFalse(D[r:, :r])
                self.assertEqual(D[r:, r:], C)

    def test_bit_parallel_rhs_elimination_wide_and_inconsistent(self):
        width, rhs = 3, 1300
        # x0+x1=all RHS; x1+x2=even RHS; x2=one RHS.
        values = [(1<<rhs)-1, sum(1<<i for i in range(0, rhs, 2)), 1<<1299]
        left = [0b011, 0b110, 0b100]
        rows = [a | (b<<width) for a, b in zip(left, values)]
        solution = _binary_solution(rows, width, rhs, 10000)
        self.assertIsNotNone(solution)
        for j, answer in enumerate(solution):
            for a, b in zip(left, values):
                self.assertEqual((a&answer).bit_count()%2, (b>>j)&1)
        self.assertIsNone(_binary_solution(rows+[1<<width], width, rhs, 10000))
        self.assertIsNone(_binary_solution(rows, width, rhs, 1))

    def test_wide_sparse_shortcut_and_no_negative_inference(self):
        ring = Ring(2, 2)
        x, y = ring.variables
        A = Matrix.from_rows(ring, [[x, 1], [0, 1]])
        B = Matrix(ring, 2, 1300, {(j%2, j): 1 for j in range(1300)})
        X = sparse_binary_solution(A, B)
        self.assertIsNotNone(X)
        self.assertEqual(reference_product(A, X), B)
        self.assertIsNone(sparse_binary_solution(A, B, max_unknowns=1))
        # Bounded misses still reach a complete solver, including localization.
        with patch('css_decoupling.algebra.sparse_linear.sparse_binary_solution',
                   return_value=None):
            backend = GeneralBackend(ring)
            self.assertEqual(A*backend.solve(A, B[:, :2]), B[:, :2])

    def test_positive_shortcut_avoids_unit_elimination(self):
        ring = Ring(2, 2)
        A = eye(ring, 5)
        with patch.object(GeneralBackend, 'reduce_units', side_effect=AssertionError):
            self.assertEqual(GeneralBackend(ring).solve(A, A), A)

    def test_optional_support_expansion_does_not_discard_affordable_solution(self):
        ring = Ring(2, 2)
        x, y = ring.variables
        A = Matrix.from_rows(ring, [[x, 0], [0, y]])
        B = eye(ring, 2)
        # Four individual contributions fit, six common-support ones do not.
        X = sparse_binary_solution(A, B, max_terms=4)
        self.assertIsNotNone(X)
        self.assertEqual(reference_product(A, X), B)


if __name__ == '__main__':
    unittest.main()
