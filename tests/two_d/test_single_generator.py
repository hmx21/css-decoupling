"""Positive scalar Laurent quotients, bounded bit planes and whole-row checks."""
from random import Random
import unittest
from unittest.mock import patch

from css_decoupling.two_d.algebra import Ring, Matrix, Lattice, zeros
from css_decoupling.two_d.single_generator import single_generator_solution
from css_decoupling.two_d.restricted_lifting import solve_restricted


class SingleGeneratorTests(unittest.TestCase):
    def test_dominant_and_graded_division_preserve_negative_valuations(self):
        R = Ring(2, 2)
        x, y = R.variables
        rng = Random(62719)
        for f, route in ((1+y+x*y**4, 'rectangular_columns'), (1+x+y**2, 'graded_columns')):
            A = Matrix.from_rows(R, [[x**-4*y**3*f], [x**2+y**-5+1]])
            for _ in range(8):
                Q = Matrix(R, 1, 12, {(0, j): R({(rng.randrange(-12, 13), rng.randrange(-9, 10)): 1
                                                 for _ in range(10)}) for j in range(12)})
                answer, stats = single_generator_solution(A, A*Q)
                self.assertEqual(answer, Q)
                self.assertGreater(stats[route], 0)
                self.assertEqual(A*answer, A*Q)

    def test_unequal_coordinate_bounds_and_block_edges(self):
        R = Ring(2, 2)
        x, y = R.variables
        for f in (1+y+x*y**31, 1+x+x**31*y):
            A = Matrix.from_rows(R, [[f], [1+x+y]])
            Q = Matrix(R, 1, 97, {(0, j): x**(j%17-8)*y**(j%11-5)+x**-3+y**2
                                  for j in range(97)})
            B = A*Q
            answer, stats = single_generator_solution(A, B)
            self.assertEqual(answer, Q)
            for j in (0, 31, 32, 63, 64, 96):
                bad = B.copy()
                bad[1, j] += 1
                missing, data = single_generator_solution(A, bad)
                self.assertIsNone(missing)
                self.assertEqual(data['status'], 'other_row_mismatch')

    def test_zero_rhs_zero_selected_row_monomial_and_unsupported_cases(self):
        R = Ring(2, 2)
        x, y = R.variables
        A = Matrix.from_rows(R, [[0], [x+y], [1+x]])
        Q = Matrix.from_rows(R, [[x**-2+y, 0]])
        answer, stats = single_generator_solution(A, A*Q)
        self.assertEqual(answer, Q)
        self.assertEqual(single_generator_solution(A, zeros(R, 3, 0))[0].shape, (1, 0))
        huge = 1 << 5000
        A = Matrix.from_rows(R, [[R.monomial((huge, -huge))]])
        Q = Matrix.from_rows(R, [[x**-4+y**5]])
        self.assertEqual(single_generator_solution(A, A*Q, max_encoded_bits=1)[0], Q)
        self.assertIsNone(single_generator_solution(zeros(R, 2, 1), zeros(R, 2, 1))[0])
        self.assertIsNone(single_generator_solution(zeros(R, 2, 2), zeros(R, 2, 1))[0])
        for ring in (Ring(3, 2), Ring(2, 1), Ring(2, 3)):
            A = Matrix.from_rows(ring, [[1]])
            self.assertIsNone(single_generator_solution(A, A)[0])

    def test_nonexactness_and_resource_caps_only_request_complete_fallback(self):
        R = Ring(2, 2)
        x, y = R.variables
        A = Matrix.from_rows(R, [[1+x+y]])
        self.assertIsNone(single_generator_solution(A, Matrix.from_rows(R, [[1]]))[0])
        Q = Matrix.from_rows(R, [[x**13+y**7+x**-9]])
        for options in ({'max_steps': 1}, {'max_terms': 1}, {'max_encoded_bits': 2}, {'timeout': 1e-20}):
            answer, stats = single_generator_solution(A, A*Q, **options)
            self.assertIsNone(answer)
            self.assertIn(stats['status'], ('resource_cap', 'encoded_span_cap'))
        L = Lattice([[2, 1], [0, 2]])
        A = Matrix.from_rows(R, [[1+x+y], [x+y**2]])
        B = L.pack(A*Q)
        stats = {}
        answer = solve_restricted(A, L.restrict(A), B, L, stats=stats)
        self.assertEqual(L.restrict(A)*answer, B)
        self.assertEqual(stats['route'], 'single_generator_quotient')
        self.assertIn('fewer columns', stats['shortcut_miss'])
        with patch('css_decoupling.two_d.single_generator.single_generator_solution',
                   return_value=(None, {'status': 'resource_cap'})):
            answer = solve_restricted(A, L.restrict(A), B, L, stats=stats)
        self.assertEqual(L.restrict(A)*answer, B)
        self.assertEqual(stats['route'], 'complete_native_fallback')

    def test_strict_budgets_before_unsupported_branch_and_divisor_preflight(self):
        R = Ring(2, 2)
        x, y = R.variables
        A = Matrix.from_rows(R, [[1+x+y]])
        B = A.copy()
        for name in ('max_steps', 'max_terms', 'max_encoded_bits', 'verify_columns'):
            for value in (True, False, 0, -1, 1.5, float('inf'), float('nan')):
                with self.assertRaises(ValueError):
                    single_generator_solution(A, B, **{name: value})
        for value in (True, False, 0, -1, float('inf'), float('nan')):
            with self.assertRaises(ValueError):
                single_generator_solution(A, B, timeout=value)
        with patch('css_decoupling.two_d.single_generator._normalize', side_effect=AssertionError):
            answer, stats = single_generator_solution(A, B, max_terms=2)
        self.assertIsNone(answer)
        self.assertEqual(stats['reason'], 'divisor_support_cap')


if __name__ == '__main__':
    unittest.main()
