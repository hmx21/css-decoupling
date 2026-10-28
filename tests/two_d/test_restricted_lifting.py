"""Exact scalar restriction and finite-quotient homotopy lift contracts."""
from random import Random
import unittest
from unittest.mock import patch

from css_decoupling.two_d.algebra import (
    Ring, Matrix, Lattice, FiniteQuotient, GeneralBackend,
    eye, zeros, NoLaurentSolution,
)
from css_decoupling.two_d.restricted_lifting import unpack, solve_restricted


class RestrictedLiftingTests(unittest.TestCase):
    def test_pack_inverse_and_module_intertwining_on_oblique_lattices(self):
        for p in (2, 3, 5):
            R, rng = Ring(p, 2), Random(881+p)
            for basis in ([[3, 2], [0, 1]], [[2, 1], [-1, 2]],
                          [[-2, 1], [1, 2]], [[0, -1], [2, 1]]):
                L = Lattice(basis)
                A = Matrix(R, 2, 3, {(i, j): R({tuple(rng.randrange(-4, 5) for _ in range(2)):
                                                rng.randrange(1, p) for _ in range(4)})
                                      for i in range(2) for j in range(3)})
                X = Matrix(R, 3, 4, {(i, j): R({tuple(rng.randrange(-4, 5) for _ in range(2)):
                                                rng.randrange(1, p) for _ in range(3)})
                                      for i in range(3) for j in range(4)})
                Y = Matrix(R, 3*L.index, 2, {(i, j): R({tuple(rng.randrange(-3, 4) for _ in range(2)):
                                                        rng.randrange(1, p) for _ in range(2)})
                                             for i in range(3*L.index) for j in range(2)})
                self.assertEqual(unpack(L, L.pack(X)), X)
                self.assertEqual(L.pack(unpack(L, Y)), Y)
                self.assertEqual(L.pack(A*X), L.restrict(A)*L.pack(X))
                self.assertEqual(A*unpack(L, Y), unpack(L, L.restrict(A)*Y))

    def test_finite_lifts_negative_powers_and_odd_prime_signs(self):
        for p in (2, 3, 5):
            R = Ring(p, 2)
            x, y = R.variables
            A = Matrix.from_rows(R, [[x-1, y-1]])
            X = Matrix.from_rows(R, [[x**-11+y**4, x**7*y**-3],
                                     [x*y**-5, x**-4*y**9]])
            for basis in ([[3, 2], [0, 1]], [[-2, 1], [1, 2]]):
                L, stats = Lattice(basis), {}
                C, B = L.restrict(A), L.pack(A*X)
                answer = solve_restricted(A, C, B, L, stats=stats)
                self.assertEqual(C*answer, B)
                self.assertEqual(A*unpack(L, answer), A*X)
                self.assertEqual(stats['route'], 'finite_quotient_homotopy')
                self.assertGreater(stats['cache_nodes'], 1)

    def test_nontrivial_actions_multiple_ambient_rows_and_reused_quotient(self):
        R = Ring(2, 2)
        x, y = R.variables
        A = Matrix.from_rows(R, [[x*x+x+1, y+x, 0, 0], [0, 0, x+1, y+1]])
        X = Matrix.from_rows(R, [[x**-8+y**9], [x**15*y**-3], [y**-5+x**7], [x**-3*y**6]])
        L, Q, stats = Lattice([[3, 2], [0, 1]]), FiniteQuotient(A), {}
        B = L.pack(A*X)
        answer = solve_restricted(A, L.restrict(A), B, L, quotient=Q, stats=stats)
        self.assertEqual(L.restrict(A)*answer, B)
        self.assertEqual(stats['quotient_dimension'], 3)

    def test_zero_quotient_infinite_quotient_and_empty_shapes(self):
        R = Ring(2, 2)
        x, y = R.variables
        L = Lattice([[2, 1], [0, 2]])
        A = Matrix.from_rows(R, [[x, 0], [0, y]])
        B = L.pack(Matrix.from_rows(R, [[x**-5, y**8], [x*y, y**-3]]))
        stats = {}
        answer = solve_restricted(A, L.restrict(A), B, L, stats=stats)
        self.assertEqual(L.restrict(A)*answer, B)
        self.assertEqual(stats['quotient_dimension'], 0)
        A = Matrix.from_rows(R, [[x+1]])
        B = L.pack(Matrix.from_rows(R, [[(x+1)*(x**-3+y)]]))
        answer = solve_restricted(A, L.restrict(A), B, L, stats=stats)
        self.assertEqual(L.restrict(A)*answer, B)
        self.assertEqual(stats['route'], 'single_generator_quotient')
        self.assertIn('InfiniteLayer', stats['shortcut_miss'])
        A, B = zeros(R, 0, 3), zeros(R, 0, 2)
        self.assertEqual(solve_restricted(A, L.restrict(A), B, L).shape, (12, 2))

    def test_resource_miss_uses_complete_fallback_and_nonmember_is_confirmed(self):
        R = Ring(2, 2)
        x, y = R.variables
        A = Matrix.from_rows(R, [[x+1, y+1]])
        L = Lattice([[3, 2], [0, 1]])
        B = L.pack(Matrix.from_rows(R, [[x**9+1]]))
        stats = {}
        answer = solve_restricted(A, L.restrict(A), B, L, max_nodes=2, stats=stats)
        self.assertEqual(L.restrict(A)*answer, B)
        self.assertEqual(stats['route'], 'complete_native_fallback')
        self.assertIn('ResourceLimit', stats['shortcut_miss'])
        with self.assertRaises(NoLaurentSolution):
            solve_restricted(A, L.restrict(A), L.pack(eye(R, 1)), L, stats=stats)
        self.assertEqual(stats['shortcut_miss'], 'nonzero_quotient_remainder')
        self.assertEqual(stats['route'], 'complete_native_fallback')

    def test_tampered_lattice_operator_context_and_primitive_are_rejected(self):
        R = Ring(2, 2)
        x, y = R.variables
        A = Matrix.from_rows(R, [[x*x+x+1, y+x]])
        L = Lattice([[3, 2], [0, 1]])
        C, B = L.restrict(A), L.pack(A)
        bad = C.copy()
        bad[0, 0] += x
        with self.assertRaises(ValueError):
            solve_restricted(A, bad, B, L)
        with self.assertRaises(ValueError):
            solve_restricted(A, C, B, Lattice([[3, 0], [0, 1]]))
        Q = FiniteQuotient(A)
        Q.actions = (Q.actions[0]+eye(R, Q.dimension), Q.actions[1])
        with self.assertRaises(ValueError):
            solve_restricted(A, C, B, L, quotient=Q)
        with self.assertRaises(ValueError):
            solve_restricted(A, C, B, L, quotient=FiniteQuotient(eye(R, 1)))
        with patch.object(GeneralBackend, 'solve', side_effect=lambda a, b: zeros(a.ring, a.ncols, b.ncols)):
            with self.assertRaises(ArithmeticError):
                solve_restricted(eye(R, 1), eye(R, L.index), L.pack(eye(R, 1)), L)

    def test_strict_caps_and_unpack_dimensions(self):
        R, L = Ring(2, 2), Lattice([[2, 1], [0, 2]])
        A = eye(R, 1)
        for kwargs in ({'max_nodes': True}, {'max_terms': 0}, {'max_steps': -1},
                       {'max_dimension': False}, {'timeout': float('nan')}, {'stats': []}):
            with self.assertRaises((ValueError, TypeError)):
                solve_restricted(A, L.restrict(A), L.pack(A), L, **kwargs)
        with self.assertRaises(ValueError):
            unpack(L, eye(R, 1))


if __name__ == '__main__':
    unittest.main()
