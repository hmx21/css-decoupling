"""Native algebra checks independent of the 2D pipeline construction."""
from copy import deepcopy
from random import Random
from unittest.mock import patch

import unittest

from css_decoupling.two_d.algebra import (
    Ring, Matrix, eye, zeros, kernel, solve, inverse, independent_columns,
    polynomial_kernel, verify_polynomial_kernel, FiniteQuotient, Lattice,
    translation_kernel, verify_translation_kernel, NoLaurentSolution,
    GeneralBackend,
)



class AlgebraTests(unittest.TestCase):
    def test_complete_kernel_witness_and_tampering(self):
        R = Ring(2, 2)
        x, y = R.variables
        A = Matrix.from_rows(R, [[x + 1, y + 1, 0]])
        witness = polynomial_kernel(A, return_witness=True)
        assert verify_polynomial_kernel(A, witness)
        assert not A * witness['kernel']
        bad = dict(witness)
        bad['kernel'] = zeros(R, 3, 0)
        with self.assertRaises((ArithmeticError, ValueError)):
            verify_polynomial_kernel(A, bad)


    def test_negative_exponent_full_kernel_and_nonmembership(self):
        R = Ring(2, 2)
        x, y = R.variables
        A = Matrix.from_rows(R, [[x**-2 * (x + 1), y**-1 * (y + 1)]])
        K = kernel(A)
        expected = Matrix.from_rows(R, [[x**2 * (y + 1)], [y * (x + 1)]])
        assert not A * K
        assert K * solve(K, expected) == expected
        assert expected * solve(expected, K) == K
        with self.assertRaises(NoLaurentSolution):
            solve(A, eye(R, 1))


    def test_free_image_basis_keeps_redundant_original_evidence(self):
        R = Ring(2, 2)
        x, y = R.variables
        B = Matrix.from_rows(R, [[x + y, x*y + 1], [x, y], [1, 1]])
        C = Matrix.from_rows(R, [[x + 1, y + 1, 1 + x + y, 0, 0],
                                 [0, 0, 0, x**-2, 0]])
        A = B * C
        result = independent_columns(A)
        G, T, S = (result[k] for k in ('basis', 'basis_in_input', 'input_in_basis'))
        assert not kernel(G)
        assert G.ncols == 2
        assert A * T == G and G * S == A
        assert G * solve(G, B) == B
        assert B * solve(B, G) == G


    def test_nonsplit_module_is_not_called_free(self):
        R = Ring(2, 2)
        x, y = R.variables
        with self.assertRaises(NoLaurentSolution):
            independent_columns(Matrix.from_rows(R, [[x + 1, y + 1]]))


    def test_laurent_resolution_actually_cancels_a_terminal_injection(self):
        R = Ring(2, 2)
        x, y = R.variables
        A = Matrix.from_rows(R, [[x, y]])
        result = independent_columns(A)
        assert result['resolution_ranks'] == (2, 1)
        assert len(result['cancellations']) == 1
        assert result['basis'].ncols == 1
        assert A * result['basis_in_input'] == result['basis']
        assert result['basis'] * result['input_in_basis'] == A


    def test_qs_split_injection_with_no_unit_entries_in_first_column(self):
        from css_decoupling.two_d.algebra import _split_injection
        R = Ring(2, 2)
        x, y = R.variables
        J = Matrix.from_rows(R, [[x + 1, 0], [y + 1, 0], [x + y + 1, 0], [0, 1]])
        assert not any(f.is_unit for f in J[:, :1].entries.values())
        W, V, left = _split_injection(J)
        assert W * V == V * W == eye(R, 4)
        assert left * J == eye(R, 2)
        assert not (W * J)[2:, :]


    def _check_current_schreyer_matches_independent_sympy_graph(self, seed):
        R = Ring(2, 2)
        x, y = R.variables
        rng = Random(9313 + seed)
        terms = (R.one(), x, y, x*y, x*x)
        A = Matrix.from_rows(R, [[sum((rng.randrange(2)*rng.choice(terms) for _ in range(2)), R.zero())
                                 for _ in range(3)] for __ in range(2)])
        A = A.hstack(A[:, :1] + y*A[:, 1:2], zeros(R, 2, 1))
        witness = polynomial_kernel(A, return_witness=True)
        # The verifier must not rerun either its GB or syzygy constructor.
        with patch('css_decoupling.algebra.schreyer._groebner', side_effect=AssertionError), \
             patch('css_decoupling.algebra.schreyer._syzygies', side_effect=AssertionError):
            assert verify_polynomial_kernel(A, witness)
        reference = GeneralBackend(R)._kernel_graph(A)
        actual = witness['kernel']
        assert reference * solve(reference, actual) == actual
        assert actual * solve(actual, reference) == reference


    def test_incomplete_groebner_witness_cannot_prove_kernel_complete(self):
        R = Ring(2, 2)
        x, y = R.variables
        A = Matrix.from_rows(R, [[x*x + y, x*y]])
        forged = dict(groebner_generators=A, basis_in_input=eye(R, 2),
                      input_in_basis=eye(R, 2), schreyer_kernel=zeros(R, 2, 0),
                      kernel=zeros(R, 2, 0), selected_columns=())
        with self.assertRaisesRegex(ArithmeticError, 'Buchberger'):
            verify_polynomial_kernel(A, forged)


    def _check_zero_free_images(self, shape):
        R = Ring(2, 2)
        A = zeros(R, *shape)
        result = independent_columns(A)
        G, T, S = (result[k] for k in ('basis', 'basis_in_input', 'input_in_basis'))
        assert G.shape == (shape[0], 0)
        assert A * T == G and G * S == A


    def test_cokernel_nontrivial_translation_full_tilted_kernel(self):
        R = Ring(2, 2)
        x, y = R.variables
        # Both translations act as the same F4 primitive element; full kernel is
        # a tilted index-three lattice, not the index-nine rectangular sublattice.
        A = Matrix.from_rows(R, [[x*x + x + 1, y + x]])
        quotient = FiniteQuotient(A)
        assert quotient.dimension == 2
        lattice = translation_kernel({1: quotient}, 2)
        assert lattice.index == 3
        assert verify_translation_kernel({1: quotient}, 2, lattice.kernel_certificate)
        bad = deepcopy(lattice.kernel_certificate)
        bad['basis_columns'] = [[3, 0], [0, 3]]
        bad['index'] = 9
        with self.assertRaises((ValueError, ArithmeticError)):
            verify_translation_kernel({1: quotient}, 2, bad)


    def test_tilted_restriction_preserves_products_dagger_and_inverse(self):
        R = Ring(2, 2)
        x, y = R.variables
        L = Lattice([[2, 1], [-1, 2]])
        A = Matrix.from_rows(R, [[1, x**-2 + y], [0, 1]])
        B = inverse(A)
        assert L.index == 5
        assert L.restrict(A * B) == L.restrict(A) * L.restrict(B)
        assert L.restrict(A.dagger()) == L.restrict(A).dagger()


    def test_external_backend_rejected(self):
        from css_decoupling.two_d.algebra import using_backend
        with self.assertRaises(ValueError):
            with using_backend('singular'):
                pass


for _seed in range(6):
    def _graph_test(self, seed=_seed):
        self._check_current_schreyer_matches_independent_sympy_graph(seed)
    setattr(AlgebraTests, f"test_current_schreyer_graph_seed_{_seed}", _graph_test)

for _shape in ((0, 0), (3, 0), (0, 4), (3, 4)):
    def _zero_test(self, shape=_shape):
        self._check_zero_free_images(shape)
    setattr(AlgebraTests, f"test_zero_free_image_{_shape[0]}_{_shape[1]}", _zero_test)

if __name__ == "__main__":
    unittest.main()
