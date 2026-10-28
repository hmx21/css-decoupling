"""Odd-prime QCA construction, weighted duality and independent witnesses."""
import unittest

from css_decoupling.two_d.algebra import (
    Ring, Matrix, eye, kernel, solve, independent_columns,
    Lattice, FiniteQuotient,
)
from css_decoupling.two_d.models import standard_code, two_generator_css
from css_decoupling.two_d.qca import (
    augmentation, clear_css_mod_j, construct_qca, verify_qca,
)
from css_decoupling.two_d.serialization import encode, decode


def scrambled(ring, toric=1, px=1, pz=1):
    """Change physical/check bases, keeping explicit physical inverses."""
    X, Z = standard_code(toric, product_x=px, product_z=pz, ring=ring)
    n = X.ncols
    x, y = ring.variables
    U, V = eye(ring, n), eye(ring, n)
    for i, j, c in ((0, 2, 2*x**-1+y), (2, 1, 2-x*y),
                    (3, 0, y**-2+2), (1, 3, x)):
        E, Ei = eye(ring, n), eye(ring, n)
        E[i, j], Ei[i, j] = c, -c
        U, V = E*U, V*Ei
    S, Si = eye(ring, n), eye(ring, n)
    S[0, 0] = 2*x**-1*y
    Si[0, 0] = S[0, 0]**-1
    U, V = S*U, V*Si
    Lx, Lz = eye(ring, X.nrows), eye(ring, Z.nrows)
    Lx[0, 0], Lx[0, 1] = 2, x**-1
    Lz[0, 0], Lz[0, 1] = 2, 2-y
    return Lx*X*V, Lz*Z*U.dagger()


class PrimeQCATests(unittest.TestCase):
    def test_signed_toric_complex_and_physical_exactness(self):
        for prime in (3, 5, 7):
            with self.subTest(prime=prime):
                R = Ring(prime, 2)
                x, y = R.variables
                X, Z = standard_code(ring=R)
                self.assertEqual(X, Matrix.from_rows(R, [[x-1, y-1]]))
                self.assertEqual(Z, Matrix.from_rows(R, [[1-y**-1, x**-1-1]]))
                self.assertFalse(X*Z.dagger())
                for H, S in ((X, Z.dagger()), (Z, X.dagger())):
                    K = kernel(H)
                    self.assertEqual(S*solve(S, K), K)
                    self.assertEqual(K*solve(K, S), S)
                self.assertTrue(verify_qca(construct_qca(X, Z, 1)))

    def test_weighted_constant_clearing_and_laurent_maps(self):
        for prime in (3, 5, 7):
            with self.subTest(prime=prime):
                X, Z = scrambled(Ring(prime, 2))
                self.assertFalse(X*Z.dagger())
                clearing = clear_css_mod_j(X, Z)
                self.assertEqual(clearing['product_x'], 1)
                self.assertEqual(clearing['product_z'], 1)
                cx, cz = clearing['cleared_HX'], clearing['cleared_HZ']
                self.assertEqual(X*clearing['xi1'], clearing['xi0']*cx)
                self.assertEqual(Z.dagger()*clearing['xi2'], clearing['xi1']*cz.dagger())
                out = construct_qca(X, Z, 1)
                self.assertTrue(verify_qca(decode(encode(out))))
                self.assertEqual(out['physical_X'].dagger()*out['physical_Z'],
                                 eye(X.ring, X.ncols))

    def test_nonzero_extension_class_is_normalized_not_identified_with_one(self):
        for prime in (3, 5, 7):
            with self.subTest(prime=prime):
                R = Ring(prime, 2)
                X, Z = standard_code(ring=R)
                out = construct_qca(X, 2*Z, 1)
                self.assertEqual(augmentation(out['phi2'])[0, 0], R(pow(2, -1, prime)))
                self.assertTrue(verify_qca(out))

    def test_products_empty_and_multiple_toric_copies(self):
        for prime in (3, 5):
            for t, px, pz in ((0, 0, 0), (0, 2, 0), (0, 0, 2),
                              (0, 2, 3), (2, 1, 1)):
                with self.subTest(prime=prime, t=t, px=px, pz=pz):
                    X, Z = standard_code(t, product_x=px, product_z=pz,
                                         ring=Ring(prime, 2))
                    self.assertTrue(verify_qca(construct_qca(X, Z, t)))

    def test_nontrivial_anyon_translation_and_restricted_lifting(self):
        for prime in (3, 5):
            with self.subTest(prime=prime):
                R = Ring(prime, 2)
                x, y = R.variables
                X, Z = two_generator_css(2*(x**prime-1), y-1)
                Q = FiniteQuotient(X)
                self.assertEqual(Q.dimension, prime)
                self.assertNotEqual(Q.actions[0], eye(R, prime))
                L = Lattice([[-prime, prime], [0, 1]])
                CX, CZ = L.restrict(X), L.restrict(Z)
                out = construct_qca(CX, CZ, prime, restriction_context={
                    'HX': X, 'HZ': Z, 'lattice': L, 'quotient': Q,
                })
                self.assertTrue(verify_qca(out))

    def test_redundant_weighted_generators_reduce_over_prime_field(self):
        for prime in (3, 5):
            with self.subTest(prime=prime):
                R = Ring(prime, 2)
                x, y = R.variables
                X, Z = standard_code(ring=R)
                for H in (X, Z):
                    A = H.dagger().hstack(2*x*H.dagger(), (y-1)*H.dagger())
                    reduced = independent_columns(A)
                    B = reduced['basis']
                    self.assertFalse(kernel(B))
                    self.assertEqual(A*reduced['basis_in_input'], B)
                    self.assertEqual(B*reduced['input_in_basis'], A)

    def test_sign_and_homotopy_tampering_is_rejected(self):
        X, Z = scrambled(Ring(5, 2))
        good = construct_qca(X, Z, 1)
        for label in ('phi1_corrected', 'phi2_corrected', 'eta', 'xi1', 'physical_X'):
            bad = dict(good)
            bad[label] = good[label].copy()
            bad[label][0, 0] += 2
            with self.subTest(label=label), self.assertRaises(ValueError):
                verify_qca(bad)
        bad = dict(good, standard_HZ=good['standard_HZ'].copy())
        bad['standard_HZ'][1, 2] = -bad['standard_HZ'][1, 2]
        with self.assertRaises(ValueError):
            verify_qca(bad)


if __name__ == '__main__':
    unittest.main()
