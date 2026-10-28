"""Native paper-QCA equations, boundary cases and certificate tampering."""
import unittest

from css_decoupling.two_d.algebra import Matrix, Ring, inverse
from css_decoupling.two_d.models import (
    BB_CODE_INSTANCES, PUBLISHED_BB_CODE_INSTANCES,
    COLOR_CODE_666, COLOR_CODE_488, standard_code,
)
from css_decoupling.two_d.qca import construct_qca, verify_qca


def eye(ring, n):
    return Matrix(ring, n, n, {(i, i): 1 for i in range(n)})


class QCATests(unittest.TestCase):
    def test_progress_reports_pending_equations_without_changing_result(self):
        HX, HZ = standard_code(1, product_x=1, product_z=1)
        messages = []
        observed = construct_qca(HX, HZ, 1,
            progress=lambda stage, details: messages.append((stage, details)))
        ordinary = construct_qca(HX, HZ, 1)
        self.assertEqual(observed, ordinary)
        self.assertEqual([stage for stage, _ in messages], [
            "clearing_mod_j", "solve_phi1", "solve_phi2", "solve_eta",
            "inverse2", "inverse1", "inverse_preimage", "inverse_homotopy",
            "inverse_assembly", "verify_qca"])
        for stage, details in messages:
            self.assertIsInstance(details["equation"], str)
            self.assertTrue(details["equation"])
            self.assertTrue(any(key.endswith("shape") for key in details))
        with self.assertRaises(TypeError):
            construct_qca(HX, HZ, 1, progress=True)

    def test_standard_and_product_boundary_cases(self):
        for t, px, pz in ((0, 0, 0), (0, 2, 0), (0, 0, 2), (0, 2, 3),
                          (1, 0, 0), (3, 0, 0), (2, 2, 1)):
            with self.subTest(t=t, px=px, pz=pz):
                HX, HZ = standard_code(t, product_x=px, product_z=pz)
                result = construct_qca(HX, HZ, t)
                self.assertTrue(verify_qca(result))
                self.assertEqual(result["product_x"], px)
                self.assertEqual(result["product_z"], pz)
                self.assertEqual(result["standard_HX"], HX)
                self.assertEqual(result["standard_HZ"], HZ)

    def test_nonconstant_laurent_change_with_shift_and_negative_exponents(self):
        ring = Ring(2, 2)
        x, y = ring.variables
        TX, TZ = standard_code(2, product_x=1, product_z=1, ring=ring)
        q = TX.ncols
        U, Ui = eye(ring, q), eye(ring, q)
        for a, b, polynomial in ((0, 3, x**-3+y), (2, 4, x*y**-2),
                                  (5, 1, 1+x), (3, 0, y**-1)):
            E = eye(ring, q)
            E[a, b] = polynomial
            U = E*U
            Ui = Ui*E  # characteristic two: E^-1 = E
        shift, shift_inverse = eye(ring, q), eye(ring, q)
        shift[0, 0], shift_inverse[0, 0] = x*y**-1, x**-1*y
        U, Ui = shift*U, Ui*shift_inverse
        LX, LZ = eye(ring, TX.nrows), eye(ring, TZ.nrows)
        LX[0, 1], LX[1, 2] = 1, 1
        LZ[0, 1], LZ[1, 2] = x, 1
        HX = LX*TX*Ui
        HZ = LZ*TZ*U.dagger()
        result = construct_qca(HX, HZ, 2)
        self.assertTrue(verify_qca(result))
        self.assertEqual(result["physical_Z"], result["psi"][1])
        self.assertEqual(result["physical_X"], result["psi_inverse"][1].dagger())
        self.assertEqual(result["psi"][1]*result["psi_inverse"][1], eye(ring, q))

    def test_directions_and_tampered_maps(self):
        HX, HZ = standard_code(1, product_x=1, product_z=1)
        good = construct_qca(HX, HZ, 1)
        for label in ("psi", "psi_inverse"):
            for degree in range(3):
                bad = dict(good)
                bad[label] = dict(good[label])
                bad[label][degree] = good[label][degree].copy()
                bad[label][degree][0, 0] += 1
                with self.subTest(label=label, degree=degree):
                    with self.assertRaises(ValueError):
                        verify_qca(bad)
        for label in ("eta", "xi1", "phi1", "phi2", "phi1_corrected",
                      "phi2_corrected", "physical_X", "physical_Z"):
            bad = dict(good)
            bad[label] = good[label].copy()
            bad[label][0, 0] += 1
            with self.subTest(label=label):
                with self.assertRaises(ValueError):
                    verify_qca(bad)

    def test_wrong_t_and_noncommuting_checks(self):
        HX, HZ = standard_code(1)
        with self.assertRaises(ValueError):
            construct_qca(HX, HZ, 2)
        HZ[0, 0] += 1
        with self.assertRaises(ValueError):
            construct_qca(HX, HZ, 1)
        ring = Ring(3, 3)
        with self.assertRaises(ValueError):
            construct_qca(Matrix(ring, 0, 0), Matrix(ring, 0, 0), 0)

    def test_tampered_toric_and_product_counts(self):
        HX, HZ = standard_code(1, product_x=1, product_z=1)
        good = construct_qca(HX, HZ, 1)
        for label in ("toric_copies", "product_x", "product_z"):
            for value in (True, -1, 1.0, 2):
                bad = dict(good, **{label: value})
                with self.subTest(label=label, value=value):
                    with self.assertRaises(ValueError):
                        verify_qca(bad)

    def test_exact_paper_input_catalog(self):
        self.assertEqual(len(BB_CODE_INSTANCES), 59)
        self.assertEqual(len({c.identifier for c in BB_CODE_INSTANCES}), 59)
        self.assertEqual(len(PUBLISHED_BB_CODE_INSTANCES), 60)
        for instance in (*PUBLISHED_BB_CODE_INSTANCES, COLOR_CODE_666, COLOR_CODE_488):
            with self.subTest(instance=instance):
                HX, HZ = instance.checks()
                self.assertFalse(HX*HZ.dagger())
        HX, HZ = COLOR_CODE_666.checks()
        x, y = HX.ring.variables
        self.assertEqual(HX, Matrix.from_rows(HX.ring, [[1+x+x*y, 1+y+x*y]]))
        self.assertEqual(HZ, Matrix.from_rows(HX.ring,
            [[1+y**-1+x**-1*y**-1, 1+x**-1+x**-1*y**-1]]))
        HX, HZ = COLOR_CODE_488.checks()
        self.assertEqual(HX, HZ)
        self.assertEqual(HX.shape, (2, 4))


if __name__ == "__main__":
    unittest.main()
