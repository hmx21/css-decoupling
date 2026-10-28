"""Independent small exact cross-checks for aggregated circuit verification."""
from copy import deepcopy
import random
import unittest
from unittest.mock import patch

from css_decoupling.two_d.algebra import Matrix, Ring, eye
from css_decoupling.two_d.circuits import (
    CircuitResourceLimit, _BinaryLaurent, _resources, _unit_pivot_determinant,
    _univariate_divmod, _univariate_determinant_valuation, circuit_product,
    compile_qca, determinant, gate_matrix, verify_circuit,
)


def gate(control, target, offset=(0, 0)):
    return {"kind": "CNOT", "control": control, "target": target,
            "offset": list(offset), "coefficient": 1}


def generic_product(ring, qubits, layers, pauli="Z", reverse=False):
    """Intentionally no aggregation or inverse/duality shortcuts."""
    product = eye(ring, qubits)
    for layer in reversed(layers) if reverse else layers:
        for family in reversed(layer) if reverse else layer:
            product = gate_matrix(ring, qubits, family, pauli=pauli, reverse=reverse) * product
    return product


def clone(value):
    if isinstance(value, Matrix):
        return value.copy()
    if isinstance(value, dict):
        return {key: clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone(item) for item in value]
    return deepcopy(value)


class AggregatedCircuitTests(unittest.TestCase):
    def setUp(self):
        self.ring = Ring(2, 2)

    def check_all_modes(self, qubits, layers):
        products = {}
        for pauli in ("Z", "X"):
            for reverse in (False, True):
                expected = generic_product(self.ring, qubits, layers, pauli, reverse)
                actual = circuit_product(self.ring, qubits, layers,
                                         pauli=pauli, reverse=reverse)
                self.assertEqual(actual, expected)
                products[pauli, reverse] = actual
        forward, backward = products["Z", False], products["Z", True]
        self.assertEqual(forward * backward, eye(self.ring, qubits))
        self.assertEqual(backward * forward, eye(self.ring, qubits))
        self.assertEqual(products["X", False], backward.dagger())
        self.assertEqual(products["X", True], forward.dagger())

    def test_actual_stream_with_dependencies_and_cancelled_terms(self):
        # Sharing a destination in consecutive layers is safe to aggregate;
        # encountering a source/destination dependency must flush first.
        layers = [[gate(0, 3, (-2, 1))], [gate(0, 4, (1, -3))],
                  [gate(2, 4, (0, 2))], [gate(0, 3, (-2, 1))],
                  [gate(4, 1, (-1, 0))], [gate(1, 0, (2, 2))],
                  [gate(3, 2, (0, -1))]]
        self.check_all_modes(5, layers)

    def test_shuffled_arbitrary_noncommuting_streams(self):
        randomizer = random.Random(260823)
        for qubits in (2, 3, 5, 6):
            # Includes dependencies, repeated species, negative exponents,
            # nontrivial multi-family layers, and shuffled chronological order.
            layers = []
            for _ in range(13):
                species = list(range(qubits))
                randomizer.shuffle(species)
                layers.append([gate(species[i], species[i + 1],
                                    (randomizer.randrange(-2, 3),
                                     randomizer.randrange(-2, 3)))
                               for i in range(0, qubits - 1, 2)])
            self.check_all_modes(qubits, layers)
            randomizer.shuffle(layers)
            self.check_all_modes(qubits, layers)

    def test_verifier_accepts_noncanonical_identity_pairs(self):
        U = eye(self.ring, 2)
        U[0, 1] = self.ring.variable(0) + self.ring.variable(1) ** -1
        bundle = compile_qca(U, U)
        # Not the compiler's six-shear stream: a same-register physical
        # gate appears twice at the beginning. Its product is identity.
        extra = gate(0, 1, (-3, 2))
        bundle["layers"][:0] = [[extra], [dict(extra)]]
        bundle["resources"] = _resources(2, bundle["layers"])
        self.assertTrue(verify_circuit(bundle))

    def test_reordered_noncommuting_factors_are_rejected(self):
        U = eye(self.ring, 2)
        U[0, 1] = 1 + self.ring.variable(0)
        bundle = compile_qca(U, U)
        original = generic_product(self.ring, 4, bundle["layers"])
        for index in range(1, len(bundle["layers"])):
            changed = clone(bundle)
            changed["layers"][0], changed["layers"][index] = (
                changed["layers"][index], changed["layers"][0])
            if generic_product(self.ring, 4, changed["layers"]) != original:
                with self.assertRaisesRegex(ValueError, "CNOT product"):
                    verify_circuit(changed)
                break
        else:
            self.fail("Fixture did not contain a noncommuting reordering")

    def test_commuting_reordering_is_accepted(self):
        U = eye(self.ring, 2)
        U[0, 1] = 1 + self.ring.variable(0)
        bundle = compile_qca(U, U)
        bundle["layers"][0:2] = reversed(bundle["layers"][0:2])
        self.assertTrue(verify_circuit(bundle))

    def test_actual_x_and_reverse_declarations_still_checked(self):
        bundle = compile_qca(eye(self.ring, 2), eye(self.ring, 2))
        for name in ("x_action", "x_inverse", "z_inverse"):
            changed = clone(bundle)
            changed[name][0, 1] = self.ring.variable(0)
            with self.assertRaises(ValueError):
                verify_circuit(changed)

    def test_retained_shear_and_action_budgets(self):
        families = [[gate(0, 1, (index, 0))] for index in range(5)]
        with self.assertRaisesRegex(CircuitResourceLimit, "shear"):
            circuit_product(self.ring, 2, families, max_matrix_terms=4)
        with self.assertRaisesRegex(CircuitResourceLimit, "product"):
            circuit_product(self.ring, 2, families[:3], max_matrix_terms=4)
        # Cancellation is exact even when a shear is the zero matrix.
        repeated = [[gate(0, 1)], [gate(0, 1)]]
        self.assertEqual(circuit_product(self.ring, 2, repeated,
                                        max_matrix_terms=2), eye(self.ring, 2))

    def test_indexed_determinant_matches_general_exact_algorithm(self):
        randomizer = random.Random(332178)
        for p in (2, 3, 5):
            ring = Ring(p, 1)
            for qubits in range(5):
                for _ in range(5):
                    entries = {}
                    for i in range(qubits):
                        for j in range(qubits):
                            if randomizer.randrange(3):
                                entries[i, j] = ring({
                                    (exponent,): randomizer.randrange(p)
                                    for exponent in (-1, 0, 1)})
                    matrix = Matrix(ring, qubits, qubits, entries)
                    self.assertEqual(_unit_pivot_determinant(matrix), determinant(matrix))

    def test_binary_determinant_fallback_restarts_unchanged_matrix(self):
        ring = Ring(2, 1)
        x, = ring.variables
        matrix = Matrix.from_rows(ring, [[1, x ** 3], [x ** 3, 1]])
        original = matrix.copy()
        # Every original entry fits four bits. A Schur subtraction exceeds
        # that width after the pivot row/column have already been removed.
        with patch.object(_BinaryLaurent, "MAX_BITS", 4):
            self.assertEqual(_unit_pivot_determinant(matrix), 1 + x ** 6)
        self.assertEqual(matrix, original)
        wide = Matrix.from_rows(ring, [[1 + x ** 65537]])
        self.assertEqual(_unit_pivot_determinant(wide), 1 + x ** 65537)

    def test_binary_polynomial_arithmetic_exact_large_negative_shifts(self):
        ring = Ring(2, 1)
        randomizer = random.Random(7741)
        for _ in range(40):
            left = ring({(randomizer.randrange(-15, 5),): 1 for _ in range(10)})
            right = ring({(randomizer.randrange(-7, 15),): 1 for _ in range(10)})
            a, b = _BinaryLaurent.from_poly(left), _BinaryLaurent.from_poly(right)
            self.assertEqual((a * b).to_poly(ring), left * right)
            self.assertEqual((a - b).to_poly(ring), left - right)
        enormous = ring.monomial((-10 ** 30,))
        shifted = _BinaryLaurent.from_poly(enormous)
        self.assertEqual((shifted ** -1).to_poly(ring), enormous ** -1)

    def test_laurent_euclidean_division_and_strict_remainder_span(self):
        randomizer = random.Random(73522)
        for p in (2, 3, 5):
            ring = Ring(p, 1)
            for _ in range(60):
                a = ring({(randomizer.randrange(-12, 9),): randomizer.randrange(1, p)
                          for _ in range(8)})
                b = ring({(randomizer.randrange(-9, 12),): randomizer.randrange(1, p)
                          for _ in range(5)})
                q, r = _univariate_divmod(a, b)
                self.assertEqual(q * b + r, a)
                if r:
                    span = lambda f: max(e[0] for e in f.terms) - min(e[0] for e in f.terms)
                    self.assertLess(span(r), span(b))
                if p == 2:
                    bq, br = _BinaryLaurent.from_poly(a).divide(_BinaryLaurent.from_poly(b))
                    self.assertEqual(bq.to_poly(ring), q)
                    self.assertEqual(br.to_poly(ring), r)

    def test_dense_unimodular_no_unit_entries_from_elementary_factors(self):
        ring = Ring(2, 1)
        x, = ring.variables
        factors = []
        # L*diag(A,B)*R, with A and B explicit three-factor elementary
        # products. Every one of the final 16 entries is nonzero/nonunit.
        L, R = eye(ring, 4), eye(ring, 4)
        for i in range(2):
            L[i, i + 2], R[i + 2, i] = 1, 1
        factors.append(L)
        for offset, powers in ((0, (1, 2, 3)), (2, (4, 5, 6))):
            for (i, j), power in zip(((0, 1), (1, 0), (0, 1)), powers):
                E = eye(ring, 4)
                E[offset + i, offset + j] = 1 + x ** power
                factors.append(E)
        factors.append(R)
        U, V = eye(ring, 4), eye(ring, 4)
        for E in factors:
            U, V = U * E, E * V
        self.assertEqual(U * V, eye(ring, 4))
        self.assertEqual(V * U, eye(ring, 4))
        self.assertEqual(len(U.entries), 16)
        self.assertTrue(all(not f.is_unit for f in U.entries.values()))
        with patch('css_decoupling.two_d.circuits.determinant',
                   side_effect=lambda M: ring.one() if M.nrows == 0 else
                   (_ for _ in ()).throw(AssertionError('unexpected general determinant'))):
            self.assertEqual(_unit_pivot_determinant(U), 1)
        self.assertEqual(determinant(U), 1)

    def test_euclidean_determinants_nonunits_singular_and_odd_signs(self):
        randomizer = random.Random(993870)
        for p in (2, 3, 5):
            ring = Ring(p, 1)
            x, = ring.variables
            for q in (2, 3, 4):
                for _ in range(8):
                    entries = {(i, j): ring({(-1,): randomizer.randrange(1, p),
                                              (randomizer.randrange(0, 3),): randomizer.randrange(1, p)})
                               for i in range(q) for j in range(q)}
                    A = Matrix(ring, q, q, entries)
                    self.assertTrue(all(not f.is_unit for f in A.entries.values()))
                    self.assertEqual(_unit_pivot_determinant(A), determinant(A))
            isolated = Matrix.from_rows(ring, [[0, x + 1], [x**-2 + x, 0]])
            self.assertEqual(_unit_pivot_determinant(isolated), -(x + 1) * (x**-2 + x))
            singular = Matrix.from_rows(ring, [[x + 1, x*x + 1], [x + 1, x*x + 1]])
            self.assertEqual(_unit_pivot_determinant(singular), 0)

    def test_determinant_valuation_packed_and_sparse_vs_full_determinants(self):
        randomizer = random.Random(438512)
        ring = Ring(2, 1)
        for q in range(5):
            for _ in range(20):
                A = Matrix(ring, q, q, {
                    (i, j): ring({(randomizer.randrange(-3, 4),): 1
                                 for _ in range(randomizer.randrange(1, 5))})
                    for i in range(q) for j in range(q) if randomizer.randrange(3)})
                full = determinant(A)
                for forced_sparse in (False, True):
                    with patch('css_decoupling.two_d.circuits._VALUATION_MAX_PACKED_BITS',
                               0 if forced_sparse else 256_000_000):
                        if not full:
                            with self.assertRaisesRegex(ValueError, 'Singular'):
                                _univariate_determinant_valuation(A)
                        else:
                            expected = min(e[0] for e in full.terms)
                            self.assertEqual(_univariate_determinant_valuation(A), expected)

    def test_valuation_huge_sparse_exponents_and_nonunit_warning(self):
        ring = Ring(2, 1)
        x, = ring.variables
        huge = 10 ** 1000
        A = Matrix.from_rows(ring, [[1, 1 + x ** huge], [1, 1]])
        self.assertEqual(_univariate_determinant_valuation(A), huge)
        A = Matrix.from_rows(ring, [[x ** -huge, x ** -huge + 1], [1, 1]])
        self.assertEqual(_univariate_determinant_valuation(A), 0)
        # Valuation zero does not imply unit determinant. The production
        # caller must retain its checked two-sided-inverse prerequisite.
        nonunit = Matrix.from_rows(ring, [[1 + x]])
        self.assertEqual(_univariate_determinant_valuation(nonunit), 0)
        self.assertFalse(determinant(nonunit).is_unit)
        delayed = Matrix.from_rows(ring, [[1 + x, 1], [1, 1 + x]])
        self.assertEqual(_univariate_determinant_valuation(delayed), 2)
        # Here the working row cycles without becoming zero. Exceeding
        # the initial determinant degree bound proves singularity exactly.
        singular = Matrix.from_rows(ring, [[1 + x, 0], [1, 0]])
        for cap in (0, 256_000_000):
            with patch('css_decoupling.two_d.circuits._VALUATION_MAX_PACKED_BITS', cap):
                with self.assertRaisesRegex(ValueError, 'Singular'):
                    _univariate_determinant_valuation(singular)


if __name__ == "__main__":
    unittest.main()
