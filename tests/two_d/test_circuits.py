"""Infinite Laurent identities and deliberate circuit-certificate tampering."""
from copy import deepcopy
import random
import unittest

from css_decoupling.two_d.algebra import Matrix, Ring, block_diag, eye, zeros
from css_decoupling.two_d.circuits import (
    CircuitResourceLimit, circuit_product, compile_qca, determinant,
    gate_matrix, unit_determinant, verify_circuit,
)


class CircuitTests(unittest.TestCase):
    def setUp(self):
        self.ring = Ring(2, 2)
        self.x, self.y = self.ring.variables

    def test_empty_and_product_state_rank_one(self):
        for q in (0, 1, 2, 4):
            U = eye(self.ring, q)
            result = compile_qca(U, U)
            self.assertTrue(verify_circuit(result))
            self.assertEqual(result["ancillas"]["count"], q)
            self.assertEqual(result["resources"]["depth"], 6 * q)
            self.assertEqual(result["z_action"], eye(self.ring, 2 * q))

    def test_shift_is_cancelled_by_real_gates_on_ancillas(self):
        U = Matrix.from_rows(self.ring, [[self.x ** -3 * self.y ** 2]])
        V = Matrix.from_rows(self.ring, [[self.x ** 3 * self.y ** -2]])
        result = compile_qca(U, V)
        self.assertEqual(result["determinant"]["shift_exponent"], [-3, 2])
        self.assertEqual(result["resources"]["range_l1_in_supercell_coordinates"], 5)
        self.assertEqual(result["resources"]["depth"], 6)
        self.assertEqual(result["z_action"], block_diag(U, V))
        self.assertNotEqual(result["z_action"], block_diag(U, eye(self.ring, 1)))
        self.assertTrue(all(gate["kind"] == "CNOT" for layer in result["layers"] for gate in layer))

    def test_cnot_physical_direction_and_antipode(self):
        gate = {"kind": "CNOT", "control": 0, "target": 1,
                "offset": [-2, 3], "coefficient": 1}
        Z = gate_matrix(self.ring, 2, gate, pauli="Z")
        X = gate_matrix(self.ring, 2, gate, pauli="X")
        self.assertEqual(Z[0, 1], self.x ** -2 * self.y ** 3)
        self.assertEqual(X[1, 0], self.x ** 2 * self.y ** -3)
        self.assertEqual(Z.dagger() * X, eye(self.ring, 2))
        self.assertEqual(circuit_product(self.ring, 2, [[gate]]), Z)

    def test_physical_finite_torus_cross_check(self):
        # Independent computational-basis permutation, not a Laurent
        # matrix recipe; supplements the infinite-module equalities.
        gate = {"kind": "CNOT", "control": 0, "target": 1,
                "offset": [-2, 3], "coefficient": 1}
        lx, ly = 3, 2

        def site(species, a, b):
            return species * lx * ly + (a % lx) * ly + b % ly

        def physical_cnot(bits):
            result = bits
            for a in range(lx):
                for b in range(ly):
                    control = site(0, a - 2, b + 3)
                    if (bits >> control) & 1:
                        result ^= 1 << site(1, a, b)
            return result

        def support_bits(column):
            value = 0
            for species in range(2):
                for (a, b), coefficient in column[species, 0].terms.items():
                    if coefficient:
                        value ^= 1 << site(species, a, b)
            return value

        Z = gate_matrix(self.ring, 2, gate, pauli="Z")
        X = gate_matrix(self.ring, 2, gate, pauli="X")
        for species in range(2):
            original = 1 << site(species, 0, 0)
            z_support = support_bits(Z[:, species])
            x_support = support_bits(X[:, species])
            for bits in range(1 << (2 * lx * ly)):
                transformed = physical_cnot(bits)
                # Z conjugation equality tested as eigenvalue parity.
                self.assertEqual((transformed & original).bit_count() % 2,
                                 (bits & z_support).bit_count() % 2)
                # X conjugation equality tested as bit-flip permutation.
                self.assertEqual(physical_cnot(bits ^ original),
                                 transformed ^ x_support)

    def test_low_rank_cohn_matrix(self):
        # This polynomial SL_2 example is deliberately not a mere shear.
        # The chosen route works without claiming an unstabilized SL_2
        # elementary decomposition (nor asserting Laurent nonrealizability).
        x, y = self.x, self.y
        U = Matrix.from_rows(self.ring, [[1 + x * y, x ** 2],
                                        [y ** 2, 1 + x * y]])
        self.assertEqual(U * U, eye(self.ring, 2))
        result = compile_qca(U, U)
        self.assertTrue(verify_circuit(result))
        self.assertEqual(result["determinant"]["shift_exponent"], [0, 0])

    def test_negative_exponents_and_noncommuting_shears(self):
        x, y = self.x, self.y
        A, B, S = eye(self.ring, 3), eye(self.ring, 3), eye(self.ring, 3)
        A[0, 2] = x ** -3 + y ** 2 + x * y
        B[2, 1] = x ** -2 * y ** 3 + 1
        S[1, 1] = x ** 2 * y ** -4
        Sinv = eye(self.ring, 3)
        Sinv[1, 1] = x ** -2 * y ** 4
        U, V = S * B * A, A * B * Sinv
        result = compile_qca(U, V)
        self.assertEqual(result["determinant"]["shift_exponent"], [2, -4])
        self.assertTrue(verify_circuit(result))

    def test_determinant_exact_prime_arithmetic(self):
        # Determinants and physical compilers use exact prime arithmetic.
        for p in (2, 3, 5):
            ring = Ring(p, 2)
            x, y = ring.variables
            A = Matrix.from_rows(ring, [[x ** -2, 1], [0, 2 * y ** 3]])
            self.assertEqual(determinant(A), 2 * x ** -2 * y ** 3)
            self.assertEqual(determinant(zeros(ring, 0, 0)), 1)
            self.assertEqual(determinant(zeros(ring, 2, 2)), 0)

    def test_unit_determinant_requires_actual_inverse_and_has_general_fallback(self):
        # All entries of the x specialization are nonunits; this forces
        # the exact residual determinant fallback instead of unit pivots.
        x = self.x
        a, b = x + 1, x**2 + x + 1
        c, d = x**3 + x**2 + x, x**4 + x**3 + 1
        U = Matrix.from_rows(self.ring, [[a, b], [c, d]])
        V = Matrix.from_rows(self.ring, [[d, b], [c, a]])
        self.assertTrue(all(not value.is_unit for value in U.entries.values()))
        self.assertEqual(U * V, eye(self.ring, 2))
        self.assertEqual(unit_determinant(U, V), determinant(U))
        self.assertEqual(unit_determinant(U, V), 1)
        # Both axis specializations of x+y+xy equal 1, despite this
        # polynomial being a nonunit.  An unchecked shortcut would lie.
        bad = Matrix.from_rows(self.ring, [[x + self.y + x*self.y]])
        with self.assertRaises(ValueError):
            unit_determinant(bad, eye(self.ring, 1))

    def test_indexed_gate_product_matches_generic_multiplication(self):
        randomizer = random.Random(812693)
        for q in (2, 3, 4, 5):
            layers = []
            for _ in range(14):
                species = list(range(q))
                randomizer.shuffle(species)
                layer = [{"kind": "CNOT", "control": species[i],
                          "target": species[i+1],
                          "offset": [randomizer.randrange(-2, 3), randomizer.randrange(-2, 3)],
                          "coefficient": 1}
                         for i in range(0, q - 1, 2)]
                layers.append(layer)
            for pauli in ("Z", "X"):
                for reverse in (False, True):
                    expected = eye(self.ring, q)
                    for layer in reversed(layers) if reverse else layers:
                        for gate in reversed(layer) if reverse else layer:
                            expected = gate_matrix(self.ring, q, gate, pauli=pauli) * expected
                    actual = circuit_product(self.ring, q, layers, pauli=pauli, reverse=reverse)
                    self.assertEqual(actual, expected)
        with self.assertRaises(CircuitResourceLimit):
            circuit_product(self.ring, 2, [], max_matrix_terms=1)

    def test_prime_identity_supported_and_bad_inverse_rejected(self):
        self.assertTrue(verify_circuit(compile_qca(eye(Ring(3, 2), 1), eye(Ring(3, 2), 1))))
        with self.assertRaises(ValueError):
            compile_qca(eye(self.ring, 2), zeros(self.ring, 2, 2))
        with self.assertRaises(ValueError):
            compile_qca(zeros(self.ring, 2, 1), zeros(self.ring, 1, 2))

    def test_tampered_gate_mapping_inverse_ancilla_and_witness_rejected(self):
        U = eye(self.ring, 2)
        U[0, 1] = 1 + self.x ** -2 * self.y
        result = compile_qca(U, U)

        def mutate_gate(bundle):
            bundle["layers"][0][0]["offset"][0] += 1

        def mutate_action(bundle):
            bundle["z_action"][0, 0] = self.x

        def mutate_inverse(bundle):
            bundle["original_inverse"][0, 0] = self.y

        def mutate_ancilla_type(bundle):
            bundle["ancillas"]["type"] = "X"

        def mutate_generation(bundle):
            bundle["ancillas"]["forward_generator_change"][0, 0] = self.x

        def mutate_projection(bundle):
            bundle["equivalence"]["output_projection"][0, 2] = 1

        def mutate_determinant(bundle):
            bundle["determinant"]["shift_exponent"][0] = 7

        def mutate_resources(bundle):
            bundle["resources"]["depth"] -= 1

        for mutation in (mutate_gate, mutate_action, mutate_inverse,
                         mutate_ancilla_type, mutate_generation,
                         mutate_projection, mutate_determinant, mutate_resources):
            with self.subTest(mutation=mutation.__name__):
                tampered = clone(result)
                mutation(tampered)
                with self.assertRaises(ValueError):
                    verify_circuit(tampered)

    def test_overlapping_layer_is_rejected_even_for_identity_product(self):
        gate = {"kind": "CNOT", "control": 0, "target": 1,
                "offset": [0, 0], "coefficient": 1}
        with self.assertRaisesRegex(ValueError, "overlapping"):
            circuit_product(self.ring, 2, [[gate, gate]])

    def test_metadata_boolean_float_integer_substitutions_are_rejected(self):
        result = compile_qca(eye(self.ring, 1), eye(self.ring, 1))
        substitutions = [
            (("field_characteristic",), 2.0),
            (("dimension",), 2.0),
            (("ancillas", "count"), True),
            (("ancillas", "count"), 1.0),
            (("determinant", "constant_unit"), True),
            (("determinant", "shift_exponent", 0), 0.0),
            (("stabilized_determinant", "constant_unit"), 1.0),
            (("stabilized_determinant", "shift_exponent", 1), False),
            (("resources", "depth"), 6.0),
            (("resources", "added_Z_ancillas_per_cell"), True),
            (("resources", "added_X_ancillas_per_cell"), False),
            (("resources", "range_l1_in_supercell_coordinates"), 0.0),
            (("resources", "translation_subgroup_in_supercell_coordinates", 0, 0), True),
            (("resources", "translation_subgroup_in_supercell_coordinates", 1, 0), 0.0),
            (("layers", 0, 0, "control"), False),
            (("layers", 0, 0, "target"), 1.0),
            (("layers", 0, 0, "coefficient"), True),
            (("layers", 0, 0, "offset", 0), 0.0),
        ]
        for path, replacement in substitutions:
            with self.subTest(path=path, replacement=replacement):
                tampered = clone(result)
                container = tampered
                for key in path[:-1]:
                    container = container[key]
                container[path[-1]] = replacement
                with self.assertRaises(ValueError):
                    verify_circuit(tampered)

    def test_resource_limit_is_not_nonexistence(self):
        U = eye(self.ring, 2)
        with self.assertRaises(CircuitResourceLimit):
            compile_qca(U, U, max_families=11)
        result = compile_qca(U, U)
        with self.assertRaises(CircuitResourceLimit):
            verify_circuit(result, max_families=11)


def clone(value):
    """Deepcopy only mutable records; Laurent polynomials are immutable."""
    if isinstance(value, Matrix):
        return value.copy()
    if isinstance(value, dict):
        return {key: clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone(item) for item in value]
    return deepcopy(value)


if __name__ == "__main__":
    unittest.main()
