"""Raw-input integration, serialized verification and adversarial evidence."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest

from css_decoupling.two_d.algebra import Matrix, Ring, ResourceLimit, eye
from css_decoupling.two_d.adapter import dispatch_contract
from css_decoupling.two_d.models import (
    COLOR_CODE_666, COLOR_CODE_488, standard_code, two_generator_css,
)
from css_decoupling.two_d.pipeline import InvalidCSS, decouple
from css_decoupling.two_d.serialization import decode, encode, load, save
from css_decoupling.two_d.verify import verify, verify_file


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory = os.environ.get("CSS_DECOUPLING_2D_TEST_OUTPUT")
        cls.run_dir = Path(tempfile.mkdtemp(prefix="pipeline-unittest-", dir=directory))

    def assert_complete(self, HX, HZ, **kwargs):
        # Explicitly exercise the preserved doubled-register compatibility route.
        kwargs.setdefault("compile_circuit", True)
        result = decouple(HX, HZ, **kwargs)
        answer = verify(result, expected_HX=HX, expected_HZ=HZ)
        self.assertTrue(answer["qca"])
        self.assertTrue(answer["circuit"])
        self.assertEqual(result["circuit_status"], "complete_stabilized")
        return result

    def test_raw_standard_products_and_empty(self):
        for t, px, pz in ((0, 0, 0), (0, 2, 0), (0, 0, 2),
                          (0, 2, 1), (1, 0, 0), (3, 0, 0), (2, 1, 2)):
            with self.subTest(t=t, px=px, pz=pz):
                HX, HZ = standard_code(t, product_x=px, product_z=pz)
                result = self.assert_complete(HX, HZ)
                self.assertEqual(result["qca"]["toric_copies"], t)
                self.assertEqual(result["qca"]["product_x"], px)
                self.assertEqual(result["qca"]["product_z"], pz)
                self.assertEqual(result["lattice"]["index"], 1)

    def test_paper_color_codes_and_serialize_reload(self):
        for code, index in ((COLOR_CODE_666, 3), (COLOR_CODE_488, 2)):
            with self.subTest(code=code.label):
                HX, HZ = code.checks()
                result = self.assert_complete(HX, HZ)
                self.assertEqual(result["qca"]["toric_copies"], 2)
                self.assertEqual(result["lattice"]["index"], index)
                path = self.run_dir/f"color-{code.label}.json"
                save(result, path)
                self.assertNotIn(b"\r", path.read_bytes())
                self.assertTrue(verify_file(path, expected_HX=HX, expected_HZ=HZ)["circuit"])
                self.assertEqual(encode(load(path)), encode(result))
                with self.assertRaises(FileExistsError):
                    save(result, path)

    def test_oblique_and_orientation_reversing_lattices(self):
        HX, HZ = standard_code(1)
        for basis, index in ((((2, 1), (0, 1)), 2), (((0, 1), (1, 0)), 1),
                             (((1, -1), (1, 1)), 2)):
            with self.subTest(basis=basis):
                result = self.assert_complete(HX, HZ, lattice_basis=basis)
                self.assertEqual(result["lattice"]["index"], index)
                self.assertEqual(result["qca"]["toric_copies"], 1)

    def test_nontrivial_translation_and_negative_exponents(self):
        ring = Ring(2, 2)
        x, y = ring.variables
        HX, HZ = two_generator_css(1+x**2, 1+y)
        result = self.assert_complete(HX, HZ)
        self.assertEqual(result["lattice"]["index"], 2)
        self.assertEqual(result["qca"]["toric_copies"], 2)
        with self.assertRaises(InvalidCSS):
            decouple(HX, HZ, lattice_basis=((1, 0), (0, 1)))
        HX, HZ = two_generator_css(1+x**-1, 1+y**-1)
        self.assert_complete(HX, HZ)

    def test_redundant_zero_and_nonunit_generators(self):
        HX, HZ = standard_code(1)
        zero = Matrix(HX.ring, 1, HX.ncols)
        self.assert_complete(HX.vstack(HX).vstack(zero), HZ.vstack(zero).vstack(HZ))
        x, y = HX.ring.variables
        # No selected single check row generates the original stabilizer.
        # Yet (1+x, 1+x+x^2) and its y counterpart are unimodular rows.
        RX = ((1+x)*HX).vstack((1+x+x**2)*HX).vstack(zero)
        RZ = ((1+y)*HZ).vstack((1+y+y**2)*HZ).vstack(zero)
        result = self.assert_complete(RX, RZ)
        self.assertEqual(result["reduced_HX"].nrows, 1)
        self.assertEqual(result["reduced_HZ"].nrows, 1)

    def test_original_redundant_chain_retractions_and_tampering(self):
        HX, HZ = standard_code(1)
        x, y = HX.ring.variables
        zero = Matrix(HX.ring, 1, 2)
        RX = ((1+x)*HX).vstack((1+x+x**2)*HX).vstack(zero)
        RZ = ((1+y)*HZ).vstack((1+y+y**2)*HZ).vstack(zero)
        good = self.assert_complete(RX, RZ)
        comparison = good["original_chain_comparison"]
        forward, backward = comparison["to_standard"], comparison["to_original"]
        for degree in (0, 2):
            self.assertEqual(forward[degree].shape, (1, 3))
            self.assertEqual(backward[degree].shape, (3, 1))
            self.assertEqual(forward[degree]*backward[degree], eye(HX.ring, 1))
            projector = comparison["original_endpoint_projectors"][degree]
            self.assertEqual(projector, backward[degree]*forward[degree])
            self.assertEqual(projector*projector, projector)
            self.assertNotEqual(projector, eye(HX.ring, 3))
        for name, degrees in (("to_standard", (0, 1, 2)),
                              ("to_original", (0, 1, 2)),
                              ("original_endpoint_projectors", (0, 2))):
            for degree in degrees:
                with self.subTest(name=name, degree=degree):
                    bad = decode(encode(good))
                    bad["original_chain_comparison"][name][degree][0, 0] += 1
                    with self.assertRaises(ValueError):
                        verify(bad)
        bad = decode(encode(good))
        bad["original_chain_comparison"]["kind"] = "square_chain_isomorphism"
        with self.assertRaises(ValueError):
            verify(bad)

    def test_adapter_preserves_options_and_rejects_unhandled_parameters(self):
        HX, HZ = standard_code(1)
        payload = {"algorithm": "arxiv-2608.09915v1-native-stabilized",
                   "D": 2, "p": 2, "HX": encode(HX), "HZ": encode(HZ),
                   "lattice_basis": [[2, 1], [0, 1]],
                   "options": {"compile_circuit": False}}
        result = dispatch_contract(payload)
        self.assertTrue(verify(result, expected_HX=HX, expected_HZ=HZ)["qca"])
        self.assertIsNone(result["circuit"])
        self.assertEqual(result["circuit_status"], "not_requested")
        self.assertEqual(result["lattice"]["basis_columns"], payload["lattice_basis"])
        self.assertEqual(result["lattice"]["index"], 2)
        limited = {**payload, "options": {"max_matrix_size": 1}}
        with self.assertRaises(ResourceLimit):
            dispatch_contract(limited)
        limited = {**payload, "options": {"max_families": 0}}
        result = dispatch_contract(limited)
        self.assertEqual(result["circuit_status"], "resource_incomplete")
        for key, value in (("algorithm", "generic-D2"), ("p", True), ("D", True),
                           ("p", 2.0), ("D", 2.0), ("p", 3), ("D", 3)):
            with self.subTest(key=key, value=value):
                with self.assertRaises(ValueError):
                    dispatch_contract({**payload, key: value})
        for name in ("profile", "gluing", "backend", "assume_exact"):
            with self.subTest(unhandled=name):
                with self.assertRaises(ValueError):
                    dispatch_contract({**payload, name: None})
                with self.assertRaises((TypeError, ValueError)):
                    dispatch_contract({**payload, "options": {name: None}})

    def test_invalid_inputs(self):
        HX, HZ = standard_code(1)
        bad_z = HZ.copy()
        bad_z[0, 0] += 1
        with self.assertRaises(InvalidCSS):
            decouple(HX, bad_z)
        with self.assertRaises(InvalidCSS):
            decouple(Matrix(HX.ring, 0, 1), Matrix(HX.ring, 0, 1))
        x, y = HX.ring.variables
        with self.assertRaises(InvalidCSS):
            decouple(Matrix.from_rows(HX.ring, [[1+x, 0]]),
                     Matrix.from_rows(HX.ring, [[0, 1+y**-1]]))
        for ring in (Ring(2, 3),):
            with self.assertRaises(InvalidCSS):
                decouple(Matrix(ring, 0, 0), Matrix(ring, 0, 0))

    def test_qca_survives_explicit_circuit_budget(self):
        HX, HZ = COLOR_CODE_666.checks()
        result = decouple(HX, HZ, compile_circuit=True, max_families=0)
        self.assertEqual(result["circuit_status"], "resource_incomplete")
        self.assertIsNone(result["circuit"])
        self.assertTrue(verify(result)["qca"])
        self.assertFalse(verify(result)["circuit"])

    def test_tampering_is_rejected(self):
        HX, HZ = COLOR_CODE_666.checks()
        good = self.assert_complete(HX, HZ)

        def change_matrix(record, path):
            value = record
            for key in path[:-1]:
                value = value[key]
            A = value[path[-1]]
            A[0, 0] += 1

        mutations = [
            ("selected lattice", lambda r: r["lattice"]["basis_columns"][0].__setitem__(0, 99)),
            ("lattice index", lambda r: r["lattice"].__setitem__("index", 99)),
            ("qca forward", lambda r: change_matrix(r, ("qca", "psi", 1))),
            ("qca inverse", lambda r: change_matrix(r, ("qca", "psi_inverse", 1))),
            ("raw generation", lambda r: change_matrix(r, ("check_reduction", "X", "input_in_basis"))),
            ("QCA Z generation", lambda r: change_matrix(r, ("qca_stabilizers", "Z", "source_in_target"))),
            ("QCA X generation", lambda r: change_matrix(r, ("qca_stabilizers", "X", "target_in_source"))),
            ("circuit Z generation", lambda r: change_matrix(r, ("circuit", "stabilizers", "Z", "source_in_target"))),
            ("gate deletion", lambda r: r["circuit"]["layers"].pop()),
            ("gate offset", lambda r: r["circuit"]["layers"][0][0]["offset"].__setitem__(0, 17)),
        ]
        for label, mutate in mutations:
            with self.subTest(label=label):
                bad = decode(encode(good))
                mutate(bad)
                with self.assertRaises((ValueError, ArithmeticError)):
                    verify(bad)
        other_x, other_z = standard_code(1)
        with self.assertRaises(ValueError):
            verify(good, expected_HX=other_x, expected_HZ=other_z)

    def test_strict_matrix_and_json_parser(self):
        HX, _ = standard_code(1)
        encoded = encode(HX)
        corruptions = []
        for key, value in (("__matrix__", True), ("p", True), ("D", 2.0)):
            item = json.loads(json.dumps(encoded)); item[key] = value
            corruptions.append(item)
        item = json.loads(json.dumps(encoded)); item["shape"][0] = -1
        corruptions.append(item)
        item = json.loads(json.dumps(encoded)); item["entries"].append(item["entries"][0])
        corruptions.append(item)
        item = json.loads(json.dumps(encoded)); item["entries"][0][2].append(item["entries"][0][2][0])
        corruptions.append(item)
        item = json.loads(json.dumps(encoded)); item["entries"][0][2][0][-1] = 2
        corruptions.append(item)
        for i, record in enumerate(corruptions):
            with self.subTest(record=i):
                with self.assertRaises(ValueError):
                    decode(record)
        for i, text in enumerate(('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}')):
            path = self.run_dir/f"invalid-{i}.json"
            path.write_text(text, encoding="utf8", newline="\n")
            with self.assertRaises(ValueError):
                load(path)
        with self.assertRaises(ValueError):
            decode({"__integer_mapping__": [[1, None], [1, None]]})


if __name__ == "__main__":
    unittest.main()
