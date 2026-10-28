"""Exact representative coordinates, compact-cell witnesses and legacy v1 reads."""
from copy import deepcopy
from random import Random
import unittest
from unittest.mock import patch

from sympy import Matrix as IntegerMatrix

from css_decoupling.two_d.algebra import Matrix, Ring, Lattice, ResourceLimit, eye
from css_decoupling.two_d.compact_cell import (
    choose_compact_cell, verify_cell_geometry, lattice_from_metadata,
    coset_gauge, gauss_reduce_columns, change_variables,
)
from css_decoupling.two_d.models import two_generator_css, standard_code
from css_decoupling.two_d.pipeline import decouple
from css_decoupling.two_d.serialization import encode, decode
from css_decoupling.two_d.verify import verify
from css_decoupling.two_d.restricted_lifting import unpack


def repeated(W, count):
    n = W.nrows
    return Matrix(W.ring, count*n, count*n, {(k*n+i, k*n+j): f for k in range(count)
                                           for (i, j), f in W.entries.items()})


class CompactCellTests(unittest.TestCase):
    def test_legacy_lattice_metadata_and_reduction_are_unchanged(self):
        L = Lattice([[3, 2], [0, 2]])
        expected = {'basis_columns': [[3, 2], [0, 2]], 'index': 6,
                    'representatives': [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]]}
        self.assertEqual(L.metadata(), expected)
        self.assertEqual(L.reduce((-3, 4)), (4, (-3, 2)))
        self.assertEqual(lattice_from_metadata(expected).metadata(), expected)
        HX, HZ = standard_code(1)
        legacy = decouple(HX, HZ, lattice_basis=[[1, 0], [0, 1]], compile_circuit=False)
        self.assertNotIn('cell_geometry', legacy)
        self.assertNotIn('representative_strategy', legacy['lattice'])
        self.assertTrue(verify(decode(encode(legacy)))['qca'])

    def test_arbitrary_ordered_representatives_preserve_all_module_operations(self):
        for p in (2, 3):
            R = Ring(p, 2)
            x, y = R.variables
            for basis in ([[3, 2], [0, 1]], [[2, 1], [-1, 2]], [[-2, 1], [1, 2]]):
                old = Lattice(basis)
                rng = Random(109+p+old.index)
                reps = list(reversed(old.representatives))
                chosen = []
                for rep in reps:
                    shift = [rng.randrange(-3, 4) for _ in range(2)]
                    chosen.append(tuple(rep[i]+sum(basis[i][j]*shift[j] for j in range(2)) for i in range(2)))
                L = Lattice(basis, representatives=chosen)
                self.assertEqual(L.metadata()['representative_strategy'], 'explicit-v1')
                self.assertEqual(lattice_from_metadata(L.metadata()).metadata(), L.metadata())
                for i, rep in enumerate(chosen):
                    self.assertEqual(L.reduce(rep), (i, (0, 0)))
                for _ in range(20):
                    exponent = tuple(rng.randrange(-20, 21) for _ in range(2))
                    index, shift = L.reduce(exponent)
                    self.assertEqual(exponent, tuple(chosen[index][i]+sum(basis[i][j]*shift[j] for j in range(2))
                                                     for i in range(2)))
                A = Matrix.from_rows(R, [[x**-2+y, 1], [y**-1, x-y]])
                B = Matrix.from_rows(R, [[1+y**-3, x], [x**-1, y+1]])
                X = Matrix.from_rows(R, [[x**-7+y**4], [x**2*y**-3]])
                W = coset_gauge(L, R)
                W2 = repeated(W, 2)
                self.assertEqual(W.dagger()*W, eye(R, L.index))
                self.assertEqual(L.restrict(A), W2.dagger()*old.restrict(A)*W2)
                self.assertEqual(L.pack(X), W2.dagger()*old.pack(X))
                self.assertEqual(L.restrict(A*B), L.restrict(A)*L.restrict(B))
                self.assertEqual(L.restrict(A.dagger()), L.restrict(A).dagger())
                self.assertEqual(L.pack(A*X), L.restrict(A)*L.pack(X))
                self.assertEqual(unpack(L, L.pack(X)), X)

    def test_incomplete_duplicate_noninteger_and_metadata_tampering(self):
        basis = [[2, 0], [0, 1]]
        for representatives in ([], [(0, 0)], [(0, 0), (1, 0), (2, 0)],
                                [(0, 0), (2, 0)], [(0, 0), (True, 0)],
                                [(0, 0), (1.0, 0)], [(0, 0), (1,) ]):
            with self.assertRaises((TypeError, ValueError)):
                Lattice(basis, representatives=representatives)
        metadata = Lattice(basis, representatives=[(0, 0), (1, 0)]).metadata()
        for key, value in (('index', True), ('index', 3), ('representative_strategy', None),
                           ('representative_strategy', 'trust-me'), ('representatives', [[False, 0], [1, 0]])):
            bad = deepcopy(metadata)
            bad[key] = value
            with self.assertRaises((TypeError, ValueError)):
                lattice_from_metadata(bad)
        with self.assertRaises(ValueError):
            lattice_from_metadata({**metadata, 'ignored_extra': 1})

    def test_gauss_witness_and_variable_gauge_identities(self):
        R = Ring(2, 2)
        x, y = R.variables
        A = Matrix.from_rows(R, [[1+x**-2, y], [x*y**-1, 1]])
        for basis in ([[3, 2], [0, 1]], [[-2, 1], [1, 2]], [[186, 178], [0, 2]]):
            full = Lattice(basis)
            selected, proof = choose_compact_cell(full, R)
            self.assertTrue(full.contains(selected) and selected.contains(full))
            self.assertTrue(verify_cell_geometry(full, selected, proof, R, checks=((A, selected.restrict(A)),)))
            transformed = change_variables(full.restrict(A), proof['inverse_basis_change'])
            gauge = repeated(proof['coset_gauge'], 2)
            self.assertEqual(selected.restrict(A), gauge.dagger()*transformed*gauge)
        for options in ({'max_steps': 0}, {'max_steps': True}):
            with self.assertRaises(ValueError):
                gauss_reduce_columns([[3, 2], [0, 1]], **options)
        with self.assertRaises(ResourceLimit):
            gauss_reduce_columns([[889, 198], [0, 1]], max_steps=1)

    def test_full_lattice_equivalence_not_basis_bytes(self):
        R = Ring(2, 2)
        full = Lattice([[3, 2], [0, 1]])
        selected, proof = choose_compact_cell(full, R)
        V = IntegerMatrix([[1, 1], [0, 1]])
        old_U = IntegerMatrix(proof['basis_change'])
        alternate = dict(proof)
        alternate['full_kernel_basis_columns'] = (full.B*V).tolist()
        alternate['basis_change'] = (V.inv()*old_U).tolist()
        alternate['inverse_basis_change'] = (old_U.inv()*V).tolist()
        # Convert SymPy integers to the strict public integer format.
        for name in ('full_kernel_basis_columns', 'basis_change', 'inverse_basis_change'):
            alternate[name] = [[int(value) for value in row] for row in alternate[name]]
        self.assertNotEqual(alternate['full_kernel_basis_columns'], [list(row) for row in full.basis])
        self.assertTrue(verify_cell_geometry(full, selected, alternate, R))
        child, child_proof = choose_compact_cell(Lattice([[2, 0], [0, 2]]), R)
        with self.assertRaises(ValueError):
            verify_cell_geometry(Lattice([[1, 0], [0, 1]]), child, child_proof, R)

    def test_geometry_proof_tampering_and_complete_pipeline_binding(self):
        R = Ring(2, 2)
        x, y = R.variables
        HX, HZ = two_generator_css(x*x+x+1, y+x)
        result = decouple(HX, HZ, compile_circuit=False)
        self.assertEqual(result['lattice']['representative_strategy'], 'explicit-v1')
        self.assertIn('cell_geometry', result)
        self.assertTrue(verify(result, expected_HX=HX, expected_HZ=HZ)['qca'])
        for name in ('basis_change', 'inverse_basis_change', 'full_kernel_basis_columns'):
            bad = decode(encode(result))
            bad['cell_geometry'][name][0][0] += 1
            with self.assertRaises((ValueError, ArithmeticError)):
                verify(bad)
        bad = decode(encode(result))
        bad['cell_geometry']['coset_gauge'][0, 0] += 1
        with self.assertRaises(ValueError):
            verify(bad)
        bad = decode(encode(result))
        bad['lattice']['representatives'].reverse()
        with self.assertRaises(ValueError):
            verify(bad)
        bad = decode(encode(result))
        bad['cell_geometry']['basis_algorithm'] = 'unverified'
        with self.assertRaises(ValueError):
            verify(bad)
        bad = decode(encode(result))
        del bad['cell_geometry']
        with self.assertRaises(ValueError):
            verify(bad)

    def test_explicit_user_basis_and_expansion_preflight_remain_exact(self):
        R = Ring(2, 2)
        x, y = R.variables
        HX, HZ = two_generator_css(x*x+x+1, y+x)
        basis = [[-3, 2], [0, 1]]
        result = decouple(HX, HZ, lattice_basis=basis, compile_circuit=False)
        self.assertEqual(result['lattice']['basis_columns'], basis)
        self.assertNotIn('representative_strategy', result['lattice'])
        self.assertNotIn('cell_geometry', result)
        self.assertTrue(verify(result)['qca'])
        with patch.object(Lattice, 'restrict', side_effect=AssertionError('Expanded before preflight')):
            with self.assertRaises(ResourceLimit):
                decouple(HX, HZ, max_matrix_size=1, compile_circuit=False)


if __name__ == '__main__':
    unittest.main()
