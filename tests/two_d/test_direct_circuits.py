"""Exact supplied-factor compilation, actual gates, shifts and adversarial data."""
from copy import deepcopy
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from css_decoupling.two_d.algebra import Matrix, Ring, block_diag, eye, zeros
from css_decoupling.two_d.circuits import CircuitResourceLimit, circuit_product, gate_matrix
from css_decoupling.two_d.direct_circuits import (
    compile_direct_circuit, direct_gate_product, elementary_factor, elementary_product, verify_direct_circuit,
)
from css_decoupling.two_d.serialization import decode, encode, load, save


def clone(value):
    return decode(encode(value))


class DirectCircuitTests(unittest.TestCase):
    def setUp(self):
        self.ring = Ring(2,2)
        self.x,self.y = self.ring.variables

    def factor(self,i,j,a):
        return elementary_factor(i,j,Matrix.from_rows(self.ring,[[a]]))

    def balanced_pair(self, unit):
        # Reverse the written six-shear identity, now on two existing species.
        return [self.factor(0,1,1),self.factor(1,0,1),self.factor(0,1,1),
                self.factor(0,1,unit),self.factor(1,0,unit**-1),self.factor(0,1,unit)]

    def test_zero_and_identity_ranks_have_no_added_qubits_or_gates(self):
        for q in (0,1,2,4):
            with self.subTest(q=q):
                U=eye(self.ring,q)
                C=compile_direct_circuit(U,U,[])
                self.assertTrue(verify_direct_circuit(C))
                self.assertEqual(C['layers'],[])
                self.assertEqual(C['z_action'],U)
                self.assertEqual(C['ancillas']['count'],0)
                self.assertEqual(C['resources']['depth'],0)
                self.assertTrue(C['equivalence']['original_qca_implemented_by_circuit'])
                self.assertTrue(C['equivalence']['extended_qca_implemented_by_circuit'])
        self.assertTrue(verify_direct_circuit(compile_direct_circuit(
            eye(self.ring,0),eye(self.ring,0),[],max_matrix_terms=0,max_factors=0,max_families=0)))

    def test_rank_one_global_shift_stays_separate(self):
        unit=self.x**-3*self.y**2
        U=Matrix.from_rows(self.ring,[[unit]])
        V=Matrix.from_rows(self.ring,[[unit**-1]])
        C=compile_direct_circuit(U,V,[],shift_exponents=[[-3,2]])
        self.assertTrue(verify_direct_circuit(C))
        self.assertEqual(C['z_action'],eye(self.ring,1))
        self.assertEqual(C['layers'],[])
        self.assertEqual(C['mode'],'shift_separated')
        self.assertFalse(C['equivalence']['original_qca_implemented_by_circuit'])
        self.assertFalse(C['equivalence']['extended_qca_implemented_by_circuit'])
        self.assertEqual(C['resources']['range_l1_in_supercell_coordinates'],0)
        self.assertEqual(C['resources']['separate_shift_range_l1'],5)
        with self.assertRaises(ValueError):
            compile_direct_circuit(U,V,[])

    def test_one_ancilla_has_actual_inverse_shift_action_not_identity(self):
        unit=self.x**-2*self.y**3
        U=Matrix.from_rows(self.ring,[[unit]])
        V=Matrix.from_rows(self.ring,[[unit**-1]])
        C=compile_direct_circuit(U,V,self.balanced_pair(unit),added_ancillas=1,
                                 ancilla_action=V,ancilla_inverse=U)
        self.assertEqual(C['mode'],'exact')
        self.assertEqual(C['z_action'],block_diag(U,V))
        self.assertNotEqual(C['z_action'],block_diag(U,eye(self.ring,1)))
        self.assertEqual(C['ancillas']['forward_generator_change'],V)
        self.assertTrue(verify_direct_circuit(C))

    def test_cohn_transvection_has_identity_single_ancilla_commutator(self):
        # No claim that the non-E2 polynomial theorem applies after localization.
        for a,b in [(self.x,self.y),(1+self.x,1+self.y),
                    (1+self.x**-2*self.y,1+self.x*self.y**-1)]:
            U=Matrix.from_rows(self.ring,[[1+a*b,a*a],[b*b,1+a*b]])
            self.assertEqual(U*U,eye(self.ring,2))
            # A(v) B(w) A(v)^-1 B(w)^-1 = diag(I+vw,1), wv=0.
            # Chronological list reverses that written product.
            B=[self.factor(2,0,b),self.factor(2,1,a)]
            A=[self.factor(0,2,a),self.factor(1,2,b)]
            C=compile_direct_circuit(U,U,B+A+B+A,added_ancillas=1)
            self.assertEqual(C['z_action'],block_diag(U,eye(self.ring,1)))
            self.assertEqual(C['ancilla_action'],eye(self.ring,1))
            self.assertEqual(C['resources']['circuit_qubits_per_cell'],3)
            self.assertTrue(verify_direct_circuit(C))

    def test_standard_cohn_can_factor_without_ancillas_in_laurent_ring(self):
        a=d=1+self.x*self.y
        b=self.x**2
        U=Matrix.from_rows(self.ring,[[a,b],[self.y**2,d]])
        # U=E21(d/b)*diag(b,b^-1)*E12(a/b)*SWAP; b is a Laurent unit.
        swap=[self.factor(0,1,1),self.factor(1,0,1),self.factor(0,1,1)]
        factors=swap+[self.factor(0,1,a*b**-1)]+self.balanced_pair(b)+[self.factor(1,0,d*b**-1)]
        C=compile_direct_circuit(U,U,factors)
        self.assertEqual(C['ancillas']['count'],0)
        self.assertEqual(C['z_action'],U)
        self.assertTrue(verify_direct_circuit(C))

    def test_transport_side_noncommuting_factors_and_physical_dagger(self):
        factors=[self.factor(0,1,1+self.x**-2*self.y),self.factor(1,0,self.y**-1)]
        F=elementary_product(self.ring,2,factors)
        Finv=elementary_product(self.ring,2,factors,reverse=True)
        S=Matrix.from_rows(self.ring,[[self.x,0],[0,self.y**-2]])
        Sinv=Matrix.from_rows(self.ring,[[self.x**-1,0],[0,self.y**2]])
        U,V=S*F,Finv*Sinv
        C=compile_direct_circuit(U,V,factors,shift_exponents=[[1,0],[0,-2]])
        self.assertNotEqual(F*S,U)
        self.assertEqual(C['z_action'],F)
        self.assertEqual(C['x_action'],Finv.dagger())
        for pauli,reverse,key in [('Z',False,'z_action'),('Z',True,'z_inverse'),
                                  ('X',False,'x_action'),('X',True,'x_inverse')]:
            self.assertEqual(circuit_product(self.ring,2,C['layers'],pauli=pauli,reverse=reverse),C[key])
        self.assertEqual(F*C['x_action'].dagger(),eye(self.ring,2))

    def test_countershifts_are_not_called_identity_transport(self):
        S=Matrix.from_rows(self.ring,[[self.x,0],[0,self.x**-1]])
        C=compile_direct_circuit(S,S.dagger(),[],shift_exponents=[[1,0],[-1,0]])
        self.assertEqual(C['transport']['determinant_exponent'],[0,0])
        self.assertEqual(C['mode'],'shift_separated')
        self.assertFalse(C['equivalence']['original_qca_implemented_by_circuit'])
        # Zero total determinant alone does not authorize deleting species shifts.
        bad=clone(C);bad['transport']['exponents']=[[0,0],[0,0]]
        with self.assertRaises(ValueError):verify_direct_circuit(bad)

    def test_shifted_ancilla_uses_actual_circuit_generator_change(self):
        # F is the balanced two-species shift, W=S F=I, including the ancilla.
        u=self.x**2*self.y**-1
        C=compile_direct_circuit(eye(self.ring,1),eye(self.ring,1),self.balanced_pair(u),
            added_ancillas=1,shift_exponents=[[-2,1],[2,-1]])
        self.assertEqual(C['ancilla_action'],eye(self.ring,1))
        self.assertEqual(C['ancillas']['forward_generator_change'],Matrix.from_rows(self.ring,[[u**-1]]))
        self.assertTrue(verify_direct_circuit(C))

    def test_ancilla_only_transport_keeps_exact_original_system_action(self):
        u=self.x**-2*self.y
        U=Matrix.from_rows(self.ring,[[u]])
        V=Matrix.from_rows(self.ring,[[u**-1]])
        C=compile_direct_circuit(U,V,self.balanced_pair(u),added_ancillas=1,
                                 shift_exponents=[[0,0],[-2,1]])
        self.assertEqual(C['mode'],'shift_separated')
        self.assertTrue(C['equivalence']['original_qca_implemented_by_circuit'])
        self.assertFalse(C['equivalence']['extended_qca_implemented_by_circuit'])
        self.assertEqual(C['ancilla_action'],eye(self.ring,1))
        self.assertEqual(C['ancillas']['forward_generator_change'],V)
        self.assertEqual(C['extended_qca'],block_diag(U,eye(self.ring,1)))
        self.assertEqual(C['z_action'],block_diag(U,V))
        self.assertNotEqual(C['extended_qca'],C['z_action'])
        self.assertTrue(verify_direct_circuit(C))
        for flag in ('original_qca_implemented_by_circuit','extended_qca_implemented_by_circuit'):
            bad=clone(C);bad['equivalence'][flag]=not bad['equivalence'][flag]
            with self.assertRaises(ValueError):verify_direct_circuit(bad)
        # An empty original system is also distinguished from shifted ancillas.
        C=compile_direct_circuit(eye(self.ring,0),eye(self.ring,0),[],added_ancillas=1,
            ancilla_action=U,ancilla_inverse=V,shift_exponents=[[-2,1]])
        self.assertTrue(C['equivalence']['original_qca_implemented_by_circuit'])
        self.assertFalse(C['equivalence']['extended_qca_implemented_by_circuit'])
        self.assertTrue(verify_direct_circuit(C))

    def test_factor_replay_against_generic_matrix_products(self):
        rng=random.Random(372341)
        for n in (2,3,4):
            factors=[]
            for _ in range(8):
                i,j=rng.sample(range(n),2)
                a=sum((self.ring.monomial((rng.randrange(-2,3),rng.randrange(-2,3))) for _ in range(3)),self.ring.zero())
                factors.append(self.factor(i,j,a))
            for reverse in (False,True):
                expected=eye(self.ring,n)
                for f in reversed(factors) if reverse else factors:
                    E=eye(self.ring,n);E[f['row'],f['column']]=f['coefficient'][0,0]
                    expected=E*expected
                self.assertEqual(elementary_product(self.ring,n,factors,reverse=reverse),expected)

    def test_zero_factors_and_serialized_roundtrip(self):
        factors=[self.factor(0,1,0),self.factor(0,1,self.x**-5+self.y**2)]
        U=elementary_product(self.ring,2,factors)
        C=compile_direct_circuit(U,U,factors)
        self.assertEqual(len(C['layers']),2)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'direct.json'
            save(C,path)
            self.assertTrue(verify_direct_circuit(load(path)))

    def test_verifier_accepts_alternative_actual_gate_stream_not_recipe(self):
        factors=[self.factor(0,1,self.x+self.y)]
        U=elementary_product(self.ring,2,factors)
        C=compile_direct_circuit(U,U,factors)
        C['layers'].reverse()  # Same-species same-direction monomials commute.
        self.assertTrue(verify_direct_circuit(C))
        # Distinct, disjoint factors can be in one actual layer.
        fs=[self.factor(0,1,self.x),self.factor(2,3,self.y)]
        F=elementary_product(self.ring,4,fs)
        C=compile_direct_circuit(F,F,fs)
        C['layers']=[C['layers'][0]+C['layers'][1]]
        C['resources']['depth']=1
        self.assertTrue(verify_direct_circuit(C))

    def test_tampered_factor_gate_order_inverse_shift_and_metadata_reject(self):
        fs=[self.factor(0,1,self.x**-1),self.factor(1,0,self.y)]
        U=elementary_product(self.ring,2,fs)
        V=elementary_product(self.ring,2,fs,reverse=True)
        C=compile_direct_circuit(U,V,fs)
        def bad_factor(c):c['factors'][0]['coefficient'][0,0]+=1
        def factor_order(c):c['factors'].reverse()
        def gate_order(c):c['layers'].reverse()
        def bad_gate(c):c['layers'][0][0]['offset'][0]+=1
        def bad_inverse(c):c['z_inverse'][0,0]+=1
        def bad_x(c):c['x_action']=c['z_inverse'].transpose()
        def bad_shift(c):c['transport']['z_action'][0,0]=self.x
        def bad_original(c):c['original_qca'][0,0]+=1
        def overlap(c):c['layers']=[c['layers'][0]+c['layers'][1]]
        for change in [bad_factor,factor_order,gate_order,bad_gate,bad_inverse,bad_x,bad_shift,bad_original,overlap]:
            with self.subTest(change=change.__name__):
                altered=clone(C);change(altered)
                with self.assertRaises(ValueError):verify_direct_circuit(altered)
        for path,value in [(('field_characteristic',),2.0),(('ancillas','count'),False),
            (('transport','constant_unit'),True),(('transport','exponents',0,0),0.0),
            (('transport','included_in_gate_layers'),0),(('resources','depth'),2.0),
            (('equivalence','original_qca_implemented_by_circuit'),1),
            (('factors',0,'row'),False),(('layers',0,0,'coefficient'),True)]:
            altered=clone(C);cursor=altered
            for key in path[:-1]:cursor=cursor[key]
            cursor[path[-1]]=value
            with self.subTest(path=path):
                with self.assertRaises(ValueError):verify_direct_circuit(altered)
        altered=clone(C);altered['trusted_factorizer_success']=True
        with self.assertRaises(ValueError):verify_direct_circuit(altered)

    def test_ancilla_witness_and_equivalence_tampering(self):
        u=self.x*self.y**-1
        U=Matrix.from_rows(self.ring,[[u]])
        V=Matrix.from_rows(self.ring,[[u**-1]])
        C=compile_direct_circuit(U,V,self.balanced_pair(u),added_ancillas=1,ancilla_action=V,ancilla_inverse=U)
        for section,key in [('ancillas','forward_generator_change'),('ancillas','backward_generator_change'),
                            ('ancillas','z_generators'),('equivalence','input_injection'),('equivalence','output_projection')]:
            altered=clone(C);altered[section][key][0,0]+=1
            with self.assertRaises(ValueError):verify_direct_circuit(altered)
        altered=clone(C);altered['ancilla_action']=eye(self.ring,1)
        with self.assertRaises(ValueError):verify_direct_circuit(altered)

    def test_invalid_domains_shapes_factors_and_caps(self):
        I=eye(self.ring,2)
        self.assertTrue(verify_direct_circuit(compile_direct_circuit(eye(Ring(3,2),1),eye(Ring(3,2),1),[])))
        with self.assertRaises(ValueError):compile_direct_circuit(I,zeros(self.ring,2,2),[])
        with self.assertRaises(ValueError):compile_direct_circuit(I,I,[],ancilla_action=eye(self.ring,0))
        with self.assertRaises(ValueError):elementary_factor(0,0,Matrix.from_rows(self.ring,[[1]]))
        with self.assertRaises(ValueError):elementary_factor(0,1,self.x)
        with self.assertRaises(ValueError):compile_direct_circuit(I,I,[self.factor(0,2,0)])
        with self.assertRaises(ValueError):compile_direct_circuit(I,I,[],shift_exponents=[[0,0]])
        with self.assertRaises(ValueError):compile_direct_circuit(I,I,[],max_factors=True)
        with self.assertRaises(CircuitResourceLimit):compile_direct_circuit(I,I,[],added_ancillas=10**100,max_matrix_terms=10)
        fs=[self.factor(0,1,1+self.x)]
        U=elementary_product(self.ring,2,fs)
        for kwargs in ({'max_families':1},{'max_factors':0},{'max_matrix_terms':2}):
            with self.assertRaises(CircuitResourceLimit):compile_direct_circuit(U,U,fs,**kwargs)

    def test_checked_shift_equations_prove_actual_inverse_without_reexpansion(self):
        factors=[self.factor(0,1,1+self.x**-2*self.y),self.factor(1,0,self.y**-1)]
        F=elementary_product(self.ring,2,factors)
        Finv=elementary_product(self.ring,2,factors,reverse=True)
        identity=eye(self.ring,2)
        for exponents in ([[0,0],[0,0]],[[-2,1],[1,-3]]):
            with self.subTest(exponents=exponents):
                S=Matrix(self.ring,2,2,{(i,i):self.ring.monomial(e) for i,e in enumerate(exponents)})
                Sinv=Matrix(self.ring,2,2,{(i,i):self.ring.monomial(tuple(-a for a in e)) for i,e in enumerate(exponents)})
                C=compile_direct_circuit(S*F,Finv*Sinv,factors,shift_exponents=exponents)
                actual,backward=C['z_action'],C['z_inverse']
                # Actual exact products independently establish the desired
                # condition before observing that the verifier avoids repeats.
                self.assertEqual(actual*backward,identity)
                self.assertEqual(backward*actual,identity)
                self.assertEqual(actual*C['x_action'].dagger(),identity)
                self.assertEqual(C['x_action'].dagger()*actual,identity)
                multiply=Matrix.__mul__
                def checked_multiply(left,right):
                    if ((left is actual and right is backward) or
                            (left is backward and right is actual)):
                        raise AssertionError('Redundant circuit inverse product expanded')
                    return multiply(left,right)
                with patch.object(Matrix,'__mul__',checked_multiply):
                    self.assertTrue(verify_direct_circuit(C))
                for pauli,reverse,key in [('Z',False,'z_action'),('Z',True,'z_inverse'),
                                         ('X',False,'x_action'),('X',True,'x_inverse')]:
                    self.assertEqual(direct_gate_product(self.ring,2,C['layers'],pauli=pauli,reverse=reverse),C[key])
                for key in ('z_action','z_inverse'):
                    altered=clone(C);altered[key][0,0]+=1
                    with self.assertRaisesRegex(ValueError,'shift/QCA/circuit equations'):
                        verify_direct_circuit(altered)

    def test_both_actual_action_and_inverse_require_matrix_domain_shape_and_cap(self):
        identity=eye(self.ring,2);good=compile_direct_circuit(identity,identity,[])
        invalid=[None,[],{'shape':[2,2]},self.x,eye(self.ring,1),zeros(self.ring,2,3),
                 eye(Ring(3,2),2),eye(Ring(2,3),2)]
        for key in ('z_action','z_inverse'):
            for value in invalid:
                with self.subTest(key=key,type=type(value).__name__,shape=getattr(value,'shape',None)):
                    altered=clone(good);altered[key]=value
                    with self.assertRaisesRegex(ValueError,'wrong type, dimension or domain'):
                        verify_direct_circuit(altered)
            altered=clone(good);altered[key][0,1]=self.x+self.y
            with self.assertRaises(CircuitResourceLimit):
                verify_direct_circuit(altered,max_matrix_terms=2)

    def test_sparse_gate_replay_all_modes_noncommuting_words_and_layers(self):
        rng=random.Random(304105)
        for n in (0,1,2,3,5,6):
            layers=[]
            if n>=2:
                for _ in range(13):
                    species=list(range(n));rng.shuffle(species)
                    layer=[{'kind':'CNOT','control':species[i],'target':species[i+1],
                            'offset':[rng.randrange(-3,4),rng.randrange(-3,4)],'coefficient':1}
                           for i in range(0,n-1,2)]
                    layers.append(layer)
            for shuffle in (False,True):
                if shuffle:rng.shuffle(layers)
                before=deepcopy(layers)
                for pauli in ('Z','X'):
                    for reverse in (False,True):
                        expected=eye(self.ring,n)
                        for layer in reversed(layers) if reverse else layers:
                            for gate in reversed(layer) if reverse else layer:
                                expected=gate_matrix(self.ring,n,gate,pauli=pauli)*expected
                        actual=direct_gate_product(self.ring,n,layers,pauli=pauli,reverse=reverse)
                        self.assertEqual(actual,expected)
                        self.assertEqual(actual,circuit_product(self.ring,n,layers,pauli=pauli,reverse=reverse))
                self.assertEqual(layers,before)

    def test_sparse_replay_avoids_full_matrix_multiplication_and_term_rescans(self):
        layers=[[{'kind':'CNOT','control':0,'target':1,'offset':[-1,2],'coefficient':1}],
                [{'kind':'CNOT','control':1,'target':0,'offset':[2,-1],'coefficient':1}]]*20
        expected=circuit_product(self.ring,2,layers)
        with patch.object(Matrix,'__mul__',side_effect=AssertionError('global matrix product')):
            with patch('css_decoupling.two_d.direct_circuits._terms',side_effect=AssertionError('global term rescan')):
                actual=direct_gate_product(self.ring,2,layers)
        self.assertEqual(actual,expected)

    def test_sparse_replay_intermediate_caps_overlap_and_exact_types(self):
        gate={'kind':'CNOT','control':0,'target':1,'offset':[-2,3],'coefficient':1}
        cancel=[[gate],[deepcopy(gate)]]
        self.assertEqual(direct_gate_product(self.ring,2,cancel),eye(self.ring,2))
        # Unlike grouped replay, sequential replay retains the first gate.
        self.assertEqual(circuit_product(self.ring,2,cancel,max_matrix_terms=2),eye(self.ring,2))
        with self.assertRaises(CircuitResourceLimit):
            direct_gate_product(self.ring,2,cancel,max_matrix_terms=2)
        with self.assertRaises(CircuitResourceLimit):
            direct_gate_product(self.ring,2,cancel,max_families=1)
        with self.assertRaises(CircuitResourceLimit):
            direct_gate_product(self.ring,2,[],max_matrix_terms=1)
        with self.assertRaises(ValueError):
            direct_gate_product(self.ring,2,[[gate,deepcopy(gate)]])
        for kwargs in ({'qubits':True},{'reverse':1},{'max_families':2.0},
                       {'max_matrix_terms':True},{'pauli':'Y'}):
            options={'qubits':2,**kwargs}
            with self.assertRaises(ValueError):
                direct_gate_product(self.ring,layers=cancel,**options)
        for R in (Ring(2,1),):
            with self.assertRaises(ValueError):direct_gate_product(R,0,[])
        for field,value in [('control',False),('target',0),('offset',[0.0,1]),
                            ('coefficient',True),('kind','SWAP')]:
            bad=deepcopy(gate);bad[field]=value
            with self.assertRaises(ValueError):direct_gate_product(self.ring,2,[[bad]])


if __name__ == '__main__':
    unittest.main()
