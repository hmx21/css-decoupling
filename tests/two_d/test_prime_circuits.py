"""Prime-field gate physics, inverse streams and complete Laurent-unit transport."""
from itertools import product
import random
import unittest
from unittest.mock import patch

from css_decoupling.two_d.algebra import Matrix, Ring, block_diag, eye
from css_decoupling.two_d.circuits import (
    circuit_product, compile_qca, determinant, gate_matrix, unit_determinant,
    verify_circuit,
)
from css_decoupling.two_d.direct_circuits import (
    compile_direct_circuit, direct_gate_product, elementary_factor,
    elementary_product, verify_direct_circuit,
)
from css_decoupling.two_d.serialization import decode, encode


def factor(ring, i, j, value):
    return elementary_factor(i, j, Matrix.from_rows(ring, [[value]]))


def balanced(ring, unit):
    return [factor(ring,0,1,-1), factor(ring,1,0,1), factor(ring,0,1,-1),
            factor(ring,0,1,unit), factor(ring,1,0,-unit**-1), factor(ring,0,1,unit)]


def clone(value):
    return decode(encode(value))


class PrimeCircuitTests(unittest.TestCase):
    def test_weighted_sum_by_computational_basis_permutation(self):
        # Four qudits on a 2x1 periodic lattice. The permutation below is
        # independently defined on computational basis labels, not matrices.
        for p in (3,5):
            R=Ring(p,2)
            for weight in range(1,p):
                gate={'kind':'SUM','control':0,'target':1,
                      'offset':[-1,2],'coefficient':weight}
                Z=gate_matrix(R,2,gate,pauli='Z')
                X=gate_matrix(R,2,gate,pauli='X')
                self.assertEqual(Z[0,1],-weight*R.monomial((-1,2)))
                self.assertEqual(X[1,0],weight*R.monomial((1,-2)))
                def act(state):
                    return tuple(state[:2])+tuple((state[2+a]+weight*state[(a-1)%2])%p for a in range(2))
                def support(column):
                    result=[0]*4
                    for species in range(2):
                        for (a,b),c in column[species,0].terms.items():
                            index=2*species+a%2
                            result[index]=(result[index]+c)%p
                    return result
                for species in range(2):
                    zs,xs=support(Z[:,species]),support(X[:,species])
                    for state in product(range(p),repeat=4):
                        transformed=act(state)
                        self.assertEqual(state[2*species],sum(a*b for a,b in zip(zs,transformed))%p)
                        shifted=list(state);shifted[2*species]=(shifted[2*species]+1)%p
                        self.assertEqual(act(shifted),tuple((a+b)%p for a,b in zip(transformed,xs)))
                self.assertEqual(gate_matrix(R,2,gate,reverse=True)*Z,eye(R,2))
                self.assertEqual(Z.dagger()*X,eye(R,2))

    def test_both_stream_replayers_match_generic_products_in_all_modes(self):
        rng=random.Random(48290)
        for p in (3,5):
            R=Ring(p,2)
            layers=[]
            for _ in range(12):
                i,j=rng.sample(range(4),2)
                layers.append([{'kind':'SUM','control':i,'target':j,
                    'offset':[rng.randrange(-2,3),rng.randrange(-2,3)],
                    'coefficient':rng.randrange(1,p)}])
            for pauli in ('X','Z'):
                for reverse in (False,True):
                    expected=eye(R,4)
                    for layer in reversed(layers) if reverse else layers:
                        for g in reversed(layer) if reverse else layer:
                            expected=gate_matrix(R,4,g,pauli=pauli,reverse=reverse)*expected
                    self.assertEqual(circuit_product(R,4,layers,pauli=pauli,reverse=reverse),expected)
                    self.assertEqual(direct_gate_product(R,4,layers,pauli=pauli,reverse=reverse),expected)

    def test_six_shears_cancel_entire_determinant_unit(self):
        for p in (3,5):
            R=Ring(p,2);x,y=R.variables
            A,Ainv,B,Binv=tuple(eye(R,2) for _ in range(4))
            A[0,1]=1+2*x**-1*y;Ainv[0,1]=-A[0,1]
            B[1,0]=2*y**-2;Binv[1,0]=-B[1,0]
            unit=2*x**-2*y**3
            S=Matrix.from_rows(R,[[unit,0],[0,1]])
            Sinv=Matrix.from_rows(R,[[unit**-1,0],[0,1]])
            U,V=S*B*A,Ainv*Binv*Sinv
            C=compile_qca(U,V)
            self.assertEqual(C['schema'],'css-decoupling-2d-circuit-v2')
            self.assertEqual(unit_determinant(U,V),determinant(U))
            self.assertEqual(unit_determinant(U,V),unit)
            self.assertEqual(C['determinant']['constant_unit'],2)
            self.assertEqual(C['determinant']['shift_exponent'],[-2,3])
            self.assertEqual(C['z_action'],block_diag(U,V))
            self.assertTrue(verify_circuit(clone(C)))
            for pauli,reverse,key in [('Z',False,'z_action'),('Z',True,'z_inverse'),
                                      ('X',False,'x_action'),('X',True,'x_inverse')]:
                self.assertEqual(circuit_product(R,4,C['layers'],pauli=pauli,reverse=reverse),C[key])

    def test_direct_signed_factors_and_one_balancing_ancilla(self):
        for p in (3,5):
            R=Ring(p,2);x,y=R.variables
            unit=2*x**-3*y**2
            U=Matrix.from_rows(R,[[unit]]);V=Matrix.from_rows(R,[[unit**-1]])
            factors=balanced(R,unit)
            C=compile_direct_circuit(U,V,factors,added_ancillas=1,
                                      ancilla_action=V,ancilla_inverse=U)
            self.assertEqual(C['schema'],'css-decoupling-2d-direct-circuit/v2')
            self.assertEqual(C['mode'],'exact')
            self.assertEqual(C['z_action'],block_diag(U,V))
            self.assertEqual(C['ancillas']['forward_generator_change'],V)
            self.assertEqual(C['transport']['coefficients'],[1,1])
            self.assertTrue(verify_direct_circuit(clone(C)))
            self.assertEqual(elementary_product(R,2,factors,reverse=True),C['z_inverse'])
            one=[factor(R,0,1,x)]
            self.assertNotEqual(elementary_product(R,2,one),elementary_product(R,2,one,reverse=True))

    def test_noncommuting_factors_with_full_separate_transport(self):
        for p in (3,5):
            R=Ring(p,2);x,y=R.variables
            fs=[factor(R,0,1,2*x**-2+y),factor(R,1,0,1+2*y**-1)]
            F=elementary_product(R,2,fs);Finv=elementary_product(R,2,fs,reverse=True)
            S=Matrix.from_rows(R,[[2*x,0],[0,2*y**-2]])
            Sinv=Matrix.from_rows(R,[[(2*x)**-1,0],[0,(2*y**-2)**-1]])
            C=compile_direct_circuit(S*F,Finv*Sinv,fs,
                shift_exponents=[[1,0],[0,-2]],transport_coefficients=[2,2])
            self.assertEqual(C['mode'],'transport_separated')
            self.assertEqual(C['transport']['constant_unit'],4%p)
            self.assertEqual(C['transport']['x_action'],Sinv.dagger())
            self.assertEqual(C['z_action'],F)
            self.assertFalse(C['equivalence']['original_qca_implemented_by_circuit'])
            with patch('css_decoupling.two_d.direct_circuits.compile_direct_circuit',side_effect=AssertionError):
                self.assertTrue(verify_direct_circuit(clone(C)))

    def test_counter_scalings_are_not_identity_transport(self):
        R=Ring(5,2);S=Matrix.from_rows(R,[[2,0],[0,3]])
        Sinv=Matrix.from_rows(R,[[3,0],[0,2]])
        C=compile_direct_circuit(S,Sinv,[],transport_coefficients=[2,3])
        self.assertEqual(C['transport']['constant_unit'],1)
        self.assertEqual(C['transport']['determinant_exponent'],[0,0])
        self.assertEqual(C['resources']['separate_shift_species_count'],0)
        self.assertEqual(C['resources']['separate_unit_species_count'],2)
        self.assertFalse(C['equivalence']['extended_qca_implemented_by_circuit'])
        self.assertEqual(C['mode'],'transport_separated')

    def test_schema_weight_inverse_and_transport_tampering(self):
        R=Ring(5,2);x,y=R.variables
        U=Matrix.from_rows(R,[[2*x**-1*y]]);V=Matrix.from_rows(R,[[(2*x**-1*y)**-1]])
        double=compile_qca(U,V)
        direct=compile_direct_circuit(U,V,[],shift_exponents=[[-1,1]],transport_coefficients=[2])
        for C,verify,schema in [(double,verify_circuit,'css-decoupling-2d-circuit-v1'),
                               (direct,verify_direct_circuit,'css-decoupling-2d-direct-circuit/v1')]:
            bad=clone(C);bad['schema']=schema
            with self.assertRaises(ValueError):verify(bad)
            bad=clone(C);bad['field_characteristic']=2
            with self.assertRaises(ValueError):verify(bad)
            bad=clone(C);bad['x_action'][0,0]+=1
            with self.assertRaises(ValueError):verify(bad)
        for value in (True,0,5,-1,1.0):
            bad=clone(double);bad['layers'][0][0]['coefficient']=value
            with self.assertRaises(ValueError):verify_circuit(bad)
        bad=clone(double);bad['layers'][0][0]['coefficient']=2
        with self.assertRaises(ValueError):verify_circuit(bad)
        for coefficients in ([1],[0],[True],[5]):
            bad=clone(direct);bad['transport']['coefficients']=coefficients
            with self.assertRaises(ValueError):verify_direct_circuit(bad)
        bad=clone(direct);del bad['transport']['coefficients']
        with self.assertRaises(ValueError):verify_direct_circuit(bad)
        bad=clone(direct);bad['mode']='exact'
        with self.assertRaises(ValueError):verify_direct_circuit(bad)

    def test_binary_schemas_stay_binary_and_empty_prime_inputs_work(self):
        for p in (2,3,5):
            R=Ring(p,2)
            A=compile_qca(eye(R,0),eye(R,0))
            B=compile_direct_circuit(eye(R,0),eye(R,0),[])
            self.assertTrue(verify_circuit(A));self.assertTrue(verify_direct_circuit(B))
            self.assertTrue(A['schema'].endswith('v1' if p==2 else 'v2'))
            self.assertTrue(B['schema'].endswith('/v1' if p==2 else '/v2'))


if __name__=='__main__':
    unittest.main()
