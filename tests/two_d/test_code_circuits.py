"""Same-register decoupling, target-code freedom and physical scalar gates."""
from itertools import product
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from css_decoupling.two_d.algebra import Ring,Matrix,eye
from css_decoupling.two_d.models import standard_code,COLOR_CODE_666,COLOR_CODE_488
from css_decoupling.two_d.pipeline import decouple
from css_decoupling.two_d.elementary import factor_unit_reduction
from css_decoupling.two_d.code_circuits import (
    SCHEMA,compile_code_circuit,compile_qca_no_ancilla,verify_no_ancilla_result,
    target_transport_correction,code_gate_product,
)
from css_decoupling.two_d.circuits import CircuitResourceLimit,unit_determinant
from css_decoupling.two_d.serialization import encode,decode,digest,save,load


def clone(value):
    return decode(encode(value))


def factor_data(source):
    Q=source['qca']
    return factor_unit_reduction(Q['psi'][1],Q['psi_inverse'][1],
                                 transport='separate',residual='stop')


def diagonal(ring,units):
    return Matrix(ring,len(units),len(units),{(i,i):u for i,u in enumerate(units)})


class TargetCorrectionTests(unittest.TestCase):
    def test_signed_toric_pairs_and_existing_products(self):
        rng=random.Random(260815)
        for p in (2,3,5,7):
            R=Ring(p,2)
            for t,px,pz in ((0,0,0),(0,1,0),(0,0,1),(0,2,1),
                            (1,0,0),(2,0,0),(1,1,0),(2,0,1),(2,1,2)):
                TX,TZ=standard_code(t,product_x=px,product_z=pz,ring=R)
                q=TX.ncols
                for _ in range(3):
                    S=diagonal(R,[R.monomial((rng.randrange(-2,3),rng.randrange(-2,3)),
                                             rng.randrange(1,p)) for i in range(q)])
                    C,K=target_transport_correction(TX,TZ,t,px,pz,S)
                    G,Ginv=C['z_action'],C['z_inverse']
                    self.assertEqual(G.shape,(q,q))
                    self.assertEqual(G*Ginv,eye(R,q))
                    self.assertEqual(K['z_action']*S,G)
                    if px+pz:
                        self.assertFalse(any(g['kind']=='MULTIPLY' for L in C['layers'] for g in L))
                        for i in range(px+pz,q):
                            self.assertEqual(K['z_action'][:,i],eye(R,q)[:,i])
                    for sector,A,B,H in (('Z',K['z_action'],K['z_inverse'],TZ),
                                         ('X',K['x_action'],K['x_inverse'],TX)):
                        changes=K['stabilizers'][sector]
                        self.assertEqual(A*H.dagger(),H.dagger()*changes['source_in_target'])
                        self.assertEqual(B*H.dagger(),H.dagger()*changes['target_in_source'])
                    for pauli,rev,key in (('Z',False,'z_action'),('Z',True,'z_inverse'),
                                          ('X',False,'x_action'),('X',True,'x_inverse')):
                        self.assertEqual(code_gate_product(R,q,C['layers'],pauli=pauli,reverse=rev),C[key])

    def test_product_only_transport_needs_no_gates_or_new_species(self):
        R=Ring(5,2);x,y=R.variables
        TX,TZ=standard_code(0,product_x=2,product_z=2,ring=R)
        S=diagonal(R,[2*x,3*y**-1,4*x**2*y**-3,x**-4])
        C,K=target_transport_correction(TX,TZ,0,2,2,S)
        self.assertEqual(C['layers'],[])
        self.assertEqual(C['z_action'],eye(R,4))
        self.assertNotEqual(K['z_action'],eye(R,4))

    def test_p5_nonsquare_toric_scalar_needs_explicit_onsite_gate(self):
        R=Ring(5,2);TX,TZ=standard_code(ring=R)
        S=diagonal(R,[R(2),R(1)])
        C,K=target_transport_correction(TX,TZ,1,0,0,S)
        self.assertEqual(C['layers'],[[{'kind':'MULTIPLY','qudit':0,'multiplier':3}]])
        self.assertEqual(C['z_action'],S)
        self.assertEqual(K['z_action'],eye(R,2))
        self.assertNotIn(2,{a*a%5 for a in range(1,5)})
        self.assertIn(-1%5,{a*a%5 for a in range(1,5)})
        # An existing target product coordinate avoids this scalar gate:
        PX,PZ=standard_code(1,product_z=1,ring=R)
        D,_=target_transport_correction(PX,PZ,1,0,1,diagonal(R,[R(1),R(2),R(1)]))
        self.assertFalse(any(g['kind']=='MULTIPLY' for L in D['layers'] for g in L))

    def test_multiplication_gate_actual_basis_conjugation_and_inverse(self):
        for p in (3,5,7):
            R=Ring(p,2)
            for multiplier in range(1,p):
                layers=[[{'kind':'MULTIPLY','qudit':0,'multiplier':multiplier}]]
                Z=code_gate_product(R,1,layers);X=code_gate_product(R,1,layers,pauli='X')
                self.assertEqual(Z[0,0],pow(multiplier,-1,p))
                self.assertEqual(X[0,0],multiplier)
                for state in range(p):
                    transformed=multiplier*state%p
                    # P Z P^-1 eigenvalue and P X = X^a P, independently
                    # checked as computational-basis permutations/phases.
                    self.assertEqual(state,pow(multiplier,-1,p)*transformed%p)
                    self.assertEqual(multiplier*(state+1)%p,(transformed+multiplier)%p)
                self.assertEqual(code_gate_product(R,1,layers,reverse=True)*Z,eye(R,1))
                self.assertEqual(Z.dagger()*X,eye(R,1))

    def test_disjoint_gate_schedule_and_correction_caps(self):
        R=Ring(5,2);TX,TZ=standard_code(ring=R);x,y=R.variables
        with self.assertRaises(CircuitResourceLimit):
            target_transport_correction(TX,TZ,1,0,0,diagonal(R,[x**100,y]),max_matrix_terms=10)
        with self.assertRaises(CircuitResourceLimit):
            target_transport_correction(TX,TZ,1,0,0,diagonal(R,[R(2),R(1)]),max_families=0)
        scale={'kind':'MULTIPLY','qudit':0,'multiplier':2}
        gate={'kind':'SUM','control':0,'target':1,'offset':[0,0],'coefficient':1}
        with self.assertRaisesRegex(ValueError,'Overlapping'):
            code_gate_product(R,2,[[scale,gate]])
        for scalar in (0,5,True,1.0):
            with self.assertRaises(ValueError):
                code_gate_product(R,1,[[{**scale,'multiplier':scalar}]])


class NoInputAncillaTests(unittest.TestCase):
    def raw(self,p,t=1,px=0,pz=0):
        R=Ring(p,2);TX,TZ=standard_code(t,product_x=px,product_z=pz,ring=R)
        q=TX.ncols
        units=[R.monomial((i%2,-(i+1)%2),2 if p>2 and i==0 else 1) for i in range(q)]
        U=diagonal(R,units);V=diagonal(R,[u**-1 for u in units])
        return TX*U,TZ*V.dagger()

    def test_raw_same_register_empty_product_toric_and_skew_fixed(self):
        for p in (2,3,5):
            for counts in ((0,0,0),(0,1,0),(0,0,1),(1,0,0),(1,1,1)):
                with self.subTest(p=p,counts=counts):
                    HX,HZ=self.raw(p,*counts)
                    source=decouple(HX,HZ,compile_circuit=False,lattice_mode='fixed',
                                    lattice_basis=[[1,1],[0,1]])
                    before=digest(source)
                    result=compile_qca_no_ancilla(source,factor_data=factor_data(source))
                    report=verify_no_ancilla_result(clone(result),expected_HX=HX,expected_HZ=HZ)
                    self.assertTrue(report['verified'] and report['stabilizer_equivalent'])
                    self.assertEqual(report['input_ancillas'],0)
                    self.assertEqual(report['physical_qudits'],HX.ncols)
                    self.assertEqual(result['schema'],SCHEMA)
                    self.assertEqual(digest(source),before)
                    actual=result['actual_circuit']
                    for pauli,rev,key in (('Z',False,'z_action'),('Z',True,'z_inverse'),
                                          ('X',False,'x_action'),('X',True,'x_inverse')):
                        self.assertEqual(code_gate_product(HX.ring,HX.ncols,actual['layers'],
                                        pauli=pauli,reverse=rev),actual[key])

    def test_low_rank_identity_uses_code_equality_without_factoring_initial_qca(self):
        for p in (2,3,5):
            R=Ring(p,2);x,y=R.variables
            for t,px,pz in ((0,0,0),(0,1,0),(0,0,1),(1,0,0)):
                HX,HZ=standard_code(t,product_x=px,product_z=pz,ring=R)
                # Noncanonical redundant generator lists still present the
                # identical physical code; no row-count shortcut suffices.
                HX=(x*HX).vstack((x+y)*HX)
                HZ=(y**-1*HZ).vstack((x-1)*HZ)
                source=decouple(HX,HZ,compile_circuit=False,lattice_mode='fixed',lattice_basis=[[1,0],[0,1]])
                with patch('css_decoupling.two_d.code_circuits.compile_code_circuit',side_effect=AssertionError('factor path')):
                    C=compile_qca_no_ancilla(source)
                self.assertIsNone(C['initial_factorization']);self.assertIsNone(C['correction'])
                self.assertEqual(C['actual_circuit']['layers'],[])
                self.assertEqual(C['target_automorphism']['z_action'],source['qca']['psi_inverse'][1])
                self.assertTrue(verify_no_ancilla_result(clone(C),expected_HX=HX,expected_HZ=HZ)['verified'])
                bad=clone(C);bad['construction']['reason']='q=2 therefore elementary'
                with self.assertRaises(ValueError):verify_no_ancilla_result(bad)

    def test_nonstandard_rank_two_cohn_input_is_not_called_identity(self):
        from css_decoupling.two_d.code_circuits import _low_rank_identity
        for p in (2,3,5):
            R=Ring(p,2);x,y=R.variables;a=x-1;b=y-1
            HX,HZ=standard_code(ring=R)
            C=Matrix.from_rows(R,[[1+a*b,a*a],[-b*b,1-a*b]])
            Ci=Matrix.from_rows(R,[[1-a*b,-a*a],[b*b,1+a*b]])
            source=decouple(HX*C,HZ*Ci.dagger(),compile_circuit=False,lattice_mode='fixed',lattice_basis=[[1,0],[0,1]])
            self.assertIsNone(_low_rank_identity(source,max_matrix_terms=1000000,max_families=200000))

    def test_color_codes_and_serialized_independent_reader(self):
        for model in (COLOR_CODE_666,COLOR_CODE_488):
            HX,HZ=model.checks()
            source=decouple(HX,HZ,compile_circuit=False)
            result=compile_code_circuit(source,factor_data(source))
            q=source['qca']['psi'][1].nrows
            self.assertEqual(result['actual_circuit']['z_action'].shape,(q,q))
            self.assertEqual(result['resources']['MULTIPLY_family_count'],0)
            with tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'code-circuit.json';save(result,path)
                with (patch('css_decoupling.two_d.code_circuits.compile_code_circuit',side_effect=AssertionError('constructor')),
                      patch('css_decoupling.two_d.code_circuits.target_transport_correction',side_effect=AssertionError('correction constructor'))):
                    report=verify_no_ancilla_result(load(path),expected_HX=HX,expected_HZ=HZ)
                    self.assertEqual(report['physical_qudits'],q)

    def test_p5_scalar_code_and_original_qca_are_distinguished(self):
        R=Ring(5,2);TX,TZ=standard_code(ring=R)
        S=diagonal(R,[R(2),R(1)]);Sinv=diagonal(R,[R(3),R(1)])
        source=decouple(TX*S,TZ*Sinv.dagger(),compile_circuit=False)
        result=compile_code_circuit(source,factor_data(source))
        report=verify_no_ancilla_result(result)
        self.assertEqual(report['input_ancillas'],0)
        self.assertEqual(report['physical_qudits'],2)
        self.assertGreater(result['resources']['MULTIPLY_family_count'],0)
        U,V=source['qca']['psi'][1],source['qca']['psi_inverse'][1]
        det=unit_determinant(U,V)
        self.assertNotIn(next(iter(det.terms.values())),{a*a%5 for a in range(1,5)})
        self.assertEqual(result['actual_circuit']['z_action'],result['target_automorphism']['z_action']*U)
        self.assertEqual(report['original_qca_implemented_by_circuit'],result['actual_circuit']['z_action']==U)

    def test_complete_sum_multiply_stream_on_computational_basis_torus(self):
        R=Ring(5,2);TX,TZ=standard_code(ring=R)
        S=diagonal(R,[R(2),R(1)]);Sinv=diagonal(R,[R(3),R(1)])
        source=decouple(TX*S,TZ*Sinv.dagger(),compile_circuit=False)
        C=compile_code_circuit(source,factor_data(source))['actual_circuit']
        # Two cells in x and one in y, four physical qudits. This oracle
        # interprets gate records as computational-basis permutations only.
        def site(species,x,y):return 2*species+x%2
        def apply(state):
            state=list(state)
            for layer in C['layers']:
                for gate in layer:
                    if gate['kind']=='MULTIPLY':
                        for x in range(2):
                            i=site(gate['qudit'],x,0)
                            state[i]=gate['multiplier']*state[i]%5
                    else:
                        dx,dy=gate['offset']
                        for x in range(2):
                            control=site(gate['control'],x+dx,dy)
                            target=site(gate['target'],x,0)
                            state[target]=(state[target]+gate['coefficient']*state[control])%5
            return tuple(state)
        def support(column):
            answer=[0]*4
            for species in range(2):
                for (x,y),c in column[species,0].terms.items():
                    i=site(species,x,y);answer[i]=(answer[i]+c)%5
            return answer
        for species in range(2):
            z=support(C['z_action'][:,species]);x=support(C['x_action'][:,species])
            for state in product(range(5),repeat=4):
                output=apply(state)
                self.assertEqual(state[2*species],sum(a*b for a,b in zip(output,z))%5)
                moved=list(state);moved[2*species]=(moved[2*species]+1)%5
                self.assertEqual(apply(moved),tuple((a+b)%5 for a,b in zip(output,x)))

    def test_added_register_and_altered_equivalence_or_witness_are_rejected(self):
        HX,HZ=self.raw(5,1,1,1)
        source=decouple(HX,HZ,compile_circuit=False)
        data=factor_data(source)
        for count in (1,False,0.0):
            with self.assertRaises(ValueError):
                compile_code_circuit(source,{**data,'added_ancillas':count})
        good=compile_code_circuit(source,data)
        def ancilla(c):c['equivalence']['input_ancillas']=1
        def size(c):c['resources']['physical_qudits']+=1
        def phase(c):c['actual_circuit']['conventions']['pure_X_Z_phase']=1
        def raw(c):c['qca_result']['original']['HX'][0,0]+=1
        def original(c):c['initial_factorization']['original_qca'][0,0]+=1
        def action(c):c['actual_circuit']['z_action'][0,0]+=1
        def automorphism(c):c['target_automorphism']['z_action'][0,0]+=1
        def target(c):c['code_binding']['target_HX'][0,0]+=1
        def scope(c):c['equivalence']['original_qca_implemented_by_circuit']=not c['equivalence']['original_qca_implemented_by_circuit']
        def generation(c):c['code_binding']['stabilizers']['Z']['source_in_target'][0,0]+=1
        def gate(c):c['actual_circuit']['layers'][0][0]['offset'][0]+=1
        for mutate in (ancilla,size,phase,raw,original,action,automorphism,target,scope,generation,gate):
            with self.subTest(mutate=mutate.__name__):
                bad=clone(good);mutate(bad)
                with self.assertRaises((ValueError,ArithmeticError)):
                    verify_no_ancilla_result(bad)


if __name__=='__main__':
    unittest.main()
