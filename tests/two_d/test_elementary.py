"""Independent exact products for the elementary fast path and its limits."""
import random
import unittest
from unittest.mock import patch

from css_decoupling.two_d.algebra import Ring, Matrix, eye, direct_sum
from css_decoupling.two_d.elementary import factor_unit_reduction, FactorizationIncomplete
from css_decoupling.errors import ResourceLimit


class ElementaryReductionTests(unittest.TestCase):
    def setUp(self):
        self.R = Ring(2,2)
        self.x,self.y = self.R.variables

    def check(self,U,V,mode):
        result = factor_unit_reduction(U,V,transport=mode)
        n = U.nrows+result['added_ancillas']
        P = eye(self.R,n)
        for item in result['factors']:
            E = eye(self.R,n)
            E[item['row'],item['column']] = item['coefficient'][0,0]
            P = E*P
        Pinv = eye(self.R,n)
        for item in reversed(result['factors']):
            E = eye(self.R,n)
            E[item['row'],item['column']] = item['coefficient'][0,0]
            Pinv = E*Pinv
        S = Matrix(self.R,n,n,{(i,i):self.R.monomial(e)
                              for i,e in enumerate(result['shift_exponents'])})
        Sinv = Matrix(self.R,n,n,{(i,i):self.R.monomial(e)**-1
                                 for i,e in enumerate(result['shift_exponents'])})
        self.assertEqual(S*P,direct_sum(U,result['ancilla_action']))
        self.assertEqual(Pinv*Sinv,direct_sum(V,result['ancilla_inverse']))
        self.assertEqual(P*Pinv,eye(self.R,n))
        self.assertEqual(Pinv*P,eye(self.R,n))
        self.assertFalse(result['complete_general_suslin_algorithm'])
        return result

    def test_zero_one_two_rank_and_negative_transport(self):
        x,y,R=self.x,self.y,self.R
        for U,V in [(eye(R,0),eye(R,0)),
                    (Matrix.from_rows(R,[[x*y**-2]]),Matrix.from_rows(R,[[x**-1*y**2]])),
                    (Matrix.from_rows(R,[[x,1+y],[0,x**-1]]),
                     Matrix.from_rows(R,[[x**-1,1+y],[0,x]]))]:
            for mode in ('separate','balance'):
                with self.subTest(rank=U.nrows,mode=mode):self.check(U,V,mode)

    def test_random_elementary_words_and_determinant_transport(self):
        rng=random.Random(78123);R=self.R
        for n in (2,3,5):
            for trial in range(5):
                U,V=eye(R,n),eye(R,n)
                for _ in range(8):
                    i,j=rng.sample(range(n),2)
                    c=R.monomial((rng.randrange(-2,3),rng.randrange(-2,3)))
                    E=eye(R,n);E[i,j]=c
                    U,V=E*U,V*E
                D,Di=eye(R,n),eye(R,n)
                D[0,0]=self.x**-2*self.y;Di[0,0]=self.x**2*self.y**-1
                U,V=D*U,V*Di
                for mode in ('separate','balance'):
                    with self.subTest(n=n,trial=trial,mode=mode):self.check(U,V,mode)

    def test_classical_cohn_has_laurent_unit_pivot(self):
        x,y,R=self.x,self.y,self.R
        C=Matrix.from_rows(R,[[1+x*y,x*x],[y*y,1+x*y]])
        self.check(C,C,'separate')

    def test_nonunit_cohn_stop_is_algorithmic_not_nonexistence(self):
        a,b=1+self.x,1+self.y;R=self.R
        C=Matrix.from_rows(R,[[1+a*b,a*a],[b*b,1+a*b]])
        with self.assertRaises(FactorizationIncomplete) as caught:
            factor_unit_reduction(C,C,cancellation=False)
        self.assertEqual(caught.exception.residual,C)
        self.assertEqual(caught.exception.pivot_count,0)

    def test_higman_residual_embeds_after_tracked_prefix(self):
        from css_decoupling.two_d.direct_circuits import compile_direct_circuit,verify_direct_circuit
        R=self.R;a,b=1+self.x,1+self.y
        C=Matrix.from_rows(R,[[1+a*b,a*a],[b*b,1+a*b]])
        U=direct_sum(Matrix.from_rows(R,[[self.x*self.y**-1]]),C)
        V=direct_sum(Matrix.from_rows(R,[[self.x**-1*self.y]]),C)
        left,right=eye(R,3),eye(R,3)
        left[1,0]=1+self.x;right[0,2]=self.y**-1
        U,V=left*U*right,right*V*left
        for mode in ('separate','balance'):
            data=factor_unit_reduction(U,V,transport=mode,residual='higman',cancellation=False)
            self.assertTrue(data['reduction_statistics']['higman_used'])
            self.assertEqual(data['reduction_statistics']['unit_pivots'],1)
            bundle=compile_direct_circuit(U,V,data['factors'],added_ancillas=data['added_ancillas'],
                ancilla_action=data['ancilla_action'],ancilla_inverse=data['ancilla_inverse'],
                shift_exponents=[list(e) for e in data['shift_exponents']])
            self.assertTrue(verify_direct_circuit(bundle))

    def test_inverse_higman_word_and_transport_are_pulled_back_exactly(self):
        from css_decoupling.two_d.suslin import factor_suslin
        from css_decoupling.two_d.elementary import _invert_factor_data
        from css_decoupling.two_d.direct_circuits import compile_direct_circuit,verify_direct_circuit
        from css_decoupling.two_d.serialization import canonical_bytes
        R=self.R;a,b=1+self.x,1+self.y
        C=Matrix.from_rows(R,[[1+a*b,a*a],[b*b,1+a*b]])
        D,Di=eye(R,2),eye(R,2)
        D[0,0]=self.x**-2*self.y;Di[0,0]=self.x**2*self.y**-1
        U,V=D*C,C*Di
        for mode in ('separate','balance'):
            original=factor_suslin(V,U,transport=mode)
            before=canonical_bytes(original)
            data=_invert_factor_data(original,V)
            self.assertEqual(canonical_bytes(original),before)
            self.assertEqual(data['input_inverse'],V)
            Cc=compile_direct_circuit(U,V,data['factors'],added_ancillas=data['added_ancillas'],
                ancilla_action=data['ancilla_action'],ancilla_inverse=data['ancilla_inverse'],
                shift_exponents=[list(e) for e in data['shift_exponents']])
            self.assertTrue(verify_direct_circuit(Cc))

    def test_auto_inverse_unit_direction_has_exact_terminal_product(self):
        from css_decoupling.two_d.direct_circuits import compile_direct_circuit,verify_direct_circuit
        R=self.R;a,b=1+self.x,1+self.y
        small=Matrix.from_rows(R,[[1+a*b,a*a],[b*b,1+a*b]])
        C=direct_sum(*[small for _ in range(9)])
        data=factor_unit_reduction(C,C,cancellation=False,residual='higman',timeout=30)
        self.assertTrue(data['reduction_statistics']['inverse_unit_direction'])
        bundle=compile_direct_circuit(C,C,data['factors'],added_ancillas=data['added_ancillas'],
            ancilla_action=data['ancilla_action'],ancilla_inverse=data['ancilla_inverse'],
            shift_exponents=[list(e) for e in data['shift_exponents']])
        self.assertTrue(verify_direct_circuit(bundle))

    def test_nested_higman_shares_remaining_operation_budget(self):
        from css_decoupling.two_d.suslin import factor_suslin
        from css_decoupling.two_d.elementary import _Budget
        a,b=1+self.x,1+self.y;R=self.R
        C=Matrix.from_rows(R,[[1+a*b,a*a],[b*b,1+a*b]])
        with patch('css_decoupling.two_d.suslin.factor_suslin',wraps=factor_suslin) as called:
            data=factor_unit_reduction(C,C,cancellation=False,residual='higman',
                max_steps=100000,suslin_options={'max_steps':1000000000})
        self.assertEqual(called.call_count,1)
        self.assertLess(called.call_args.kwargs['max_steps'],100000)
        self.assertLessEqual(data['reduction_statistics']['operations'],100000)
        with self.assertRaises(ResourceLimit):
            factor_unit_reduction(C,C,cancellation=False,residual='higman',max_steps=10)
        exhausted=_Budget(10,1,100,100)
        with self.assertRaises(ResourceLimit):exhausted.subcall()

    def test_tampered_inverse_domain_and_budgets_rejected(self):
        R=self.R;U=eye(R,2);U[0,1]=self.x+self.y
        with self.assertRaises(ValueError):factor_unit_reduction(U,eye(R,2))
        for name,value in [('max_terms',True),('max_factors',0),('timeout',float('nan')),
                           ('transport','unknown'),('cancellation',1)]:
            with self.subTest(name=name),self.assertRaises(ValueError):
                factor_unit_reduction(U,U,**{name:value})
        with self.assertRaises(ResourceLimit):factor_unit_reduction(U,U,max_steps=1)
        for Rother in (Ring(2,1),Ring(3,3)):
            with self.assertRaises(ValueError):factor_unit_reduction(eye(Rother,2),eye(Rother,2))


if __name__=='__main__':unittest.main()
