"""Exact normalization/replay and adversarial certificate checks, without CAS."""
import random
import json
import unittest

from css_decoupling.two_d.algebra import Matrix, Ring, eye, zeros
from css_decoupling.errors import ResourceLimit
from css_decoupling.two_d.laurent_to_polynomial import (
    laurent_column_to_polynomial, verify_laurent_column_to_polynomial,
)
from css_decoupling.two_d.serialization import encode, decode


def clone(value):
    if isinstance(value, Matrix):
        return value.copy()
    if isinstance(value, dict):
        return {key:clone(item) for key,item in value.items()}
    if isinstance(value, list):
        return [clone(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone(item) for item in value)
    return value


class LaurentPolynomializationTests(unittest.TestCase):
    def setUp(self):
        self.ring = Ring(2, 2)
        self.x, self.y = self.ring.variables

    def matrix(self, rows):
        return Matrix.from_rows(self.ring, rows)

    def assert_exact_replay(self, v, h, result):
        self.assertTrue(verify_laurent_column_to_polynomial(v, h, result))
        transform = eye(self.ring, v.nrows)
        inverse = eye(self.ring, v.nrows)
        for factor in result['factors']:
            E = eye(self.ring, v.nrows)
            E[factor['row'], factor['column']] = factor['coefficient'][0, 0]
            transform = E*transform
            inverse = inverse*E
        self.assertEqual(transform*inverse, eye(self.ring, v.nrows))
        self.assertEqual(inverse*transform, eye(self.ring, v.nrows))
        B = result['coordinate_map']
        def change(matrix):
            return Matrix(self.ring, matrix.nrows, matrix.ncols,
                          {ij: self.ring({(B[0][0]*a+B[0][1]*b,
                                          B[1][0]*a+B[1][1]*b): c
                                         for (a,b),c in f.terms.items()})
                           for ij,f in matrix.entries.items()})
        self.assertEqual(change(transform*v), result['polynomial_column'])
        self.assertEqual(change(h*inverse), result['tracked_laurent_bezout'])
        self.assertEqual(result['polynomial_bezout']*result['polynomial_column'], eye(self.ring,1))
        self.assertFalse(result['complete_suslin_factorization'])

    def test_zero_first_component_negative_powers_and_actual_swap(self):
        v = self.matrix([[0], [self.x**-3*self.y**2], [self.x+self.y**-2]])
        h = self.matrix([[0, self.x**3*self.y**-2, 0]])
        result = laurent_column_to_polynomial(v, h)
        self.assert_exact_replay(v,h,result)
        self.assertGreaterEqual(len(result['factors']),3)
        self.assertEqual([(f['row'],f['column']) for f in result['factors'][:3]],[(0,1),(1,0),(0,1)])

    def test_rank_two_and_nonnormalized_laurent_bezout(self):
        a = self.x**-1+self.y
        v = self.matrix([[a], [1]])
        t = self.x**-2*self.y**-1
        h = self.matrix([[t, 1+t*a]])
        result = laurent_column_to_polynomial(v,h)
        self.assert_exact_replay(v,h,result)
        self.assertGreater(result['clearing_power'],0)

    def test_polynomial_normalization_does_not_claim_completion(self):
        v=self.matrix([[1+self.x*self.y], [self.x**2]])
        h=self.matrix([[1+self.x*self.y, self.y**2]])
        result=laurent_column_to_polynomial(v,h)
        self.assert_exact_replay(v,h,result)
        self.assertNotEqual(result['polynomial_column'], self.matrix([[1],[0]]))

    def test_unit_column_has_polynomial_witness_even_at_clearing_zero(self):
        v=self.matrix([[1],[0]])
        h=self.matrix([[1,0]])
        result=laurent_column_to_polynomial(v,h)
        self.assertEqual(result['clearing_power'],0)
        self.assert_exact_replay(v,h,result)

    def test_serialized_certificate_has_independent_input_binding(self):
        v=self.matrix([[self.x**-1+self.y],[1]])
        h=self.matrix([[0,1]])
        certificate=laurent_column_to_polynomial(v,h)
        restored=decode(json.loads(json.dumps(encode(certificate))))
        self.assertTrue(verify_laurent_column_to_polynomial(v,h,restored))
        with self.assertRaises(ValueError):
            verify_laurent_column_to_polynomial(self.matrix([[self.x**-1+self.y+1],[1]]),h,restored)

    def test_generated_unimodular_columns_and_inverse_witnesses(self):
        rng=random.Random(99101)
        for n in (2,3,4):
            for repeat in range(4):
                T=eye(self.ring,n);Tinverse=eye(self.ring,n)
                for _ in range(4):
                    i,j=rng.sample(range(n),2)
                    c=self.ring.monomial((rng.randrange(-1,2),rng.randrange(-1,2)))
                    E=eye(self.ring,n);E[i,j]=c
                    T=E*T;Tinverse=Tinverse*E
                v,h=T[:,0],Tinverse[0,:]
                result=laurent_column_to_polynomial(v,h)
                self.assert_exact_replay(v,h,result)

    def test_bad_inputs_and_restricted_applicability(self):
        v=self.matrix([[1],[0]]);h=self.matrix([[1,0]])
        cases=[(v,zeros(self.ring,1,2)),(zeros(self.ring,2,1),h),
               (self.matrix([[1]]),self.matrix([[1]])),
               (zeros(self.ring,0,1),zeros(self.ring,1,0)),(v,h.T)]
        for column,row in cases:
            with self.assertRaises(ValueError):laurent_column_to_polynomial(column,row)
        for ring in (Ring(2,1),Ring(2,3)):
            with self.assertRaises(ValueError):
                laurent_column_to_polynomial(Matrix.from_rows(ring,[[1],[0]]),Matrix.from_rows(ring,[[1,0]]))

    def test_certificate_tampering(self):
        v=self.matrix([[self.x**-1+self.y],[1]])
        h=self.matrix([[0,1]])
        original=laurent_column_to_polynomial(v,h)
        mutations=[
            lambda c:c.__setitem__('schema','other'),
            lambda c:c.__setitem__('complete_suslin_factorization',True),
            lambda c:c.__setitem__('unrecognized',1),
            lambda c:c.__setitem__('coordinate_map',((1,0),(0,2))),
            lambda c:c.__setitem__('coordinate_inverse',((1,0),(0,1))),
            lambda c:c.__setitem__('coordinate_map',((True,0),(0,1))),
            lambda c:c.__setitem__('clearing_power',True),
            lambda c:c.__setitem__('clearing_power',c['clearing_power']+1),
            lambda c:c['polynomial_column'].__setitem__((0,0),c['polynomial_column'][0,0]+1),
            lambda c:c['polynomial_bezout'].__setitem__((0,0),c['polynomial_bezout'][0,0]+self.x**-1),
            lambda c:c['tracked_laurent_bezout'].__setitem__((0,0),c['tracked_laurent_bezout'][0,0]+1),
            lambda c:c['cleared_row'].__setitem__((0,0),c['cleared_row'][0,0]+1),
            lambda c:c['factors'][0].__setitem__('row',True),
            lambda c:c['factors'][0].__setitem__('column',c['factors'][0]['row']),
            lambda c:c['factors'][0].__setitem__('coefficient',self.matrix([[0]])),
            lambda c:c['factors'][0].__setitem__('coefficient',self.matrix([[self.x+1]])),
        ]
        for mutate in mutations:
            changed=clone(original);mutate(changed)
            with self.assertRaises(ValueError):verify_laurent_column_to_polynomial(v,h,changed)
        # A determinant-one change and its correct inverse are still rejected
        # when they disagree with the recorded actual column transformation.
        changed=clone(original)
        (a,b),(c,d)=changed['coordinate_map']
        changed['coordinate_map']=((a,b),(c+a,d+b))
        changed['coordinate_inverse']=((d+b,-b),(-c-a,a))
        with self.assertRaises(ValueError):verify_laurent_column_to_polynomial(v,h,changed)

    def test_resource_stops_and_strict_caps(self):
        v=self.matrix([[self.x**-3*self.y**2],[1]])
        h=self.matrix([[0,1]])
        for key in ('max_steps','max_terms','max_products','max_factors','max_exponent'):
            with self.assertRaises(ResourceLimit):laurent_column_to_polynomial(v,h,**{key:1})
            for bad in (False,0,-1,1.5):
                with self.assertRaises(ValueError):laurent_column_to_polynomial(v,h,**{key:bad})
        for bad in (False,0,-1,float('inf'),float('nan')):
            with self.assertRaises(ValueError):laurent_column_to_polynomial(v,h,timeout=bad)
        with self.assertRaises(ResourceLimit):laurent_column_to_polynomial(v,h,timeout=1e-12)
        result=laurent_column_to_polynomial(v,h)
        with self.assertRaises(ResourceLimit):verify_laurent_column_to_polynomial(v,h,result,max_factors=1)


if __name__ == '__main__':
    unittest.main()
