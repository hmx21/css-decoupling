"""Independent full-matrix replay of constructive Higman/PID factor words."""
import random
import unittest

from css_decoupling.two_d.algebra import Matrix, Ring, direct_sum, eye
from css_decoupling.two_d.suslin import factor_suslin
from css_decoupling.errors import ResourceLimit


def replay(U, result):
    """Use generic matrix products, independent of engine row-update helpers."""
    ring = U.ring
    q = U.nrows + result['added_ancillas']
    product = eye(ring, q)
    for item in result['factors']:
        i, j = item['row'], item['column']
        if i == j or not 0 <= i < q or not 0 <= j < q:
            raise ValueError('Invalid elementary indices')
        E = eye(ring, q)
        E[i, j] = item['coefficient'][0, 0]
        product = E * product
    inverse = eye(ring, q)
    for item in reversed(result['factors']):
        E = eye(ring, q)
        E[item['row'], item['column']] = -item['coefficient'][0, 0]
        inverse = E * inverse
    S = Matrix(ring, q, q, {(i, i): ring.monomial(e)
                           for i, e in enumerate(result['shift_exponents'])})
    Si = Matrix(ring, q, q, {(i, i): ring.monomial(e)**-1
                            for i, e in enumerate(result['shift_exponents'])})
    return product, inverse, S, Si


class SuslinTests(unittest.TestCase):
    def setUp(self):
        self.R = Ring(2, 2)
        self.x, self.y = self.R.variables

    def assert_contract(self, U, V, result, mode):
        P, Pi, S, Si = replay(U, result)
        q = U.nrows + result['added_ancillas']
        self.assertEqual(S * P, direct_sum(U, result['ancilla_action']))
        self.assertEqual(Pi * Si, direct_sum(V, result['ancilla_inverse']))
        self.assertEqual(P * Pi, eye(self.R, q))
        self.assertEqual(Pi * P, eye(self.R, q))
        self.assertEqual(result['input_inverse'], V)
        if mode == 'separate':
            self.assertEqual(result['ancilla_action'], eye(self.R, result['added_ancillas']))
        else:
            self.assertEqual(S, eye(self.R, q))
        self.assertTrue(result['complete'])
        self.assertTrue(result['complete_general_suslin_algorithm'])
        self.assertEqual(result['algorithm'], 'higman-pid-f2-laurent-v1')
        self.assertFalse(result['statistics']['minimal_stabilization_claimed'])

    def check(self, U, V, mode='separate', **options):
        result = factor_suslin(U, V, transport=mode, **options)
        self.assert_contract(U, V, result, mode)
        return result

    def cohn(self):
        a, b = 1+self.x, 1+self.y
        return Matrix.from_rows(self.R, [[1+a*b, a*a], [b*b, 1+a*b]])

    def test_empty_and_rank_one_units(self):
        R, x, y = self.R, self.x, self.y
        for U, V in [(eye(R, 0), eye(R, 0)), (eye(R, 1), eye(R, 1)),
                     (Matrix.from_rows(R, [[x**-3*y**2]]),
                      Matrix.from_rows(R, [[x**3*y**-2]]))]:
            for mode in ('separate', 'balance'):
                with self.subTest(rank=U.nrows, mode=mode):
                    self.check(U, V, mode)

    def test_univariate_pid_units_and_no_given_inverse(self):
        R, x = self.R, self.x
        U = Matrix.from_rows(R, [[x, 1+x**-2], [0, x**-1]])
        V = Matrix.from_rows(R, [[x**-1, 1+x**-2], [0, x]])
        result = factor_suslin(U)
        self.assert_contract(U, V, result, 'separate')
        self.assertEqual(result['statistics']['polynomial_degree'], 0)
        self.assertEqual(result['added_ancillas'], 0)

    def test_nonunit_cohn_identity_stabilization(self):
        C = self.cohn()
        for mode in ('separate', 'balance'):
            result = self.check(C, C, mode)
            self.assertEqual(result['statistics']['linearized_rank'], 4)
            self.assertEqual(result['statistics']['identity_ancillas'], 2)
            self.assertEqual(result['statistics']['specialization_pid_rank'], 2)
            self.assertTrue(result['statistics']['specialization_reuses_higman_trace'])
            self.assertTrue(result['statistics']['kernel_splits'])
            self.assertEqual(result['ancilla_action'], eye(self.R, 2))

    def test_companion_specialization_reuses_original_rank(self):
        R, x, y = self.R, self.x, self.y
        U = eye(R, 2)
        U[0, 1] = (x+1)*y**3 + x**-2*y
        result = self.check(U, U, axis=1)
        self.assertEqual(result['statistics']['linearized_rank'], 6)
        self.assertEqual(result['statistics']['specialization_pid_rank'], 2)
        self.assertEqual(result['statistics']['identity_ancillas'], 4)

    def test_mixed_projector_with_nonconstant_pid_basis(self):
        R, x, y = self.R, self.x, self.y
        A, B = eye(R, 2), eye(R, 2)
        A[1, 0], B[0, 1] = x, x+1
        K, Ki = A*B, B*A
        D, Di = eye(R, 2), eye(R, 2)
        D[0, 0], Di[0, 0] = y, y**-1
        U, V = K*D*Ki, K*Di*Ki
        for mode in ('separate', 'balance'):
            result = self.check(U, V, mode, axis=1)
            self.assertEqual(result['statistics']['projector_rank'], 1)
            self.assertEqual(result['statistics']['identity_ancillas'], 0)
            self.assertEqual(result['statistics']['balance_ancillas'], int(mode == 'balance'))

    def test_nilpotent_nonunit_kernel_and_longer_flag(self):
        R, x, y = self.R, self.x, self.y
        a, b = x+1, x*x+x+1
        N = Matrix.from_rows(R, [[a*b, a*a], [b*b, a*b]])
        self.assertEqual(N*N, Matrix(R, 2, 2, {}))
        U = eye(R, 2)+N*y
        result = self.check(U, U, axis=1)
        self.assertTrue(result['statistics']['kernel_splits'])
        N = Matrix(R, 3, 3, {(0, 1): a, (1, 2): b})
        U, V = eye(R, 3)+N*y, eye(R, 3)+N*y+(N*N)*(y*y)
        result = self.check(U, V, axis=1)
        self.assertGreaterEqual(len(result['statistics']['kernel_splits']), 2)

    def test_negative_powers_and_both_axis_choices(self):
        R, x, y, C = self.R, self.x, self.y, self.cohn()
        D, Di = eye(R, 2), eye(R, 2)
        D[0, 0], Di[0, 0] = x*y**-1, x**-1*y
        U = D*C*Di
        for axis in (0, 1):
            result = self.check(U, U, axis=axis)
            self.assertGreater(result['statistics']['polynomial_shift_power'], 0)

    def test_seeded_mixed_laurent_words(self):
        R, rng = self.R, random.Random(920317)
        for rank in (2, 3):
            for trial in range(3):
                U, V = eye(R, rank), eye(R, rank)
                for _ in range(3):
                    i, j = rng.sample(range(rank), 2)
                    c = R.one()+R.monomial((rng.randrange(-1, 2), rng.randrange(-1, 2)))
                    E = eye(R, rank)
                    E[i, j] = c
                    U, V = E*U, V*E
                D, Di = eye(R, rank), eye(R, rank)
                D[0, 0], Di[0, 0] = self.x*self.y**-1, self.x**-1*self.y
                with self.subTest(rank=rank, trial=trial):
                    self.check(D*U, V*Di, 'balance')

    def test_input_validation_and_resource_stops(self):
        R, C = self.R, self.cohn()
        with self.assertRaises(ValueError):
            factor_suslin(C, eye(R, 2))
        for U in (Matrix(R, 1, 2, {}), eye(Ring(2, 1), 2),
                  Matrix(R, 2, 2, {}), Matrix.from_rows(R, [[1+self.x+self.y]])):
            with self.subTest(shape=U.shape, ring=U.ring), self.assertRaises(ValueError):
                factor_suslin(U)
        for key in ('max_steps', 'max_terms', 'max_factors', 'max_linearized_size', 'max_products'):
            for bad in (True, 0, -1, 1.0, float('inf')):
                with self.subTest(key=key, bad=bad), self.assertRaises(ValueError):
                    factor_suslin(C, C, **{key: bad})
        for kwargs in ({'timeout': float('nan')}, {'timeout': float('inf')},
                       {'timeout': True}, {'timeout': 0}, {'axis': True},
                       {'axis': 2}, {'transport': 'swap-shift'}, {'progress': 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                factor_suslin(C, C, **kwargs)
        for kwargs in ({'max_steps': 1}, {'max_terms': 1}, {'max_factors': 1},
                       {'max_products': 1}, {'max_linearized_size': 2}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ResourceLimit) as caught:
                factor_suslin(C, C, **kwargs)
            self.assertIn('phase=', str(caught.exception))
            self.assertIn('linearized_rank=', str(caught.exception))
        with self.assertRaises(ResourceLimit) as caught:
            factor_suslin(C, C, max_linearized_size=2)
        self.assertIn('N=4', str(caught.exception))
        self.assertIn('max_linearized_size=2', str(caught.exception))

    def test_progress_and_output_tamper_detection(self):
        C, events = self.cohn(), []
        result = self.check(C, C, progress=lambda stage, details: events.append((stage, details)))
        self.assertEqual(events[0][0], 'validation')
        self.assertEqual(events[-1][0], 'complete')
        self.assertIn('projector_basis', [stage for stage, _ in events])
        self.assertTrue(all(d['input_rank'] == 2 for _, d in events))
        self.assertEqual(events[-1][1]['linearized_rank'], 4)
        for mutation in ('factor', 'shift', 'ancilla'):
            changed = {**result,
                       'factors': [{**item, 'coefficient': item['coefficient'].copy()}
                                   for item in result['factors']],
                       'shift_exponents': list(result['shift_exponents']),
                       'ancilla_inverse': result['ancilla_inverse'].copy()}
            if mutation == 'factor':
                changed['factors'][0]['coefficient'][0, 0] += 1
            elif mutation == 'shift':
                changed['shift_exponents'][0] = (1, 0)
            else:
                changed['ancilla_inverse'][0, 0] = self.x
            with self.subTest(mutation=mutation), self.assertRaises(AssertionError):
                self.assert_contract(C, C, changed, 'separate')


if __name__ == '__main__':
    unittest.main()
