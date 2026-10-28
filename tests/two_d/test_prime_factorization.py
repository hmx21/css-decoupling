"""Prime-field controls replay physical matrix words without constructor helpers."""
import random

import pytest

from css_decoupling.two_d.algebra import Matrix, Ring, direct_sum, eye
from css_decoupling.two_d.elementary import factor_unit_reduction, _invert_factor_data
from css_decoupling.two_d.suslin import factor_suslin, _pid_divide, _Budget
from css_decoupling.two_d.laurent_to_polynomial import (
    laurent_column_to_polynomial, verify_laurent_column_to_polynomial,
)
from css_decoupling.errors import ResourceLimit


def elementary(ring, n, i, j, c):
    E, Ei = eye(ring, n), eye(ring, n)
    E[i, j], Ei[i, j] = c, -ring(c)
    return E, Ei


def cohn(ring):
    x, y = ring.variables
    a, b = x+2, y+1
    C = Matrix.from_rows(ring, [[1+a*b, a*a], [-b*b, 1-a*b]])
    Ci = Matrix.from_rows(ring, [[1-a*b, -a*a], [b*b, 1+a*b]])
    assert C*Ci == Ci*C == eye(ring, 2)
    return C, Ci


def assert_factorization(U, V, result, mode):
    ring = U.ring
    q = U.nrows+result['added_ancillas']
    P, Pi = eye(ring, q), eye(ring, q)
    for factor in result['factors']:
        E, Ei = elementary(ring, q, factor['row'], factor['column'], factor['coefficient'][0, 0])
        P, Pi = E*P, Pi*Ei
    coefficients = result.get('transport_coefficients', [1]*q)
    assert len(coefficients) == q
    assert all(type(c) is int and 0 < c < ring.p for c in coefficients)
    units = [ring.monomial(e, c) for e, c in zip(result['shift_exponents'], coefficients)]
    S = Matrix(ring, q, q, {(i, i): u for i, u in enumerate(units)})
    Si = Matrix(ring, q, q, {(i, i): u**-1 for i, u in enumerate(units)})
    assert S*P == direct_sum(U, result['ancilla_action'])
    assert Pi*Si == direct_sum(V, result['ancilla_inverse'])
    assert P*Pi == Pi*P == eye(ring, q)
    if mode == 'balance':
        assert S == eye(ring, q)
    else:
        assert result['ancilla_action'] == eye(ring, result['added_ancillas'])
    return P, Pi, S


@pytest.mark.parametrize('p', [2, 3, 5, 7])
def test_pid_division_nonmonic_coefficients_and_negative_valuations(p):
    R = Ring(p, 2)
    x, y = R.variables
    budget = _Budget(10, 100000, 100000, 100000, 32, 1000000)
    for axis, t in enumerate((x, y)):
        b = (p-1)*t**-2 + t + (p-1)*t**3
        expected_q, expected_r = 2*t**-3 + (p-1)*t**2, R(1)
        a = b*expected_q+expected_r
        q, r = _pid_divide(a, b, axis, budget)
        assert q*b+r == a
        if r:
            assert max(e[axis] for e in r.terms)-min(e[axis] for e in r.terms) < 5


@pytest.mark.parametrize('p', [3, 5])
@pytest.mark.parametrize('mode', ['separate', 'balance'])
@pytest.mark.parametrize('axis', [0, 1])
def test_forced_higman_cohn_negative_exponents_and_nontrivial_scalar(p, mode, axis):
    R = Ring(p, 2)
    x, y = R.variables
    C, Ci = cohn(R)
    D, Di = eye(R, 2), eye(R, 2)
    D[0, 0] = 2*x**-1*y
    Di[0, 0] = D[0, 0]**-1
    U, V = D*C, Ci*Di
    result = factor_suslin(U, V, axis=axis, transport=mode)
    assert_factorization(U, V, result, mode)
    assert result['algorithm'] == 'higman-pid-fp-laurent-v1'
    assert result['statistics']['identity_ancillas'] > 0
    assert result['statistics']['projector_power'] >= result['statistics']['linearized_rank']
    power = result['statistics']['projector_power']
    while power > 1:
        assert power % p == 0
        power //= p
    assert result['determinant'][0, 0] == D[0, 0]
    if mode == 'separate':
        assert result['transport_coefficients'][-1] == 2
    else:
        assert result['statistics']['balance_ancillas'] == 1


@pytest.mark.parametrize('p', [3, 5])
def test_mixed_projector_and_nonconstant_nilpotent_kernel(p):
    R = Ring(p, 2)
    x, y = R.variables
    A, Ai = elementary(R, 2, 1, 0, 2*x)
    B, Bi = elementary(R, 2, 0, 1, x+2)
    K, Ki = A*B, Bi*Ai
    D, Di = eye(R, 2), eye(R, 2)
    D[0, 0], Di[0, 0] = y, y**-1
    U, V = K*D*Ki, K*Di*Ki
    result = factor_suslin(U, V, axis=1, transport='balance')
    assert_factorization(U, V, result, 'balance')
    assert result['statistics']['projector_rank'] == 1
    a, b = x+2, x*x+2*x+1
    N = Matrix.from_rows(R, [[a*b, a*a], [-b*b, -a*b]])
    assert not N*N
    U, V = eye(R, 2)+N*y, eye(R, 2)-N*y
    result = factor_suslin(U, V, axis=1)
    assert_factorization(U, V, result, 'separate')
    assert result['statistics']['kernel_splits']
    N = Matrix(R, 3, 3, {(0, 1): 2*a, (1, 2): b})
    U = eye(R, 3)+N*y
    V = eye(R, 3)-N*y+(N*N)*y**2
    result = factor_suslin(U, V, axis=1)
    assert_factorization(U, V, result, 'separate')
    assert len(result['statistics']['kernel_splits']) >= 2


@pytest.mark.parametrize('p', [3, 5, 7])
@pytest.mark.parametrize('engine', [factor_suslin, factor_unit_reduction])
def test_actual_swaps_scalar_units_and_empty_registers(p, engine):
    R = Ring(p, 2)
    x, y = R.variables
    permutation = Matrix.from_rows(R, [[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    swap = Matrix.from_rows(R, [[0, 1], [1, 0]])
    scalar = Matrix.from_rows(R, [[2*x**-2*y]])
    constant = Matrix.from_rows(R, [[2]])
    cases = [(eye(R, 0), eye(R, 0)), (scalar, Matrix.from_rows(R, [[scalar[0, 0]**-1]])),
             (constant, Matrix.from_rows(R, [[R(2)**-1]])),
             (swap, swap), (permutation, permutation.T)]
    for U, V in cases:
        for mode in ('balance', 'separate'):
            result = engine(U, V, transport=mode)
            assert_factorization(U, V, result, mode)
            if U == swap:
                assert result['determinant'][0, 0] == R(-1)


@pytest.mark.parametrize('p', [3, 5])
@pytest.mark.parametrize('mode', ['balance', 'separate'])
def test_higman_residual_pulls_back_signed_prefix_and_inverse_direction(p, mode):
    R = Ring(p, 2)
    x, y = R.variables
    C, Ci = cohn(R)
    unit = 2*x*y**-1
    U = direct_sum(Matrix.from_rows(R, [[unit]]), C)
    V = direct_sum(Matrix.from_rows(R, [[unit**-1]]), Ci)
    L, Li = elementary(R, 3, 1, 0, 2+x)
    Q, Qi = elementary(R, 3, 0, 2, 2*y**-1)
    U, V = L*U*Q, Qi*V*Li
    result = factor_unit_reduction(U, V, residual='higman', cancellation=False, transport=mode)
    assert_factorization(U, V, result, mode)
    assert result['reduction_statistics']['higman_used']
    assert result['reduction_statistics']['unit_pivots'] == 1
    inverse_result = factor_suslin(V, U, transport=mode)
    pulled_back = _invert_factor_data(inverse_result, V)
    assert_factorization(U, V, pulled_back, mode)


@pytest.mark.parametrize('p', [3, 5])
def test_seeded_scalar_weighted_cancellation_and_forced_words(p):
    R = Ring(p, 2)
    rng = random.Random(78123+p)
    for n in (2, 3):
        for trial in range(3):
            U, V = eye(R, n), eye(R, n)
            for _ in range(5):
                i, j = rng.sample(range(n), 2)
                c = R.monomial((rng.randrange(-1, 2), rng.randrange(-1, 2)), rng.randrange(1, p))
                E, Ei = elementary(R, n, i, j, c)
                U, V = E*U, V*Ei
            for engine in (factor_unit_reduction, factor_suslin):
                result = engine(U, V, transport='balance')
                assert_factorization(U, V, result, 'balance')


@pytest.mark.parametrize('p', [3, 5])
def test_park_signed_witness_constant_normalization_and_geometric_correction(p):
    R = Ring(p, 2)
    x, y = R.variables
    a, t = x**-1+2*y, 2*x**-2*y**-1
    cases = [
        (Matrix.from_rows(R, [[a], [1]]), Matrix.from_rows(R, [[t, 1-t*a]])),
        (Matrix.from_rows(R, [[0], [2*x**-3*y**2], [x+2*y**-2]]),
         Matrix.from_rows(R, [[0, R(2)**-1*x**3*y**-2, 0]])),
    ]
    for v, h in cases:
        result = laurent_column_to_polynomial(v, h)
        assert verify_laurent_column_to_polynomial(v, h, result)
        T, Ti = eye(R, v.nrows), eye(R, v.nrows)
        for factor in result['factors']:
            E, Ei = elementary(R, v.nrows, factor['row'], factor['column'], factor['coefficient'][0, 0])
            T, Ti = E*T, Ti*Ei
        B = result['coordinate_map']
        def change(M):
            return Matrix(R, M.nrows, M.ncols, {ij: R({
                (B[0][0]*a+B[0][1]*b, B[1][0]*a+B[1][1]*b): c
                for (a, b), c in f.terms.items()}) for ij, f in M.entries.items()})
        assert T*Ti == Ti*T == eye(R, v.nrows)
        assert change(T*v) == result['polynomial_column']
        assert change(h*Ti) == result['tracked_laurent_bezout']
        assert result['polynomial_bezout']*result['polynomial_column'] == eye(R, 1)
        changed = dict(result)
        changed['polynomial_bezout'] = result['polynomial_bezout'].copy()
        changed['polynomial_bezout'][0, 0] += 1
        with pytest.raises(ValueError):
            verify_laurent_column_to_polynomial(v, h, changed)


@pytest.mark.parametrize('p', [3, 5])
def test_prime_resource_stops_remain_incomplete(p):
    R = Ring(p, 2)
    C, Ci = cohn(R)
    for options in ({'max_steps': 1}, {'max_terms': 1}, {'max_products': 1},
                    {'max_factors': 1}, {'max_linearized_size': 2}):
        with pytest.raises(ResourceLimit):
            factor_suslin(C, Ci, **options)
    with pytest.raises(ResourceLimit):
        factor_unit_reduction(C, Ci, residual='higman', cancellation=False, max_steps=1)


def test_nested_inverse_route_budget_includes_child_terminal_replay():
    R = Ring(3, 2)
    C, Ci = cohn(R)
    U, V = direct_sum(*([C]*9)), direct_sum(*([Ci]*9))
    options = dict(residual='higman', cancellation=False, timeout=30)
    first = factor_unit_reduction(U, V, max_steps=1000000, **options)
    assert first['reduction_statistics']['inverse_unit_direction']
    assert_factorization(U, V, first, 'separate')
    used = first['reduction_statistics']['operations']
    # A complete repeated computation fits exactly its declared shared budget.
    # Omitting a child's terminal replay from the parent's absorption makes
    # the declared number too small and this exact-boundary run fail.
    exact = factor_unit_reduction(U, V, max_steps=used, **options)
    assert exact['reduction_statistics']['operations'] == used
    assert_factorization(U, V, exact, 'separate')
    with pytest.raises(ResourceLimit):
        factor_unit_reduction(U, V, max_steps=used-1, **options)
