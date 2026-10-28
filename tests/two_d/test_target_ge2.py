"""Exact target-subgroup decomposition over prime fields, independently replayed."""
import pytest

from css_decoupling.two_d.algebra import Ring, Matrix, eye
from css_decoupling.two_d.rank_two_budget import RankTwoBudget, RankTwoResourceLimit
from css_decoupling.two_d.target_ge2 import factor_target_automorphism, separate_atoms


def budget(**options):
    return RankTwoBudget(**{**dict(timeout=20, max_steps=1000000, max_terms=1000000,
                                  max_products=30000000, max_factors=200000), **options})


def sample(p):
    R = Ring(p, 2)
    x, y = R.variables
    h, d = Matrix.from_rows(R, [[x-1, y-1]]), Matrix.from_rows(R, [[1-y], [x-1]])
    f = 1+x**-1*y+(p-1)*x*y**-2
    P, Pi = eye(R, 2)+d*(h*f), eye(R, 2)-d*(h*f)
    Tx, Txi = Matrix.from_rows(R, [[1, 1-y], [0, x]]), Matrix.from_rows(R, [[1, (y-1)*x**-1], [0, x**-1]])
    Ty, Tyi = Matrix.from_rows(R, [[y, 0], [1-x, 1]]), Matrix.from_rows(R, [[y**-1, 0], [(x-1)*y**-1, 1]])
    lam = (p-1)*x**-1*y
    B, Bi = (Tx*Tx*Tyi)*lam, (Ty*Txi*Txi)*(lam**-1)
    return B*P, Pi*Bi


@pytest.mark.parametrize('p', [2, 3, 5, 7])
def test_nontrivial_endpoint_units_and_negative_exponents_have_original_rank_words(p):
    K, Ki = sample(p)
    cap = budget()
    atoms = factor_target_automorphism(K, Ki, cap)
    A = eye(K.ring, 2)
    for kind, i, j, c in atoms:
        E = eye(K.ring, 2)
        if kind == 'e':
            assert i != j and i in (0, 1) and j in (0, 1)
            E[i, j] = c
        else:
            assert c.is_unit and i in (0, 1)
            E[i, i] = c
        A = A*E
    assert A == K
    data = separate_atoms(K, Ki, atoms, cap, algorithm='test-exact-target-subgroup')
    F, Fi = eye(K.ring, 2), eye(K.ring, 2)
    for factor in data['factors']:
        i, j, c = factor['row'], factor['column'], factor['coefficient'][0, 0]
        E, Ei = eye(K.ring, 2), eye(K.ring, 2)
        E[i, j], Ei[i, j] = c, -c
        F, Fi = E*F, Fi*Ei
    units = [K.ring.monomial(tuple(e), c) for e, c in zip(data['shift_exponents'], data['transport_coefficients'])]
    S = Matrix(K.ring, 2, 2, {(i, i): u for i, u in enumerate(units)})
    Si = Matrix(K.ring, 2, 2, {(i, i): u**-1 for i, u in enumerate(units)})
    assert S*F == K and Fi*Si == Ki and F*Fi == Fi*F == eye(K.ring, 2)


def test_not_a_target_automorphism_and_tampered_inverse_are_rejected():
    K, Ki = sample(3)
    bad = Ki.copy()
    bad[0, 0] += 1
    with pytest.raises(ValueError, match='inverse'):
        factor_target_automorphism(K, bad, budget())
    R = K.ring
    x = R.variable(0)
    A, Ai = eye(R, 2), eye(R, 2)
    A[1, 1], Ai[1, 1] = x, x**-1
    with pytest.raises(ValueError, match='preserve'):
        factor_target_automorphism(A, Ai, budget())


def test_factor_and_product_limits_apply_to_endpoint_and_transvection_work():
    K, Ki = sample(5)
    for limits, reason in (({'max_factors': 1}, 'factor_limit'), ({'max_products': 1}, 'product_limit')):
        with pytest.raises(RankTwoResourceLimit) as error:
            factor_target_automorphism(K, Ki, budget(**limits))
        assert error.value.reason == reason
