"""Independent original-rank replay of elementary column constructions."""
from itertools import permutations
from random import Random

import pytest

from css_decoupling.two_d.algebra import Matrix, Ring, eye
from css_decoupling.two_d.elementary_columns import (
    elementary_column_reduce, _Budget, _characteristic, _conjugate_pair,
    _resultant_relation, _support_monic_shear,
)
from css_decoupling.algebra.closed_points import apply_shear
from css_decoupling.errors import ResourceLimit


def budget(**overrides):
    options = dict(timeout=20, max_steps=1000000, max_terms=200000,
                   max_products=10000000, max_factors=100000,
                   max_exponent=1000000, max_patches=128,
                   max_resultant_size=64, max_field_trials=128)
    options.update(overrides)
    return _Budget(**options)


def replay(ring, n, factors):
    T, Ti = eye(ring, n), eye(ring, n)
    for item in factors:
        if isinstance(item, dict):
            i, j, a = item['row'], item['column'], item['coefficient'][0, 0]
            assert item['coefficient'].shape == (1, 1)
        else:
            i, j, a = item
        assert 0 <= i < n and 0 <= j < n and i != j
        E, Ei = eye(ring, n), eye(ring, n)
        E[i, j], Ei[i, j] = a, -a
        T, Ti = E*T, Ti*Ei
    return T, Ti


def check(column, bezout, result):
    n, R = column.nrows, column.ring
    assert result['added_ancillas'] == 0
    T, Ti = replay(R, n, result['factors'])
    assert T.shape == Ti.shape == (n, n)
    assert T == result['transform'] and Ti == result['inverse']
    assert T*Ti == Ti*T == eye(R, n)
    assert T*column == Matrix(R, n, 1, {(0, 0): 1})
    assert result['tracked_bezout'] == bezout*Ti
    assert result['tracked_bezout']*result['reduced_column'] == eye(R, 1)


@pytest.mark.parametrize('p', [2, 3, 5, 7])
def test_division_free_characteristic_and_resultant_have_exact_signs(p):
    R, rng = Ring(p, 2), Random(100+p)
    x, y = R.variables
    for n in (1, 2, 3, 4):
        M = [[R(rng.randrange(p))+rng.randrange(p)*x for _ in range(n)] for _ in range(n)]
        actual = _characteristic(M, R, budget())[-1]*((-1)**n)
        expected = R.zero()
        for order in permutations(range(n)):
            inversions = sum(order[i] > order[j] for i in range(n) for j in range(i+1, n))
            term = R((-1)**inversions)
            for i, j in enumerate(order):
                term *= M[i][j]
            expected += term
        assert actual == expected
    f, g = y*y+x*y+1, y+x
    r, a, b = _resultant_relation(f, g, budget())
    assert a*f+b*g == r
    assert r == 1  # Res_y(y²+xy+1,y+x) = f(-x) = 1.


@pytest.mark.parametrize('p', [2, 3, 5])
def test_normality_uses_existing_third_slot_for_a_nontrivial_sl2_block(p):
    R = Ring(p, 2)
    x, y = R.variables
    B = Matrix.from_rows(R, [[1+x*y, x*x], [-y*y, 1-x*y]])
    Bi = Matrix.from_rows(R, [[1-x*y, -x*x], [y*y, 1+x*y]])
    large, inverse = eye(R, 3), eye(R, 3)
    large[:2, :2], inverse[:2, :2] = B, Bi
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            a = 2*x+y+1
            factors = _conjugate_pair(B, Bi, (i, j, a), 3, budget())
            actual, actual_inverse = replay(R, 3, factors)
            E = eye(R, 3)
            E[i, j] = a
            assert actual == large*E*inverse
            assert actual*actual_inverse == eye(R, 3)


@pytest.mark.parametrize('p', [2, 3, 5, 7])
def test_monic_patching_and_nontrivial_polynomial_column(p):
    R = Ring(p, 2)
    x, y = R.variables
    a, b = x+1, y+1
    for values, witness in (([x-1, y-1, 3-x-y], [1, 1, 1]),
                            ([1+a*b, -b*b, 0], [1-a*b, -a*a, 0])):
        column = Matrix.from_rows(R, [[v] for v in values])
        row = Matrix.from_rows(R, [witness])
        result = elementary_column_reduce(column, row, timeout=20)
        check(column, row, result)
        if all(values):
            assert result['statistics']['patches'] > 0


@pytest.mark.parametrize('p,constant', [(2, 1), (3, 1), (5, 2)])
def test_cover_must_visit_an_irreducible_degree_two_residue_field(p, constant):
    R = Ring(p, 2)
    x, y = R.variables
    g = x*x+x+1 if p == 2 else x*x+constant
    s, t = (R.one(), -x-1) if p == 2 else (R(pow(constant, -1, p)), -pow(constant, -1, p)*x)
    witness = [-x*y*t, s, t]
    column = Matrix.from_rows(R, [[y+1], [g], [x*(y*y+y+1)]])
    row = Matrix.from_rows(R, [witness])
    result = elementary_column_reduce(column, row, timeout=20)
    check(column, row, result)
    points = result['statistics']['closed_points']
    assert [len(point['residue_modulus'])-1 for point in points] == [1, 2]
    assert [point['cover_gcd_degree'] for point in points] == [2, 0]


@pytest.mark.parametrize('p', [2, 3, 5])
def test_laurent_coordinates_and_only_laurent_bezout_remain_at_original_rank(p):
    R = Ring(p, 2)
    x, y = R.variables
    examples = [([x**-1, y, 1-x**-1-y], [1, 1, 1]),
                ([x, 0, 0], [x**-1, 0, 0]),
                ([0, (p-1)*x**-2*y, x+y**-1], [0, (R(p-1)*x**-2*y)**-1, 0])]
    for values, witness in examples:
        column, row = Matrix.from_rows(R, [[v] for v in values]), Matrix.from_rows(R, [witness])
        result = elementary_column_reduce(column, row, timeout=20)
        check(column, row, result)


@pytest.mark.parametrize('p', [2, 3, 5])
def test_laurent_nonunit_column_composes_polynomialization_and_fixed_rank_word(p):
    R = Ring(p, 2)
    x, y = R.variables
    v = Matrix.from_rows(R, [[x**-1*(x-1)], [y**-1*(y-1)],
                            [(3-x-y)*x**-1*y**-1]])
    h = Matrix.from_rows(R, [[x, y, x*y]])
    assert not any(f.is_unit for f in v.entries.values())
    result = elementary_column_reduce(v, h, timeout=20)
    check(v, h, result)
    assert result['statistics']['row_choices'][0]['stage'] == 'Laurent'


@pytest.mark.parametrize('n', [4, 5])
def test_larger_original_registers_keep_all_slots_and_weighted_inverse(n):
    R = Ring(5, 2)
    x, y = R.variables
    values = [x-1, y-1, 3-x-y]+[R.zero()]*(n-3)
    witness = [1, 1, 1]+[0]*(n-3)
    column, row = Matrix.from_rows(R, [[v] for v in values]), Matrix.from_rows(R, [witness])
    E, Ei = eye(R, n), eye(R, n)
    E[n-1, 1], Ei[n-1, 1] = 2*x*y, -2*x*y
    column, row = E*column, row*Ei
    result = elementary_column_reduce(column, row, timeout=20)
    check(column, row, result)


def test_invalid_inputs_and_resource_stops_do_not_return_gl_completions():
    R = Ring(3, 2)
    x, y = R.variables
    v, h = Matrix.from_rows(R, [[x-1], [y-1], [3-x-y]]), Matrix.from_rows(R, [[1, 1, 1]])
    with pytest.raises(ValueError, match='n>=3'):
        elementary_column_reduce(v[:2, :], h[:, :2])
    with pytest.raises(ValueError, match='Bezout'):
        elementary_column_reduce(v, 2*h)
    for limits in ({'max_steps': 1}, {'max_products': 1}, {'max_terms': 2},
                   {'max_factors': 1}, {'max_resultant_size': 1}, {'max_patches': 1}):
        with pytest.raises(ResourceLimit):
            elementary_column_reduce(v, h, **limits)


def test_complete_column_route_uses_no_completion_or_external_oracle(monkeypatch):
    import css_decoupling.two_d.algebra as algebra
    import css_decoupling.algebra.quillen_suslin as qs

    def forbidden(*args, **kwargs):
        pytest.fail('Fixed-rank elementary construction used an external/completion oracle')

    for target in ('subprocess.Popen', 'os.system', 'os.popen', 'shutil.which'):
        monkeypatch.setattr(target, forbidden)
    monkeypatch.setattr(algebra, 'solve', forbidden)
    monkeypatch.setattr(algebra, 'inverse', forbidden)
    monkeypatch.setattr(qs, 'complete_laurent_row', forbidden)
    monkeypatch.setattr(qs, 'complete_polynomial_row', forbidden)
    R = Ring(3, 2)
    x, y = R.variables
    v = Matrix.from_rows(R, [[y+1], [x*x+1], [x*(y*y+y+1)]])
    h = Matrix.from_rows(R, [[x*x*y, 1, -x]])
    check(v, h, elementary_column_reduce(v, h, timeout=20))


@pytest.mark.parametrize('p', [2, 3, 5])
def test_laurent_unit_fast_path_avoids_polynomial_coordinate_expansion(monkeypatch, p):
    import css_decoupling.two_d.elementary_columns as columns

    def forbidden(*args, **kwargs):
        pytest.fail('A Laurent unit does not need polynomialization or resultant patches')

    monkeypatch.setattr(columns, '_laurent_column_to_polynomial', forbidden)
    monkeypatch.setattr(columns, '_polynomial_reduce', forbidden)
    R = Ring(p, 2)
    x, y = R.variables
    unit = (p-1)*x**-7*y**11
    for pivot in range(3):
        v = Matrix.from_rows(R, [[x**-3+y], [x*y**-2+1], [x+y**-1]])
        v[pivot, 0] = unit
        h = Matrix(R, 1, 3, {(0, pivot): unit**-1})
        result = elementary_column_reduce(v, h, timeout=5)
        check(v, h, result)
        assert result['statistics']['patches'] == 0
        assert len(result['factors']) <= 11


@pytest.mark.parametrize('p', [2, 3, 5])
@pytest.mark.parametrize('n', [3, 5, 9])
def test_existing_zero_slot_uses_checked_bezout_without_enlargement(monkeypatch, p, n):
    import css_decoupling.two_d.elementary_columns as columns

    def forbidden(*args, **kwargs):
        pytest.fail('An existing zero slot and Bezout row need no resultant or completion')

    monkeypatch.setattr(columns, '_laurent_column_to_polynomial', forbidden)
    monkeypatch.setattr(columns, '_polynomial_reduce', forbidden)
    R = Ring(p, 2)
    x, y = R.variables
    a, b = x+1, y+1
    for offset in (0, 1):
        v, h = Matrix(R, n, 1), Matrix(R, 1, n)
        v[offset, 0], v[offset+1, 0] = 1+a*b, -b*b
        h[0, offset], h[0, offset+1] = 1-a*b, -a*a
        assert not any(f.is_unit for f in v.entries.values())
        result = elementary_column_reduce(v, h, timeout=5, max_resultant_size=1)
        check(v, h, result)
        assert result['statistics']['patches'] == 0
        assert len(result['factors']) <= 2*n


@pytest.mark.parametrize('p', [3, 5])
@pytest.mark.parametrize('laurent', [False, True])
def test_existing_row_selection_prefers_small_monic_degree_and_tracks_actual_swap(p, laurent):
    R = Ring(p, 2)
    x, y = R.variables
    unit = x**-1 if laurent else R.one()
    v = Matrix.from_rows(R, [[unit*(x+y**6)], [unit*(y+1)], [unit*(y+2)]])
    h = Matrix.from_rows(R, [[0, -unit**-1, unit**-1]])
    assert not any(f.is_unit for f in v.entries.values())
    result = elementary_column_reduce(v, h, timeout=20, max_resultant_size=2)
    check(v, h, result)
    selected = result['statistics']['row_choices'][0]
    assert selected['row'] == 1
    assert selected['score'][0] == 1


@pytest.mark.parametrize('p', [2, 3, 5])
def test_support_vertex_bound_gives_a_unique_scalar_leading_term(p):
    R = Ring(p, 2)
    x, y = R.variables
    f = (p-1)*x**3*y**2+x**2*y**5+x*y**8+1
    weights, scalar = _support_monic_shear(f)
    assert weights == {0: 4}
    transformed = apply_shear(f, 1, weights)
    degree = max(e[1] for e in transformed.terms)
    assert degree == 14
    assert {e: c for e, c in transformed.terms.items() if e[1] == degree} == {(0, 14): scalar}
    assert _support_monic_shear(y**3+x*y**2+x*x) == ({}, 1)
