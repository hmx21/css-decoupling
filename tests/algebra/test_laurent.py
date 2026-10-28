"""Exact Laurent arithmetic and constructive module checks."""
from itertools import combinations

import pytest

from css_decoupling.algebra.laurent import Ring, Matrix, block, direct_sum, eye, zeros
from css_decoupling.algebra.modules import GeneralBackend, NoLaurentSolution, kernel, solve


def test_prime_characteristic_dimension_and_mixed_ring_rejection():
    with pytest.raises(ValueError, match="prime"):
        Ring(9, 4)
    with pytest.raises(ValueError, match="variable"):
        Ring(3, 0)
    with pytest.raises(TypeError, match="integer"):
        Ring(3.0, 4)
    r = Ring(3, 4)
    x = r.variable(0)
    assert 3 * x == 0
    assert 2 * x + x == 0
    assert (x + 1) ** 3 == x ** 3 + 1
    assert (2 * x ** -3) ** -1 * (2 * x ** -3) == 1
    with pytest.raises(ValueError, match="different"):
        _ = x + Ring(5, 4).variable(0)
    with pytest.raises(ValueError, match="units"):
        _ = (x + 1) ** -1
    with pytest.raises(ValueError, match="dimension"):
        r.monomial((1, 0, 0))
    with pytest.raises(TypeError, match="integer"):
        r(1.5)


def test_sparse_matrix_operations_dagger_and_empty_shapes():
    r = Ring(5, 4)
    x, y, z, w = r.variables
    A = Matrix.from_rows(r, [[2*x, y**-1], [z-w, 0]])
    B = Matrix.from_rows(r, [[x**-1, 1], [w, 3*y]])
    assert (A * B).dagger() == B.dagger() * A.dagger()
    assert A.dagger().dagger() == A
    assert A + (-A) == 0
    assert A[:, ::-1][:, ::-1] == A
    assert A[[0, 0], [1]][0, 0] == A[0, 1]
    assert A[[0, 0], [1]][1, 0] == A[0, 1]
    C = block([[A, zeros(r, 2, 0)], [B, zeros(r, 2, 0)]])
    assert C == A.vstack(B)
    assert direct_sum(A, B) == block([[A, zeros(r, 2, 2)], [zeros(r, 2, 2), B]])
    assert zeros(r, 3, 0) * zeros(r, 0, 4) == zeros(r, 3, 4)
    assert eye(r, 0).inverse() == eye(r, 0)
    with pytest.raises(ValueError, match="ring"):
        _ = A * eye(Ring(3, 4), 2)
    with pytest.raises(IndexError):
        A[2, 0] = 1


def test_nonbinary_laurent_membership_not_fraction_field_solvability():
    r = Ring(3, 4)
    x, y, z, w = r.variables
    A = Matrix.from_rows(r, [[2*(x-1), y-1, 0], [0, 0, z+1]])
    expected = Matrix.from_rows(r, [[x**-2 + 2*w, y], [2*x*w**-1, 1], [w**-3, 0]])
    B = A * expected
    found = solve(A, B)
    assert A * found == B
    with pytest.raises(NoLaurentSolution):
        solve(Matrix.from_rows(r, [[x-1]]), eye(r, 1))
    # x itself is a Laurent unit; its inverse is not an ordinary polynomial.
    assert solve(Matrix.from_rows(r, [[2*x]]), eye(r, 1))[0, 0] == 2*x**-1
    with pytest.raises(NoLaurentSolution):
        solve(zeros(r, 2, 0), Matrix.from_rows(r, [[0], [1]]))


def test_localization_relation_uses_minus_one_in_odd_characteristic():
    r = Ring(5, 4)
    x, y, z, w = r.variables
    backend = GeneralBackend(r)
    A = Matrix.from_rows(r, [[x-2, y-3, z-4, w-2]])
    B = Matrix.from_rows(r, [[3*x**-2*y**3*z**-1*w + y**-1]])
    expected = (3*pow(2, -2, 5)*pow(3, 3, 5)*pow(4, -1, 5)*2 + pow(3, -1, 5)) % 5
    reduced = backend.normal_form(A, B)
    assert reduced == Matrix.from_rows(r, [[expected]])
    assert A * backend.solve(A, B-reduced) == B-reduced
    normal_sdm = backend.reduce_quotient(A, B)
    assert all(isinstance(c, int) and 0 < c < 5 for column in normal_sdm for _, c in column)
    data = backend.quotient_groebner(A)
    assert data["arithmetic"] == "GF(5)"
    assert all(len(m) == r.D+2 for m in data["leading_terms"])


def test_full_normal_form_reduces_nonleading_terms():
    r = Ring(3, 4)
    x, y, _, _ = r.variables
    A = Matrix.from_rows(r, [[y-1]])
    backend = GeneralBackend(r)
    B = Matrix.from_rows(r, [[x**4 + y**2 + 2*y]])
    assert backend.normal_form(A, B) == Matrix.from_rows(r, [[x**4]])


@pytest.mark.parametrize("p,D", [(3, 4), (5, 5)])
def test_syzygies_generate_all_koszul_relations(p, D):
    r = Ring(p, D)
    generators = [x-1 for x in r.variables]
    A = Matrix.from_rows(r, [generators])
    pairs = list(combinations(range(D), 2))
    koszul = zeros(r, D, len(pairs))
    for col, (i, j) in enumerate(pairs):
        koszul[i, col] = generators[j]
        koszul[j, col] = -generators[i]
    computed = kernel(A)
    assert A * computed == 0
    # Mutual module inclusions check the complete kernel, not just A*K=0.
    assert computed * solve(computed, koszul) == koszul
    assert koszul * solve(koszul, computed) == computed


def test_denominator_clearing_preserves_entire_kernel_with_redundant_columns():
    r = Ring(3, 4)
    x, y, _, _ = r.variables
    a, b = x**-2*(x-y), y**-3*(x+y)
    A = Matrix.from_rows(r, [[a, b, a+b, 0]])
    expected = Matrix.from_rows(r, [[b, -1, 0], [-a, -1, 0], [0, 1, 0], [0, 0, 1]])
    found = kernel(A)
    assert A * found == 0
    assert found * solve(found, expected) == expected
    assert expected * solve(expected, found) == found


def test_unit_reduction_tracks_odd_characteristic_row_and_column_signs():
    r = Ring(5, 4)
    x, y, z, _ = r.variables
    A = Matrix.from_rows(r, [[0, x-1, 2*y], [z-1, x+y, x*z], [0, 0, 0]])
    backend = GeneralBackend(r)
    count, reduced, left, right = backend.reduce_units(A)
    assert count >= 1
    expected = block([[eye(r, count), zeros(r, count, reduced.ncols)],
                      [zeros(r, reduced.nrows, count), reduced]])
    assert left * A * right == expected
    assert left.inverse() * left == eye(r, A.nrows)
    assert right * right.inverse() == eye(r, A.ncols)
    found = backend.kernel(A)
    assert A * found == 0


def test_groebner_caches_do_not_alias_mutable_output():
    r = Ring(3, 4)
    x, y, _, _ = r.variables
    A = Matrix.from_rows(r, [[x-1, y-1]])
    backend = GeneralBackend(r)
    original = backend.kernel(A)
    corrupted = backend.kernel(A)
    corrupted[0, 0] = 0
    assert backend.kernel(A) == original
    assert A * original == 0
    data = backend.quotient_groebner(A)
    data["basis"].clear()
    assert backend.quotient_groebner(A)["basis"]
