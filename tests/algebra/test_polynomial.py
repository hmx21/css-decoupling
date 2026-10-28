"""Polynomial membership must remain distinct from Laurent membership."""
import pytest

from css_decoupling.algebra.laurent import Ring, Matrix, eye, zeros
from css_decoupling.algebra.modules import solve
from css_decoupling.algebra.polynomial import solve_polynomial, NoPolynomialSolution
from css_decoupling.errors import NoLaurentSolution, ResourceLimit


def polynomial(matrix):
    return all(exponent >= 0 for f in matrix.entries.values() for powers in f.terms for exponent in powers)


def test_laurent_unit_is_not_a_polynomial_unit():
    ring = Ring(3, 2)
    x = ring.variable(0)
    A, B = Matrix.from_rows(ring, [[x]]), eye(ring, 1)
    assert solve(A, B) == Matrix.from_rows(ring, [[x ** -1]])
    with pytest.raises(NoPolynomialSolution):
        solve_polynomial(A, B)
    # The completed basis has size one. A nonzero target remainder is not a
    # new basis element and must not be mistaken for a cap violation.
    with pytest.raises(NoPolynomialSolution):
        solve_polynomial(A, B, max_basis=1)
    assert not issubclass(NoPolynomialSolution, NoLaurentSolution)


@pytest.mark.parametrize("p,D", [(2, 2), (3, 3), (5, 5), (2147483659, 2)])
def test_polynomial_bezout_in_every_prime_characteristic(p, D, monkeypatch):
    import subprocess
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: (_ for _ in ()).throw(AssertionError("native arithmetic only")))
    ring = Ring(p, D)
    x = ring.variable(D - 1)
    row = Matrix.from_rows(ring, [[x, 1 - x]])
    result = solve_polynomial(row, eye(ring, 1))
    assert polynomial(result) and row * result == eye(ring, 1)


def test_multivariate_membership_requires_polynomial_combinations():
    ring = Ring(5, 3)
    x, y, z = ring.variables
    A = Matrix.from_rows(ring, [[x * y - 1, y * z - 1]])
    B = Matrix.from_rows(ring, [[x - z, x * y - 1]])
    # x-z = z*(xy-1) - x*(yz-1), with an odd-characteristic minus sign.
    result = solve_polynomial(A, B)
    assert polynomial(result) and A * result == B


def test_true_module_membership_not_just_the_ideal_of_all_entries():
    ring = Ring(3, 2)
    x, y = ring.variables
    A = Matrix.from_rows(ring, [[x, 1 - x, 0], [0, y, 1 - y]])
    known = Matrix.from_rows(ring, [[x + y, 1], [x * y, y], [1 - x, x * x]])
    B = A * known
    answer = solve_polynomial(A, B)
    assert polynomial(answer) and A * answer == B
    diagonal = Matrix.from_rows(ring, [[x, 0], [0, 1]])
    with pytest.raises(NoPolynomialSolution):
        solve_polynomial(diagonal, Matrix.from_rows(ring, [[1], [0]]))


def test_zero_source_columns_keep_correct_transition_indices():
    ring = Ring(5, 2)
    x, y = ring.variables
    row = Matrix.from_rows(ring, [[0, x, 0, 1 - x, 0]])
    rhs = Matrix.from_rows(ring, [[1, y, 0]])
    answer = solve_polynomial(row, rhs)
    assert answer.shape == (5, 3) and row * answer == rhs
    assert not answer[0:1, :] and not answer[2:3, :] and not answer[4:5, :]


@pytest.mark.parametrize("m,n,k", [(0, 0, 0), (0, 3, 2), (3, 0, 2), (3, 2, 0), (0, 0, 4)])
def test_zero_shapes(m, n, k):
    ring = Ring(2, 2)
    A, B = zeros(ring, m, n), zeros(ring, m, k)
    result = solve_polynomial(A, B)
    assert result.shape == (n, k) and A * result == B


def test_empty_source_cannot_generate_nonzero_rhs():
    ring = Ring(3, 2)
    with pytest.raises(NoPolynomialSolution, match="zero polynomial image"):
        solve_polynomial(zeros(ring, 2, 0), Matrix.from_rows(ring, [[1], [0]]))


def test_negative_exponents_are_rejected_even_when_a_laurent_solution_exists():
    ring = Ring(3, 2)
    x = ring.variable(0)
    negative = Matrix.from_rows(ring, [[x ** -1]])
    with pytest.raises(ValueError, match="negative"):
        solve_polynomial(negative, eye(ring, 1))
    with pytest.raises(ValueError, match="negative"):
        solve_polynomial(eye(ring, 1), negative)


def test_nonmembership_of_one_rhs_column_rejects_the_whole_system():
    ring = Ring(5, 2)
    x = ring.variable(0)
    with pytest.raises(NoPolynomialSolution, match="column 1"):
        solve_polynomial(Matrix.from_rows(ring, [[x]]), Matrix.from_rows(ring, [[x, 1]]))


def test_resource_limits_do_not_become_nonmembership():
    ring = Ring(3, 2)
    x, y = ring.variables
    A = Matrix.from_rows(ring, [[x * x - y, x * y - 1]])
    with pytest.raises(ResourceLimit, match="max_basis"):
        solve_polynomial(A, A[:, :1], max_basis=2)
    with pytest.raises(ResourceLimit, match="max_terms"):
        solve_polynomial(A, A[:, :1], max_terms=2)
    result = solve_polynomial(A, A[:, :1], max_basis=None, max_terms=None)
    assert A * result == A[:, :1]


def test_parameter_and_shape_validation():
    ring = Ring(3, 2)
    with pytest.raises(ValueError, match="positive"):
        solve_polynomial(eye(ring, 1), eye(ring, 1), max_basis=True)
    with pytest.raises(ValueError, match="matching"):
        solve_polynomial(eye(ring, 1), eye(ring, 2))
    with pytest.raises(ValueError, match="common"):
        solve_polynomial(eye(ring, 1), eye(Ring(5, 2), 1))
