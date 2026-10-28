"""Positive sparse coefficient solves must never replace full membership."""
from unittest.mock import patch

import pytest

from css_decoupling.algebra.laurent import Matrix, Ring, eye, zeros
from css_decoupling.algebra.modules import GeneralBackend, NoLaurentSolution
from css_decoupling.algebra.sparse_linear import sparse_binary_solution


def test_binary_laurent_multiple_rhs_shortcut_has_exact_witness_without_groebner():
    R = Ring(2, 3)
    x, y, z = R.variables
    A = Matrix.from_rows(R, [[x + 1, y + 1, 0], [0, z + 1, 0]])
    X = Matrix.from_rows(R, [[y ** -2 + z, 1], [x ** -1, y], [0, 0]])
    B = A * X
    with patch.object(GeneralBackend, '_basis', side_effect=AssertionError('GB not needed')):
        found = GeneralBackend(R).solve(A, B)
    assert A * found == B
    assert found.shape == X.shape


def test_failed_finite_support_falls_back_to_complete_groebner():
    R = Ring(2, 2)
    x, _ = R.variables
    A = Matrix.from_rows(R, [[1 + x]])
    B = Matrix.from_rows(R, [[1 + x ** 9]])
    # A solution needs the intermediate exponents absent from the one-shot
    # support differences.  Its restricted inconsistency is not nonmembership.
    assert sparse_binary_solution(A, B) is None
    backend = GeneralBackend(R)
    found = backend.solve(A, B)
    assert A * found == B
    assert found[0, 0] == sum((x ** j for j in range(9)), R.zero())


def test_negative_result_still_requires_full_module_nonmembership():
    R = Ring(2, 2)
    A = Matrix.from_rows(R, [[R.variable(0) + 1]])
    assert sparse_binary_solution(A, eye(R, 1)) is None
    backend = GeneralBackend(R)
    with patch.object(backend, '_basis', wraps=backend._basis) as full:
        with pytest.raises(NoLaurentSolution):
            backend.solve(A, eye(R, 1))
    assert full.called


@pytest.mark.parametrize('prime', [3, 5, 2147483659])
def test_odd_primes_keep_complete_exact_backend(prime):
    R = Ring(prime, 2)
    x, y = R.variables
    A = Matrix.from_rows(R, [[2 * (x - 1), y - 1]])
    B = A * Matrix.from_rows(R, [[x ** -2], [2 * y]])
    assert sparse_binary_solution(A, B) is None
    assert A * GeneralBackend(R).solve(A, B) == B


def test_support_work_does_not_depend_on_exponent_box_width():
    R = Ring(2, 3)
    x, y, _ = R.variables
    huge = 10 ** 30
    A = Matrix.from_rows(R, [[1 + x ** huge, 0]])
    B = Matrix.from_rows(R, [[y ** -huge * (1 + x ** huge)]])
    found = sparse_binary_solution(A, B, max_unknowns=8, max_equations=20)
    assert found is not None
    assert A * found == B


@pytest.mark.parametrize('limit,value', [
    ('max_unknowns', 0), ('max_unknowns', 1), ('max_equations', 1),
    ('max_support_pairs', 1), ('max_terms', 1), ('max_coefficient_bits', 1),
    ('max_steps', 1), ('max_rhs', 1), ('max_solution_bits', 1),
    ('max_product_work', 1),
])
def test_every_resource_cap_means_fallback_not_nonmembership(limit, value):
    R = Ring(2, 2)
    x, y = R.variables
    A = Matrix.from_rows(R, [[x + 1, y + 1], [0, x + y]])
    B = A * Matrix.from_rows(R, [[x ** -1, y], [y ** -1, x]])
    assert sparse_binary_solution(A, B, **{limit: value}) is None
    assert A * GeneralBackend(R).solve(A, B) == B


def test_no_shortcut_for_zero_or_mismatched_input():
    R = Ring(2, 2)
    assert sparse_binary_solution(zeros(R, 1, 2), zeros(R, 1, 3)) is None
    with pytest.raises(ValueError):
        sparse_binary_solution(eye(R, 1), eye(R, 2))


def test_extremely_wide_sparse_rhs_is_rejected_before_elimination_allocation():
    R = Ring(2, 2)
    x = R.variable(0)
    A = Matrix.from_rows(R, [[1 + x]])
    B = Matrix(R, 1, 10 ** 9, {(0, 10 ** 9 - 1): 1 + x})
    with patch('css_decoupling.algebra.sparse_linear._binary_solution',
               side_effect=AssertionError('RHS width must be capped first')):
        assert sparse_binary_solution(A, B) is None


def test_rhs_equation_limit_precedes_operator_support_enumeration():
    R = Ring(2, 2)
    A = Matrix.from_rows(R, [[1], [1]])
    B = A.copy()
    with patch.object(A, 'entries') as entries:
        entries.__bool__.return_value = True
        entries.items.side_effect = AssertionError('RHS equations already exceed the limit')
        assert sparse_binary_solution(A, B, max_equations=1) is None


def test_exact_product_guard_declines_before_materializing_the_verification_product(monkeypatch):
    R = Ring(2, 3)
    x = R.variable(0)
    A = Matrix.from_rows(R, [[1+x+x*x]])
    B = A.copy()
    multiplied = []
    original = Matrix.__mul__

    def forbidden_over_budget(self, other):
        if self is A and isinstance(other, Matrix):
            multiplied.append(other)
            raise AssertionError('The over-budget sparse verification product was evaluated')
        return original(self, other)

    monkeypatch.setattr(Matrix, '__mul__', forbidden_over_budget)
    assert sparse_binary_solution(A, B, max_product_work=2) is None
    assert not multiplied


@pytest.mark.parametrize('budget', [3, None])
def test_exact_product_at_the_declared_cap_is_still_multiplied_and_verified(monkeypatch, budget):
    R = Ring(2, 3)
    x = R.variable(0)
    A = Matrix.from_rows(R, [[1+x+x*x]])
    B = A.copy()
    multiplied = []
    original = Matrix.__mul__

    def observed(self, other):
        if self is A and isinstance(other, Matrix):
            multiplied.append(other.copy())
        return original(self, other)

    monkeypatch.setattr(Matrix, '__mul__', observed)
    result = sparse_binary_solution(A, B, max_product_work=budget)
    assert result == eye(R, 1)
    assert multiplied == [result]


def test_zero_product_budget_returns_unknown_instead_of_an_unchecked_positive():
    R = Ring(2, 2)
    assert sparse_binary_solution(eye(R, 1), eye(R, 1), max_product_work=0) is None
