"""Presentation choices preserve complete Laurent module results."""
from unittest.mock import patch

import pytest

from css_decoupling.algebra.laurent import Matrix, Ring, eye
from css_decoupling.algebra.modules import GeneralBackend
from css_decoupling.errors import ResourceLimit
from css_decoupling.topology.cohomology import FiniteQuotient, InfiniteLayer


def _fill_example():
    R = Ring(2, 2)
    x, y = R.variables
    return Matrix.from_rows(R, [
        [y, 0, 1 + y, 0, 1, 1 + x],
        [1 + y, 0, 1 + x, 1 + y, x, 1 + y],
        [x, 1, 0, y, 1 + y, 0],
        [1 + y, 1 + x, 0, 0, y, y],
    ])


def test_fill_aware_kernel_keeps_original_sparse_presentation_and_complete_image():
    A = _fill_example()
    backend = GeneralBackend(A.ring)
    count, reduced, _, _ = backend.reduce_units(A)
    assert count == 2
    assert sum(len(f.terms) for f in reduced.entries.values()) > sum(
        len(f.terms) for f in A.entries.values())
    with patch.object(backend, '_kernel_reduced', wraps=backend._kernel_reduced) as chosen:
        K = backend.kernel(A)
    assert chosen.call_args.args[0] == A
    independent = backend._kernel_graph(A)
    assert K * backend.solve(K, independent) == independent
    assert independent * backend.solve(independent, K) == K


def test_fill_aware_solve_keeps_original_presentation():
    A = _fill_example()
    x, y = A.ring.variables
    X = Matrix.from_rows(A.ring, [[x ** -1], [y], [1], [x + y], [0], [y ** -2]])
    B = A * X
    backend = GeneralBackend(A.ring)
    # Bypass the bounded F2 shortcut, which answers before any presentation choice.
    with patch('css_decoupling.algebra.sparse_linear.sparse_binary_solution', return_value=None), \
            patch.object(backend, '_solve_reduced', wraps=backend._solve_reduced) as chosen:
        found = backend.solve(A, B)
    assert chosen.call_args.args[0] == A
    assert A * found == B


@pytest.mark.parametrize('prime', [2, 3, 5])
@pytest.mark.parametrize('dimension', [2, 3])
def test_tracked_native_kernel_matches_independent_graph_with_laurent_units(prime, dimension):
    R = Ring(prime, dimension)
    x, y = R.variables[:2]
    A = Matrix.from_rows(R, [
        [x ** -2 * (x - 1), y - 1, 0, (x - 1) + y * (y - 1)],
        [0, x - 1, y ** -1 * (y - 1), y * (x - 1)],
    ])
    backend = GeneralBackend(R)
    native = backend.kernel(A)
    graph = backend._kernel_graph(A)
    assert A * native == 0
    assert native * backend.solve(native, graph) == graph
    assert graph * backend.solve(graph, native) == native


def test_native_kernel_does_not_suppress_completeness_verification_or_budget_failures():
    R = Ring(3, 2)
    x, y = R.variables
    A = Matrix.from_rows(R, [[x - 1, y - 1]])
    with patch('css_decoupling.algebra.schreyer._verify_polynomial_kernel',
               side_effect=ArithmeticError('independent completeness check')):
        with pytest.raises(ArithmeticError, match='independent completeness'):
            GeneralBackend(R).kernel(A)
    with patch('css_decoupling.algebra.schreyer.polynomial_kernel',
               side_effect=ResourceLimit('native algebra budget')):
        with pytest.raises(ResourceLimit, match='native algebra budget'):
            GeneralBackend(R).kernel(A)


@pytest.mark.parametrize('prime', [2, 3, 5])
def test_zero_quotient_uses_an_exact_right_inverse_not_fraction_field_rank(prime):
    R = Ring(prime, 2)
    x, y = R.variables
    A = Matrix.from_rows(R, [[x, y - 1], [0, y]])
    with patch('css_decoupling.algebra.modules.sdm_groebner',
               side_effect=AssertionError('An explicit right inverse proves the quotient zero')):
        assert FiniteQuotient(A, backend=GeneralBackend(R)).dimension == 0
    # A rank-one map over the fraction field can still have infinite cokernel.
    with pytest.raises(InfiniteLayer):
        FiniteQuotient(Matrix.from_rows(R, [[x - 1]]), backend=GeneralBackend(R))


def test_nonzero_quotient_uses_plain_groebner_and_keeps_exact_coordinates():
    import css_decoupling.algebra.modules as modules
    R = Ring(3, 2)
    x, y = R.variables
    A = Matrix.from_rows(R, [[x - 1, y - 1]])
    with patch.object(modules, 'sdm_groebner', wraps=modules.sdm_groebner) as groebner:
        quotient = FiniteQuotient(A, backend=GeneralBackend(R))
    assert quotient.dimension == 1
    assert quotient.coordinates(Matrix.from_rows(R, [[x ** -3 * y ** 2]])) == eye(R, 1)
    assert all(not call.kwargs.get('extended', False) for call in groebner.call_args_list)


@pytest.mark.parametrize('prime', [2, 3, 5])
def test_positive_polynomial_membership_avoids_inverse_variable_groebner(prime):
    R = Ring(prime, 2)
    x, y = R.variables
    A = Matrix.from_rows(R, [[x - 1, y - 1]])
    B = A * Matrix.from_rows(R, [[x ** -2 * (1 + y ** 5)], [2 * y ** -1]])
    backend = GeneralBackend(R)
    with patch('css_decoupling.algebra.sparse_linear.sparse_binary_solution', return_value=None), \
            patch.object(backend, '_basis', side_effect=AssertionError('Polynomial witness suffices')):
        found = backend.solve(A, B)
    assert A * found == B


def test_polynomial_nonmembership_is_not_laurent_nonmembership():
    R = Ring(2, 2)
    x, y = R.variables
    # The polynomial image is (x), while x is invertible in the Laurent ring.
    A = Matrix.from_rows(R, [[x * (1 + y), x * y * (1 + y + y ** 2)]])
    B = eye(R, 1)
    backend = GeneralBackend(R)
    assert backend._solve_polynomial(A, B) is None
    with patch('css_decoupling.algebra.sparse_linear.sparse_binary_solution', return_value=None), \
            patch.object(backend, '_basis', wraps=backend._basis) as localized:
        found = backend.solve(A, B)
    assert localized.called
    assert A * found == B


def test_polynomial_attempt_budget_failure_falls_back_without_negative_claim():
    R = Ring(2, 2)
    x = R.variable(0)
    A, B = Matrix.from_rows(R, [[1 + x]]), Matrix.from_rows(R, [[1 + x ** 9]])
    with patch('css_decoupling.algebra.sparse_linear.sparse_binary_solution', return_value=None), \
            patch('css_decoupling.algebra.schreyer.polynomial_groebner',
                  side_effect=ResourceLimit('positive attempt budget')):
        found = GeneralBackend(R).solve(A, B)
    assert A * found == B
