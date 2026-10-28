"""Expanded positive solves are checked shortcuts, never kernel or negative proofs."""
from unittest.mock import patch

import pytest

from css_decoupling.algebra.laurent import Matrix, Ring, zeros
from css_decoupling.algebra.modules import GeneralBackend
from css_decoupling.algebra.sparse_linear import sparse_binary_solution
from css_decoupling.errors import NoLaurentSolution
from css_decoupling.lattice.retained import Lattice
from css_decoupling.lattice.restriction_backend import RestrictionBackend


def _case():
    R, lattice = Ring(2, 2), Lattice(((2, 0), (0, 1)))
    x, y = R.variables
    source = Matrix.from_rows(R, [[x + 1, y + 1]])
    A = lattice.restrict(source)
    X = Matrix.from_rows(R, [[1, 0], [0, 1], [1, 1], [0, 1]])
    return R, lattice, source, A, X, A * X


def test_actual_expanded_solution_is_accepted_before_structure_or_groebner_work():
    R, lattice, _, A, _, B = _case()
    base = GeneralBackend(R)
    backend = RestrictionBackend(base, lattice)
    with patch('css_decoupling.algebra.sparse_linear.sparse_binary_solution',
               wraps=sparse_binary_solution) as early, \
            patch.object(backend, '_trim_zero_rows', side_effect=AssertionError('positive route is first')), \
            patch.object(backend, 'source_matrix', side_effect=AssertionError('no reconstruction needed')), \
            patch.object(base, 'solve', side_effect=AssertionError('no Groebner fallback needed')):
        found = backend.solve(A, B)
    assert early.call_args.args == (A, B)
    assert found.ring == R and found.shape == (A.ncols, B.ncols)
    assert A * found == B


@pytest.mark.parametrize('fault', ['wrong-equation', 'wrong-shape', 'wrong-ring'])
def test_false_positive_candidates_cannot_pass_the_original_equation(fault):
    R, lattice, _, A, _, B = _case()
    candidate = zeros(R, A.ncols, B.ncols)
    if fault == 'wrong-shape':
        candidate = zeros(R, A.ncols + 1, B.ncols)
    elif fault == 'wrong-ring':
        candidate = zeros(Ring(3, R.D), A.ncols, B.ncols)
    with patch('css_decoupling.algebra.sparse_linear.sparse_binary_solution', return_value=candidate):
        with pytest.raises(ArithmeticError, match='Expanded sparse solution'):
            RestrictionBackend(GeneralBackend(R), lattice).solve(A, B)


def test_none_preserves_the_recognized_source_route():
    R, lattice, source, A, _, B = _case()
    base = GeneralBackend(R)
    backend = RestrictionBackend(base, lattice)
    with patch('css_decoupling.algebra.sparse_linear.sparse_binary_solution', return_value=None), \
            patch.object(base, 'solve', wraps=base.solve) as fallback:
        found = backend.solve(A, B)
    assert fallback.call_args.args[0] == source
    assert A * found == B


def test_none_preserves_ordinary_fallback_and_exact_negative_membership():
    R, lattice, _, _, _, _ = _case()
    x, y = R.variables
    A = Matrix.from_rows(R, [[x + 1, y + 1]])
    X = Matrix.from_rows(R, [[x ** -2], [y]])
    base = GeneralBackend(R)
    backend = RestrictionBackend(base, lattice)
    with patch('css_decoupling.algebra.sparse_linear.sparse_binary_solution', return_value=None), \
            patch.object(base, 'solve', wraps=base.solve) as fallback:
        assert A * backend.solve(A, A * X) == A * X
    assert fallback.call_args.args[0] == A
    nonmember = lattice.restrict(Matrix.from_rows(R, [[x + 1]]))
    rhs = Matrix(R, nonmember.nrows, 1, {(0, 0): 1})
    with patch('css_decoupling.algebra.sparse_linear.sparse_binary_solution', return_value=None):
        with pytest.raises(NoLaurentSolution):
            backend.solve(nonmember, rhs)


def test_positive_shortcut_is_never_used_to_establish_kernel_completeness():
    R, lattice, _, A, _, _ = _case()
    backend = RestrictionBackend(GeneralBackend(R), lattice)
    with patch('css_decoupling.algebra.sparse_linear.sparse_binary_solution',
               side_effect=AssertionError('kernel completeness cannot use a positive ansatz')):
        K = backend.kernel(A)
    reference_backend = GeneralBackend(R)
    reference = reference_backend.kernel(A)
    assert K * reference_backend.solve(K, reference) == reference
    assert reference * reference_backend.solve(reference, K) == K
