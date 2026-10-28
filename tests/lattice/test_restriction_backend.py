"""Restriction recognition and exact algebra with arbitrary RHS columns."""
from unittest.mock import patch

import pytest

from css_decoupling.algebra.bounded_linear import BoundedLinearBackend
from css_decoupling.algebra.laurent import Matrix, Ring, direct_sum, eye, zeros
from css_decoupling.algebra.modules import GeneralBackend, kernel, solve, using_backend
from css_decoupling.errors import NoLaurentSolution, ResourceLimit
from css_decoupling.lattice.retained import Lattice
from css_decoupling.lattice.restriction_backend import RestrictionBackend, unpack_columns
from css_decoupling.topology.cohomology import FiniteQuotient, InfiniteLayer


BASES = [((2, 0), (0, 1)), ((2, 1), (1, -1)), ((0, -1), (1, 0))]


@pytest.mark.parametrize('prime', [2, 3, 5])
@pytest.mark.parametrize('basis', BASES)
def test_pack_unpack_both_directions_for_negative_and_skew_coordinates(prime, basis):
    R, lattice = Ring(prime, 2), Lattice(basis)
    x, y = R.variables
    original = Matrix.from_rows(R, [[x ** -3 + 2 * y, y ** -2], [x * y, x ** 4 - y]])
    assert unpack_columns(lattice.pack(original), lattice) == original
    # Every S-column is allowed, including columns with no Res-matrix pattern.
    packed = Matrix(R, 2 * lattice.index, 3,
                    {(0, 0): x ** -2 + y, (lattice.index - 1, 1): 2 * y ** -3,
                     (2 * lattice.index - 1, 2): x * y ** -1 + 1})
    assert lattice.pack(unpack_columns(packed, lattice)) == packed


@pytest.mark.parametrize('prime', [2, 3, 5])
@pytest.mark.parametrize('basis', BASES[:2])
def test_recognized_solve_uses_smaller_source_operator_and_arbitrary_rhs(prime, basis):
    R, lattice = Ring(prime, 2), Lattice(basis)
    x, y = R.variables
    source = Matrix.from_rows(R, [[x - 1, y - 1, 0], [0, x + y, 1]])
    A = lattice.restrict(source)
    # Four RHS columns need not arise by restricting another R-matrix.
    X = Matrix(R, A.ncols, 4, {(i, j): x ** (-i % 3) * y ** (j - 2) + (i + j)
                              for i in range(A.ncols) for j in range(4)})
    B = A * X
    base = GeneralBackend(R)
    backend = RestrictionBackend(base, lattice)
    assert backend.source_matrix(A) == source
    with patch('css_decoupling.algebra.sparse_linear.sparse_binary_solution', return_value=None), \
            patch.object(base, 'solve', wraps=base.solve) as smaller:
        found = backend.solve(A, B)
    assert smaller.call_args.args[0] == source
    assert smaller.call_args.args[1].shape == (source.nrows, B.ncols)
    assert A * found == B
    assert found.shape == X.shape


def test_recognized_negative_membership_is_an_exact_negative():
    R, lattice = Ring(3, 2), Lattice(((2, 1), (0, 1)))
    source = Matrix.from_rows(R, [[R.variable(0) - 1]])
    A = lattice.restrict(source)
    B = Matrix(R, A.nrows, 1, {(0, 0): 1})
    with pytest.raises(NoLaurentSolution):
        RestrictionBackend(GeneralBackend(R), lattice).solve(A, B)


@pytest.mark.parametrize('prime', [2, 3, 5])
def test_complete_kernel_restricts_all_translated_generators(prime):
    R, lattice = Ring(prime, 2), Lattice(((2, 1), (1, -1)))
    x, y = R.variables
    source = Matrix.from_rows(R, [[x - 1, y - 1, 0]])
    A = lattice.restrict(source)
    base = GeneralBackend(R)
    backend = RestrictionBackend(base, lattice)
    with patch.object(base, 'kernel', wraps=base.kernel) as smaller:
        K = backend.kernel(A)
    assert smaller.call_args.args[0] == source
    reference = GeneralBackend(R).kernel(A)
    independent = GeneralBackend(R)
    assert K * independent.solve(K, reference) == reference
    assert reference * independent.solve(reference, K) == K
    assert A * K == 0
    # A single R-free direction contributes N S-generators, not just one.
    simple = lattice.restrict(Matrix.from_rows(R, [[1, 0]]))
    assert backend.kernel(simple) == zeros(R, lattice.index, lattice.index).vstack(
        eye(R, lattice.index))
    assert backend.kernel_generators(simple) == backend.kernel(simple)


def test_full_reconstruction_rejects_changes_outside_coset_zero_columns():
    R, lattice = Ring(2, 2), Lattice(((2, 0), (0, 1)))
    x, y = R.variables
    source = Matrix.from_rows(R, [[x + 1, y + 1]])
    A = lattice.restrict(source)
    base, backend = GeneralBackend(R), None
    backend = RestrictionBackend(base, lattice)
    assert backend.source_matrix(A) == source
    # The sampled columns remain unchanged; checking them alone is unsound.
    A[0, 1] += y
    assert unpack_columns(A[:, [0, 2]], lattice) == source
    assert backend.source_matrix(A) is None
    X = Matrix.from_rows(R, [[x], [y], [1], [x ** -1]])
    B = A * X
    with patch('css_decoupling.algebra.sparse_linear.sparse_binary_solution', return_value=None), \
            patch.object(base, 'solve', wraps=base.solve) as fallback:
        answer = backend.solve(A, B)
    assert fallback.call_args.args[0] == A
    assert A * answer == B


def test_nondimensional_or_nonmultiplication_operator_uses_plain_backend():
    R, lattice = Ring(5, 2), Lattice(((2, 0), (0, 1)))
    x, y = R.variables
    A = Matrix.from_rows(R, [[1, x, 0, 0], [0, 1, y, 0], [0, 0, 1, y]])
    base = GeneralBackend(R)
    backend = RestrictionBackend(base, lattice)
    assert backend.source_matrix(A) is None
    X = Matrix.from_rows(R, [[1], [x], [y ** -1], [2]])
    with patch('css_decoupling.algebra.sparse_linear.sparse_binary_solution', return_value=None), \
            patch.object(base, 'solve', wraps=base.solve) as fallback:
        found = backend.solve(A, A * X)
    assert fallback.call_args.args[0] == A
    assert A * found == A * X
    with patch.object(base, 'kernel', wraps=base.kernel) as fallback_kernel:
        assert A * backend.kernel(A) == 0
    assert fallback_kernel.call_args.args[0] == A


def test_zero_constraint_rows_are_checked_and_trimmed_before_recognition():
    R, lattice = Ring(3, 2), Lattice(((2, 0), (0, 1)))
    x, y = R.variables
    source = Matrix.from_rows(R, [[x - 1, y - 1]])
    A0 = lattice.restrict(source)
    A = A0.vstack(zeros(R, 3, A0.ncols))
    X = Matrix.from_rows(R, [[x], [y], [x ** -1], [2]])
    B = A * X
    base = GeneralBackend(R)
    backend = RestrictionBackend(base, lattice)
    with patch('css_decoupling.algebra.sparse_linear.sparse_binary_solution', return_value=None), \
            patch.object(base, 'solve', wraps=base.solve) as smaller:
        assert A * backend.solve(A, B) == B
    assert smaller.call_args.args[0] == source
    with patch.object(base, 'kernel', wraps=base.kernel) as smaller_kernel:
        K = backend.kernel(A)
    assert smaller_kernel.call_args.args[0] == source
    assert A * K == 0
    bad = B.copy()
    bad[A.nrows - 1, 0] = 1
    with patch.object(base, 'solve', side_effect=AssertionError('zero-row obstruction is immediate')):
        with pytest.raises(NoLaurentSolution, match='zero constraint'):
            backend.solve(A, bad)
    # The quotient path must retain those zero rows as free summands.
    with pytest.raises(InfiniteLayer):
        FiniteQuotient(A, backend=backend)


def test_recognition_and_kernel_caches_do_not_alias_or_ignore_mutation():
    R, lattice = Ring(2, 2), Lattice(((2, 0), (0, 1)))
    source = Matrix.from_rows(R, [[1, 0]])
    A = lattice.restrict(source)
    backend = RestrictionBackend(GeneralBackend(R), lattice)
    restored = backend.source_matrix(A)
    restored[0, 0] = 0
    assert backend.source_matrix(A) == source
    first = backend.kernel(A)
    bad = backend.kernel(A)
    bad[0, 0] = 1
    assert backend.kernel(A) == first
    lattice.basis = ((3, 0), (0, 1))
    assert backend.lattice.basis == ((2, 0), (0, 1))
    assert backend.source_matrix(A) == source


def test_source_resource_failures_remain_unfinished():
    # The binary positive shortcut can solve this equation independently;
    # use an odd field to exercise propagation from the actual fallback.
    R, lattice = Ring(3, 2), Lattice(((2, 0), (0, 1)))
    source = Matrix.from_rows(R, [[R.variable(0) + 1, R.variable(1) + 1]])
    A = lattice.restrict(source)
    base = GeneralBackend(R)
    backend = RestrictionBackend(base, lattice)
    with patch.object(base, 'solve', side_effect=ResourceLimit('source solve budget')):
        with pytest.raises(ResourceLimit, match='source solve budget'):
            backend.solve(A, A[:, :1])
    with patch.object(base, 'kernel', side_effect=ResourceLimit('source kernel budget')):
        with pytest.raises(ResourceLimit, match='source kernel budget'):
            backend.kernel(A)


def test_actual_source_and_expanded_equations_cannot_be_replaced_by_success_flags():
    R, lattice = Ring(3, 2), Lattice(((2, 0), (0, 1)))
    source = Matrix.from_rows(R, [[R.variable(0) + 1, R.variable(1) + 1]])
    A = lattice.restrict(source)
    base = GeneralBackend(R)
    backend = RestrictionBackend(base, lattice)
    with patch.object(base, 'solve', return_value=zeros(R, 2, 1)):
        with pytest.raises(ArithmeticError, match='Source-ring solution'):
            backend.solve(A, A[:, :1])
    with patch.object(base, 'kernel', return_value=Matrix.from_rows(R, [[1], [0]])):
        with pytest.raises(ArithmeticError, match='source kernel'):
            backend.kernel(A)


def test_unknown_or_incomplete_custom_backends_and_bounded_route_are_not_accepted():
    R, lattice = Ring(3, 2), Lattice(((2, 0), (0, 1)))

    class Incomplete(GeneralBackend):
        def kernel(self, A):
            return zeros(self.ring, A.ncols, 0)

    for base in (Incomplete(R), BoundedLinearBackend(R), object()):
        with pytest.raises(TypeError, match='native practical'):
            RestrictionBackend(base, lattice)
    with pytest.raises(ValueError, match='dimensions'):
        RestrictionBackend(GeneralBackend(Ring(3, 3)), lattice)


def test_empty_shapes_bad_unpack_and_context_selection():
    R, lattice = Ring(5, 2), Lattice(((2, 0), (0, 1)))
    backend = RestrictionBackend(GeneralBackend(R), lattice)
    assert unpack_columns(zeros(R, 0, 3), lattice).shape == (0, 3)
    assert backend.solve(zeros(R, 3, 4), zeros(R, 3, 2)) == zeros(R, 4, 2)
    assert backend.kernel(zeros(R, 3, 4)) == eye(R, 4)
    with pytest.raises(ValueError, match='Packed rows'):
        unpack_columns(eye(R, 1), lattice)
    with pytest.raises(ValueError, match='row counts'):
        backend.solve(eye(R, 2), eye(R, 1))
    A = lattice.restrict(Matrix.from_rows(R, [[1, 0]]))
    with using_backend(backend):
        assert A * solve(A, A[:, :1]) == A[:, :1]
        assert A * kernel(A) == 0


@pytest.mark.parametrize('prime', [2, 3, 5])
def test_interleaved_independent_unit_disks_leave_a_recognized_restricted_code(prime):
    R, lattice = Ring(prime, 2), Lattice(((2, 1), (1, -1)))
    x, y = R.variables
    source = Matrix.from_rows(R, [[x - 1, y - 1]])
    core = lattice.restrict(source)
    weight = 1 if prime == 2 else 2
    units = (weight * x ** -2 * y ** 3, -y ** -1)
    augmented = direct_sum(core, Matrix.from_rows(R, [[units[0], 0], [0, units[1]]]))
    row_order = [3, 0, 4, 1, 2]
    column_order = [0, 6, 1, 2, 7, 3, 4, 5]
    A = augmented[row_order, column_order].vstack(zeros(R, 2, augmented.ncols))
    X = Matrix(R, A.ncols, 4,
               {(i, j): x ** (-(i % 3)) * y ** (j - 2) + i + j
                for i in range(A.ncols) for j in range(4)})
    B = A * X
    base = GeneralBackend(R)
    backend = RestrictionBackend(base, lattice)
    with patch('css_decoupling.algebra.sparse_linear.sparse_binary_solution', return_value=None), \
            patch.object(base, 'solve', wraps=base.solve) as smaller:
        found = backend.solve(A, B)
    assert smaller.call_args.args[0] == source
    assert A * found == B
    for k, unit in enumerate(units):
        row = row_order.index(core.nrows + k)
        coordinate = column_order.index(core.ncols + k)
        assert found[coordinate:coordinate + 1, :] == unit ** -1 * B[row:row + 1, :]
    with patch.object(base, 'kernel', wraps=base.kernel) as smaller_kernel:
        K = backend.kernel(A)
    assert smaller_kernel.call_args.args[0] == source
    assert A * K == 0
    assert not K[[column_order.index(core.ncols), column_order.index(core.ncols + 1)], :]
    reference = GeneralBackend(R).kernel(A)
    independent = GeneralBackend(R)
    assert K * independent.solve(K, reference) == reference
    assert reference * independent.solve(reference, K) == K


def test_only_disks_and_free_columns_need_no_general_module_solve():
    R, lattice = Ring(3, 2), Lattice(((2, 0), (0, 1)))
    x, y = R.variables
    A = Matrix(R, 3, 4, {(0, 1): 2 * x ** -1, (2, 3): y})
    B = Matrix.from_rows(R, [[1 + y, x], [0, 0], [x ** -1, 2]])
    base = GeneralBackend(R)
    backend = RestrictionBackend(base, lattice)
    with patch.object(base, 'solve', side_effect=AssertionError('Independent disks solve explicitly')), \
            patch.object(base, 'kernel', side_effect=AssertionError('Free coordinates form the kernel')):
        X = backend.solve(A, B)
        K = backend.kernel(A)
    assert A * X == B
    assert not X[[0, 2], :]
    assert K == Matrix(R, 4, 2, {(0, 0): 1, (2, 1): 1})


def test_isolated_nonunit_is_not_removed_as_a_disk():
    R, lattice = Ring(3, 2), Lattice(((2, 0), (0, 1)))
    x, y = R.variables
    core = lattice.restrict(Matrix.from_rows(R, [[x - 1, y - 1]]))
    A = direct_sum(core, Matrix.from_rows(R, [[x + 1]]))
    B = Matrix(R, A.nrows, 1, {(A.nrows - 1, 0): 1})
    base = GeneralBackend(R)
    backend = RestrictionBackend(base, lattice)
    with patch('css_decoupling.algebra.sparse_linear.sparse_binary_solution', return_value=None), \
            patch.object(base, 'solve', wraps=base.solve) as fallback:
        with pytest.raises(NoLaurentSolution):
            backend.solve(A, B)
    assert fallback.call_args.args[0] == A


@pytest.mark.parametrize('row_isolated', [False, True])
def test_a_unit_must_be_isolated_in_both_its_row_and_column(row_isolated):
    R, lattice = Ring(5, 2), Lattice(((2, 0), (0, 1)))
    x, y = R.variables
    A = Matrix.from_rows(R, [[1, 0], [x, y + 1]]) if row_isolated else Matrix.from_rows(
        R, [[1, x], [0, y + 1]])
    B = A * Matrix.from_rows(R, [[x ** -1], [2 * y]])
    base = GeneralBackend(R)
    backend = RestrictionBackend(base, lattice)
    with patch('css_decoupling.algebra.sparse_linear.sparse_binary_solution', return_value=None), \
            patch.object(base, 'solve', wraps=base.solve) as fallback:
        X = backend.solve(A, B)
    assert fallback.call_args.args[0] == A
    assert A * X == B
