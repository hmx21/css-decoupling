"""Exact compact coordinates for a two-dimensional retained translation lattice.

Automatic selection uses a unimodular Gauss column reduction and monomial
representatives in the selected basis's fundamental parallelogram. It does
not change the translation subgroup. Explicit user bases are handled by the
caller and are never reduced here implicitly. Legacy lattice metadata keeps
its original HNF interpretation; explicit representative metadata is validated
as a complete set of integer cosets before it is used.
"""
from numbers import Integral

from sympy import Matrix as IntegerMatrix

from .algebra import Matrix, Lattice, ResourceLimit, eye


_GEOMETRY_FIELDS = {'schema', 'selection', 'basis_algorithm', 'representative_algorithm',
                    'full_kernel_basis_columns', 'basis_change', 'inverse_basis_change', 'coset_gauge'}


def _integer_square(rows, size=2):
    rows = tuple(tuple(row) for row in rows)
    if (len(rows) != size or any(len(row) != size for row in rows) or
            any(isinstance(value, bool) or not isinstance(value, Integral) for row in rows for value in row)):
        raise ValueError('Expected a square integer coordinate matrix')
    return IntegerMatrix([[int(value) for value in row] for row in rows])


def _rows(matrix):
    return tuple(tuple(int(matrix[i, j]) for j in range(matrix.cols)) for i in range(matrix.rows))


def gauss_reduce_columns(basis, *, max_steps=10000):
    """Return B,U with B=input*U and det(U)=+-1, by exact integer arithmetic."""
    if type(max_steps) is not int or max_steps < 1:
        raise ValueError('Gauss step cap must be a positive integer')
    original = _integer_square(basis)
    if not original.det():
        raise ValueError('Lattice basis must have full rank')
    B, U, steps = original.copy(), IntegerMatrix.eye(2), 0
    while True:
        first, second = B[:, 0], B[:, 1]
        norm_first, norm_second = int(first.dot(first)), int(second.dot(second))
        if norm_second < norm_first:
            B.col_swap(0, 1)
            U.col_swap(0, 1)
            first, second = B[:, 0], B[:, 1]
            norm_first = int(first.dot(first))
        product = int(first.dot(second))
        multiple = (2*product+norm_first)//(2*norm_first)
        if not multiple:
            break
        if steps >= max_steps:
            raise ResourceLimit('Exact Gauss reduction step budget')
        B[:, 1] -= multiple*B[:, 0]
        U[:, 1] -= multiple*U[:, 0]
        steps += 1
    if B.det() < 0:
        B[:, 1] = -B[:, 1]
        U[:, 1] = -U[:, 1]
    if original*U != B or abs(int(U.det())) != 1:
        raise ArithmeticError('Gauss reduction lost its unimodular witness')
    return _rows(B), _rows(U)


def parallelogram_representatives(basis, *, max_index=4096):
    """Integer coset representatives r-B floor(B^-1 r), in legacy HNF order."""
    legacy = Lattice(basis, max_index=max_index)
    if legacy.D != 2:
        raise ValueError('Compact cell construction is two-dimensional')
    answer = []
    for representative in legacy.representatives:
        coordinates = legacy.inverse*IntegerMatrix(representative)
        floor = IntegerMatrix([int(value.p//value.q) for value in coordinates])
        answer.append(tuple(map(int, IntegerMatrix(representative)-legacy.B*floor)))
    return tuple(answer)


def lattice_from_metadata(metadata, *, max_index=None):
    """Reconstruct declared coordinates, rejecting unknown/altered metadata.

    Missing representative_strategy means the unchanged legacy HNF strategy.
    explicit-v1 accepts exactly one integer representative of each coset,
    including arbitrary ordering and retained-lattice shifts. This helper
    validates coordinates only; pipeline certificates declaring explicit-v1
    additionally require their checked automatic compact-cell proof.
    """
    if not isinstance(metadata, dict):
        raise ValueError('Lattice metadata must be an object')
    base_fields = {'basis_columns', 'index', 'representatives'}
    strategy = metadata.get('representative_strategy')
    expected = base_fields if 'representative_strategy' not in metadata else base_fields | {'representative_strategy'}
    if set(metadata) != expected or (strategy is not None and strategy != 'explicit-v1'):
        raise ValueError('Unknown lattice representative metadata')
    if 'representative_strategy' in metadata and strategy != 'explicit-v1':
        raise ValueError('An explicit representative strategy must be named')
    if type(metadata['index']) is not int or metadata['index'] < 1:
        raise ValueError('Lattice index metadata must be a positive integer')
    saved = metadata['representatives']
    if not isinstance(saved, (list, tuple)) or any(
            not isinstance(row, (list, tuple)) or any(type(value) is not int for value in row)
            for row in saved):
        raise ValueError('Lattice representative metadata must contain integer vectors')
    result = Lattice(metadata['basis_columns'], max_index=max_index,
                     representatives=saved if strategy == 'explicit-v1' else None)
    if result.index != metadata['index'] or result.representatives != tuple(tuple(row) for row in saved):
        raise ValueError('Altered lattice index or representatives')
    return result


def coset_gauge(lattice, ring):
    """Map selected coset coordinates to legacy HNF coordinates at the same B."""
    if ring.D != lattice.D:
        raise ValueError('Coset gauge ring/lattice dimensions differ')
    legacy = Lattice(lattice.basis, max_index=None)
    entries = {}
    for column, representative in enumerate(lattice.representatives):
        row, shift = legacy.reduce(representative)
        entries[row, column] = ring.monomial(shift)
    gauge = Matrix(ring, lattice.index, lattice.index, entries)
    if gauge.dagger()*gauge != eye(ring, lattice.index) or gauge*gauge.dagger() != eye(ring, lattice.index):
        raise ArithmeticError('Coset coordinate change is not unitary')
    return gauge


def change_variables(matrix, inverse_basis_change):
    """Apply a unimodular coarse-variable substitution without changing shape."""
    U = _integer_square(inverse_basis_change, matrix.ring.D)
    if abs(int(U.det())) != 1:
        raise ValueError('Coarse variable change must be unimodular')
    rows = _rows(U)
    return Matrix(matrix.ring, matrix.nrows, matrix.ncols,
                  {ij: matrix.ring({tuple(sum(rows[i][j]*exponent[j] for j in range(matrix.ring.D))
                                         for i in range(matrix.ring.D)): coefficient
                                    for exponent, coefficient in polynomial.terms.items()})
                   for ij, polynomial in matrix.entries.items()})


def _repeat_gauge(gauge, components):
    n = gauge.nrows
    return Matrix(gauge.ring, components*n, components*n,
                  {(component*n+i, component*n+j): value for component in range(components)
                   for (i, j), value in gauge.entries.items()})


def choose_compact_cell(full, ring, *, max_index=4096):
    """Choose a compact basis of exactly the full lattice, with explicit proof."""
    if not isinstance(full, Lattice) or full.D != 2 or ring.D != 2:
        raise ValueError('Automatic compact cells require a two-dimensional lattice/ring')
    basis, U = gauss_reduce_columns(full.basis)
    selected = Lattice(basis, max_index=max_index,
                       representatives=parallelogram_representatives(basis, max_index=max_index))
    U_inverse = _rows(IntegerMatrix(U).inv())
    proof = dict(schema='css-decoupling-2d-cell-geometry/v1', selection='automatic_full_kernel',
                 basis_algorithm='gauss-2d-v1', representative_algorithm='fundamental-parallelogram-v1',
                 full_kernel_basis_columns=[list(row) for row in full.basis],
                 basis_change=[list(row) for row in U], inverse_basis_change=[list(row) for row in U_inverse],
                 coset_gauge=coset_gauge(selected, ring))
    verify_cell_geometry(full, selected, proof, ring)
    return selected, proof


def verify_cell_geometry(full, selected, proof, ring, *, checks=()):
    """Check subgroup equality, basis units, representatives and optional gauges.

    checks contains (fine_matrix, selected_restriction) pairs, normally after
    the caller's expansion resource preflights. Equivalent integer bases of
    the same full kernel are accepted through actual unimodular identities;
    no byte equality with the kernel's chosen basis is assumed.
    """
    if not isinstance(proof, dict) or set(proof) != _GEOMETRY_FIELDS:
        raise ValueError('Unexpected compact-cell proof fields')
    if (proof['schema'] != 'css-decoupling-2d-cell-geometry/v1' or
            proof['selection'] != 'automatic_full_kernel' or
            proof['basis_algorithm'] != 'gauss-2d-v1' or
            proof['representative_algorithm'] != 'fundamental-parallelogram-v1'):
        raise ValueError('Unknown compact-cell proof strategy')
    if full.D != 2 or selected.D != 2 or ring.D != 2:
        raise ValueError('Compact-cell proof dimension mismatch')
    original = Lattice(proof['full_kernel_basis_columns'], max_index=None)
    U, U_inverse = _integer_square(proof['basis_change']), _integer_square(proof['inverse_basis_change'])
    if (abs(int(U.det())) != 1 or U*U_inverse != IntegerMatrix.eye(2) or
            U_inverse*U != IntegerMatrix.eye(2) or original.B*U != selected.B):
        raise ValueError('Compact-cell unimodular basis identities failed')
    if not (full.contains(original) and original.contains(full) and
            full.contains(selected) and selected.contains(full)):
        raise ValueError('Automatic compact cell changed the full translation subgroup')
    first, second = selected.B[:, 0], selected.B[:, 1]
    norm_first, norm_second = int(first.dot(first)), int(second.dot(second))
    if (selected.B.det() <= 0 or norm_first > norm_second or
            2*abs(int(first.dot(second))) > norm_first):
        raise ValueError('Claimed Gauss basis is not reduced with positive orientation')
    expected_reps = parallelogram_representatives(selected.basis, max_index=None)
    if selected.representatives != expected_reps:
        raise ValueError('Compact-cell representatives do not follow the declared algorithm')
    expected_gauge = coset_gauge(selected, ring)
    if proof['coset_gauge'] != expected_gauge:
        raise ValueError('Compact-cell monomial gauge witness failed')
    for fine, coarse in checks:
        if fine.ring != ring or coarse.ring != ring:
            raise ValueError('Compact-cell check ring mismatch')
        changed = change_variables(original.restrict(fine), _rows(U_inverse))
        left = _repeat_gauge(expected_gauge, fine.nrows)
        right = _repeat_gauge(expected_gauge, fine.ncols)
        if coarse != left.dagger()*changed*right:
            raise ValueError('Compact-cell coarse variable/gauge identity failed')
    return True
