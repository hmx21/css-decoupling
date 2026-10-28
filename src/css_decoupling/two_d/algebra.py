"""Exact native algebra shared by the independent two-dimensional algorithms.

These are the package's shared algebra modules, re-exported for the
two-dimensional code. They depend only on CPython and SymPy; no external CAS
is imported or invoked.
"""
from ..algebra.laurent import (
    Ring, Poly, Matrix, eye, zeros, hstack, vstack, block, direct_sum,
)
from ..algebra.modules import GeneralBackend, solve, kernel, using_backend
from ..algebra.quillen_suslin import complete_laurent_row
from ..algebra.schreyer import (
    polynomial_kernel, verify_polynomial_kernel, polynomial_image_resolution,
)
from ..topology.cohomology import FiniteQuotient
from ..lattice.retained import Lattice
from ..lattice.kernel import translation_kernel
from ..certificates.translation_kernel import verify_translation_kernel
from ..errors import ResourceLimit, NoLaurentSolution, InfiniteLayer

block_diag = direct_sum


def inverse(A):
    """Return an actual two-sided Laurent inverse, not a fraction-field inverse."""
    if A.nrows != A.ncols:
        raise ValueError('A Laurent inverse requires a square matrix')
    result = solve(A, eye(A.ring, A.nrows))
    # solve's exact postcondition already checked A*result=I; check the other
    # side here rather than recomputing the same large product.
    if result * A != eye(A.ring, A.nrows):
        raise ArithmeticError('Laurent inverse failed its two exact identities')
    return result


def _split_injection(J, *, qs_options=None):
    """Find W with W*im(J) the first coordinate summand, retaining W inverse.

    The explicit left inverse proves each successive residual column is
    unimodular. QS completes that column's transposed row. Column subtractions
    only change the basis of im(J); no choice by fraction-field rank is made.
    """
    r, n = J.ncols, J.nrows
    if r > n:
        raise ValueError('A split injection cannot have more columns than rows')
    left = solve(J.T, eye(J.ring, r)).T
    if left * J != eye(J.ring, r):
        raise ArithmeticError('Terminal injection has no checked left inverse')
    W, Winv, current = eye(J.ring, n), eye(J.ring, n), J.copy()
    for k in range(r):
        U, V = complete_laurent_row(current[k:, k].T, **(qs_options or {}))
        step, step_inverse = direct_sum(eye(J.ring, k), U.T), direct_sum(eye(J.ring, k), V.T)
        current = step * current
        W, Winv = step * W, Winv * step_inverse
        for j in range(k + 1, r):
            coefficient = current[k, j]
            if coefficient:
                for i in range(n):
                    current[i, j] -= current[i, k] * coefficient
    canonical = eye(J.ring, r).vstack(zeros(J.ring, n - r, r))
    if current != canonical or W * Winv != eye(J.ring, n) or Winv * W != eye(J.ring, n):
        raise ArithmeticError('Injection completion failed its actual inverse identities')
    transformed = W * J
    if transformed[r:, :]:
        raise ArithmeticError('Injection completion did not isolate its image')
    inverse(transformed[:r, :])
    return W, Winv, left


def independent_columns(A, *, timeout=120, max_basis=100000,
                        max_terms=2000000, max_pairs=1000000, qs_options=None):
    """Construct a free basis for im(A), with exact two-way generator witnesses.

    This is a full-resolution/cancellation algorithm, not selection of a maximal
    fraction-field independent subset. It succeeds for free images. A genuinely
    nonsplit last injection raises ``NoLaurentSolution``; budget exhaustion is
    ``ResourceLimit`` and does not imply nonfreeness. In the 2D CSS pipeline the
    required freeness must follow from, and be checked with, its input hypotheses.
    """
    if not isinstance(A, Matrix):
        raise TypeError('Expected a Laurent Matrix')
    if not A:
        return dict(basis=zeros(A.ring, A.nrows, 0), basis_in_input=zeros(A.ring, A.ncols, 0),
                    input_in_basis=zeros(A.ring, 0, A.ncols), resolution_ranks=(0,), cancellations=())
    polynomial = A.copy()
    for j, column in enumerate(A.columns()):
        support = [exponent for value in column for exponent in value.terms]
        shift = tuple(max(0, -min((e[k] for e in support), default=0)) for k in range(A.ring.D))
        unit = A.ring.monomial(shift)
        for i, value in enumerate(column):
            polynomial[i, j] = value * unit
    resolution = polynomial_image_resolution(
        polynomial, timeout=timeout, max_basis=max_basis, max_terms=max_terms,
        max_pairs=max_pairs, return_image_witness=True)
    maps = list(resolution['maps'])
    ranks = tuple(M.ncols for M in maps)
    cancellations = []
    # Check completeness over Laurent R before manipulating the resolution.
    for previous, following in zip(maps, maps[1:]):
        if previous * following:
            raise ArithmeticError('Native free resolution is not a complex')
        solve(following, kernel(previous))
    if kernel(maps[-1]):
        raise ArithmeticError('Native free resolution did not end in an injection')
    while len(maps) > 1:
        J, previous = maps[-1], maps[-2]
        W, Winv, left = _split_injection(J, qs_options=qs_options)
        transformed = previous * Winv
        if transformed[:, :J.ncols]:
            raise ArithmeticError('Cancellation failed to isolate a zero block')
        reduced = transformed[:, J.ncols:]
        if kernel(reduced):
            raise ArithmeticError('Terminal cancellation did not preserve injectivity')
        cancellations.append({'injection': J, 'left_inverse': left,
                              'coordinate_map': W, 'coordinate_inverse': Winv})
        maps[-2:] = [reduced]
    basis = maps[0]
    if kernel(basis):
        raise ArithmeticError('Final image generators are not a free basis')
    T, S = solve(A, basis), solve(basis, A)
    if A * T != basis or basis * S != A:
        raise ArithmeticError('Original checks failed both image-generation identities')
    return dict(basis=basis, basis_in_input=T, input_in_basis=S,
                resolution_ranks=ranks, cancellations=tuple(cancellations))


__all__ = [
    'Ring', 'Poly', 'Matrix', 'eye', 'zeros', 'hstack', 'vstack', 'block',
    'direct_sum', 'block_diag', 'GeneralBackend', 'solve', 'kernel', 'using_backend',
    'inverse', 'independent_columns', 'FiniteQuotient',
    'Lattice', 'translation_kernel', 'verify_translation_kernel',
    'complete_laurent_row', 'polynomial_kernel', 'verify_polynomial_kernel',
    'ResourceLimit', 'NoLaurentSolution', 'InfiniteLayer',
]
