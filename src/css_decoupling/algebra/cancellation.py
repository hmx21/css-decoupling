"""Cancel all extra free summands using certified row-completion oracles.

The matrix identities here do not themselves solve general Laurent
unimodular-row completion.  Every oracle result is checked, and a failed
completion never returns the still-stabilized module as a free basis of E.
"""
from css_decoupling.algebra.laurent import Matrix, direct_sum, eye, zeros
from css_decoupling.algebra.modules import solve as module_solve
from css_decoupling.errors import BackendUnavailable


class UnimodularCompletionUnavailable(BackendUnavailable):
    """The selected row oracle does not cover this unimodular row."""


def _row_target(ring, size):
    target = zeros(ring, 1, size)
    if size:
        target[0, 0] = 1
    return target


def _check_completion(row, U, Uinv):
    size, ring = row.ncols, row.ring
    if not isinstance(U, Matrix) or not isinstance(Uinv, Matrix):
        raise ValueError("Row completion must return two Laurent matrices")
    if U.ring != ring or Uinv.ring != ring or U.shape != (size, size) or Uinv.shape != (size, size):
        raise ValueError("Row-completion matrices have incompatible rings or shapes")
    if U * Uinv != eye(ring, size) or Uinv * U != eye(ring, size):
        raise ValueError("Row-completion matrices are not two-sided inverses")
    if row * U != _row_target(ring, size):
        raise ValueError("Row completion must satisfy row * U = e_first")


def complete_unimodular_row_basic(row, *, solve=module_solve):
    """Return U,Uinv with row*U=e_first for the exact elementary base cases.

    A row of any length with a Laurent-unit entry is eliminated directly.
    Every unimodular two-entry row is completed by one Bezout solve.  No
    candidate polynomial, column combination, or row subset is enumerated.
    Longer rows without a unit need a stronger completion oracle.
    """
    if not isinstance(row, Matrix) or row.nrows != 1 or row.ncols < 1:
        raise ValueError("Expected a nonempty Laurent row")
    ring, size = row.ring, row.ncols
    pivot = next((j for j in range(size) if row[0, j].is_unit), None)
    if pivot is not None:
        permutation = eye(ring, size)
        if pivot:
            permutation[0, 0], permutation[pivot, pivot] = 0, 0
            permutation[0, pivot], permutation[pivot, 0] = 1, 1
        changed = row * permutation
        unit_inverse = changed[0, 0] ** -1
        eliminate, restore = eye(ring, size), eye(ring, size)
        eliminate[0, 0], restore[0, 0] = unit_inverse, changed[0, 0]
        for j in range(1, size):
            eliminate[0, j] = -unit_inverse * changed[0, j]
            restore[0, j] = changed[0, j]
        U, Uinv = permutation * eliminate, restore * permutation
    elif size == 2:
        bezout = solve(row, eye(ring, 1))
        if bezout.shape != (2, 1) or row * bezout != eye(ring, 1):
            raise ValueError("Bezout solver did not produce a right inverse")
        a, b, x, y = row[0, 0], row[0, 1], bezout[0, 0], bezout[1, 0]
        U = Matrix.from_rows(ring, [[x, -b], [y, a]])
        Uinv = Matrix.from_rows(ring, [[a, b], [-y, x]])
    elif size == 1:
        # A one-entry unimodular row over a Laurent polynomial ring is a
        # Laurent unit.  The exact solve reports nonmembership on bad input.
        solve(row, eye(ring, 1))
        raise ArithmeticError("A nonunit Laurent scalar unexpectedly had an inverse")
    else:
        raise UnimodularCompletionUnavailable(
            "A Laurent row of length greater than two without a unit entry "
            "requires a general unimodular-row completion oracle")
    _check_completion(row, U, Uinv)
    return U, Uinv


def _cancel_last_direction(E, A, B, extra, U, Uinv):
    """Remove the last extra free direction of a checked stable frame.

    ``A*B=I_m`` and ``B*A=diag(E,I_extra)``; ``(U,Uinv)`` completes the row
    of B belonging to that direction.  Return the smaller frame and a record.
    """
    ring, m, ambient = E.ring, A.nrows, B.nrows
    u, v = A[:, ambient - 1:ambient], B[ambient - 1:ambient, :]
    if v * u != eye(ring, 1):
        raise ArithmeticError("Extra summand did not give a unimodular pair")
    _check_completion(v, U, Uinv)
    w = Uinv * u
    if w[0, 0] != 1:
        raise ArithmeticError("Completed unimodular pair has incorrect first coordinate")
    # w=(1,c).  L e_first=w, e_first^T L=e_first^T, so W=U L
    # simultaneously satisfies W e_first=u and v W=e_first^T.
    L, Linv = eye(ring, m), eye(ring, m)
    for i in range(1, m):
        L[i, 0], Linv[i, 0] = w[i, 0], -w[i, 0]
    W, Winv = U * L, Linv * Uinv
    first = _row_target(ring, m)
    if W * first.T != u or v * W != first or W * Winv != eye(ring, m) or Winv * W != eye(ring, m):
        raise ArithmeticError("Unimodular pair alignment failed")
    aligned_A, aligned_B = Winv * A, B * W
    last = zeros(ring, 1, ambient)
    last[0, ambient - 1] = 1
    if (aligned_A[:1, :] != last or aligned_A[:, ambient - 1:ambient] != first.T or
            aligned_B[:, :1] != last.T or aligned_B[ambient - 1:ambient, :] != first):
        raise ArithmeticError("Aligned free direction is not an isolated identity block")
    next_A, next_B = aligned_A[1:m, :ambient - 1], aligned_B[:ambient - 1, 1:m]
    if (next_A * next_B != eye(ring, m - 1) or
            next_B * next_A != direct_sum(E, eye(ring, extra - 1))):
        raise ArithmeticError("Cancellation did not preserve the smaller stable-frame identities")
    return next_A, next_B, {"extra_before": extra, "extra_after": extra - 1,
                            "row": v, "dual_column": u,
                            "completion": U, "completion_inverse": Uinv,
                            "alignment": W, "alignment_inverse": Winv}


def cancel_stable_frame(E, A, B, complete_row=complete_unimodular_row_basic):
    """Cancel a checked stable frame completely, returning a basis of im(E).

    Inputs satisfy ``A*B=I_m`` and ``B*A=diag(E,I_s)``, with shapes
    A: m x (n+s), B: (n+s) x m, E: n x n.  ``complete_row(v)`` must return
    (U,Uinv) with v*U=e_first and U*Uinv=Uinv*U=I.  The callback can be any
    exact general algorithm; these identities are always checked independently.

    On success the returned P,Q satisfy P*Q=E and Q*P=I_(m-s).  Every extra
    direction has been removed.  Failure propagates without returning a
    larger module or modifying the caller's matrices.
    """
    if not isinstance(E, Matrix) or E.nrows != E.ncols or E * E != E:
        raise ValueError("Expected an idempotent Laurent matrix E")
    if not isinstance(A, Matrix) or not isinstance(B, Matrix):
        raise ValueError("Stable frame must consist of Laurent matrices")
    ring, n = E.ring, E.nrows
    if A.ring != ring or B.ring != ring or A.ncols != B.nrows or A.nrows != B.ncols:
        raise ValueError("Stable-frame rings or shapes are incompatible")
    extra, m = B.nrows - n, A.nrows
    if extra < 0 or m < extra:
        raise ValueError("Stable frame has an invalid number of extra free directions")
    if A * B != eye(ring, m) or B * A != direct_sum(E, eye(ring, extra)):
        raise ValueError("Stable-frame identities do not hold")
    if not callable(complete_row):
        raise ValueError("Row completion oracle must be callable")
    rank, initial_extra = m - extra, extra
    A, B, steps = A.copy(), B.copy(), []
    while extra:
        ambient = B.nrows
        u, v = A[:, ambient - 1:ambient], B[ambient - 1:ambient, :]
        if v * u != eye(ring, 1):
            raise ArithmeticError("Extra summand did not give a unimodular pair")
        answer = complete_row(v.copy())
        if not isinstance(answer, (tuple, list)) or len(answer) != 2:
            raise ValueError("Row completion oracle must return (U,Uinv)")
        A, B, step = _cancel_last_direction(E, A, B, extra, *answer)
        steps.append(step)
        extra -= 1
    P, Q = B, A
    if P.shape != (n, rank) or Q.shape != (rank, n) or P * Q != E or Q * P != eye(ring, rank):
        raise ArithmeticError("Final cancellation did not produce a basis of the original summand")
    return {"basis": P, "coordinates": Q, "P": P, "Q": Q, "rank": rank,
            "from_free": P, "to_free": Q, "odd_free_rank": 0,
            "even_free_rank": rank, "cancelled_free_rank": initial_extra,
            "method": "complete-stable-frame-cancellation", "steps": steps}


def cancel_unit_directions(E, A, B):
    """Cancel every extra free direction whose frame row has a Laurent unit.

    The frame satisfies ``A*B=I_m`` and ``B*A=diag(E,I_s)`` as in
    ``cancel_stable_frame``.  Directions whose row of B contains a unit are
    removed one at a time with the elementary completion; no general
    unimodular-row oracle or Quillen-Suslin step is used, and the remaining
    directions stay in the frame.  Among eligible directions, the sparsest
    row is taken first.  Return the smaller frame and the cancelled
    directions, as indices into the original extra coordinates.
    """
    if not isinstance(E, Matrix) or E.nrows != E.ncols or E * E != E:
        raise ValueError("Expected an idempotent Laurent matrix E")
    ring, n = E.ring, E.nrows
    extra = B.nrows - n
    if A * B != eye(ring, A.nrows) or B * A != direct_sum(E, eye(ring, extra)):
        raise ValueError("Stable-frame identities do not hold")
    A, B, labels, cancelled = A.copy(), B.copy(), list(range(extra)), []
    while extra:
        weights = {}
        for (i, j), value in B.entries.items():
            if i >= n:
                count, unit = weights.get(i, (0, False))
                weights[i] = count + 1, unit or value.is_unit
        eligible = [(count, i) for i, (count, unit) in weights.items() if unit]
        if not eligible:
            break
        _, row = min(eligible)
        last = n + extra - 1
        if row != last:
            # Extra directions are interchangeable: swap them in both maps.
            order = list(range(B.nrows))
            order[row], order[last] = last, row
            A, B = A[:, order], B[order, :]
            labels[row - n], labels[last - n] = labels[last - n], labels[row - n]
        U, Uinv = complete_unimodular_row_basic(B[last:last + 1, :])
        A, B, _ = _cancel_last_direction(E, A, B, extra, U, Uinv)
        cancelled.append(labels.pop())
        extra -= 1
    return A, B, cancelled
