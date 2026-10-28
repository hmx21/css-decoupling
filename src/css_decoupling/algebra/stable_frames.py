"""Exact stable frames for an idempotent image from a finite free resolution.

A stable frame is intermediate algebraic data. An actual free basis of the
original image requires cancelling every auxiliary free direction afterward.
"""
from css_decoupling.algebra.laurent import block, direct_sum, eye, zeros


def stably_trivialize_idempotent(E, resolution, solve):
    """Return explicit im(E) + F_odd <-> F_even stable trivializations.

    ``resolution`` is a nonempty finite list [q,d1,...,dn], where
    q:F0 -> ambient(E), im(q)=im(E), and di:Fi -> F(i-1).  No claim of
    exactness is trusted: a split contraction is constructed with Laurent
    solves and all identities are checked.  The matrix representing im(E)
    in these formulas uses its ambient coordinates and identity E.
    """
    ring = E.ring
    if E.nrows != E.ncols or E * E != E:
        raise ValueError("The projective presentation must be an idempotent")
    differentials = list(resolution)
    if not differentials or differentials[0].nrows != E.nrows:
        raise ValueError("A resolution must start with F0 -> ambient(E)")
    if any(d.ring != ring for d in differentials):
        raise ValueError("Resolution matrices belong to another ring")
    if E * differentials[0] != differentials[0]:
        raise ValueError("Resolution image is not contained in the idempotent image")
    ranks = [d.ncols for d in differentials]
    for i in range(1, len(differentials)):
        if differentials[i].nrows != ranks[i - 1]:
            raise ValueError("Resolution differential has wrong dimensions")
        if differentials[i - 1] * differentials[i] != zeros(
                ring, differentials[i - 1].nrows, differentials[i].ncols):
            raise ValueError("Resolution does not form a complex")

    h = {-1: solve(differentials[0], E) * E}
    if differentials[0] * h[-1] != E:
        raise ValueError("Resolution generators do not span im(E)")
    for i, d in enumerate(differentials):
        rhs = eye(ring, ranks[i]) - h[i - 1] * d
        next_d = differentials[i + 1] if i + 1 < len(differentials) else zeros(ring, ranks[i], 0)
        h[i] = solve(next_d, rhs)
        if next_d * h[i] != rhs:
            raise ValueError(f"Resolution is not split exact at F_{i}")

    # h d h is a contraction with square zero.  This normalization makes
    # d+h an involution, so its two parity blocks are explicit inverses.
    normalized = {-1: h[-1] * differentials[0] * h[-1]}
    for i in range(len(differentials)):
        next_d = differentials[i + 1] if i + 1 < len(differentials) else zeros(ring, ranks[i], 0)
        normalized[i] = h[i] * next_d * h[i]
    h = normalized
    for i, d in enumerate(differentials):
        next_d = differentials[i + 1] if i + 1 < len(differentials) else zeros(ring, ranks[i], 0)
        if next_d * h[i] + h[i - 1] * d != eye(ring, ranks[i]):
            raise ValueError("Normalized contraction failed")
        if h[i] * h[i - 1] != zeros(ring, h[i].nrows, h[i - 1].ncols):
            raise ValueError("Normalized contraction is not square zero")

    even = list(range(0, len(ranks), 2))
    odd = [-1] + list(range(1, len(ranks), 2))
    augmented_ranks = {-1: E.nrows, **dict(enumerate(ranks))}

    def entry(target, source):
        if target == source - 1:
            return differentials[source]
        if target == source + 1:
            return h[source]
        return zeros(ring, augmented_ranks[target], augmented_ranks[source])

    to_free = block([[entry(i, j) for j in odd] for i in even])
    from_free = block([[entry(i, j) for j in even] for i in odd])
    odd_free_rank = sum(ranks[i] for i in odd if i >= 0)
    even_free_rank = sum(ranks[i] for i in even)
    if to_free * from_free != eye(ring, even_free_rank):
        raise ValueError("Stable projective trivialization has no right inverse")
    if from_free * to_free != direct_sum(E, eye(ring, odd_free_rank)):
        raise ValueError("Stable projective trivialization has no left inverse")
    return dict(to_free=to_free, from_free=from_free,
                odd_free_rank=odd_free_rank, even_free_rank=even_free_rank,
                resolution=differentials, contraction=h)
