"""Appendix C4 stable complement from the comparison's bounded mapping cone.

No free basis of the projective complement is computed.  The existing
complexes supply every free term; the only algebraic operations requested
from a backend are Laurent module equations A X = B.
"""
from __future__ import annotations
import logging

from css_decoupling.algebra.laurent import Matrix, block, direct_sum, eye, zeros
from css_decoupling.topology.complexes import (
    CochainComplex, CochainMap, identity_map, mapping_cone, validate_homotopy,
)

log = logging.getLogger(__name__)


def _product_work(A, B):
    """Number of scalar monomial pairs, without enumerating their sums."""
    rows = {}
    for (i, _), value in B.entries.items():
        rows[i] = rows.get(i, 0) + len(value.terms)
    return sum(len(value.terms) * rows.get(j, 0)
               for (_, j), value in A.entries.items())


def _contracting_sandwich(H, d):
    """Compute exact H*d*H using the cheaper actual intermediate.

    The work estimate selects an association; it is not a product budget.
    Both intermediates involve the sparse differential. Their actual
    supports include cancellations before the last multiplication.
    """
    left, right = H * d, d * H
    if left == eye(H.ring, H.nrows) or right == eye(H.ring, H.ncols):
        return H.copy()
    return left * H if _product_work(left, H) <= _product_work(H, right) else H * right


def _vertical(A, B):
    if A.ring != B.ring or A.ncols != B.ncols:
        raise ValueError("Incompatible vertical blocks")
    return Matrix(A.ring, A.nrows + B.nrows, A.ncols,
                  {**A.entries, **{(A.nrows + i, j): v for (i, j), v in B.entries.items()}})


def _horizontal(A, B):
    if A.ring != B.ring or A.nrows != B.nrows:
        raise ValueError("Incompatible horizontal blocks")
    return Matrix(A.ring, A.nrows, A.ncols + B.ncols,
                  {**A.entries, **{(i, A.ncols + j): v for (i, j), v in B.entries.items()}})


def _homotopy_component(C, h, n):
    return h.get(n, zeros(C.ring, C.rank(n - 1), C.rank(n)))


def _validate_comparison(phi, psi, hC, hT):
    C, T = phi.target, phi.source
    if psi.source != C or psi.target != T:
        raise ValueError("Comparison maps have incompatible complexes")
    C.validate()
    T.validate()
    phi.validate()
    psi.validate()
    for X, h in ((C, hC), (T, hT)):
        for n, value in h.items():
            if (not isinstance(n, int) or isinstance(n, bool) or value.ring != X.ring
                    or value.shape != (X.rank(n - 1), X.rank(n))):
                raise ValueError("Malformed full-complex homotopy")
    validate_homotopy(C, C, identity_map(C), CochainMap(C, C, {
        n: phi.at(n) * psi.at(n) for n in C.degrees}), hC)
    validate_homotopy(T, T, identity_map(T), CochainMap(T, T, {
        n: psi.at(n) * phi.at(n) for n in T.degrees}), hT)
    return C, T


def _contract_around_projector(C, projectors, solve):
    """Contract I-P via an exact unit-disk retract, preserving the given P.

    P has zero differential on both sides. If 1-ip=dH+Hd and the residual
    contraction is 1-pPi, H+i*h*p contracts 1-ipPip. Subtracting H*P and
    ip*P*H corrects this to the original 1-P; it does not change the saved
    homology section/projection. All identities are checked in C afterwards.
    """
    from css_decoupling.topology.contractions import cancel_unit_disks
    reduction = cancel_unit_disks(C)
    E, inclusion, projection = reduction.residual, reduction.inclusion, reduction.projection
    small_P = {}
    for n in C.degrees:
        P = projectors[n]
        if P.shape != (C.rank(n), C.rank(n)) or P.ring != C.ring or (
                P * P != P or C.d(n) * P or P * C.d(n - 1)):
            raise ValueError("Invalid homology projector for cone splitting")
        small_P[n] = projection.at(n) * P * inclusion.at(n)
        if small_P[n] * small_P[n] != small_P[n]:
            raise ArithmeticError("Transferred cone homology projector is not idempotent")
    small_h = {}
    for n in reversed(E.degrees):
        next_h = small_h.get(n + 1, zeros(C.ring, E.rank(n), E.rank(n + 1)))
        rhs = eye(C.ring, E.rank(n)) - small_P[n] - next_h * E.d(n)
        small_h[n] = solve(E.d(n - 1), rhs)
        if E.d(n - 1) * small_h[n] != rhs:
            raise ValueError("Residual cone splitting failed its exact identity")
    h = {}
    for n in C.degrees:
        H = reduction.homotopy[n]
        previous_P = projectors.get(n - 1, zeros(C.ring, C.rank(n - 1), C.rank(n - 1)))
        h[n] = (H + inclusion.at(n - 1) * small_h[n] * projection.at(n)
                - H * projectors[n]
                - inclusion.at(n - 1) * projection.at(n - 1) * previous_P * H)
    for n in C.degrees:
        next_h = h.get(n + 1, zeros(C.ring, C.rank(n), C.rank(n + 1)))
        if C.d(n - 1) * h[n] + next_h * C.d(n) != eye(C.ring, C.rank(n)) - projectors[n]:
            raise ValueError("Pulled-back cone splitting changed the homology projector")
    return h


def stable_complement_from_cone(phi, psi, hC, hT, solve, *, comparison_contraction=None,
                                strategy="normalized"):
    """Return Q + R^k <-> R^a from Cone(rho), including split witnesses.

    ``phi:T->C`` and ``psi:C->T`` are maps of complete bounded complexes,
    with ``I-phi*psi=d*hC+hC*d`` and its T analogue in every degree.
    The input comparison supplies the degree <=1 resolutions used by C4.
    Q is represented as the image of an explicit idempotent in T^2+C^2.
    Free summands are *retained*: k is the total odd-degree cone rank,
    including degree one, and a is its total even-degree rank.

    ``normalized`` uses an exact weak retract followed by hdh, or normalizes
    and transfers the supplied full Gamma contraction. ``direct`` solves
    the strong section equations on the original cone. Both check the same
    final strong identities and parity frame. Normalized products have no
    direct solver's fixed support/product admission caps: backend resource
    failures propagate, and external time/memory supervision is separate.
    """
    if strategy not in ("normalized", "direct"):
        raise ValueError("Schanuel strategy must be 'normalized' or 'direct'")
    C, T = _validate_comparison(phi, psi, hC, hT)
    R = C.ring
    f, g = phi.at(1), psi.at(1)
    x, xp = C.d(1), T.d(1)
    z, zp = _homotopy_component(C, hC, 2), _homotopy_component(T, hT, 2)
    inclusion = _vertical(g, x)
    rho_top = _horizontal(f, z)
    j = direct_sum(xp, eye(R, C.rank(2)))
    v = block([[zp, -g * z],
               [zeros(R, C.rank(2), T.rank(2)), eye(R, C.rank(2))]])
    projection = eye(R, T.rank(1) + C.rank(2)) - inclusion * rho_top
    complement_inclusion = projection * v
    E = j * complement_inclusion
    coordinates = j * projection
    if E * E != E or E * coordinates != coordinates:
        raise ValueError("C4 complement formulas did not yield an idempotent image")

    # rho is already lifted in all negative degrees by phi: the extra
    # C^2 summand is isolated at degree 1 and never requires a Q resolution.
    low = min((0, *C.degrees, *T.degrees))
    degrees = range(low, 2)
    A = CochainComplex(R, {n: C.rank(n) for n in degrees},
                       {n: C.d(n) for n in range(low, 1)})
    B = CochainComplex(R, {n: T.rank(n) + (C.rank(2) if n == 1 else 0)
                          for n in degrees},
                       {n: (_vertical(T.d(0), zeros(R, C.rank(2), T.rank(0)))
                            if n == 0 else T.d(n)) for n in range(low, 1)})
    rho = CochainMap(B, A, {n: rho_top if n == 1 else phi.at(n) for n in degrees})
    cone = mapping_cone(rho)

    # q and s exhibit H^0(cone) as im(E), without choosing a projective basis.
    q0 = _horizontal(zeros(R, E.nrows, C.rank(0)), coordinates)
    lift = (solve(C.d(0), rho_top * complement_inclusion) if comparison_contraction is None else
            _homotopy_component(C, hC, 1) * rho_top * v)
    if C.d(0) * lift != rho_top * complement_inclusion:
        raise ValueError("Invalid complement cycle lift")
    s0 = _vertical(-lift, complement_inclusion) * E
    if (q0 * s0 != E or cone.d(0) * s0
            or q0 * cone.d(-1) or E * q0 != q0 or s0 * E != s0):
        raise ValueError("C4 cone homology section or projection is invalid")
    P = {n: s0 * q0 if n == 0 else zeros(R, cone.rank(n), cone.rank(n))
         for n in cone.degrees}
    N = {n: eye(R, cone.rank(n)) - P[n] for n in cone.degrees}

    # A supplied witness belongs to the actual complete Gamma. It need not
    # be strong; each strategy still checks the final original C4 cone.
    if comparison_contraction is not None:
        gamma_cone = mapping_cone(phi)
        validate_homotopy(gamma_cone, gamma_cone, identity_map(gamma_cone),
                          CochainMap(gamma_cone, gamma_cone, {}), comparison_contraction)
    differentials = {n: cone.d(n) for n in range(min(cone.degrees) - 1,
                                               max(cone.degrees) + 1)}
    if strategy == "direct":
        from css_decoupling.topology.direct_splitting import solve_strong_contraction
        # dS^0 [hC^1;iota] = dC^0 hC^1 + phi^1 psi^1 + hC^2 dC^1 = I.
        # q0 [hC^1;iota] = j iota dC^0 hC^1 = 0, by the chain identities.
        # The checked solver accepts only the original four equations;
        # a resource-declined seed retains its original fallback.
        known_top = _vertical(_homotopy_component(C, hC, 1), inclusion)
        h = solve_strong_contraction(cone, solve, section=s0, projection=q0,
                                     idempotent=E, name="schanuel", initial_h={1: known_top})
    elif comparison_contraction is None:
        # Work on an auxiliary unit-disk retract, retaining the original
        # cone's homology projector and all its stable free summands.
        h = _contract_around_projector(cone, P, solve)
        # dh+hd=N implies (dh)^2=dh, (hd)^2=hd and dh^2d=0. Thus hdh
        # is square-zero and annihilates s and q without first forming NhN.
        h = {n: _contracting_sandwich(h[n], differentials[n - 1]) for n in cone.degrees}
    else:
        # Normalize the complete acyclic comparison contraction first.
        # K=H*d*H satisfies dK+Kd=I and K^2=0; both are checked below.
        normalized = {}
        for n in gamma_cone.degrees:
            current = comparison_contraction.get(n, zeros(
                R, gamma_cone.rank(n - 1), gamma_cone.rank(n)))
            normalized[n] = _contracting_sandwich(current, gamma_cone.d(n - 1))
            log.info("normalized comparison contraction degree %s: %s monomials", n,
                     sum(len(f.terms) for f in normalized[n].entries.values()))
        validate_homotopy(gamma_cone, gamma_cone, identity_map(gamma_cone),
                          CochainMap(gamma_cone, gamma_cone, {}), normalized)
        for n in gamma_cone.degrees:
            previous = normalized.get(n - 1, zeros(
                R, gamma_cone.rank(n - 2), gamma_cone.rank(n - 1)))
            if previous * normalized[n]:
                raise ArithmeticError("Normalized comparison contraction is not square-zero")
        h = {1: _vertical(_homotopy_component(C, hC, 1), inclusion)}
        for n in reversed(cone.degrees):
            next_h = h.get(n + 1, zeros(R, cone.rank(n), cone.rank(n + 1)))
            rhs = N[n] - next_h * cone.d(n)
            if n != 1:
                if n == 0:
                    if rhs[gamma_cone.rank(0):, :]:
                        raise ArithmeticError("C4 lifting cycle has a nonzero extra free component")
                    cycle = rhs[:gamma_cone.rank(0), :]
                else:
                    cycle = rhs
                if gamma_cone.d(n) * cycle:
                    raise ArithmeticError("C4 lifting right hand side is not a Gamma-cone cycle")
                existing = normalized.get(n, zeros(
                    R, gamma_cone.rank(n - 1), gamma_cone.rank(n)))
                # Below zero, K[n]*h[n+1]=0, so K[n]*(I-h[n+1]*d[n])=K[n].
                # Degree zero instead restricts to the actual lifting cycle.
                h[n] = existing if n < 0 else existing * cycle
            if cone.d(n - 1) * h[n] != rhs:
                raise ValueError(f"Transported cone splitting failed in degree {n}")
        # The transported h is already strong; do not expand another hdh.
        # All strong identities are still checked on the original cone.
    products = {}

    def multiply(A, B):
        # References in each value keep object identities alive.  This
        # local cache is never reused after mutable matrices leave this call.
        key = id(A), id(B)
        if key not in products:
            products[key] = (A, B, A * B)
        return products[key][2]

    for n in cone.degrees:
        hnxt = h.get(n + 1, zeros(R, cone.rank(n), cone.rank(n + 1)))
        hprev = h.get(n - 1, zeros(R, cone.rank(n - 2), cone.rank(n - 1)))
        if (multiply(differentials[n - 1], h[n]) + multiply(hnxt, differentials[n]) != N[n]
                or multiply(hprev, h[n])):
            raise ValueError(f"{strategy.capitalize()} cone splitting failed")
    if multiply(h[0], s0) or multiply(q0, h[1]):
        raise ValueError("Cone splitting does not annihilate its homology summand")

    even = tuple(n for n in cone.degrees if n % 2 == 0)
    odd = tuple(n for n in cone.degrees if n % 2)
    a, k = sum(cone.rank(n) for n in even), sum(cone.rank(n) for n in odd)

    def off_diagonal(target, source):
        if target == source + 1:
            return differentials[source]
        if target == source - 1:
            return h[source]
        return zeros(R, cone.rank(target), cone.rank(source))

    to_blocks = [[s0 if n == 0 else zeros(R, cone.rank(n), E.nrows),
                  *(off_diagonal(n, m) for m in odd)] for n in even]
    from_blocks = [[q0 if n == 0 else zeros(R, E.nrows, cone.rank(n)) for n in even],
                   *[[off_diagonal(m, n) for n in even] for m in odd]]
    to_free, from_free = block(to_blocks), block(from_blocks)

    def block_product(left, right):
        result = []
        for row in left:
            output_row = []
            for j in range(len(right[0])):
                value = zeros(R, row[0].nrows, right[0][j].ncols)
                for middle, entry in enumerate(row):
                    other = right[middle][j]
                    if entry and other:
                        value = value + multiply(entry, other)
                output_row.append(value)
            result.append(output_row)
        return block(result)

    # Exact block multiplication reuses the local d/h identities already
    # checked above. The serialized verifier multiplies independently.
    if (block_product(to_blocks, from_blocks) != eye(R, a)
            or block_product(from_blocks, to_blocks) != direct_sum(E, eye(R, k))):
        raise ValueError("Schanuel parity frame has no exact two-sided inverse")
    return dict(input_phi=phi, input_psi=psi, input_hC=dict(hC), input_hT=dict(hT),
                rho=rho, cone=cone, section=s0, projection=q0, contraction=h,
                idempotent=E, to_free=to_free, from_free=from_free,
                odd_free_rank=k, even_free_rank=a, method="paper-cone-schanuel",
                even_degrees=even, odd_degrees=odd,
                complement_inclusion=complement_inclusion, complement_coordinates=coordinates,
                cokernel_section=inclusion, cokernel_retraction=rho_top)
