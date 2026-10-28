"""Check-cokernel comparisons with explicit stabilization witnesses.

``paper_stable_cokernel_isomorphism`` follows current Appendix C4 and retains
Schanuel free directions from the existing bounded comparison cone.  The
older ``stable_cokernel_isomorphism`` remains an optional actual-free-basis
route with its original auxiliary-direction cancellation semantics.
"""
from __future__ import annotations

from css_decoupling.algebra.laurent import Matrix, block, direct_sum, eye, zeros
from css_decoupling.algebra.stable_frames import stably_trivialize_idempotent


def _vertical(A, B):
    if A.ncols != B.ncols:
        raise ValueError("Vertical blocks have different column counts")
    entries = dict(A.entries)
    entries.update({(A.nrows + i, j): v for (i, j), v in B.entries.items()})
    return Matrix(A.ring, A.nrows + B.nrows, A.ncols, entries)


def _horizontal(A, B):
    if A.nrows != B.nrows:
        raise ValueError("Horizontal blocks have different row counts")
    entries = dict(A.entries)
    entries.update({(i, A.ncols + j): v for (i, j), v in B.entries.items()})
    return Matrix(A.ring, A.nrows, A.ncols + B.ncols, entries)


def stable_cokernel_isomorphism(phi, psi, hC, hT, solve, image_resolution=None, *, window=False, qs_options=None):
    """Lift a bounded free homotopy equivalence to stable cokernel inverses.

    phi:T->C and psi:C->T must obey I-phi psi=d hC+hC d and
    I-psi phi=d hT+hT d.  The output includes all four comparison lifts,
    two degree-one inverse homotopies, and an actual free basis P,Q of the
    Appendix-C complement, satisfying PQ=E, QP=I.  The optional resolution
    callback may build an intermediate stable frame; all of its auxiliary
    directions must be cancelled. ``qs_options`` sets row-completion limits.
    With ``window=True``, phi and psi compare only nonnegative truncations;
    their inverse homotopies are required from degree one upwards.  These
    are exactly the identities used for the check-cokernel construction.
    """
    from css_decoupling.topology.complexes import CochainMap, identity_map, validate_homotopy

    C, T = phi.target, phi.source
    if psi.source != C or psi.target != T:
        raise ValueError("Comparison maps have incompatible complexes")
    phi.validate()
    psi.validate()
    # These degree-zero chain equations remain mandatory for a truncated
    # comparison, even though a degree-zero inverse homotopy is unnecessary.
    if C.d(0) * phi.at(0) != phi.at(1) * T.d(0) or T.d(0) * psi.at(0) != psi.at(1) * C.d(0):
        raise ValueError("Comparison does not commute at the physical differential")
    ring = C.ring
    phipsi = CochainMap(C, C, {n: phi.at(n) * psi.at(n) for n in C.degrees})
    psiphi = CochainMap(T, T, {n: psi.at(n) * phi.at(n) for n in T.degrees})
    validate_homotopy(C, C, identity_map(C), phipsi, hC, min_degree=1 if window else None)
    validate_homotopy(T, T, identity_map(T), psiphi, hT, min_degree=1 if window else None)
    f, g = phi.at(1), psi.at(1)
    x, xp = C.d(1), T.d(1)
    z = hC.get(2, zeros(ring, C.rank(1), C.rank(2)))
    zp = hT.get(2, zeros(ring, T.rank(1), T.rank(2)))

    # i:A->B+C2 and r:B+C2->A obey r i=I on the cokernel A.
    i = _vertical(g, x)
    r = _horizontal(f, z)
    j = direct_sum(xp, eye(ring, C.rank(2)))
    v = block([[zp, -g * z],
               [zeros(ring, C.rank(2), T.rank(2)), eye(ring, C.rank(2))]])
    projection = eye(ring, T.rank(1) + C.rank(2)) - i * r
    k = projection * v
    E = j * k
    if E * E != E:
        raise ValueError("Appendix-C complement did not produce an idempotent")
    Jp = j * projection
    if E * Jp != Jp:
        raise ValueError("Complement coordinates are outside im(E)")

    from css_decoupling.algebra.projective import free_basis
    from css_decoupling.certificates.projective import verify_projective_basis
    trivialization = None
    if T.rank(2) == 0:
        # In this case the Appendix-C formulas give E=I-d_C^1*h_C^2.
        # The degree-two homotopy identity gives E=h_C^3*d_C^2.  If degree
        # three is terminal, its identity also gives d_C^2*h_C^3=I, so the
        # full contraction already supplies a basis of L.  Check both
        # identities explicitly; they may also hold in a longer complex.
        P = hC.get(3, zeros(ring, C.rank(2), C.rank(3)))
        Q = C.d(2)
        if P * Q == E and Q * P == eye(ring, C.rank(3)):
            trivialization = dict(
                basis=P, coordinates=Q, rank=P.ncols,
                method="paper-free-complement",
                steps=[{"method": "terminal-contraction", "rank_removed": P.ncols}],
                from_free=P, to_free=Q, odd_free_rank=0, even_free_rank=P.ncols)
    if trivialization is None:
        trivialization = free_basis(E, solve=solve, image_resolution=image_resolution, qs_options=qs_options)
    verify_projective_basis(E, trivialization['basis'], trivialization['coordinates'])
    # These are precisely Appendix C's free summands, with no parity-based
    # stabilization: (t,y) -> (rho(t,y), Q*j*(I-i*r)(t,y)), whose inverse is
    # (c,l) -> i(c)+k*P(l), on cokernel classes.
    a, b = trivialization['rank'], C.rank(2)
    phi1 = _vertical(r, trivialization['coordinates'] * Jp)
    psi1 = _horizontal(i, k * trivialization['basis'])
    dC = _vertical(C.d(0), zeros(ring, a, C.rank(0)))
    dT = _vertical(T.d(0), zeros(ring, b, T.rank(0)))

    phi0 = solve(dC, phi1 * dT)
    psi0 = solve(dT, psi1 * dC)
    etaC = solve(dC, eye(ring, dC.nrows) - phi1 * psi1)
    etaT = solve(dT, eye(ring, dT.nrows) - psi1 * phi1)
    for A, X, B in ((dC, phi0, phi1 * dT),
                    (dT, psi0, psi1 * dC),
                    (dC, etaC, eye(ring, dC.nrows) - phi1 * psi1),
                    (dT, etaT, eye(ring, dT.nrows) - psi1 * phi1)):
        if A * X != B:
            raise ValueError("Stable cokernel comparison failed exact verification")
    return dict(dC=dC, dT=dT, phi0=phi0, phi1=phi1, psi0=psi0, psi1=psi1,
                etaC=etaC, etaT=etaT, free_C=a, free_T=b,
                idempotent=E, trivialization=trivialization,
                complement_inclusion=k, complement_coordinates=Jp)


def paper_stable_cokernel_isomorphism(phi, psi, hC, hT, solve, *,
                                    comparison_contraction=None, target_preimage=None,
                                    contraction_strategy="normalized", cancel_units=True):
    """Current Appendix C4: the cone's Schanuel frame, less its unit directions.

    Unlike ``stable_cokernel_isomorphism``, this never requests an actual
    free basis of Q.  Complete bounded comparisons are mandatory; every
    negative cone term enters the stable frame.  With ``cancel_units``, each
    extra free direction whose frame row has a Laurent unit is then removed
    by an elementary completion, which needs no Quillen-Suslin step and
    lowers the adjoined checks and ancillas. All solves use the injected
    Laurent backend, with no independent complement resolution.
    """
    from css_decoupling.algebra.cancellation import cancel_unit_directions
    from css_decoupling.decoupling.schanuel import stable_complement_from_cone

    frame = stable_complement_from_cone(phi, psi, hC, hT, solve,
                                       comparison_contraction=comparison_contraction,
                                       strategy=contraction_strategy)
    C, T, ring = phi.target, phi.source, phi.target.ring
    E, k, a = frame['idempotent'], frame['odd_free_rank'], frame['even_free_rank']
    r, i = frame['cokernel_retraction'], frame['cokernel_section']
    P, Q = frame['to_free'], frame['from_free']
    cancelled = []
    if cancel_units:
        P, Q, cancelled = cancel_unit_directions(E, P, Q)
    k, a = k - len(cancelled), a - len(cancelled)
    b = C.rank(2) + k
    J, K = frame['complement_coordinates'], frame['complement_inclusion']
    phi1 = block([[r, zeros(ring, C.rank(1), k)],
                  [P[:, :E.nrows] * J, P[:, E.nrows:]]])
    psi1 = block([[i, K * Q[:E.nrows, :]],
                  [zeros(ring, k, C.rank(1)), Q[E.nrows:, :]]])
    dC = _vertical(C.d(0), zeros(ring, a, C.rank(0)))
    dT = _vertical(T.d(0), zeros(ring, b, T.rank(0)))
    phi0 = solve(dC, phi1 * dT)
    def lift_target(rhs):
        if target_preimage is None:
            return solve(dT, rhs)
        if rhs[T.rank(1):, :]:
            raise ValueError("Toric lift has a nonzero adjoined-check component")
        result = target_preimage(rhs[:T.rank(1), :])
        if dT * result != rhs:
            raise ValueError("Explicit toric lift failed its exact identity")
        return result
    psi0 = lift_target(psi1 * dC)
    defect_C = eye(ring, dC.nrows) - phi1 * psi1
    defect_T = eye(ring, dT.nrows) - psi1 * phi1
    etaC, etaT = solve(dC, defect_C), lift_target(defect_T)
    for A, X, B in ((dC, phi0, phi1 * dT), (dT, psi0, psi1 * dC),
                    (dC, etaC, defect_C), (dT, etaT, defect_T)):
        if A * X != B:
            raise ValueError("Schanuel cokernel lift failed exact verification")
    return dict(dC=dC, dT=dT, phi0=phi0, phi1=phi1, psi0=psi0, psi1=psi1,
                etaC=etaC, etaT=etaT, free_C=a, free_T=b, idempotent=E,
                trivialization=frame, schanuel=frame,
                unit_cancellation=dict(directions=cancelled, to_free=P, from_free=Q),
                complement_inclusion=K, complement_coordinates=J)
