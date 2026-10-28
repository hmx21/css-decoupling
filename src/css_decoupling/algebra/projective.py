"""Actual free bases of prime Laurent projective summands.

Cheap exact reductions precede the general constructive route: a native
finite free resolution gives an intermediate stable frame, and Laurent
Quillen--Suslin row completion cancels every auxiliary free direction.
The result always describes the original image, with checked two-sided maps.
"""
from __future__ import annotations

from sympy import GF
from sympy.polys.rings import ring as polynomial_ring

from css_decoupling.algebra.laurent import Matrix, eye, zeros
from css_decoupling.algebra.modules import get_backend, solve as module_solve
from css_decoupling.errors import BackendUnavailable, NoLaurentSolution


class ProjectiveBasisUnavailable(BackendUnavailable):
    """Compatibility exception for a caller-supplied incomplete basis method."""

def _primitive_column(column):
    """Divide a Laurent column by its exact polynomial content.

    A common monomial clears all negative powers.  A multivariate gcd then
    removes the common nonunit content; exact divisions are verified in the
    original Laurent ring.  This is a gcd operation, not a support search.
    """
    r = column.ring
    entries = [f for f in column.entries.values() if f]
    if not entries:
        return column.copy()
    shift = tuple(-min(e[k] for f in entries for e in f.terms) for k in range(r.D))
    poly_ring = polynomial_ring([f"x{k}" for k in range(r.D)], GF(r.p))[0]
    polys = {i: poly_ring.from_dict({tuple(a + b for a, b in zip(e, shift)): c
                                  for e, c in f.terms.items()})
             for (i, _), f in column.entries.items()}
    content = poly_ring.zero
    for f in polys.values():
        content = content.gcd(f)
        if content == poly_ring.one:
            break
    answer = zeros(r, column.nrows, 1)
    for i, f in polys.items():
        quotient, remainder = divmod(f, content)
        if remainder:
            raise ArithmeticError("Polynomial-content division was not exact")
        answer[i, 0] = r({e: int(c) for e, c in quotient.items()})
    common = r({tuple(a - b for a, b in zip(e, shift)): int(c)
                for e, c in content.items()})
    if answer * common != column:
        raise ArithmeticError("Primitive-column content certificate failed")
    return answer


def _generator_basis(E, G, solve):
    """Extract a free generating matrix when units remove all relations."""
    backend = get_backend(E.ring)
    if E * G != G:
        raise ValueError("Proposed projective generators are outside im(E)")
    solve(G, E)  # Every original column must remain in the new image.
    relations = backend.kernel(G)
    count, remainder, L, _ = backend.reduce_units(relations)
    if remainder:
        return None
    # L*relations has image in the first count coordinates, and contains the
    # coordinate basis there.  The quotient basis is consequently represented
    # by columns count: of L^-1.  No alternative column subsets are searched.
    if count:
        lift = solve(L, eye(E.ring, L.nrows)[:, count:])
        P = G * lift
    else:
        P = G
    Q = solve(P, E)
    if P * Q != E or Q * P != eye(E.ring, P.ncols):
        raise ArithmeticError("Projective generator reduction is not a basis")
    return P, Q


def free_basis(E, *, solve=module_solve, image_resolution=None, method="auto", qs_options=None):
    """Return P,Q with P*Q=E and Q*P=I_rank(im E), or fail explicitly.

    ``method='auto'`` tries unit, primitive-column and generator reductions
    first; ``method='quillen-suslin'`` directly uses a finite free resolution
    and cancels its intermediate stable frame completely. The optional
    resolution callback must return an exact finite free resolution; its
    split contraction is checked. ``qs_options`` passes resource limits to
    row completion. No support, degree, point or candidate-basis enumeration
    is used. Resource exhaustion raises without returning a stabilized answer.
    """
    if not isinstance(E, Matrix) or E.nrows != E.ncols or E * E != E:
        raise ValueError("A projective summand must be specified by an idempotent")
    if method not in ("auto", "quillen-suslin"):
        raise ValueError("Projective-basis method must be 'auto' or 'quillen-suslin'")
    from functools import partial
    from .resolutions import image_resolution as native_resolution
    from .stable_frames import stably_trivialize_idempotent
    from .cancellation import cancel_stable_frame
    from .quillen_suslin import complete_laurent_row
    resolution_builder = image_resolution or native_resolution
    row_completion = partial(complete_laurent_row, **(qs_options or {}))
    r, n = E.ring, E.nrows
    backend = get_backend(r)
    P, Q, F, steps = zeros(r, n, 0), zeros(r, 0, n), E.copy(), []
    general_constructions = []

    def general_basis(remainder, resolution=None):
        maps = resolution if resolution is not None else resolution_builder(remainder)
        frame = stably_trivialize_idempotent(remainder, maps, solve)
        result = cancel_stable_frame(remainder, frame["to_free"], frame["from_free"],
                                     complete_row=row_completion)
        general_constructions.append({"resolution": maps, "stable_frame": frame,
                                      "cancellation": result})
        return result["basis"], result["coordinates"]

    def accept(A, B, method):
        nonlocal P, Q, F
        if B * A != eye(r, A.ncols) or F * A != A or B * F != B:
            raise ArithmeticError("Projective line splitting certificate failed")
        if Q * A or B * P:
            raise ArithmeticError("Projective splittings are not orthogonal")
        P, Q, F = P.hstack(A), Q.vstack(B), F - A * B
        if F * F != F:
            raise ArithmeticError("Projective remainder is not idempotent")
        steps.append({"method": method, "rank_removed": A.ncols})

    while F:
        if method == "quillen-suslin":
            accept(*general_basis(F), "quillen-suslin-complete-cancellation")
            continue
        count, remainder, L, V = backend.reduce_units(F)
        if count:
            A = F * V[:, :count]
            B = L[:count, :] * F
            accept(A, B, "laurent-unit-elimination")
            continue
        # A rank-one direct summand has a primitive column whose entries
        # generate the unit ideal.  Thus this step is complete in rank one.
        accepted = False
        for j in sorted({j for _, j in F.entries}):
            A = _primitive_column(F[:, j:j + 1])
            if F * A != A:
                raise ArithmeticError("Primitive column left its saturated summand")
            try:
                B = solve(A.T, eye(r, 1)).T * F
            except NoLaurentSolution:
                continue
            accept(A, B, "primitive-column-bezout")
            accepted = True
            break
        if accepted:
            continue
        # Kernel generators need not resemble columns of E.  Reducing their
        # complete relation module can expose a basis even when every column
        # of E vanishes somewhere on the algebraic torus.
        candidates = [backend.kernel(eye(r, n) - F)]
        basis = _generator_basis(F, candidates[0], solve)
        resolution = None
        if basis is None:
            resolution = resolution_builder(F)
            if not resolution:
                raise ArithmeticError("A nonzero projective module has empty resolution")
            basis = _generator_basis(F, resolution[0], solve)
        if basis is not None:
            accept(*basis, "syzygy-generator-reduction")
            continue
        accept(*general_basis(F, resolution), "quillen-suslin-complete-cancellation")
    if P * Q != E or Q * P != eye(r, P.ncols):
        raise ArithmeticError("Final projective basis identities failed")
    return {"basis": P, "coordinates": Q, "rank": P.ncols,
            "method": "paper-free-complement", "steps": steps,
            "general_constructions": general_constructions,
            "from_free": P, "to_free": Q,
            "odd_free_rank": 0, "even_free_rank": P.ncols}
