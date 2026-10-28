"""C4 sections, constructed from the top degree down in original coordinates.

Only the current h^n is unknown.  Higher components have already passed all
strong-retract identities; no weak contraction is frozen or normalized by
h*d*h.  The default solver is a practical Laurent backend, not a claim about
its complexity.  Finite-support/backend resource failures remain incomplete.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import logging
from time import perf_counter

from css_decoupling.algebra.laurent import Matrix, eye, zeros
from css_decoupling.errors import ResourceLimit
from .complexes import solve_matrix_equations

log = logging.getLogger(__name__)
_observer = ContextVar("direct_splitting_observer", default=None)
_structural_options = ContextVar("c4_structural_sections", default=None)


@contextmanager
def structural_sections(*, enabled=True, **options):
    """Temporarily enable or disable the native structural coefficient solver.

    Legacy/custom and strictly bounded callers keep their explicit solver
    unless their enclosing API deliberately selects this context.
    """
    if type(enabled) is not bool:
        raise ValueError("enabled must be a boolean")
    token = _structural_options.set(dict(options) if enabled else None)
    try:
        yield
    finally:
        _structural_options.reset(token)


@contextmanager
def observe_direct_splitting(observer):
    """Observe equations before a possibly expensive solve (no proof caching)."""
    token = _observer.set(observer)
    try:
        yield
    finally:
        _observer.reset(token)


def _terms(A):
    return sum(len(f.terms) for f in A.entries.values())


def product_work(A, B):
    """Scalar monomial pairs before cancellation; no exponent box allocation."""
    rows = {}
    for (i, _), value in B.entries.items():
        rows[i] = rows.get(i, 0) + len(value.terms)
    return sum(len(value.terms)*rows.get(j, 0) for (_, j), value in A.entries.items())


def _right_annihilator_by_units(B, max_terms, max_product_work):
    """Eliminate known right constraints, retaining their exact row quotient.

    A pivot unit in row i eliminates that row/column of B. The same row
    operations on N=I give a full row annihilator if the residual is zero.
    The surviving original coordinates J obey N*J=I. Thus R*B=0 implies
    R=(R*J)*N, which is explicitly checked by the caller. This is ordinary
    linear elimination of the CURRENT equation, not cancellation of the
    complex or an assumed free basis of Q. A nonzero residual just declines
    this shortcut; the complete joint Laurent equations remain available.
    """
    from .contractions import _unit_pivot
    R, residual = B.ring, B.copy()
    active, N = list(range(B.nrows)), eye(B.ring, B.nrows)
    while residual:
        at = _unit_pivot(residual)
        if at is None:
            return None
        i, j = at
        rows = [k for k in range(residual.nrows) if k != i]
        cols = [k for k in range(residual.ncols) if k != j]
        column = residual[rows, j:j+1] * (residual[i, j] ** -1)
        row, old = residual[i:i+1, cols], N[i:i+1, :]
        work = product_work(column, row) + product_work(column, old)
        if max_product_work is not None and work > max_product_work:
            return None
        residual = residual[rows, cols] - column*row
        N = N[rows, :] - column*old
        active.pop(i)
        if max_terms is not None and _terms(residual)+_terms(N) > max_terms:
            return None
    if N[:, active] != eye(R, len(active)):
        raise ArithmeticError("Right constraint elimination lost its surviving coordinates")
    return N, active


def _homology(C, section, projection, idempotent):
    R = C.ring
    if section is projection is idempotent is None:
        return zeros(R, C.rank(0), 0), zeros(R, 0, C.rank(0)), zeros(R, 0, 0)
    if not all(isinstance(A, Matrix) and A.ring == R
               for A in (section, projection, idempotent)):
        raise ValueError("Provide section, projection and idempotent over the complex ring")
    s, q, e = section, projection, idempotent
    if (e.nrows != e.ncols or s.shape != (C.rank(0), e.nrows)
            or q.shape != (e.nrows, C.rank(0))):
        raise ValueError("Malformed degree-zero projective homology coordinates")
    if (e*e != e or q*s != e or s*e != s or e*q != q
            or C.d(0)*s or q*C.d(-1)):
        raise ValueError("Invalid projective homology section/projection")
    return s, q, e


def validate_strong_contraction(C, h, *, section=None, projection=None, idempotent=None):
    """Check actual matrices in every degree, including all projective identities.

    The full dh+hd identity also proves that s/q account for ALL homology;
    merely supplying a retract qs=e does not establish that assertion.
    """
    C.validate()
    s, q, e = _homology(C, section, projection, idempotent)
    R = C.ring
    for n, A in h.items():
        if (type(n) is not int or not isinstance(A, Matrix) or A.ring != R
                or A.shape != (C.rank(n-1), C.rank(n))):
            raise ValueError("Malformed strong contraction component")
    def at(n):
        return h.get(n, zeros(R, C.rank(n-1), C.rank(n)))
    for n in C.degrees:
        P = s*q if n == 0 else zeros(R, C.rank(n), C.rank(n))
        if C.d(n-1)*at(n) + at(n+1)*C.d(n) != eye(R, C.rank(n))-P:
            raise ValueError(f"Invalid strong contraction identity in degree {n}")
        if at(n-1)*at(n):
            raise ValueError(f"Strong contraction is not square-zero in degree {n}")
    if at(0)*s or q*at(1):
        raise ValueError("Strong contraction violates a homology side condition")
    return True


def solve_strong_contraction(C, solve, *, section=None, projection=None, idempotent=None,
                             max_unknowns=250000, max_terms=2000000, observer=None,
                             name="direct", max_product_work=50000000, initial_h=None):
    """Solve dprev X=R, X hnext=0, X s=0, qprev X=0, descending.

    R=I-sq-hnext*d is the boundary projection once the higher-degree
    induction holds.  Its image is projective.  dprev:ker(qprev)->im(R)
    is onto: replace a preimage z by (I-s_prev*qprev)z.  A section sigma
    therefore exists and sigma*R solves all four linear equations.  We
    solve for X directly and do not form this potentially dense product.

    A complete, checked right quotient is used before an expensive weak
    left solve. If that shortcut is unavailable or exceeds a local budget,
    the original left/joint routes remain available. No higher h is changed
    or accepted before all its tests pass.
    """
    for value in (max_unknowns, max_terms, max_product_work):
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError("Direct splitting limits must be nonnegative integers or None")
    C.validate()
    s0, q0, e = _homology(C, section, projection, idempotent)
    R, h = C.ring, {}
    seeds = {} if initial_h is None else dict(initial_h)
    if any(type(n) is not int or n not in C.degrees for n in seeds):
        raise ValueError("Known sections use invalid complex degrees")
    if seeds and set(seeds) != set(C.degrees[-len(seeds):]):
        raise ValueError("Known sections must form a complete descending prefix")
    callback = observer if observer is not None else _observer.get()

    def emit(event, **data):
        if callback is not None:
            callback(dict(event=event, name=name, C=C, section=s0, projection=q0,
                          e=e, completed_h=dict(h), limits=dict(
                              max_unknowns=max_unknowns, max_terms=max_terms,
                              max_product_work=max_product_work), **data))

    def multiply(A, B):
        work = product_work(A, B)
        if max_product_work is not None and work > max_product_work:
            emit("product-incomplete", degree=n, left=A, right=B, product_work=work)
            raise ResourceLimit(f"Direct section degree {n} product needs {work} monomial pairs")
        return A*B

    def check_size(X):
        if (not isinstance(X, Matrix) or X.ring != R or X.shape != (rows, cols)):
            raise ValueError(f"Invalid direct section solver result in degree {n}")
        if max_terms is not None and _terms(X) > max_terms:
            emit("solution-incomplete", degree=n, h=X)
            raise ResourceLimit(f"Direct section degree {n} solution exceeds support budget {max_terms}")

    def candidate_right_conditions(X, following, section):
        # This X is only a proposed shortcut. If even testing it is too
        # costly, decline it and solve the constrained system instead of
        # exhausting the whole stage on an unaccepted weak candidate.
        if max_product_work is not None and any(
                product_work(X, B) > max_product_work for B in (following, section)):
            emit("candidate-declined", degree=n, candidate=X,
                 reason="right-condition-check-budget")
            return False
        return not multiply(X, following) and not multiply(X, section)

    for n in reversed(C.degrees):
        started = perf_counter()
        rows, cols = C.rank(n-1), C.rank(n)
        next_h = h.get(n+1, zeros(R, cols, C.rank(n+1)))
        s = s0 if n == 0 else zeros(R, cols, e.nrows)
        q = q0 if n == 1 else zeros(R, e.nrows, rows)
        emit("projection-start", degree=n, dprev=C.d(n-1), hnext=next_h, s=s, q=q)
        P = multiply(s0, q0) if n == 0 else zeros(R, cols, cols)
        rhs = eye(R, cols)-P-multiply(next_h, C.d(n))
        d = C.d(n-1)
        data = dict(degree=n, dprev=d, hnext=next_h, s=s, q=q, R=rhs)
        emit("solve-start", **data)
        log.info("%s direct section degree %s: X=%sx%s, rhs=%s monomials",
                 name, n, rows, cols, _terms(rhs))
        if max_unknowns is not None and rows*cols > max_unknowns:
            raise ResourceLimit(f"Direct section degree {n} exceeds {max_unknowns} Laurent unknowns")
        if max_terms is not None and sum(_terms(A) for A in (d, rhs, next_h, s, q)) > max_terms:
            raise ResourceLimit(f"Direct section degree {n} exceeds input support budget {max_terms}")
        # These are necessary induction checks, not assumptions based on an
        # old contraction. At the very bottom nonzero R is detected by solve.
        # R*hnext=0 follows from the previously checked contraction,
        # hnext*hnextnext=0 and both homology side conditions. Expanding
        # it again can dwarf all differential products. Every NEW
        # X*hnext=0 is still multiplied below and in the final validation.
        if multiply(C.d(n), rhs) or multiply(rhs, s):
            raise ValueError(f"Invalid higher-degree splitting induction at {n}")
        if n in seeds:
            candidate = seeds[n]
            try:
                check_size(candidate)
                if (multiply(d, candidate) != rhs or multiply(candidate, next_h)
                        or multiply(candidate, s) or multiply(q, candidate)):
                    raise ValueError("Known section violates an original constraint")
            except ResourceLimit as exc:
                emit("known-section-declined", reason=str(exc), **data)
                # Later supplied sections may depend on this particular seed.
                # Once it is declined, solve the remaining degrees normally.
                seeds.clear()
            else:
                h[n] = candidate
                emit("solve-complete", h=candidate, strategy="verified-known-section",
                     seconds=perf_counter()-started, **data)
                continue
        structural_options = _structural_options.get()
        quotient_witness = None
        if structural_options is not None:
            from css_decoupling.algebra.structural_c4 import solve_structural_section
            previous_P = s0*q0 if n == 1 else zeros(R, rows, rows)
            local_options = dict(structural_options)
            local_options.setdefault("try_sparse_seed", True)
            for key, outer, default in (("max_product_work", max_product_work, 50000000),
                                         ("max_terms", max_terms, 5000000)):
                if outer is not None:
                    local_options[key] = min(local_options.get(key, default), outer)
            try:
                X, certificate = solve_structural_section(
                    d, rhs, next_h, s, q, previous_P, return_certificate=True,
                    observer=lambda info: emit("structural-attempt", attempt=info, **data),
                    **local_options)
            except ResourceLimit as exc:
                quotient_witness = getattr(exc, "structural_quotient", None)
                emit("structural-incomplete", reason=str(exc), **data)
            else:
                check_size(X)
                h[n] = X
                emit("solve-complete", h=X, strategy="structural-coefficient-section",
                     structural_certificate=certificate, seconds=perf_counter()-started, **data)
                log.info("%s direct section degree %s: structural coefficients, h=%s monomials, %.3fs",
                         name, n, _terms(X), perf_counter()-started)
                continue
        left = d.vstack(q) if q else d
        right = rhs.vstack(zeros(R, q.nrows, cols)) if q else rhs
        right_constraints = next_h.hstack(s)

        def quotient_section(N, active, *, reused):
            if (not isinstance(N, Matrix) or N.ring != R
                    or not isinstance(active, (tuple, list))
                    or any(type(j) is not int or not 0 <= j < cols for j in active)
                    or len(set(active)) != len(active)
                    or N.shape != (len(active), cols)):
                raise ValueError("Malformed right constraint quotient")
            if max_terms is not None and _terms(N) > max_terms:
                raise ResourceLimit("Right constraint quotient exceeds support budget")
            if (N[:, active] != eye(R, len(active)) or multiply(N, right_constraints)
                    or multiply(rhs[:, active], N) != rhs):
                raise ValueError("Right constraint quotient failed its exact identities")
            emit("right-quotient-ready", quotient=N, active=tuple(active), reused=reused, **data)
            Y = solve(left, right[:, active])
            if (not isinstance(Y, Matrix) or Y.ring != R
                    or Y.shape != (rows, len(active))):
                raise ValueError("Invalid reduced direct section solver result")
            if max_terms is not None and _terms(Y) > max_terms:
                raise ResourceLimit("Reduced direct section exceeds support budget")
            if multiply(d, Y) != rhs[:, active] or multiply(q, Y):
                raise ValueError("Reduced direct section failed its original left equations")
            answer = multiply(Y, N)
            check_size(answer)
            # Keep an unaffordable positive candidate from suppressing an
            # older, possibly sparser route. All four products are still
            # actually evaluated before accepting answer below.
            if max_product_work is not None and any(product_work(A, B) > max_product_work
                    for A, B in ((d, answer), (answer, next_h), (answer, s), (q, answer))):
                raise ResourceLimit("Right quotient solution verification exceeds product budget")
            return answer

        X, right_valid = None, False
        strategy = "shared-left-solve"
        if quotient_witness is not None:
            if (not isinstance(quotient_witness, dict)
                    or set(quotient_witness) != {"constraints", "rhs", "N", "active"}
                    or quotient_witness["constraints"] != right_constraints
                    or quotient_witness["rhs"] != rhs):
                raise ValueError("Reused right quotient changed its original equation")
            try:
                X = quotient_section(quotient_witness["N"], quotient_witness["active"], reused=True)
            except ResourceLimit as exc:
                emit("right-quotient-declined", reason=str(exc), reused=True, **data)
        if X is None and quotient_witness is None and right_constraints:
            try:
                eliminated = _right_annihilator_by_units(right_constraints, max_terms, max_product_work)
                if eliminated is not None:
                    X = quotient_section(*eliminated, reused=False)
            except ResourceLimit as exc:
                emit("right-quotient-declined", reason=str(exc), reused=False, **data)
        if X is not None:
            strategy, right_valid = "right-unit-linear-elimination", True
        else:
            X = solve(left, right)
            check_size(X)
            if multiply(d, X) != rhs or multiply(q, X):
                raise ValueError(f"Invalid direct section solver result in degree {n}")
            right_valid = candidate_right_conditions(X, next_h, s)
        # A missing full quotient is only a failed local strategy; it does
        # not establish nonexistence. The coupled Laurent system is retained.
        if not right_valid:
            # Preflight the sparse Kronecker operator before materializing
            # it. Identity factors never multiply polynomial supports.
            operator_terms = cols*(_terms(d)+_terms(q)) + rows*(_terms(next_h)+_terms(s))
            if max_terms is not None and operator_terms+_terms(rhs) > max_terms:
                emit("solve-incomplete", strategy="joint-linear", candidate=X,
                     operator_terms=operator_terms, **data)
                raise ResourceLimit(f"Direct section degree {n} joint operator needs {operator_terms} terms")
            emit("solve-joint", candidate=X, operator_terms=operator_terms, **data)
            Irow, Icol = eye(R, rows), eye(R, cols)
            equations = [([(d, "X", Icol)], rhs)]
            if next_h:
                equations.append(([(Irow, "X", next_h)], zeros(R, rows, next_h.ncols)))
            if s:
                equations.append(([(Irow, "X", s)], zeros(R, rows, s.ncols)))
            if q:
                equations.append(([(q, "X", Icol)], zeros(R, q.nrows, cols)))
            X = solve_matrix_equations(R, {"X": (rows, cols)}, equations, solve)["X"]
            strategy = "joint-linear"
        check_size(X)
        if (multiply(d, X) != rhs or multiply(X, next_h)
                or multiply(X, s) or multiply(q, X)):
            raise ValueError(f"Direct section violates a linear constraint in degree {n}")
        h[n] = X
        emit("solve-complete", h=X, strategy=strategy, seconds=perf_counter()-started, **data)
        log.info("%s direct section degree %s: %s, h=%s monomials, %.3fs",
                 name, n, strategy, _terms(X), perf_counter()-started)
    validate_strong_contraction(C, h, section=s0, projection=q0, idempotent=e)
    return h
