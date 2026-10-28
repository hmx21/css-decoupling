"""Constructive unimodular-row completion over prime Laurent rings.

Laurent rows reduce to polynomial rows by a finite support normalization.
Polynomial completion uses monic triangular changes, local Horrocks degree
reduction, a strictly growing denominator-cover ideal, and exact patching.
Closed points are constructed by elimination and finite residue extensions;
there is no search over points, candidate bases, or polynomial degrees.
"""
from functools import partial

from .laurent import Matrix, eye
from .cancellation import complete_unimodular_row_basic, _check_completion
from .closed_points import closed_point, monic_shear, apply_shear
from .local_completion import AlgebraBudget, RationalContext, local_completion, patch_completions
from .normalization import normalize_unimodular_row
from .polynomial import solve_polynomial, NoPolynomialSolution
from .univariate import complete_unimodular_row as univariate_completion
from css_decoupling.errors import ResourceLimit


def _ordinary(row):
    if not isinstance(row, Matrix) or row.nrows != 1 or row.ncols < 1:
        raise ValueError("Expected a nonempty row of polynomials")
    if any(a < 0 for f in row.entries.values() for e in f.terms for a in e):
        raise ValueError("Polynomial Quillen--Suslin requires nonnegative exponents")


def _check_polynomial_completion(row, U, V):
    _check_completion(row, U, V)
    if any(a < 0 for M in (U, V) for f in M.entries.values() for e in f.terms for a in e):
        raise ArithmeticError("A Laurent inverse was returned for a polynomial completion problem")


def _constant_completion(row, j):
    r, n = row.ring, row.ncols
    permutation = eye(r, n)
    if j:
        permutation[0, 0] = permutation[j, j] = 0
        permutation[0, j] = permutation[j, 0] = 1
    # The first entry is now a constant, so the elementary routine cannot
    # inadvertently select an earlier nonconstant Laurent-unit monomial.
    U, V = complete_unimodular_row_basic(row * permutation)
    return permutation * U, V * permutation


def complete_polynomial_row(row, *, timeout=120, max_terms=2000000,
                            max_steps=1000000, max_patches=128,
                            max_field_trials=128, seed=0, trace=None):
    """Return polynomial U,V with row*U=e_first and UV=VU=I.

    Resource caps can stop a valid problem without an answer. Finite-field
    factorization is Las Vegas with an explicit trial cap and checked output.
    The cover loop grows an ideal strictly at each step; its termination is
    Noetherian, not a guessed degree bound or a polynomial-complexity claim.
    """
    _ordinary(row)
    if max_patches is not None and (isinstance(max_patches, bool) or max_patches < 1):
        raise ValueError("max_patches must be positive or None")
    if max_terms is None:
        raise ValueError("Polynomial expansion requires an explicit positive term cap")
    budget = AlgebraBudget(timeout=timeout, max_terms=max_terms, max_steps=max_steps)
    context = RationalContext(row.ring, budget)
    records = trace if trace is not None else []
    member_solve = partial(solve_polynomial, max_terms=max_terms)
    # Establish the premise in the polynomial ring itself. A Laurent Bezout
    # column is insufficient, even when the input entries are polynomial.
    member_solve(row, eye(row.ring, 1))

    def complete(current):
        budget.check()
        r, n = current.ring, current.ncols
        constant = next((j for j in range(n) if current[0, j] and current[0, j].is_constant), None)
        if constant is not None:
            U, V = _constant_completion(current, constant)
            _check_polynomial_completion(current, U, V)
            return U, V
        axes = tuple(sorted({i for f in current.entries.values() for e in f.terms for i, a in enumerate(e) if a}))
        if len(axes) <= 1:
            U, V = univariate_completion(current, axis=axes[0] if axes else 0, polynomial=True)
            _check_polynomial_completion(current, U, V)
            return U, V
        if n == 2:
            a = member_solve(current, eye(r, 1))
            f, g = current[0, 0], current[0, 1]
            U = Matrix.from_rows(r, [[a[0, 0], -g], [a[1, 0], f]])
            V = Matrix.from_rows(r, [[f, g], [-a[1, 0], a[0, 0]]])
            _check_polynomial_completion(current, U, V)
            return U, V
        if n == 1:
            raise ArithmeticError("A nonconstant polynomial scalar passed the unit-ideal check")
        first = min(j for i, j in current.entries)
        permutation = eye(r, n)
        if first:
            permutation[0, 0] = permutation[first, first] = 0
            permutation[0, first] = permutation[first, 0] = 1
        selected = current * permutation
        axis = axes[-1]
        weights, scalar = monic_shear(selected[0, 0], axes)
        scaling, scaling_inverse = eye(r, n), eye(r, n)
        scaling[0, 0] = pow(scalar, -1, r.p)
        scaling_inverse[0, 0] = scalar
        change, change_inverse = permutation * scaling, scaling_inverse * permutation
        monic = apply_shear(current, axis, weights, max_terms=max_terms) * change
        solutions, denominators = [], []
        while True:
            budget.check()
            if max_patches is not None and len(solutions) >= max_patches:
                raise ResourceLimit("Quillen--Suslin exceeded its local-patch cap")
            # Each denominator already chosen vanishes here; the next one
            # is a unit here. Thus the accumulated ideal grows strictly.
            from time import perf_counter
            remaining = None if budget.deadline is None else max(0, budget.deadline - perf_counter())
            if remaining is not None and remaining <= 0:
                raise ResourceLimit("Quillen--Suslin exceeded its time budget")
            point = closed_point(r, denominators, axes=axes[:-1], seed=seed + len(solutions),
                                 max_trials=max_field_trials, timeout=remaining, max_terms=max_terms)
            if any(point.evaluate(d) != point.field.zero for d in denominators):
                raise ArithmeticError("New local point is outside the uncovered locus")
            solution = local_completion(monic, axis, point, context=context)
            solutions.append(solution)
            denominators.append(solution.denominator)
            try:
                cover = member_solve(Matrix.from_rows(r, [denominators]), eye(r, 1))
            except NoPolynomialSolution:
                continue
            break
        W, Winv = patch_completions(monic, axis, solutions, cover, context=context)
        specialized = monic * W
        if any(e[axis] for f in specialized.entries.values() for e in f.terms):
            raise ArithmeticError("The patched row still depends on its eliminated variable")
        records.append({"axes": axes, "main_axis": axis, "shear_weights": weights,
                        "patch_denominators": tuple(denominators), "cover_weights": cover,
                        "local_degree_sequences": tuple(s.degrees for s in solutions)})
        smaller, smaller_inverse = complete(specialized)
        transformed = change * W * smaller
        transformed_inverse = smaller_inverse * Winv * change_inverse
        U = apply_shear(transformed, axis, weights, sign=-1, max_terms=max_terms)
        V = apply_shear(transformed_inverse, axis, weights, sign=-1, max_terms=max_terms)
        _check_polynomial_completion(current, U, V)
        return U, V

    U, V = complete(row)
    _check_polynomial_completion(row, U, V)
    return U, V


def complete_laurent_row(row, *, timeout=120, max_terms=2000000,
                         max_steps=1000000, max_patches=128,
                         max_field_trials=128, seed=0, trace=None):
    """Complete any prime-field Laurent unimodular row, within resource caps."""
    if not isinstance(row, Matrix) or row.nrows != 1 or row.ncols < 1:
        raise ValueError("Expected a nonempty Laurent row")
    if row.ncols <= 2 or any(f.is_unit for f in row.entries.values()):
        return complete_unimodular_row_basic(row)
    axes = {i for f in row.entries.values() for e in f.terms for i, a in enumerate(e) if a}
    if len(axes) <= 1:
        return univariate_completion(row, axis=next(iter(axes), 0))
    # Already polynomial-unimodular rows can use the smaller direct route.
    # Polynomial nonmembership is not Laurent nonmembership.
    if all(a >= 0 for f in row.entries.values() for e in f.terms for a in e):
        try:
            solve_polynomial(row, eye(row.ring, 1), max_terms=max_terms)
        except NoPolynomialSolution:
            pass
        else:
            return complete_polynomial_row(row, timeout=timeout, max_terms=max_terms,
                                           max_steps=max_steps, max_patches=max_patches,
                                           max_field_trials=max_field_trials, seed=seed, trace=trace)
    normalized = normalize_unimodular_row(row)
    U, V = complete_polynomial_row(normalized.row, timeout=timeout, max_terms=max_terms,
                                   max_steps=max_steps, max_patches=max_patches,
                                   max_field_trials=max_field_trials, seed=seed, trace=trace)
    return normalized.lift_completion(U, V)
