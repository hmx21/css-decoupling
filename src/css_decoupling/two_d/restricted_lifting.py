"""Exact comparison-map lifts through restriction of Laurent scalars.

For a retained lattice with column basis B, ``unpack`` inverts ``L.pack``:
the coarse term in coset r and exponent e is the old monomial x^(r+B e).
Thus pack(A X)=restrict(A) pack(X), for arbitrary integer lattices.

A finite cokernel permits compact lifts using its *actual* translation
actions. If E represents its finite-field basis and A H_i=x_i E-E T_i,
then m e_j=E c+A h implies x_i m e_j=E T_i c+A(x_i h+H_i c).
The same identity applies to inverse translations. Only the small defining
homotopies use the complete Laurent solver; large powers use this recurrence.
All accepted answers satisfy the original fine and coarse Laurent equations.

This is a positive, bounded construction, not a new negative decision rule.
Infinite cokernels, shortcut budgets, and nonzero quotient remainders fall
back to the complete native solver. Its own resource failures retain their
usual meaning. No degree search, finite torus, or external CAS is used.
"""
from __future__ import annotations

from math import isfinite
from time import perf_counter

from .algebra import (
    Matrix, Poly, Lattice, GeneralBackend, FiniteQuotient,
    ResourceLimit, InfiniteLayer, eye, zeros,
)


def unpack(lattice, matrix):
    """Convert individual coarse columns to old-ring vectors, without replication."""
    if not isinstance(lattice, Lattice) or not isinstance(matrix, Matrix):
        raise TypeError('unpack requires a Lattice and a Laurent Matrix')
    if matrix.ring.D != lattice.D or matrix.nrows % lattice.index:
        raise ValueError('Coarse vector rows and ring must match the lattice')
    entries, coordinates = {}, {}
    basis, prime = lattice.basis, matrix.ring.p
    for (row, col), polynomial in matrix.entries.items():
        component, coset = divmod(row, lattice.index)
        representative = lattice.representatives[coset]
        target = entries.setdefault((component, col), {})
        for exponent, coefficient in polynomial.terms.items():
            key = coset, exponent
            original = coordinates.get(key)
            if original is None:
                original = tuple(representative[i] + sum(basis[i][j]*exponent[j]
                                                         for j in range(lattice.D))
                                 for i in range(lattice.D))
                coordinates[key] = original
            value = (target.get(original, 0)+coefficient) % prime
            if value:
                target[original] = value
            else:
                target.pop(original, None)
    return Matrix(matrix.ring, matrix.nrows//lattice.index, matrix.ncols,
                  {ij: Poly(matrix.ring, terms) for ij, terms in entries.items()})


def _terms(matrix):
    return sum(len(polynomial.terms) for polynomial in matrix.entries.values())


class _LiftBudget:
    """Bounds retained coefficient support and recurrence/accumulation work.

    These are algorithm-selection caps, not an operating-system memory cap.
    Complete primitive solves and the exact final products are independently
    checked; an external process budget is still required for hard limits.
    """
    def __init__(self, max_nodes, max_terms, max_steps, timeout, stats):
        for name, value in (('max_nodes', max_nodes), ('max_terms', max_terms),
                            ('max_steps', max_steps)):
            if type(value) is not int or value < 1:
                raise ValueError(name+' must be a positive integer')
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not isfinite(timeout) or timeout <= 0:
            raise ValueError('timeout must be positive and finite')
        self.max_nodes, self.max_terms, self.max_steps = max_nodes, max_terms, max_steps
        self.timeout, self.start, self.stats = timeout, perf_counter(), stats
        stats.update(steps=0, cache_nodes=0, cached_terms=0, peak_stored_terms=0,
                     caps=dict(max_nodes=max_nodes, max_terms=max_terms,
                               max_steps=max_steps, timeout=timeout))

    def work(self, amount=1):
        self.stats['steps'] += amount
        if self.stats['steps'] > self.max_steps:
            raise ResourceLimit('Finite quotient lifting recurrence/work budget')
        if perf_counter()-self.start > self.timeout:
            raise ResourceLimit('Finite quotient lifting wall-time budget')

    def storage(self, amount):
        self.stats['peak_stored_terms'] = max(self.stats['peak_stored_terms'], amount)
        if amount > self.max_terms:
            raise ResourceLimit('Finite quotient lifting stored coefficient budget')


def _finite_lift(A, B, backend, quotient, budget, max_dimension):
    """Return a checked answer or None; never infer nonmembership from a miss."""
    stats, start = budget.stats, perf_counter()
    if quotient is None:
        # Over the fraction field, rank(A)<=ncols<nrows leaves a positive
        # free-rank quotient. For D>0 its F_p dimension is infinite. This
        # exact dimension bound avoids constructing an unnecessary GB;
        # an explicitly supplied quotient still follows full validation.
        if A.ring.D > 0 and A.ncols < A.nrows:
            raise InfiniteLayer('Positive free-rank lower bound: fewer columns than rows')
        quotient = FiniteQuotient(A, max_dimension=max_dimension, backend=backend)
    elif not isinstance(quotient, FiniteQuotient) or quotient.presentation != A or quotient.ring != A.ring:
        raise ValueError('Supplied quotient must belong to the exact fine presentation')
    E, dimension = quotient.basis, quotient.dimension
    if (not isinstance(E, Matrix) or E.ring != A.ring or
            type(dimension) is not int or dimension < 0 or E.shape != (A.nrows, dimension)):
        raise ValueError('Malformed finite quotient basis')
    if dimension > max_dimension:
        raise ResourceLimit('Finite quotient lifting dimension budget')
    if (quotient.coordinates(E) != eye(A.ring, dimension) or
            quotient.coordinates(A) != zeros(A.ring, dimension, A.ncols)):
        raise ValueError('Supplied quotient basis does not verify against its presentation')
    if len(quotient.actions) != A.ring.D:
        raise ValueError('Malformed finite quotient translation actions')
    for axis, action in enumerate(quotient.actions):
        if action != quotient.coordinates(A.ring.variable(axis)*E):
            raise ValueError('Supplied quotient translation action was altered')
    stats.update(quotient_seconds=perf_counter()-start, quotient_dimension=dimension)
    identity = eye(A.ring, A.nrows)
    C0 = quotient.coordinates(identity)
    seed_target = identity-E*C0
    H0 = backend.solve(A, seed_target)
    if A*H0 != seed_target:
        raise ArithmeticError('Finite quotient seed homotopy failed')
    shifts = {(axis, 1 if exponent[axis] > 0 else -1)
              for polynomial in B.entries.values() for exponent in polynomial.terms
              for axis in range(A.ring.D) if exponent[axis]}
    homotopies = {}
    retained = _terms(E)+_terms(C0)+_terms(H0)
    budget.storage(retained)
    for axis, direction in sorted(shifts):
        budget.work()
        unit = A.ring.variable(axis)**direction
        action = quotient.coordinates(unit*E)
        if direction == 1 and action != quotient.actions[axis]:
            raise ValueError('Supplied quotient translation action was altered')
        target = unit*E-E*action
        homotopy = backend.solve(A, target)
        if A*homotopy != target:
            raise ArithmeticError('Finite quotient translation homotopy failed')
        homotopies[axis, direction] = unit, action, homotopy
        retained += _terms(action)+_terms(homotopy)
        budget.storage(retained)
    stats['primitive_lift_seconds'] = perf_counter()-start-stats['quotient_seconds']
    zero = (0,)*A.ring.D
    if A.nrows > budget.max_nodes:
        raise ResourceLimit('Finite quotient lifting seed-node budget')
    cache = {(i, zero): (C0[:, i:i+1], H0[:, i:i+1]) for i in range(A.nrows)}
    cached_terms = sum(_terms(c)+_terms(h) for c, h in cache.values())
    budget.storage(retained+cached_terms)

    def get(component, exponent):
        nonlocal cached_terms
        key, path = (component, exponent), []
        while key not in cache:
            if len(cache)+len(path) >= budget.max_nodes:
                raise ResourceLimit('Finite quotient lifting node budget')
            current = key[1]
            axis = max(i for i, value in enumerate(current) if value)
            direction = 1 if current[axis] > 0 else -1
            parent = list(current)
            parent[axis] -= direction
            path.append((key, axis, direction))
            key = component, tuple(parent)
        coordinates, lift = cache[key]
        for key, axis, direction in reversed(path):
            budget.work(1+_terms(lift)+_terms(coordinates))
            unit, action, homotopy = homotopies[axis, direction]
            lift = unit*lift+homotopy*coordinates
            coordinates = action*coordinates
            cached_terms += _terms(lift)+_terms(coordinates)
            budget.storage(retained+cached_terms)
            cache[key] = coordinates, lift
            stats.update(cache_nodes=len(cache), cached_terms=cached_terms)
        return cache[component, exponent]

    try:
        requested = set()
        for (i, _), polynomial in B.entries.items():
            budget.work(len(polynomial.terms))
            for exponent in polynomial.terms:
                requested.add((i, exponent))
                if len(requested)+len(cache) > budget.max_nodes:
                    raise ResourceLimit('Finite quotient lifting requested-node budget')
        for component, exponent in sorted(requested, key=lambda key: (sum(map(abs, key[1])), key)):
            get(component, exponent)
        answer = zeros(A.ring, A.ncols, B.ncols)
        remainder = zeros(A.ring, dimension, B.ncols)
        answer_terms = remainder_terms = 0
        for (i, j), polynomial in B.entries.items():
            for exponent, coefficient in polynomial.terms.items():
                coordinates, lift = cache[i, exponent]
                budget.work(1+_terms(coordinates)+_terms(lift))
                for matrix, vector, counter in ((remainder, coordinates, 'remainder'), (answer, lift, 'answer')):
                    for (row, _), value in vector.entries.items():
                        old = matrix[row, j]
                        new = old+coefficient*value
                        matrix[row, j] = new
                        if counter == 'remainder':
                            remainder_terms += len(new.terms)-len(old.terms)
                        else:
                            answer_terms += len(new.terms)-len(old.terms)
                budget.storage(retained+cached_terms+answer_terms+remainder_terms)
        stats.update(cache_nodes=len(cache), cached_terms=cached_terms, solution_terms=answer_terms)
        if remainder:
            stats['shortcut_miss'] = 'nonzero_quotient_remainder'
            return None
        budget.work()
        if A*answer != B:
            raise ArithmeticError('Finite quotient recurrence failed the original Laurent equation')
        return answer
    finally:
        cache.clear()
        homotopies.clear()


def solve_restricted(fine_A, coarse_A, B, lattice, *, quotient=None,
                     max_dimension=10000, max_nodes=20000, max_terms=2000000,
                     max_steps=50000000, timeout=60, stats=None):
    """Solve a restricted equation using exact fine coordinates and optional coker.

    ``quotient``, when provided, must be a FiniteQuotient of exactly fine_A.
    ``stats`` is an optional caller-owned dictionary populated with the actual
    route, retained support, and timings. Shortcut resource exhaustion requests
    the unchanged complete fine-ring solver; it never becomes nonmembership.
    All temporary backend caches are closed before this function returns.
    """
    if not all(isinstance(matrix, Matrix) for matrix in (fine_A, coarse_A, B)):
        raise TypeError('Restricted lifting requires Laurent matrices')
    if not isinstance(lattice, Lattice):
        raise TypeError('Restricted lifting requires an explicit Lattice')
    if type(max_dimension) is not int or max_dimension < 1:
        raise ValueError('max_dimension must be a positive integer')
    if stats is not None and not isinstance(stats, dict):
        raise TypeError('stats must be a dictionary or None')
    stats = {} if stats is None else stats
    stats.clear()
    budget = _LiftBudget(max_nodes, max_terms, max_steps, timeout, stats)
    if fine_A.ring != coarse_A.ring or fine_A.ring != B.ring or B.nrows != coarse_A.nrows:
        raise ValueError('Restricted equation ring or ambient-row mismatch')
    if coarse_A != lattice.restrict(fine_A):
        raise ValueError('Coarse operator is not the declared restriction of the fine operator')
    if quotient is not None and (not isinstance(quotient, FiniteQuotient) or
                                quotient.presentation != fine_A or quotient.ring != fine_A.ring):
        raise ValueError('Supplied quotient must belong to the exact fine presentation')
    fine_B = unpack(lattice, B)
    backend = GeneralBackend(fine_A.ring)
    try:
        if not B:
            answer = zeros(fine_A.ring, fine_A.ncols, B.ncols)
            stats['route'] = 'zero_rhs'
        else:
            try:
                answer = _finite_lift(fine_A, fine_B, backend, quotient, budget, max_dimension)
            except (InfiniteLayer, ResourceLimit) as exc:
                stats.update(shortcut_miss=type(exc).__name__+': '+str(exc))
                answer = None
            if answer is None and fine_A.ncols == 1:
                from .single_generator import single_generator_solution
                answer, division_stats = single_generator_solution(
                    fine_A, fine_B, max_steps=max_steps, max_terms=max_terms, timeout=timeout)
                stats['single_generator'] = division_stats
                if answer is not None:
                    stats['route'] = 'single_generator_quotient'
            if answer is None:
                stats['route'] = 'complete_native_fallback'
                answer = backend.solve(fine_A, fine_B)
            elif stats.get('route') != 'single_generator_quotient':
                stats['route'] = 'finite_quotient_homotopy'
        # The scalar shortcut already verified every fine column in blocks.
        # Other routes retain the wrapper's exact fine check, also in blocks.
        if stats.get('route') != 'single_generator_quotient':
            for column in range(0, fine_B.ncols, 32):
                if fine_A*answer[:, column:column+32] != fine_B[:, column:column+32]:
                    raise ArithmeticError('Fine-ring restricted lift failed its exact equation')
        del fine_B
        packed = lattice.pack(answer)
        del answer
        for column in range(0, B.ncols, 32):
            if coarse_A*packed[:, column:column+32] != B[:, column:column+32]:
                raise ArithmeticError('Packed restricted lift failed its exact coarse equation')
        stats.update(solution_terms=_terms(packed), wall_seconds=perf_counter()-budget.start)
        return packed
    finally:
        backend.close()
