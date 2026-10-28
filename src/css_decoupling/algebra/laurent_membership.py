"""Bounded replay after an existing complete localized module basis.

The polynomial weak remainder uses the original global order and first
dividing basis vector. Only the positive quotient and transition are
evaluated in the Laurent ring. This module does not construct a Groebner
basis, establish its completeness, or issue a nonmembership certificate.
"""
from heapq import heapify, heappop, heappush
import math
from numbers import Integral
from time import perf_counter

from sympy import Integer
from sympy.polys.agca.modules import ModuleOrder
from sympy.polys.domains.modularinteger import ModularInteger
from sympy.polys.orderings import monomial_key
from sympy.ntheory.primetest import isprime

from .laurent import Matrix, Poly
from css_decoupling.errors import ResourceLimit


class _NonzeroRemainder(Exception):
    """Private request for an independent original-input normal-form check."""
    def __init__(self, column, remainder, stats=None):
        super().__init__(f"Polynomial weak remainder in column {column}")
        self.column, self.remainder = column, remainder
        self.stats = {} if stats is None else stats


class _Budget:
    """Cumulative work and retained logical coefficient support, not RSS.

    peak_terms includes prepared/evaluated factors, the verified output
    prefix and current coefficient workspace. Caller inputs, index/heap
    metadata and temporary constructor copies are not coefficient storage
    in this counter. Heap entries are tracked separately; multiplication
    intermediates have a cumulative pair-work preflight before their actual
    retained support is checked.
    """
    def __init__(self, *, max_steps=2000000, max_work=50000000,
                 max_terms=2000000, max_product_work=50000000, timeout=60):
        for name, value in dict(max_steps=max_steps, max_work=max_work,
                max_terms=max_terms, max_product_work=max_product_work).items():
            if type(value) is not int or value < 0:
                raise ValueError(name+" must be a nonnegative integer")
        if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
                or not math.isfinite(timeout) or timeout <= 0):
            raise ValueError("Membership shortcut timeout must be positive and finite")
        self.max_steps, self.max_work = max_steps, max_work
        self.max_terms, self.max_product_work = max_terms, max_product_work
        self.timeout, self.start = timeout, perf_counter()
        self.retained_terms = 0
        self.stats = dict(steps=0, work=0, coefficient_updates=0, product_work=0,
            peak_terms=0, peak_remainder_terms=0, peak_quotient_terms=0,
            peak_queue_entries=0, columns_verified=0, current_column=None,
            prepare_seconds=0.0, normal_form_seconds=0.0,
            transfer_seconds=0.0, verification_seconds=0.0,
            caps=dict(max_steps=max_steps, max_work=max_work, max_terms=max_terms,
                      max_product_work=max_product_work, timeout=timeout))

    def check(self):
        self.stats['elapsed_seconds'] = perf_counter()-self.start
        if self.stats['elapsed_seconds'] > self.timeout:
            raise ResourceLimit("Membership shortcut cumulative wall-time budget")

    def work(self, amount):
        self.check()
        if amount > self.max_work-self.stats['work']:
            raise ResourceLimit("Membership shortcut cumulative work budget")
        self.stats['work'] += amount

    def support(self, amount):
        count = self.retained_terms+amount
        self.stats['peak_terms'] = max(self.stats['peak_terms'], count)
        if count > self.max_terms:
            raise ResourceLimit("Membership shortcut stored coefficient support budget")

    def step(self):
        self.check()
        if self.stats['steps'] >= self.max_steps:
            raise ResourceLimit("Membership shortcut cumulative reduction-step budget")
        self.stats['steps'] += 1

    def snapshot(self):
        self.stats['elapsed_seconds'] = perf_counter()-self.start
        return dict(self.stats)


def _reverse_key(key):
    if isinstance(key, tuple):
        return tuple(_reverse_key(value) for value in key)
    if type(key) is not int:
        raise ValueError("Unsupported noninteger module-order key")
    return -key


def _check_order(order):
    # Limit heap reversal to known global module orders. An unsupported
    # capability requests the unchanged old normal form, not nonmembership.
    if type(order) is not ModuleOrder or order.is_global is not True:
        raise ResourceLimit("Membership shortcut requires a supported global ModuleOrder")
    supported = [ModuleOrder(monomial_key('lex'), monomial_key(name), top)
                 for name in ('lex', 'grlex', 'grevlex') for top in (True, False)]
    if not any(order == candidate for candidate in supported):
        raise ResourceLimit("Membership shortcut module-order layout is unsupported")


def _canonical(vector, order, prime, dimension, rows, budget):
    budget.support(len(vector))
    result, seen, previous = [], set(), None
    for monomial, coefficient in vector:
        budget.work(1)
        monomial = tuple(monomial)
        if (len(monomial) != dimension+2
                or any(type(value) is not int or value < 0 for value in monomial)
                or monomial[0] >= rows or monomial in seen):
            raise ValueError("Malformed or duplicate localized polynomial SDM monomial")
        if isinstance(coefficient, bool) or not isinstance(coefficient, (Integral, Integer, ModularInteger)):
            raise ValueError("Polynomial SDM coefficients must be exact field integers")
        try:
            integer = int(coefficient)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Malformed polynomial SDM coefficient") from exc
        if coefficient != integer or not integer % prime:
            raise ValueError("Polynomial SDM coefficient must be a nonzero field integer")
        key = order(monomial)
        _reverse_key(key)
        if previous is not None and not previous > key:
            raise ValueError("Polynomial SDM is not in canonical decreasing order")
        result.append((monomial, integer % prime))
        seen.add(monomial)
        previous = key
    return result


class _PreparedBasis:
    def __init__(self, basis, indexed, order, prime, dimension, rows, term_count):
        self.basis, self.indexed, self.order = basis, indexed, order
        self.prime, self.dimension, self.rows = prime, dimension, rows
        self.term_count = term_count


def prepare_basis(basis, order, prime, *, dimension, rows, budget=None):
    """Validate/index ordered SDMs; this does not prove basis completeness."""
    budget = _Budget() if budget is None else budget
    if type(dimension) is not int or dimension < 1 or type(rows) is not int or rows < 0:
        raise ValueError("Malformed polynomial module dimensions")
    if type(prime) is not int or not isprime(prime):
        raise ValueError("Malformed coefficient characteristic")
    _check_order(order)
    total = sum(len(vector) for vector in basis)
    budget.support(total+len(basis))
    vectors, indexed = [], {}
    for index, vector in enumerate(basis):
        if not vector:
            raise ValueError("Zero polynomial standard-basis vector")
        terms = _canonical(vector, order, prime, dimension, rows, budget)
        lead, coefficient = terms[0]
        vectors.append(terms)
        indexed.setdefault(lead[0], []).append((index, lead, pow(coefficient, -1, prime), terms))
    return _PreparedBasis(vectors, indexed, order, prime, dimension, rows, total)


def _divide(f, prepared, budget):
    p, order = prepared.prime, prepared.order
    terms = _canonical(f, order, p, prepared.dimension, prepared.rows, budget)
    remaining, quotient = dict(terms), {}
    budget.support(len(remaining))
    queue = [(_reverse_key(order(m)), m) for m in remaining]
    heapify(queue)
    budget.work(len(queue))
    budget.stats['peak_remainder_terms'] = max(budget.stats['peak_remainder_terms'], len(remaining))
    budget.stats['peak_queue_entries'] = max(budget.stats['peak_queue_entries'], len(queue))
    while remaining:
        budget.check()
        while queue and queue[0][1] not in remaining:
            budget.work(1)
            heappop(queue)
        if not queue:
            raise ArithmeticError("Normal-form queue lost a nonzero polynomial term")
        lead = queue[0][1]
        reducer = None
        for entry in prepared.indexed.get(lead[0], ()):
            budget.work(len(lead))
            if all(a <= b for a, b in zip(entry[1][1:], lead[1:])):
                reducer = entry
                break
        if reducer is None:
            break
        index, leading, inverse, generator = reducer
        budget.step()
        budget.work(len(generator))
        budget.stats['coefficient_updates'] += len(generator)
        scale = remaining[lead]*inverse % p
        shift = tuple(b-a for a, b in zip(leading[1:], lead[1:]))
        exponent = tuple(a-shift[-1] for a in shift[:-1])
        key = index, exponent
        coefficient = (quotient.get(key, 0)+scale) % p
        if coefficient:
            if key not in quotient:
                budget.support(len(remaining)+len(quotient)+1)
            quotient[key] = coefficient
        else:
            quotient.pop(key, None)
        for monomial, coefficient in generator:
            target = (monomial[0],)+tuple(a+b for a, b in zip(monomial[1:], shift))
            old = remaining.get(target, 0)
            value = (old-scale*coefficient) % p
            if value:
                if not old:
                    budget.support(len(remaining)+len(quotient)+1)
                    # Bound lazy queue storage before adding another entry.
                    if len(queue) >= max(64, 2*len(remaining)):
                        budget.work(len(remaining))
                        queue = [(_reverse_key(order(m)), m) for m in remaining]
                        heapify(queue)
                    heappush(queue, (_reverse_key(order(target)), target))
                remaining[target] = value
            else:
                remaining.pop(target, None)
        for name, value in (('peak_remainder_terms', len(remaining)),
                            ('peak_quotient_terms', len(quotient)), ('peak_queue_entries', len(queue))):
            budget.stats[name] = max(budget.stats[name], value)
    budget.work(len(remaining))
    remainder = sorted(remaining.items(), key=lambda pair: order(pair[0]), reverse=True)
    return remainder, quotient


def divide(f, prepared, *, budget=None):
    """Return polynomial weak remainder, positive Laurent quotient and totals."""
    budget = _Budget() if budget is None else budget
    start = perf_counter()
    try:
        remainder, quotient = _divide(f, prepared, budget)
    finally:
        budget.stats['normal_form_seconds'] += perf_counter()-start
    return remainder, quotient, budget.snapshot()


def _terms(matrix):
    return sum(len(value.terms) for value in matrix.entries.values())


def _product(left, right, budget):
    budget.work(len(left.entries)+len(right.entries))
    rows = {}
    for (row, _), value in right.entries.items():
        rows[row] = rows.get(row, 0)+len(value.terms)
    work = sum(len(value.terms)*rows.get(column, 0) for (_, column), value in left.entries.items())
    if work > budget.max_product_work-budget.stats['product_work']:
        raise ResourceLimit("Membership shortcut cumulative product-work budget")
    budget.stats['product_work'] += work
    # The preflight pair bound also bounds possible transient product terms;
    # check the actual retained support afterwards. Neither cap is an RSS cap.
    answer = left*right
    budget.support(_terms(answer))
    budget.check()
    return answer


def _evaluate(vectors, ring, rows, mapping, budget):
    entries, count = {}, 0
    for column, vector in enumerate(vectors):
        for monomial, coefficient in vector:
            budget.work(1)
            row = mapping(monomial[0])
            if row is None:
                continue
            exponent = tuple(value-monomial[-1] for value in monomial[1:-1])
            values = entries.setdefault((row, column), {})
            old = values.get(exponent, 0)
            value = (old+coefficient) % ring.p
            if value:
                if not old:
                    budget.support(count+1)
                    count += 1
                values[exponent] = value
            elif old:
                del values[exponent]
                count -= 1
    return Matrix(ring, rows, len(vectors), {at: Poly(ring, values) for at, values in entries.items()})


def membership_columns(A, B, basis_data, *, order, vector_sdm,
                       max_steps=2000000, max_work=50000000, max_terms=2000000,
                       max_product_work=50000000, timeout=60, observer=None):
    """Yield consecutive verified (column, solution) pairs from complete G.

    The 60-second default is cumulative across this one shortcut call, not
    renewed per RHS. Logical work/support limits are not RSS bounds. A caller
    may retain yielded columns on ResourceLimit and use its old exact normal
    form for the current and subsequent original RHS columns.
    """
    if not isinstance(A, Matrix) or not isinstance(B, Matrix) or A.ring != B.ring or A.nrows != B.nrows:
        raise ValueError("Membership replay requires one ring and matching rows")
    budget = _Budget(max_steps=max_steps, max_work=max_work, max_terms=max_terms,
                     max_product_work=max_product_work, timeout=timeout)
    def emit(event, **extra):
        if observer is not None:
            observer(dict(event=event, stats=budget.snapshot(), **extra))
    start, preparation_complete = budget.start, False
    try:
        emit('prepare-start')
        start = perf_counter()
        basis, transition, active = basis_data
        if len(basis) != len(transition):
            raise ValueError("Basis and transition column counts differ")
        budget.work(len(A.entries)+len(active))
        budget.support(len(active))
        expected = {column for _, column in A.entries}
        if len(active) != len(expected)+A.nrows:
            raise ValueError("Active mapping has the wrong number of source generators")
        expected.update(range(A.ncols, A.ncols+A.nrows))
        if (any(type(index) is not int for index in active) or len(set(active)) != len(active)
                or set(active) != expected):
            raise ValueError("Active mapping differs from original nonzero columns and localization relations")
        budget.work(len(active))
        prepared = prepare_basis(basis, order, A.ring.p, dimension=A.ring.D, rows=A.nrows, budget=budget)
        budget.retained_terms = prepared.term_count
        transition_terms = sum(len(vector) for vector in transition)
        budget.support(transition_terms+len(basis))
        tracked = [_canonical(vector, order, A.ring.p, A.ring.D, len(active), budget)
                   for vector in transition]
        budget.retained_terms += transition_terms
        G = _evaluate(prepared.basis, A.ring, A.nrows, lambda i: i, budget)
        g_terms = _terms(G)
        budget.retained_terms += g_terms
        T = _evaluate(tracked, A.ring, A.ncols,
                      lambda i: active[i] if active[i] < A.ncols else None, budget)
        budget.retained_terms += _terms(T)
        if _product(A, T, budget) != G:
            raise ArithmeticError("Membership transition failed A*T=G_eval")
        # Retain the prepared basis and evaluated transition; G and the
        # polynomial transition copy are no longer needed after this check.
        budget.retained_terms -= g_terms+transition_terms
        budget.support(0)
        del G, tracked
        budget.stats['prepare_seconds'] = perf_counter()-start
        preparation_complete = True
        emit('prepare-complete')
        for j, column in enumerate(B.columns()):
            budget.stats['current_column'] = j
            budget.check()
            emit('column-start', column=j)
            remainder, quotient, _ = divide(vector_sdm(column), prepared, budget=budget)
            if remainder:
                emit('polynomial-remainder', column=j, remainder_terms=len(remainder))
                raise _NonzeroRemainder(j, remainder, budget.snapshot())
            budget.support(2*len(quotient))
            entries = {}
            for (index, exponent), coefficient in quotient.items():
                budget.work(1)
                entries.setdefault((index, 0), {})[exponent] = coefficient
            Q = Matrix(A.ring, len(basis), 1, {at: Poly(A.ring, values) for at, values in entries.items()})
            q_terms = _terms(Q)
            budget.retained_terms += q_terms
            del quotient, entries
            start = perf_counter()
            try:
                solution = _product(T, Q, budget)  # Positive quotient; no phantom minus sign.
            finally:
                budget.stats['transfer_seconds'] += perf_counter()-start
            budget.retained_terms += _terms(solution)
            del Q
            budget.retained_terms -= q_terms
            start = perf_counter()
            try:
                target_terms = sum(len(value.terms) for value in column)
                budget.support(target_terms)
                target = Matrix(A.ring, A.nrows, 1, {(i, 0): value for i, value in enumerate(column) if value})
                budget.retained_terms += target_terms
                if _product(A, solution, budget) != target:
                    raise ArithmeticError("Membership column failed A*x_j=B_j")
                del target
                budget.retained_terms -= target_terms
            finally:
                budget.stats['verification_seconds'] += perf_counter()-start
            budget.support(0)
            budget.check()
            budget.stats['columns_verified'] += 1
            emit('column-verified', column=j)
            yield j, solution
        budget.stats['current_column'] = None
        emit('complete')
    except ResourceLimit as exc:
        if not preparation_complete:
            budget.stats['prepare_seconds'] = perf_counter()-start
        exc.membership_stats = budget.snapshot()
        emit('resource-incomplete', reason=str(exc))
        raise
