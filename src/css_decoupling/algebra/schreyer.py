"""Native exact Schreyer resolutions over prime polynomial rings.

The implementation follows the constructive Hilbert syzygy argument in
Berkesch--Schreyer, *Syzygies, finite length modules, and random curves*, §1.
At each level generators are sorted by the exponent of the last variable
occurring in their leading terms.  Minimal monomial colon generators then
produce a syzygy Gröbner basis whose leading terms omit that variable.
This strictly descending invariant, rather than a trial degree or repeated
unstructured kernel computation, bounds the number of resolution stages.

Vectors are sparse dictionaries ``(component, *exponents) -> coefficient``.
Only exact Python integer arithmetic modulo p is used; no CAS subprocess or
fraction-field division occurs.  Gröbner bases can still be expensive.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from heapq import heapify, heappop, heappush
from math import isfinite
from time import perf_counter

from css_decoupling.algebra.laurent import Matrix, Poly, eye, zeros
from css_decoupling.errors import ResourceLimit


@dataclass
class _Budget:
    timeout: float | None
    max_basis: int
    max_terms: int
    max_pairs: int
    start: float = field(default_factory=perf_counter)
    pairs: int = 0

    def __post_init__(self):
        if self.timeout is not None and (isinstance(self.timeout, bool) or
                                        not isfinite(self.timeout) or self.timeout <= 0):
            raise ValueError("Native resolution timeout must be positive or None")
        for value in (self.max_basis, self.max_terms, self.max_pairs):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("Native resolution limits must be positive integers")

    def check(self, *vectors, basis_size=0):
        if self.timeout is not None and perf_counter() - self.start > self.timeout:
            raise ResourceLimit(f"Native Schreyer resolution exceeded {self.timeout} seconds")
        if basis_size > self.max_basis:
            raise ResourceLimit(f"Native Schreyer basis exceeds {self.max_basis} generators")
        if sum(map(len, vectors)) > self.max_terms:
            raise ResourceLimit(f"Native Schreyer working vectors exceed {self.max_terms} terms")

    def pair(self):
        self.pairs += 1
        if self.pairs > self.max_pairs:
            raise ResourceLimit(f"Native Schreyer computation exceeds {self.max_pairs} critical pairs")
        self.check()


class _Order:
    """A global grevlex module order or an induced Schreyer order."""

    def __init__(self, parent=None, leading=None):
        self.parent = parent
        self.leading = leading

    def __call__(self, monomial):
        component, *exponent = monomial
        if self.parent is None:
            return (sum(exponent), tuple(-a for a in reversed(exponent)), component)
        image = self.leading[component]
        translated = (image[0],) + tuple(a + b for a, b in zip(image[1:], exponent))
        return (self.parent(translated), component)


def _divides(a, b):
    return a[0] == b[0] and all(x <= y for x, y in zip(a[1:], b[1:]))


def _add_multiple(target, source, exponent, coefficient, p, budget, *, enqueue=None):
    coefficient %= p
    if not coefficient:
        return
    for at, value in source.items():
        translated = (at[0],) + tuple(a + b for a, b in zip(at[1:], exponent))
        old = target.get(translated, 0)
        new = (old + coefficient * value) % p
        if new:
            target[translated] = new
            if not old and enqueue is not None:
                enqueue(translated)
        else:
            target.pop(translated, None)
    budget.check(target)


def _monic(vector, order, p):
    leading = max(vector, key=order)
    inverse = pow(vector[leading], -1, p)
    return {m: c * inverse % p for m, c in vector.items()}


class _DivisionTerm:
    """A max-order heap entry with one cached, possibly nested order key."""

    __slots__ = ("monomial", "key")

    def __init__(self, monomial, order):
        self.monomial = monomial
        self.key = order(monomial)

    def __lt__(self, other):
        return self.key > other.key


def _divide(vector, basis, order, p, budget, *, track=False):
    """Exact ordered division, without rescanning the whole support per step.

    The heap orders the same leading terms as max(work,key=order). Deleted
    terms are discarded lazily. A queued set avoids duplicate heap entries
    when a cancelled term reappears before its stale entry has been popped.
    Rebuilding when stale entries dominate keeps storage linear in the live
    support. Divisor order and all coefficient/quotient operations are the
    original division algorithm's operations, including for Schreyer orders.
    """
    work, remainder, quotient = vector.copy(), {}, {}
    budget.check(work)
    leading = [max(g, key=order) for g in basis]
    pending = [_DivisionTerm(monomial, order) for monomial in work]
    heapify(pending)
    queued = set(work)

    def enqueue(monomial):
        if monomial not in queued:
            queued.add(monomial)
            heappush(pending, _DivisionTerm(monomial, order))

    while work:
        budget.check(work, remainder, quotient)
        while pending:
            head = heappop(pending).monomial
            queued.remove(head)
            if head in work:
                break
        else:
            raise ArithmeticError("Ordered-division heap lost a live monomial")
        coefficient = work[head]
        for i, divisor in enumerate(leading):
            if _divides(divisor, head):
                exponent = tuple(a - b for a, b in zip(head[1:], divisor[1:]))
                factor = coefficient * pow(basis[i][divisor], -1, p) % p
                _add_multiple(work, basis[i], exponent, -factor, p, budget, enqueue=enqueue)
                if track:
                    at = (i,) + exponent
                    value = (quotient.get(at, 0) + factor) % p
                    if value:
                        quotient[at] = value
                    else:
                        quotient.pop(at, None)
                break
        else:
            remainder[head] = work.pop(head)
        if len(pending) > 2*len(work) + 64:
            pending = [_DivisionTerm(monomial, order) for monomial in work]
            heapify(pending)
            queued = set(work)
    return remainder, quotient


def _groebner(generators, order, p, budget, *, track=False):
    """Buchberger completion with optional expressions in the input columns."""
    basis, leading, pending, expressions = [], [], [], []

    def subtract_quotient(expression, quotient, source):
        for (index, *exponent), coefficient in quotient.items():
            _add_multiple(expression, source[index], exponent, -coefficient, p, budget)
        return expression

    def append(vector, expression=None):
        inverse = pow(vector[max(vector, key=order)], -1, p)
        vector = {m: c * inverse % p for m, c in vector.items()}
        if track:
            expressions.append({m: c * inverse % p for m, c in expression.items()})
        head = max(vector, key=order)
        i = len(basis)
        for j, other in enumerate(leading):
            if head[0] == other[0]:
                exponent = tuple(max(a, b) for a, b in zip(head[1:], other[1:]))
                heappush(pending, (sum(exponent), i, j, exponent))
                if len(pending) > budget.max_pairs:
                    raise ResourceLimit("Native Schreyer pending critical-pair limit exceeded")
        basis.append(vector)
        leading.append(head)
        budget.check(*basis, *expressions, basis_size=len(basis))

    for index, generator in enumerate(generators):
        remainder, quotient = _divide(generator, basis, order, p, budget, track=track)
        if remainder:
            expression = None
            if track:
                # Zero columns retain their original positions in the input.
                exponent = (0,) * (len(next(iter(generator))) - 1)
                expression = subtract_quotient({(index,) + exponent: 1}, quotient, expressions)
            append(remainder, expression)
    while pending:
        budget.pair()
        _, i, j, exponent = heappop(pending)
        pair, expression = {}, {} if track else None
        for index, coefficient in ((i, 1), (j, -1)):
            shift = tuple(a - b for a, b in zip(exponent, leading[index][1:]))
            _add_multiple(pair, basis[index], shift, coefficient, p, budget)
            if track:
                _add_multiple(expression, expressions[index], shift, coefficient, p, budget)
        remainder, quotient = _divide(pair, basis, order, p, budget, track=track)
        if remainder:
            if track:
                subtract_quotient(expression, quotient, expressions)
            append(remainder, expression)
    # Delete a generator only when another leading term divides its own;
    # equal leading terms retain the earliest representative.
    kept = [i for i, g in enumerate(basis) if not any(
        i != j and _divides(h, leading[i]) and (h != leading[i] or j < i)
        for j, h in enumerate(leading))]
    minimal = [basis[i] for i in kept]
    minimal_expressions = [expressions[i] for i in kept] if track else []
    reduced, reduced_expressions = [], []
    for i, generator in enumerate(minimal):
        remainder, quotient = _divide(generator, minimal[:i] + minimal[i + 1:],
                                      order, p, budget, track=track)
        if not remainder:
            raise ArithmeticError("Minimal Gröbner generator unexpectedly reduced to zero")
        if track:
            expression = subtract_quotient(minimal_expressions[i].copy(), quotient,
                                          minimal_expressions[:i] + minimal_expressions[i + 1:])
            inverse = pow(remainder[max(remainder, key=order)], -1, p)
            reduced_expressions.append({m: c * inverse % p for m, c in expression.items()})
        reduced.append(_monic(remainder, order, p))
        budget.check(*reduced, *reduced_expressions, basis_size=len(reduced))
    return (reduced, reduced_expressions) if track else reduced


def _last_variable(leading):
    return max((k for head in leading for k, power in enumerate(head[1:]) if power), default=-1)


def _sorted_basis(basis, order):
    leading = [max(g, key=order) for g in basis]
    last = _last_variable(leading)
    indices = sorted(range(len(basis)), key=lambda i: (
        leading[i][last + 1] if last >= 0 else 0, leading[i]))
    return [basis[i] for i in indices], [leading[i] for i in indices], last


def _syzygies(basis, leading, order, p, budget):
    """Schreyer generators from minimal generators of (LT(g_<i>):LT(g_i)."""
    syzygies = []
    induced = _Order(order, leading)
    for i, head in enumerate(leading):
        colon = []
        for earlier in leading[:i]:
            if earlier[0] != head[0]:
                continue
            exponent = tuple(max(0, a - b) for a, b in zip(earlier[1:], head[1:]))
            if any(all(a <= b for a, b in zip(old, exponent)) for old in colon):
                continue
            colon = [old for old in colon if not all(a <= b for a, b in zip(exponent, old))]
            colon.append(exponent)
        for exponent in sorted(colon):
            budget.pair()
            multiple = {}
            _add_multiple(multiple, basis[i], exponent, 1, p, budget)
            remainder, quotient = _divide(multiple, basis, order, p, budget, track=True)
            if remainder:
                raise ArithmeticError("Schreyer input is not a Gröbner basis")
            syzygy = {m: -c % p for m, c in quotient.items()}
            expected = (i,) + exponent
            coefficient = (syzygy.get(expected, 0) + 1) % p
            if coefficient:
                syzygy[expected] = coefficient
            else:
                syzygy.pop(expected, None)
            if not syzygy or max(syzygy, key=induced) != expected or coefficient != 1:
                raise ArithmeticError("Schreyer leading-term invariant failed")
            syzygies.append(syzygy)
            budget.check(*syzygies, basis_size=len(syzygies))
    return syzygies, induced


def _matrix(ring, nrows, vectors):
    entries = {}
    for j, vector in enumerate(vectors):
        for (i, *exponent), coefficient in vector.items():
            entries.setdefault((i, j), {})[tuple(exponent)] = coefficient
    return Matrix(ring, nrows, len(vectors), {at: Poly(ring, f) for at, f in entries.items()})


def _vectors(A, budget):
    """Polynomial column vectors, retaining zero columns and their indices."""
    if A.ncols > budget.max_basis:
        raise ResourceLimit(f"Native Schreyer input exceeds {budget.max_basis} columns")
    generators = [{} for _ in range(A.ncols)]
    for (i, j), f in A.entries.items():
        for exponent, coefficient in f.terms.items():
            if min(exponent, default=0) < 0:
                raise ValueError("Native polynomial resolution received a Laurent exponent")
            generators[j][(i,) + exponent] = coefficient
    budget.check(*generators, basis_size=len(generators))
    return generators


def _tracked_groebner(A, budget):
    generators = _vectors(A, budget)
    order = _Order()
    basis, expressions = _groebner(generators, order, A.ring.p, budget, track=True)
    expression_by_id = {id(vector): expression for vector, expression in zip(basis, expressions)}
    basis, _, _ = _sorted_basis(basis, order)
    expressions = [expression_by_id[id(vector)] for vector in basis]
    reverse = []
    for generator in generators:
        remainder, quotient = _divide(generator, basis, order, A.ring.p, budget, track=True)
        if remainder:
            raise ArithmeticError("Original column did not reduce by its native Gröbner basis")
        reverse.append(quotient)
        budget.check(*reverse)
    return {
        "groebner_generators": _matrix(A.ring, A.nrows, basis),
        "basis_in_input": _matrix(A.ring, A.ncols, expressions),
        "input_in_basis": _matrix(A.ring, len(basis), reverse),
    }


def _verify_image_witness(A, witness):
    G, T, S = (witness[key] for key in
               ("groebner_generators", "basis_in_input", "input_in_basis"))
    if any(not isinstance(M, Matrix) or M.ring != A.ring for M in (G, T, S)):
        raise ValueError("Native image witness requires matrices in the input ring")
    if G.nrows != A.nrows or T.shape != (A.ncols, G.ncols) or S.shape != (G.ncols, A.ncols):
        raise ValueError("Native image witness has incompatible dimensions")
    if A * T != G or G * S != A:
        raise ArithmeticError("Native Gröbner image witnesses failed their two exact identities")
    return True


def polynomial_groebner(A, *, timeout=120, max_basis=100000,
                        max_terms=2000000, max_pairs=1000000, verify=True):
    """Compute a polynomial Gröbner basis with both input-change witnesses.

    The returned dictionary contains ``groebner_generators=G``,
    ``basis_in_input=T`` and ``input_in_basis=S``, satisfying ``A*T=G`` and
    ``G*S=A``.  Tracking uses the actual Buchberger operations; the reverse
    expression uses ordinary division by G, never Laurent localization.
    ``verify`` checks both matrix identities, not a second GB computation.
    """
    budget = _Budget(timeout, max_basis, max_terms, max_pairs)
    result = _tracked_groebner(A, budget)
    if verify:
        _verify_image_witness(A, result)
    return result


def _compact_columns(A):
    """Delete only zero and literally duplicate columns; record those kept."""
    by_column = {}
    for (row, column), value in A.entries.items():
        by_column.setdefault(column, {})[row] = value
    seen, kept, entries = set(), [], {}
    for column, vector in sorted(by_column.items()):
        key = tuple((row, value.key()) for row, value in sorted(vector.items()))
        if key in seen:
            continue
        seen.add(key)
        for row, value in vector.items():
            entries[row, len(kept)] = value
        kept.append(column)
    return Matrix(A.ring, A.nrows, len(kept), entries), tuple(kept)


def _verify_polynomial_kernel(A, witness, budget):
    """Check completeness through all S-pair Schreyer relations."""
    _verify_image_witness(A, witness)
    G, T, S, H, K = (witness[key] for key in (
        "groebner_generators", "basis_in_input", "input_in_basis",
        "schreyer_kernel", "kernel"))
    if H.ring != A.ring or H.nrows != G.ncols or K.ring != A.ring or K.nrows != A.ncols:
        raise ValueError("Native kernel witness has incompatible dimensions")
    for M in (A, T, S):
        _vectors(M, budget)
    basis, syzygies = _vectors(G, budget), _vectors(H, budget)
    if any(not g for g in basis) or any(not h for h in syzygies):
        raise ValueError("Native Gröbner and Schreyer bases may not contain zero columns")
    order = _Order()
    leading = [max(g, key=order) for g in basis]
    if any(g[head] != 1 for g, head in zip(basis, leading)):
        raise ValueError("Native Gröbner witness must be monic")
    if G * H:
        raise ArithmeticError("Schreyer columns are not actual Gröbner syzygies")
    induced = _Order(order, leading)
    # Independent of the constructor's minimal colon selection: check every
    # pair of equal leading components.  Buchberger's criterion proves G is
    # complete.  The associated full Schreyer relations generate ker(G);
    # reducing all of them to zero by H proves their inclusion in im(H).
    for i, head in enumerate(leading):
        for j in range(i):
            other = leading[j]
            if head[0] != other[0]:
                continue
            budget.pair()
            exponent = tuple(max(a, b) for a, b in zip(head[1:], other[1:]))
            pair, relation = {}, {}
            for index, coefficient in ((i, 1), (j, -1)):
                shift = tuple(a - b for a, b in zip(exponent, leading[index][1:]))
                _add_multiple(pair, basis[index], shift, coefficient, A.ring.p, budget)
                relation[(index,) + shift] = coefficient % A.ring.p
            remainder, quotient = _divide(pair, basis, order, A.ring.p, budget, track=True)
            if remainder:
                raise ArithmeticError("Native Gröbner witness fails Buchberger's criterion")
            for monomial, coefficient in quotient.items():
                value = (relation.get(monomial, 0) - coefficient) % A.ring.p
                if value:
                    relation[monomial] = value
                else:
                    relation.pop(monomial, None)
            remainder, _ = _divide(relation, syzygies, induced, A.ring.p, budget)
            if remainder:
                raise ArithmeticError("Schreyer witness does not generate the complete kernel")
    full = (T * H).hstack(eye(A.ring, A.ncols) - T * S)
    expected, selected = _compact_columns(full)
    if tuple(witness["selected_columns"]) != selected or K != expected:
        raise ArithmeticError("Original kernel generators changed their complete transfer witnesses")
    if A * K:
        raise ArithmeticError("Transferred native kernel failed its exact identity")
    return True


def verify_polynomial_kernel(A, witness, *, timeout=120, max_basis=100000,
                             max_terms=2000000, max_pairs=1000000):
    """Verify a full polynomial kernel certificate without recomputing a GB.

    This checks all S-pairs and their induced relations, both image-change
    identities and the formula ``ker(A)=im[T*H, I-T*S]``.  It never accepts
    ``A*K=0`` alone as a completeness argument.  Invalid witnesses raise;
    successful verification returns True.  Resource failures stay distinct.
    """
    return _verify_polynomial_kernel(A, witness, _Budget(timeout, max_basis, max_terms, max_pairs))


def polynomial_kernel(A, *, timeout=120, max_basis=100000, max_terms=2000000,
                      max_pairs=1000000, verify=True, return_witness=False):
    """Complete kernel from one tracked GB and its first Schreyer module.

    If ``G=A*T`` and ``A=G*S``, then every v in ker(A) has S*v in ker(G).
    Therefore H spanning ker(G) gives ``v=T*H*w+(I-T*S)*v``.  Both inclusions
    are certified; no longer free resolution is constructed here.  Only
    zero and literally duplicate columns are removed from this formula.
    By default verification uses the same cumulative construction budget.
    """
    budget = _Budget(timeout, max_basis, max_terms, max_pairs)
    result = _tracked_groebner(A, budget)
    G, T, S = (result[key] for key in ("groebner_generators", "basis_in_input", "input_in_basis"))
    basis = _vectors(G, budget)
    order = _Order()
    leading = [max(g, key=order) for g in basis]
    syzygies, _ = _syzygies(basis, leading, order, A.ring.p, budget)
    H = _matrix(A.ring, G.ncols, syzygies)
    K, selected = _compact_columns((T * H).hstack(eye(A.ring, A.ncols) - T * S))
    result.update(kernel=K, schreyer_kernel=H, selected_columns=selected)
    if verify:
        _verify_polynomial_kernel(A, result, budget)
    return result if return_witness else K


def polynomial_image_resolution(A, *, timeout=120, max_basis=100000,
                                max_terms=2000000, max_pairs=1000000,
                                return_image_witness=False):
    """Resolve the image of a polynomial matrix without external algebra tools.

    Output matrices start at a Gröbner presentation of im(A), then its
    successive Schreyer syzygies.  The first presentation may use a different
    generator set from A.  A must have no negative exponents.  All limits
    raise ResourceLimit; none mean that a module or a kernel is zero.
    Default output remains the list of matrices. ``return_image_witness``
    instead returns a dict with that list under ``maps`` and matrices T,S
    satisfying ``A*T=maps[0]`` and ``maps[0]*S=A``.
    """
    budget = _Budget(timeout, max_basis, max_terms, max_pairs)
    order = _Order()
    witness = _tracked_groebner(A, budget) if return_image_witness else None
    if witness is None:
        generators = _vectors(A, budget)
        basis = _groebner([g for g in generators if g], order, A.ring.p, budget)
    else:
        basis = _vectors(witness["groebner_generators"], budget)
    if not basis:
        maps = [Matrix(A.ring, A.nrows, 0)]
        return dict(witness, maps=maps) if witness is not None else maps
    maps, ambient, previous_last = [], A.nrows, A.ring.D
    while basis:
        basis, leading, last = _sorted_basis(basis, order)
        if last >= previous_last:
            raise ArithmeticError("Schreyer resolution failed to eliminate a leading variable")
        previous_last = last
        maps.append(_matrix(A.ring, ambient, basis))
        if len(maps) > A.ring.D:
            raise ArithmeticError("Minimal Schreyer resolution exceeded the Hilbert length bound")
        ambient = len(basis)
        basis, order = _syzygies(basis, leading, order, A.ring.p, budget)
        budget.check()
    return dict(witness, maps=maps) if witness is not None else maps
