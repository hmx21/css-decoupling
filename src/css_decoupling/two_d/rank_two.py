"""Bounded same-cell rank-two factorization with an exact toric GE2 bridge.

The row search is a positive, bounded search, not a general E2 decision
algorithm. Every successful path returns actual factors of the specified U.
No construction matrix, model name, additional cell, or ancilla is supplied.
"""
import heapq

from .algebra import Matrix, eye
from .rank_two_budget import (
    RankTwoBudget, RankTwoIncomplete, RankTwoResourceLimit, ProbeStopped, natural,
)
from .target_ge2 import (
    associate_unit, invert_atoms, join, ge_replay, factor_target_automorphism,
    separate_atoms,
)


DEFAULT_OPTIONS = {'search_width': 64, 'search_depth': 32, 'max_nodes': 400000,
                   'max_quotient_steps': 1000, 'max_quotient_terms': 64}


def parse_rank_two_options(options=None):
    if options is None:
        return dict(DEFAULT_OPTIONS)
    if not isinstance(options, dict):
        raise ValueError('rank_two_options must be a dictionary')
    unknown = set(options)-set(DEFAULT_OPTIONS)
    if unknown:
        raise ValueError('Unknown rank_two_options: '+', '.join(sorted(map(str, unknown))))
    result = {**DEFAULT_OPTIONS, **options}
    for name, value in result.items():
        natural(value, name, minimum=0 if name == 'search_depth' else 1)
    return result


def _elementary(ring, i, j, coefficient):
    E = eye(ring, 2)
    E[i, j] = coefficient
    return E


def _signed_swap(ring):
    return [('e', 0, 1, ring.one()), ('e', 1, 0, -ring.one()),
            ('e', 0, 1, ring.one())]


def _row_add(A, i, j, coefficient, budget):
    result = A.copy()
    for k in range(2):
        result[i, k] += budget.multiply(coefficient, A[j, k])
    budget.check(result)
    return result


def _column_add(A, target, source, coefficient, budget):
    result = A.copy()
    for k in range(2):
        result[k, target] += budget.multiply(A[k, source], coefficient)
    budget.check(result)
    return result


def _fast_cancellation(A, budget):
    """Existing unit/monomial reduction strategy, charged to the shared budget."""
    ring, best = A.ring, None
    for side in (0, 1):
        for target, source in ((0, 1), (1, 0)):
            old = [A[target, k] if side == 0 else A[k, target] for k in range(2)]
            src = [A[source, k] if side == 0 else A[k, source] for k in range(2)]
            seen = set()
            for f, g in zip(old, src):
                for ef, cf in sorted(f.terms.items()):
                    for eg, cg in sorted(g.terms.items()):
                        budget.check()
                        shift = tuple(a-b for a, b in zip(ef, eg))
                        scalar = -cf*pow(cg, -1, ring.p) % ring.p
                        key = (shift, scalar)
                        if key in seen:
                            continue
                        seen.add(key)
                        budget.check(terms=len(seen))
                        coefficient = ring.monomial(shift, scalar)
                        updated = [a+budget.multiply(coefficient, b) for a, b in zip(old, src)]
                        change = sum(len(f.terms) for f in updated)-sum(len(f.terms) for f in old)
                        unit = any(f.is_unit for f in updated)
                        if change < 0 or unit:
                            score = (not unit, change, side, target, source, shift, scalar)
                            if best is None or score < best[0]:
                                best = (score, side, target, source, coefficient)
    return best


def _fast_atoms(U, budget):
    A, left, right = U.copy(), [], []
    ring = U.ring
    while True:
        budget.check(A)
        positions = [at for at, value in A.entries.items() if value.is_unit]
        if positions:
            i, j = min(positions)
            swap = _signed_swap(ring)
            if i:
                for _, a, b, c in swap:
                    A = _row_add(A, a, b, c, budget)
                    left.append(('e', a, b, c))
            if j:
                for _, a, b, c in swap:
                    A = _column_add(A, b, a, c, budget)
                    right.append(('e', a, b, c))
            pivot, top, bottom = A[0, 0], A[0, 1], A[1, 0]
            reciprocal = pivot**-1
            lower = budget.multiply(bottom, reciprocal)
            upper = budget.multiply(top, reciprocal)
            diagonal = A[1, 1]-budget.multiply(lower, top)
            if not diagonal.is_unit:
                raise ArithmeticError('Invertible rank-two unit elimination lost its unit complement')
            # left is chronological; its inverse is this written list.
            inverse_left = [('e', a, b, -c) for _, a, b, c in left]
            return join(budget, inverse_left,
                        [('e', 1, 0, lower), ('d', 0, -1, pivot),
                         ('d', 1, -1, diagonal), ('e', 0, 1, upper)],
                        invert_atoms(right))
        candidate = _fast_cancellation(A, budget)
        if candidate is None:
            return None
        _, side, target, source, coefficient = candidate
        if side == 0:
            A = _row_add(A, target, source, coefficient, budget)
            left.append(('e', target, source, coefficient))
        else:
            A = _column_add(A, target, source, coefficient, budget)
            right.append(('e', source, target, coefficient))
        budget.word(left+right)


def _normalize(f, budget):
    low = tuple(min(e[i] for e in f.terms) for i in range(2))
    coefficient = f.terms[min(f.terms)]
    unit = f.ring.monomial(low, coefficient)
    return budget.multiply(f, unit**-1), unit


def _key(row):
    return tuple(tuple(sorted(f.terms.items())) for f in row)


def _score(row):
    def span(f):
        return sum(max(e[i] for e in f.terms)-min(e[i] for e in f.terms) for i in range(2)) if f else 0
    sizes = [len(f.terms) for f in row]
    return sum(sizes), sum(map(span, row)), min(sizes)


def _goal(row, budget):
    x, y = row[0].ring.variables
    for swapped, targets in ((False, (x-1, y-1)), (True, (y-1, x-1))):
        if all(associate_unit(f, g, budget) is not None for f, g in zip(row, targets)):
            return swapped
    return None


def _quotient(f, g, swap, signs, budget, options):
    """Bounded ordinary division after a reversible Laurent support gauge.

    This only proposes an elementary coefficient. A missed quotient never
    establishes nonmembership; the actual candidate row is always replayed.
    """
    def transform(e):
        e = e[::-1] if swap else e
        return tuple(a*b for a, b in zip(e, signs))

    def back(e):
        e = tuple(a*b for a, b in zip(e, signs))
        return e[::-1] if swap else e

    F = {transform(e): c for e, c in f.terms.items()}
    G = {transform(e): c for e, c in g.terms.items()}
    fm = tuple(min(e[i] for e in F) for i in range(2))
    gm = tuple(min(e[i] for e in G) for i in range(2))
    F = {tuple(a-b for a, b in zip(e, fm)): c for e, c in F.items()}
    G = {tuple(a-b for a, b in zip(e, gm)): c for e, c in G.items()}
    lead = lambda e: (sum(e), e)
    eg = max(G, key=lead)
    inverse = pow(G[eg], -1, f.ring.p)
    quotient, steps = {}, 0
    while F:
        budget.check(terms=len(F)+len(G)+len(quotient))
        steps += 1
        if steps > options['max_quotient_steps'] or len(quotient) > options['max_quotient_terms']:
            return f.ring.zero()
        ef = max(F, key=lead)
        cf = F[ef]
        if all(a >= b for a, b in zip(ef, eg)):
            exponent = tuple(a-b for a, b in zip(ef, eg))
            coefficient = cf*inverse % f.ring.p
            if exponent not in quotient and len(quotient) >= options['max_quotient_terms']:
                return f.ring.zero()
            quotient[exponent] = (quotient.get(exponent, 0)+coefficient) % f.ring.p
            budget.charge(len(G))
            for e, c in G.items():
                key = tuple(a+b for a, b in zip(e, exponent))
                value = (F.get(key, 0)-coefficient*c) % f.ring.p
                if value:
                    F[key] = value
                else:
                    F.pop(key, None)
        else:
            F.pop(ef)
    return f.ring({back(tuple(e[i]+fm[i]-gm[i] for i in range(2))): c
                   for e, c in quotient.items() if c})


def _candidates(row, budget, options):
    ring = row[0].ring
    for target, source in ((0, 1), (1, 0)):
        f, g = row[target], row[source]
        seen, retained = set(), 0
        for ef, cf in sorted(f.terms.items()):
            for eg, cg in sorted(g.terms.items()):
                budget.check()
                exponent = tuple(a-b for a, b in zip(ef, eg))
                coefficient = -cf*pow(cg, -1, ring.p) % ring.p
                signature = ((exponent, coefficient),)
                if signature in seen:
                    continue
                budget.check(terms=retained+1)
                seen.add(signature)
                retained += 1
                yield target, source, ring.monomial(exponent, coefficient)
        for swap in (False, True):
            for signs in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
                q = -_quotient(f, g, swap, signs, budget, options)
                if q:
                    signature = tuple(sorted(q.terms.items()))
                    if signature in seen:
                        continue
                    budget.check(terms=retained+len(signature))
                    seen.add(signature)
                    retained += len(signature)
                    yield target, source, q


def _search(HX, budget, options, progress):
    original = (HX[0, 0], HX[0, 1])
    pairs = [_normalize(f, budget) for f in original]
    row = tuple(f for f, _ in pairs)
    gauge = tuple(unit**-1 for _, unit in pairs)
    frontier, seen = [(row, gauge, [])], {_key(row)}
    retained_terms = sum(len(f.terms) for f in row)
    best = _score(row)
    pair_limit = max(64, best[0]+16)
    serial = 0
    for level in range(options['search_depth']+1):
        heap = []
        if progress:
            progress('rank_two_row_search', {'depth': level, 'frontier': len(frontier),
                                            'generated_nodes': budget.nodes, 'best_score': list(best)})
        for current, gauge, word in frontier:
            budget.check(*current)
            found = _goal(current, budget)
            if found is not None:
                return word, found, {'depth': level, 'generated_nodes': budget.nodes,
                                     'visited_states': len(seen), 'best_score': list(best)}
            if level == options['search_depth']:
                continue
            for target, source, q in _candidates(current, budget, options):
                budget.nodes += 1
                if budget.nodes > options['max_nodes']:
                    budget.incomplete('node_limit', 'Rank-two row search reached max_nodes',
                                      depth=level, max_nodes=options['max_nodes'], best_score=list(best))
                updated = current[target]+budget.multiply(q, current[source])
                if not updated:
                    continue
                normalized, unit = _normalize(updated, budget)
                child = list(current)
                child[target] = normalized
                child = tuple(child)
                signature = _key(child)
                if signature in seen:
                    continue
                seen.add(signature)
                retained_terms += sum(len(f.terms) for f in child)
                budget.check(terms=retained_terms)
                score = _score(child)
                best = min(best, score)
                if score[0] > pair_limit:
                    continue
                physical = budget.multiply(budget.multiply(gauge[source], q), gauge[target]**-1)
                next_gauge = list(gauge)
                next_gauge[target] = budget.multiply(next_gauge[target], unit**-1)
                next_word = word+[('e', source, target, physical)]
                budget.word(next_word)
                # Keep only the declared beam width, with stable first-seen
                # ties, rather than retaining every generated child word.
                serial += 1
                priority = tuple(-c for c in score)+(-serial,)
                entry = (priority, serial, child, tuple(next_gauge), next_word)
                if len(heap) < options['search_width']:
                    heapq.heappush(heap, entry)
                elif priority > heap[0][0]:
                    heapq.heapreplace(heap, entry)
        if level == options['search_depth']:
            budget.incomplete('depth_limit', 'Rank-two row search reached search_depth',
                              depth=level, search_depth=options['search_depth'], best_score=list(best))
        if not heap:
            budget.incomplete('search_exhausted', 'Rank-two bounded candidate set produced no new frontier',
                              depth=level, best_score=list(best))
        ordered = sorted(heap, key=lambda item: item[0], reverse=True)
        frontier = [(child, gauge, word) for _, _, child, gauge, word in ordered]


def _single_toric_source(result, U, budget):
    Q = result['qca']
    x, y = U.ring.variables
    TX = Matrix.from_rows(U.ring, [[x-1, y-1]])
    TZ = Matrix.from_rows(U.ring, [[1-y**-1, x**-1-1]])
    HX, HZ = Q['input_HX'], Q['input_HZ']
    if (Q['toric_copies'] != 1 or Q['product_x'] != 0 or Q['product_z'] != 0 or
            Q['standard_HX'] != TX or Q['standard_HZ'] != TZ or
            HX.shape != (1, 2) or HZ.shape != (1, 2) or HX.ring != U.ring or HZ.ring != U.ring or
            not HX[0, 0] or not HX[0, 1]):
        budget.incomplete('unsupported_target', 'Rank-two row fallback supports one toric copy without product species',
                          physical_qudits=2, toric_copies=Q['toric_copies'],
                          product_x=Q['product_x'], product_z=Q['product_z'])
    return HX, HZ, TX, TZ


def factor_rank_two(result, *, timeout=120, max_steps=1000000, max_terms=1000000,
                    max_products=30000000, max_factors=200000,
                    rank_two_options=None, progress=None):
    """Find exact original-U factors after a bounded single-toric row search.

    Fast reductions, support search and target-GE2 realization consume one
    shared deadline and counter budget. The time limit is cooperative. Rank
    two has no claimed general Suslin guarantee; the general route needs q>=3.
    """
    options = parse_rank_two_options(rank_two_options)
    if progress is not None and not callable(progress):
        raise ValueError('progress must be callable or None')
    Q = result['qca']
    U, V = Q['psi'][1], Q['psi_inverse'][1]
    if U.shape != (2, 2) or V.shape != (2, 2) or U.ring != V.ring or U.ring.D != 2:
        raise ValueError('Rank-two factorization requires matching two-by-two Laurent QCA matrices')
    budget = RankTwoBudget(timeout=timeout, max_steps=max_steps, max_terms=max_terms,
                           max_products=max_products, max_factors=max_factors)
    if budget.mm(U, V) != eye(U.ring, 2) or budget.mm(V, U) != eye(U.ring, 2):
        raise ValueError('Rank-two QCA matrices are not two-sided inverses')
    budget.phase = 'rank_two_fast'
    fast = None
    try:
        with budget.fast_probe():
            fast = _fast_atoms(U, budget)
    except ProbeStopped:
        pass
    if fast is not None:
        data = separate_atoms(U, V, fast, budget, algorithm='rank-two-shared-budget-unit-reduction-v1')
        data['statistics']['route'] = 'fast'
        return data
    budget.phase = 'rank_two_target_admission'
    HX, HZ, TX, TZ = _single_toric_source(result, U, budget)
    budget.phase = 'rank_two_row_search'
    word, swapped, search = _search(HX, budget, options, progress)
    if swapped:
        word = join(budget, word, _signed_swap(U.ring))
    budget.phase = 'rank_two_row_witness'
    P, Pi = ge_replay(word, U.ring, budget), ge_replay(invert_atoms(word), U.ring, budget)
    actual_row = budget.mm(HX, P)
    units = [associate_unit(actual_row[0, j], TX[0, j], budget) for j in range(2)]
    if any(u is None for u in units):
        raise ArithmeticError('Rank-two physical row word does not reach its declared target')
    S = Matrix(U.ring, 2, 2, {(i, i): u for i, u in enumerate(units)})
    Si = Matrix(U.ring, 2, 2, {(i, i): u**-1 for i, u in enumerate(units)})
    A, Ai = budget.mm(S, Pi), budget.mm(P, Si)
    if budget.mm(TX, A) != HX or budget.mm(HX, Ai) != TX:
        raise ArithmeticError('Rank-two row orbit physical action failed')
    z = budget.mm(A, HZ.dagger())
    scale = associate_unit(z[0, 0], TZ.dagger()[0, 0], budget)
    if scale is None or z != budget.scale(TZ.dagger(), scale):
        raise ArithmeticError('Rank-two row orbit failed the complete Z-stabilizer binding')
    Aword = join(budget, [('d', i, -1, u) for i, u in enumerate(units)], invert_atoms(word))
    if ge_replay(Aword, U.ring, budget) != A:
        raise ArithmeticError('Rank-two intermediate GE2 action differs')
    budget.phase = 'rank_two_target_ge2'
    K, Ki = budget.mm(U, Ai), budget.mm(A, V)
    Kword = factor_target_automorphism(K, Ki, budget)
    factors = separate_atoms(U, V, join(budget, Kword, Aword), budget,
                             algorithm='rank-two-bounded-row-orbit-target-ge2-v1')
    factors['statistics'].update(route='row-orbit-target-GE2', search=search,
                                 rank_two_options=options, original_registers=2,
                                 no_additional_coarse_graining=True)
    if progress:
        progress('rank_two_complete', {'factors': len(factors['factors']),
                                      'physical_qudits': 2, 'input_ancillas': 0,
                                      'statistics': factors['statistics']})
    return factors
