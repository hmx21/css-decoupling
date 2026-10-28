"""Constructive stable elementary factorization over Fp[x^+-1,y^+-1].

The route is Higman linearization followed by exact PID bases over the other
Laurent variable. It realizes U direct-sum identity, not U direct-sum U^-1.
The stabilization is explicit and need not be minimal. No fraction-field basis,
bounded factor search, external CAS, or unexpanded completion matrix is used.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from time import perf_counter

from .algebra import Matrix, eye, direct_sum
from ..errors import ResourceLimit


def _positive(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return value


class _Budget:
    def __init__(self, timeout, max_steps, max_terms, max_factors,
                 max_linearized_size, max_products):
        if type(timeout) not in (int, float):
            raise ValueError('timeout must be finite and positive')
        try:
            timeout = float(timeout)
        except OverflowError as error:
            raise ValueError('timeout must be finite and positive') from error
        if not isfinite(timeout) or timeout <= 0:
            raise ValueError('timeout must be finite and positive')
        self.deadline = perf_counter() + timeout
        for name, value in [('max_steps', max_steps), ('max_terms', max_terms),
                            ('max_factors', max_factors), ('max_linearized_size', max_linearized_size),
                            ('max_products', max_products)]:
            setattr(self, name, _positive(value, name))
        self.steps = self.products = 0
        self.kernel_splits = []
        self.phase, self.input_rank, self.linearized_rank = 'validation', None, None

    def fail(self, reason):
        raise ResourceLimit(f'Higman/PID phase={self.phase}, input_rank={self.input_rank}, '
                            f'linearized_rank={self.linearized_rank}: {reason}')

    def check(self, *matrices, terms=0):
        self.steps += 1
        if perf_counter() > self.deadline or self.steps > self.max_steps:
            self.fail('cooperative time/operation budget exhausted')
        terms += sum(len(f.terms) for matrix in matrices for f in matrix.entries.values())
        if terms > self.max_terms:
            self.fail('retained term budget exhausted')

    def word(self, word):
        if len(word) > self.max_factors:
            self.fail('elementary factor budget exhausted')
        self.check(terms=sum(len(atom[3].terms) for atom in word))

    def pairs(self, number):
        self.products += number
        if self.products > self.max_products:
            self.fail('convolution-pair budget exhausted')
        self.check()


def _pmul(a, b, budget):
    budget.pairs(len(a.terms) * len(b.terms))
    result = a * b
    budget.check(terms=len(result.terms))
    return result


def _mm(A, B, budget):
    if A.ncols != B.nrows or A.ring != B.ring:
        raise ValueError('Incompatible exact matrix product')
    rows = {}
    for (i, j), f in B.entries.items():
        rows.setdefault(i, []).append((j, f))
    pairs = sum(len(f.terms) * len(g.terms)
                for (i, k), f in A.entries.items() for j, g in rows.get(k, ()))
    budget.pairs(pairs)
    C = A * B
    budget.check(C)
    return C


def _row_add(M, target, source, coefficient, budget):
    if target == source:
        raise ValueError('An elementary addition needs distinct indices')
    if not coefficient:
        return
    values = [(j, f) for (i, j), f in M.entries.items() if i == source]
    for j, f in values:
        M[target, j] = M[target, j] + _pmul(coefficient, f, budget)
    budget.check(M)


def _col_add(M, target, source, coefficient, budget):
    if target == source:
        raise ValueError('An elementary addition needs distinct indices')
    if not coefficient:
        return
    values = [(i, f) for (i, j), f in M.entries.items() if j == source]
    for i, f in values:
        M[i, target] = M[i, target] + _pmul(f, coefficient, budget)
    budget.check(M)


def _swap_rows(M, i, j):
    for k in range(M.ncols):
        M[i, k], M[j, k] = M[j, k], M[i, k]


def _swap_cols(M, i, j):
    for k in range(M.nrows):
        M[k, i], M[k, j] = M[k, j], M[k, i]


def _elementary(i, j, c):
    return ('e', i, j, c)


def _diagonal(i, u):
    if not u.is_unit:
        raise ArithmeticError('A nonunit diagonal cannot be a Laurent transport')
    return ('d', i, -1, u)


def _swap_word(ring, i, j):
    """Written word for the actual permutation, including its determinant -1."""
    if i == j:
        return []
    word = [_elementary(i, j, ring.one()), _elementary(j, i, -ring.one()),
            _elementary(i, j, ring.one())]
    if ring.p != 2:
        word.append(_diagonal(i, -ring.one()))
    return word


def _inverse_atom(atom):
    kind, i, j, c = atom
    return (kind, i, j, -c if kind == 'e' else c**-1)


def _inverse_word(word):
    return [_inverse_atom(atom) for atom in reversed(word)]


def _embed_word(word, offset):
    return [(kind, i + offset, j + offset if kind == 'e' else -1, c)
            for kind, i, j, c in word]


def _span(f, axis):
    degrees = [e[axis] for e in f.terms]
    return max(degrees) - min(degrees) if degrees else -1


def _pid_divide(a, b, axis, budget):
    """Euclidean division in Fp[t^+-1], with remainder span strictly smaller."""
    if not b:
        raise ZeroDivisionError('Zero Laurent divisor')
    ring = a.ring
    for f in (a, b):
        if any(e[1-axis] for e in f.terms):
            raise ValueError('PID coefficient depends on the eliminated variable')
    if not a:
        return ring.zero(), ring.zero()
    va, vb = min(e[axis] for e in a.terms), min(e[axis] for e in b.terms)
    rem = {e[axis]-va: c for e, c in a.terms.items()}
    div = {e[axis]-vb: c for e, c in b.terms.items()}
    quotient, degree = {}, max(div)
    leading_inverse = pow(div[degree], -1, ring.p)
    while rem and max(rem) >= degree:
        old_degree = max(rem)
        shift = old_degree-degree
        coefficient = rem[old_degree]*leading_inverse % ring.p
        quotient[shift] = coefficient
        for e, value in div.items():
            key = e+shift
            updated = (rem.get(key, 0)-coefficient*value) % ring.p
            if updated:
                rem[key] = updated
            else:
                rem.pop(key, None)
        budget.check(terms=len(rem)+len(quotient)+len(div))
        if rem and max(rem) >= old_degree:
            raise ArithmeticError('Euclidean degree did not strictly decrease')
    embed = lambda data, shift: ring({tuple(k+shift if i == axis else 0 for i in range(2)): c
                                     for k, c in data.items()})
    q, r = embed(quotient, va-vb), embed(rem, va)
    if a != _pmul(q, b, budget)+r or (r and _span(r, axis) >= _span(b, axis)):
        raise ArithmeticError('Laurent Euclidean division certificate failed')
    return q, r


@dataclass
class _PID:
    diagonal: Matrix
    left: Matrix
    right: Matrix
    left_inverse: Matrix
    right_inverse: Matrix
    left_operations: list
    right_operations: list
    rank: int


def _pid_diagonalize(original, axis, budget):
    """Tracked diagonalization over a Laurent PID; no divisibility ordering needed."""
    ring, n = original.ring, original.nrows
    if original.ncols != n or any(e[1-axis] for f in original.entries.values() for e in f.terms):
        raise ValueError('Expected a square matrix over the one-variable Laurent PID')
    A, L, R, Li, Ri = original.copy(), eye(ring, n), eye(ring, n), eye(ring, n), eye(ring, n)
    left, right = [], []

    def row_add(i, j, c):
        if c:
            _row_add(A, i, j, c, budget); _row_add(L, i, j, c, budget)
            _col_add(Li, j, i, -c, budget); left.append(_elementary(i, j, c))
            budget.word(left)

    def col_add(j, i, c):
        if c:
            _col_add(A, j, i, c, budget); _col_add(R, j, i, c, budget)
            _row_add(Ri, i, j, -c, budget); right.append(_elementary(i, j, c))
            budget.word(right)

    def row_swap(i, j):
        if i != j:
            _swap_rows(A, i, j); _swap_rows(L, i, j); _swap_cols(Li, i, j)
            left.extend(reversed(_swap_word(ring, i, j))); budget.word(left)

    def col_swap(i, j):
        if i != j:
            _swap_cols(A, i, j); _swap_cols(R, i, j); _swap_rows(Ri, i, j)
            right.extend(_swap_word(ring, i, j)); budget.word(right)

    k = 0
    while k < n:
        budget.check(A, L, R, Li, Ri)
        active = [(i, j, f) for (i, j), f in A.entries.items() if i >= k and j >= k]
        if not active:
            break
        row_count, col_count, row_terms, col_terms = {}, {}, {}, {}
        for i, j, f in active:
            row_count[i] = row_count.get(i, 0)+1
            col_count[j] = col_count.get(j, 0)+1
            row_terms[i] = row_terms.get(i, 0)+len(f.terms)
            col_terms[j] = col_terms.get(j, 0)+len(f.terms)
        # Keep the Euclidean norm first (units therefore always precede
        # nonunits). Within a norm tie, sparse Markowitz scores avoid choosing
        # a dense coefficient row before an available sparse identity row.
        # This changes only the choice of a legal pivot, never the divisions
        # or the exact identities, and makes no optimal-fill claim.
        i, j, _ = min(active, key=lambda item: (
            _span(item[2], axis), (row_count[item[0]]-1)*(col_count[item[1]]-1),
            row_terms[item[0]]*col_terms[item[1]], item[0], item[1]))
        row_swap(k, i); col_swap(k, j)
        while True:
            restarted = False
            for i in range(k+1, n):
                if A[i, k]:
                    q, remainder = _pid_divide(A[i, k], A[k, k], axis, budget)
                    row_add(i, k, -q)
                    if remainder:
                        row_swap(k, i); restarted = True; break
            if restarted:
                continue
            for j in range(k+1, n):
                if A[k, j]:
                    q, remainder = _pid_divide(A[k, j], A[k, k], axis, budget)
                    col_add(j, k, -q)
                    if remainder:
                        col_swap(k, j); restarted = True; break
            if not restarted:
                break
        k += 1
    if any(i != j for i, j in A.entries):
        raise ArithmeticError('PID reduction did not reach a diagonal matrix')
    I = eye(ring, n)
    if (_mm(_mm(L, original, budget), R, budget) != A or
            _mm(L, Li, budget) != I or _mm(Li, L, budget) != I or
            _mm(R, Ri, budget) != I or _mm(Ri, R, budget) != I):
        raise ArithmeticError('Tracked PID transformations failed their exact identities')
    return _PID(A, L, R, Li, Ri, left, right, k)


def _pid_factor(A, axis, budget):
    result = _pid_diagonalize(A, axis, budget)
    n, ring = A.nrows, A.ring
    if result.rank != n or any(not result.diagonal[i, i].is_unit for i in range(n)):
        raise ValueError('Specialization is not invertible over the Laurent coefficient PID')
    diagonal_inverse = Matrix(ring, n, n, {(i, i): result.diagonal[i, i]**-1 for i in range(n)})
    inverse = _mm(_mm(result.right, diagonal_inverse, budget), result.left, budget)
    word = ([_inverse_atom(a) for a in result.left_operations] +
            [_diagonal(i, result.diagonal[i, i]) for i in range(n)] +
            _inverse_word(result.right_operations))
    budget.word(word)
    if _mm(A, inverse, budget) != eye(ring, n) or _mm(inverse, A, budget) != eye(ring, n):
        raise ArithmeticError('PID inverse is not a two-sided inverse')
    return word, inverse


def _permutation(ring, order):
    n = len(order)
    P, labels, word = eye(ring, n), list(range(n)), []
    for i, wanted in enumerate(order):
        j = labels.index(wanted)
        if i != j:
            _swap_cols(P, i, j); labels[i], labels[j] = labels[j], labels[i]
            word.extend(_swap_word(ring, i, j))
    return P, word


def _nilpotent_triangular(N, axis, budget):
    """Split full PID kernels recursively, with actual unimodular bases."""
    ring, n = N.ring, N.nrows
    I = eye(ring, n)
    if not N.entries:
        return I, I, N.copy(), []
    diagonal = _pid_diagonalize(N, axis, budget)
    if diagonal.rank == n:
        raise ValueError('Required nilpotent block has trivial kernel')
    r, s = diagonal.rank, n-diagonal.rank
    budget.kernel_splits.append({'size': n, 'rank': r, 'kernel_rank': s})
    P, pword = _permutation(ring, list(range(r, n))+list(range(r)))
    Q = _mm(diagonal.right, P, budget)
    Qi = _mm(P.transpose(), diagonal.right_inverse, budget)
    transformed = _mm(_mm(Qi, N, budget), Q, budget)
    if any(j < s for i, j in transformed.entries):
        raise ArithmeticError('The computed complete kernel did not give zero columns')
    inner, inner_inverse, triangle, inner_word = _nilpotent_triangular(transformed[s:, s:], axis, budget)
    extension, extension_inverse = direct_sum(eye(ring, s), inner), direct_sum(eye(ring, s), inner_inverse)
    Q = _mm(Q, extension, budget)
    Qi = _mm(extension_inverse, Qi, budget)
    triangle = _mm(_mm(Qi, N, budget), Q, budget)
    if any(i >= j for i, j in triangle.entries) or _mm(Q, Qi, budget) != I or _mm(Qi, Q, budget) != I:
        raise ArithmeticError('Nilpotent triangularization/basis identities failed')
    word = diagonal.right_operations + pword + _embed_word(inner_word, s)
    budget.word(word)
    return Q, Qi, triangle, word


def _projector_split(E, axis, budget):
    ring, n = E.ring, E.nrows
    I = eye(ring, n)
    if _mm(E, E, budget) != E:
        raise ValueError('The Laurent pencil failed the required0/1 projection identity')
    diagonal = _pid_diagonalize(E, axis, budget)
    r = diagonal.rank
    if any(not diagonal.diagonal[i, i].is_unit for i in range(r)):
        raise ArithmeticError('An idempotent image was not primitive over the PID')
    Q = Matrix(ring, n, n, {(i, j): (diagonal.left_inverse if j < r else diagonal.right)[i, j]
                            for i in range(n) for j in range(n)})
    top = _mm(diagonal.left, E, budget)
    bottom = _mm(diagonal.right_inverse, I-E, budget)
    Qi = Matrix(ring, n, n, {(i, j): (top if i < r else bottom)[i, j]
                             for i in range(n) for j in range(n)})
    target = Matrix(ring, n, n, {(i, i): 1 for i in range(r)})
    if (_mm(Q, Qi, budget) != I or _mm(Qi, Q, budget) != I or
            _mm(_mm(Qi, E, budget), Q, budget) != target):
        raise ArithmeticError('PID image/kernel splitting failed its complete basis identities')
    word, inverse = _pid_factor(Q, axis, budget)
    if inverse != Qi:
        raise ArithmeticError('Split basis inverse differs from its PID elementary inverse')
    return Q, Qi, r, word


def _coefficient(A, axis, degree):
    ring = A.ring
    return Matrix(ring, A.nrows, A.ncols,
                  {ij: ring({tuple(0 if k == axis else a for k, a in enumerate(e)): c
                             for e, c in f.terms.items() if e[axis] == degree})
                   for ij, f in A.entries.items()})


def _at_one(A, axis):
    ring, entries = A.ring, {}
    for ij, f in A.entries.items():
        data = {}
        for e, c in f.terms.items():
            key = tuple(0 if k == axis else a for k, a in enumerate(e))
            data[key] = (data.get(key, 0)+c) % ring.p
        entries[ij] = ring(data)
    return Matrix(ring, A.nrows, A.ncols, entries)


def _word_at_one(word, axis, budget):
    """Specialize an existing elementary trace without solving a new problem."""
    result = []
    for kind, i, j, f in word:
        if kind != 'e':
            raise ArithmeticError('Higman linearization trace must be elementary')
        data = {}
        for exponent, coefficient in f.terms.items():
            key = tuple(0 if k == axis else a for k, a in enumerate(exponent))
            data[key] = (data.get(key, 0)+coefficient) % f.ring.p
        value = f.ring(data)
        if value:
            result.append(_elementary(i, j, value))
        budget.check(terms=len(data))
    budget.word(result)
    return result


def _apply_elementary_word(A, word, side, budget):
    """Apply a written elementary word on one side, using exact row/column ops."""
    result = A.copy()
    if side not in ('left', 'right'):
        raise ValueError('Invalid side for a written elementary word')
    for kind, i, j, coefficient in reversed(word) if side == 'left' else word:
        if kind != 'e':
            raise ArithmeticError('Expected an elementary Higman trace')
        if side == 'left':
            _row_add(result, i, j, coefficient, budget)
        else:
            _col_add(result, j, i, coefficient, budget)
    return result


def _specialized_higman_factor(C, P, axis, left_word, right_word, budget):
    """Factor C(1) by reusing L*C*R=P+I, reducing only the original-rank P(1).

    C(1)=L(1)^-1*(P(1)+I)*R(1)^-1, and its actual inverse is
    R(1)*(P(1)^-1+I)*L(1). All trace coefficients are specialized exactly.
    """
    ring, n, N = P.ring, P.nrows, C.nrows
    at_one, small = _at_one(C, axis), _at_one(P, axis)
    small_word, small_inverse = _pid_factor(small, 1-axis, budget)
    left, right = _word_at_one(left_word, axis, budget), _word_at_one(right_word, axis, budget)
    left_inverse, right_inverse = _inverse_word(left), _inverse_word(right)
    reconstruction = _apply_elementary_word(direct_sum(small, eye(ring, N-n)),
                                            right_inverse, 'right', budget)
    reconstruction = _apply_elementary_word(reconstruction, left_inverse, 'left', budget)
    inverse = _apply_elementary_word(direct_sum(small_inverse, eye(ring, N-n)),
                                    left, 'right', budget)
    inverse = _apply_elementary_word(inverse, right, 'left', budget)
    word = left_inverse + small_word + right_inverse
    budget.word(word)
    if (reconstruction != at_one or _mm(at_one, inverse, budget) != eye(ring, N) or
            _mm(inverse, at_one, budget) != eye(ring, N)):
        raise ArithmeticError('Reused Higman specialization trace/inverse identities failed')
    return at_one, word, inverse


def _linearize(P, axis, degree, budget):
    """Explicit L C R=P direct-sum I, for a block companion linear pencil C."""
    ring, n, t = P.ring, P.nrows, P.ring.variable(axis)
    N = n*degree
    coefficients = [_coefficient(P, axis, k) for k in range(degree+1)]
    C = Matrix(ring, N, N, {})
    for (i, j), f in (coefficients[0]+coefficients[1]*t).entries.items():
        C[i, j] = f
    for block in range(1, degree):
        for (i, j), f in coefficients[block+1].entries.items():
            C[i, block*n+j] = _pmul(t, f, budget)
        for i in range(n):
            C[block*n+i, (block-1)*n+i] = -t
            C[block*n+i, block*n+i] = 1
    reduced, left, right = C.copy(), [], []
    for block in range(1, degree):
        for i in range(n):
            c = t**block
            _col_add(reduced, i, block*n+i, c, budget)
            right.append(_elementary(block*n+i, i, c))
    for block in range(2, degree):
        for i in range(n):
            _row_add(reduced, block*n+i, (block-1)*n+i, t, budget)
            left.append(_elementary(block*n+i, (block-1)*n+i, t))
    for i in range(n):
        for j in range(n, N):
            c = -reduced[i, j]
            if c:
                _row_add(reduced, i, j, c, budget)
                left.append(_elementary(i, j, c))
    if reduced != direct_sum(P, eye(ring, N-n)):
        raise ArithmeticError('Higman elementary linearization identity failed')
    budget.word(left+right)
    return C, list(reversed(left)), right


def _upper_word(H, budget):
    ring, n = H.ring, H.nrows
    if any(i > j for i, j in H.entries) or any(not H[i, i].is_unit for i in range(n)):
        raise ArithmeticError('Expected upper triangular Laurent units on the diagonal')
    A, left = H.copy(), []
    for j in range(n-1, -1, -1):
        for i in range(j):
            c = -A[i, j] * A[j, j]**-1
            if c:
                _row_add(A, i, j, c, budget)
                left.append(_elementary(i, j, c))
    if any(i != j for i, j in A.entries):
        raise ArithmeticError('Upper triangular elimination failed')
    return [_inverse_atom(a) for a in left]+[_diagonal(i, A[i, i]) for i in range(n)]


def _unit_pair_word(ring, i, j, unit):
    # w(u) w(1)^-1 = diag(u,u^-1), where w(u)=[[0,u],[-u^-1,0]].
    return [_elementary(i, j, unit), _elementary(j, i, -unit**-1), _elementary(i, j, unit),
            _elementary(i, j, -ring.one()), _elementary(j, i, ring.one()), _elementary(i, j, -ring.one())]


def _transport_word(word, ring, n, transport, budget):
    diagonal, elementary = [ring.one() for _ in range(n)], []
    for kind, i, j, c in word:
        if kind == 'd':
            diagonal[i] = _pmul(diagonal[i], c, budget)
        else:
            coefficient = _pmul(_pmul(diagonal[i], c, budget), diagonal[j]**-1, budget)
            if coefficient:
                elementary.append(_elementary(i, j, coefficient))
    determinant = ring.one()
    for unit in diagonal:
        determinant = _pmul(determinant, unit, budget)
    extra = int(transport == 'balance' and determinant != 1)
    if extra:
        diagonal.append(determinant**-1)
    q = n+extra
    shifts = [(0, 0)]*q
    S = [ring.one() for _ in range(q)]
    if q and transport == 'separate':
        S[-1] = determinant
        shifts[-1] = next(iter(determinant.terms))
    final = [_elementary(i, j, _pmul(_pmul(c, S[j], budget), S[i]**-1, budget))
             for _, i, j, c in elementary]
    for i in range(max(0, q-1)):
        if diagonal[i] != 1:
            final.extend(_unit_pair_word(ring, i, q-1, diagonal[i]))
    budget.word(final)
    records = [{'row': i, 'column': j, 'coefficient': Matrix.from_rows(ring, [[c]])}
               for _, i, j, c in reversed(final) if c]
    coefficients = [next(iter(unit.terms.values())) for unit in S]
    return records, shifts, coefficients, determinant, extra


def factor_suslin(U, V=None, *, transport='separate', timeout=120,
                  max_steps=1000000, max_terms=1000000, max_factors=200000,
                  max_linearized_size=128, max_products=30000000, axis=None,
                  progress=None):
    """Construct actual factors of U stabilized by identity species.

    With polynomial-axis support span d, Higman uses N=n*max(1,d), hence
    N-n identity ancillas. A separate determinant shift is returned, or balance
    adds at most one further Z ancilla carrying det(U)^-1. The returned complete
    claim is for this stabilized two-variable prime-field route, not minimal-rank Suslin
    factorization or a polynomial complexity bound. Limits are resource stops.
    """
    if not isinstance(U, Matrix) or U.nrows != U.ncols or U.ring.D != 2:
        raise ValueError('Higman/PID requires a square prime-field two-variable Laurent matrix')
    if V is not None and (not isinstance(V, Matrix) or V.shape != U.shape or V.ring != U.ring):
        raise ValueError('Supplied inverse has incompatible dimensions or ring')
    if transport not in ('separate', 'balance') or (axis is not None and (type(axis) is not int or axis not in (0, 1))):
        raise ValueError('Invalid transport or polynomial axis')
    if progress is not None and not callable(progress):
        raise ValueError('progress must be callable or None')
    budget = _Budget(timeout, max_steps, max_terms, max_factors, max_linearized_size, max_products)
    ring, n = U.ring, U.nrows
    budget.input_rank = budget.linearized_rank = n

    def phase(name, **details):
        budget.phase = name
        budget.check()
        if progress is not None:
            progress(name, {'input_rank': n, 'linearized_rank': budget.linearized_rank, **details})

    phase('validation')
    budget.check(U, *([V] if V is not None else []))
    if n > max_linearized_size:
        budget.fail(f'input rank {n} exceeds max_linearized_size={max_linearized_size}')
    if V is not None and (_mm(U, V, budget) != eye(ring, n) or _mm(V, U, budget) != eye(ring, n)):
        raise ValueError('Supplied QCA inverse failed both Laurent identities')
    if n and not U.entries:
        raise ValueError('The zero matrix is not Laurent invertible')
    bounds = []
    for a in range(2):
        exponents = [e[a] for f in U.entries.values() for e in f.terms]
        bounds.append((min(exponents, default=0), max(exponents, default=0)))
    if axis is None:
        axis = min((0, 1), key=lambda a: (bounds[a][1]-bounds[a][0], -a))
    coefficient_axis, shift = 1-axis, -bounds[axis][0]
    degree = bounds[axis][1]-bounds[axis][0]
    N = n*max(1, degree)
    budget.linearized_rank = N
    if N > max_linearized_size:
        budget.fail(f'explicit Higman identity stabilization N={N} exceeds '
                    f'max_linearized_size={max_linearized_size}')
    t = ring.variable(axis)
    P = U * t**shift
    if any(e[axis] < 0 for f in P.entries.values() for e in f.terms):
        raise ArithmeticError('Global Laurent valuation was not cleared')
    word = [_diagonal(i, t**-shift) for i in range(n)]
    projector_rank, projector_power = 0, 0
    if degree == 0:
        phase('pid_inverse', polynomial_degree=degree)
        constant_word, _ = _pid_factor(P, coefficient_axis, budget)
        word += constant_word
    else:
        phase('linearization', polynomial_degree=degree, polynomial_axis=axis)
        C, left_word, right_word = _linearize(P, axis, degree, budget)
        phase('pid_inverse', specialization_pid_rank=n, reused_companion_rank=N)
        at_one, constant_word, constant_inverse = _specialized_higman_factor(
            C, P, axis, left_word, right_word, budget)
        T = _mm(constant_inverse, _coefficient(C, axis, 1), budget)
        phase('projector')
        E, projector_power = T.copy(), 1
        while projector_power < N:
            # Frobenius kills the nilpotent parts on both generalized 0/1
            # eigenspaces when p^k >= N. Binary powering avoids p multiplications.
            base, exponent, powered = E, ring.p, None
            while exponent:
                if exponent & 1:
                    powered = base.copy() if powered is None else _mm(powered, base, budget)
                exponent >>= 1
                if exponent:
                    base = _mm(base, base, budget)
            E = powered
            projector_power *= ring.p
        phase('projector_basis', projector_power=projector_power)
        Q, Qi, r, qword = _projector_split(E, coefficient_axis, budget)
        projector_rank = r
        split = _mm(_mm(Qi, T, budget), Q, budget)
        if any((i < r) != (j < r) for i, j in split.entries):
            raise ArithmeticError('The0/1 projection did not split invariant summands')
        phase('nilpotent_flags', projector_rank=r)
        Q1, Q1i, _, word1 = _nilpotent_triangular(split[:r, :r]-eye(ring, r), coefficient_axis, budget)
        Q0, Q0i, _, word0 = _nilpotent_triangular(split[r:, r:], coefficient_axis, budget)
        extra_basis, extra_inverse = direct_sum(Q1, Q0), direct_sum(Q1i, Q0i)
        full_basis, full_inverse = _mm(Q, extra_basis, budget), _mm(extra_inverse, Qi, budget)
        transformed = _mm(_mm(full_inverse, T, budget), full_basis, budget)
        H = eye(ring, N) + transformed*(t-1)
        if _mm(_mm(_mm(at_one, full_basis, budget), H, budget), full_inverse, budget) != C:
            raise ArithmeticError('Higman pencil reconstruction failed')
        full_word = qword + word1 + _embed_word(word0, r)
        word += left_word + constant_word + full_word + _upper_word(H, budget) + _inverse_word(full_word) + right_word
    budget.word(word)
    phase('word_normalization', written_atoms=len(word))
    records, shifts, coefficients, determinant, balance = _transport_word(word, ring, N, transport, budget)
    h = N-n
    q = N+balance
    B = eye(ring, h+balance)
    Bi = eye(ring, h+balance)
    if balance:
        B[h, h], Bi[h, h] = determinant**-1, determinant
    phase('exact_verification', output_rank=q, factors=len(records))
    product, product_inverse = eye(ring, q), eye(ring, q)
    for item in records:
        _row_add(product, item['row'], item['column'], item['coefficient'][0, 0], budget)
    for item in reversed(records):
        _row_add(product_inverse, item['row'], item['column'], -item['coefficient'][0, 0], budget)
    units = [ring.monomial(e, c) for e, c in zip(shifts, coefficients)]
    S = Matrix(ring, q, q, {(i, i): u for i, u in enumerate(units)})
    Si = Matrix(ring, q, q, {(i, i): u**-1 for i, u in enumerate(units)})
    actual_inverse = _mm(product_inverse, Si, budget)
    if V is None:
        V = actual_inverse[:n, :n]
    if (_mm(S, product, budget) != direct_sum(U, B) or actual_inverse != direct_sum(V, Bi) or
            _mm(product, product_inverse, budget) != eye(ring, q) or
            _mm(product_inverse, product, budget) != eye(ring, q) or
            _mm(U, V, budget) != eye(ring, n) or _mm(V, U, budget) != eye(ring, n)):
        raise ArithmeticError('Final elementary word/inverse does not realize the claimed stabilized QCA')
    phase('complete', output_rank=q, factors=len(records))
    result = {'factors': records, 'added_ancillas': h+balance, 'ancilla_action': B,
            'ancilla_inverse': Bi, 'shift_exponents': shifts, 'input_inverse': V,
            'determinant': Matrix.from_rows(ring, [[determinant]]), 'complete': True,
            'algorithm': 'higman-pid-f2-laurent-v1' if ring.p == 2 else 'higman-pid-fp-laurent-v1',
            'complete_general_suslin_algorithm': True,
            'statistics': {'input_rank': n, 'linearized_rank': N, 'polynomial_axis': axis,
                           'coefficient_axis': coefficient_axis, 'polynomial_degree': degree,
                           'polynomial_shift_power': shift, 'identity_ancillas': h,
                           'balance_ancillas': balance, 'projector_power': projector_power,
                           'specialization_pid_rank': n,
                           'specialization_reuses_higman_trace': degree > 0,
                           'projector_rank': projector_rank, 'kernel_splits': budget.kernel_splits,
                           'operations': budget.steps, 'convolution_pairs': budget.products,
                           'factors': len(records), 'minimal_stabilization_claimed': False}}
    if ring.p != 2:
        result['transport_coefficients'] = coefficients
    return result
