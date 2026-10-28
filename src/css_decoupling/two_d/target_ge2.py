"""Exact two-register GE2 words for the standard toric target automorphism.

Every intermediate matrix has the original two coordinates. Diagonal Laurent
units are recorded transport; they are never called one-layer SWAP circuits.
"""
from .algebra import Matrix, eye


def invert_atoms(atoms):
    return [(kind, i, j, -c if kind == 'e' else c**-1)
            for kind, i, j, c in reversed(atoms)]


def join(budget, *parts):
    if sum(map(len, parts)) > budget.max_factors:
        budget.stop('factor_limit', 'GE2 word exceeds the factor cap before concatenation')
    result = []
    for part in parts:
        for kind, i, j, c in part:
            if (kind == 'e' and not c) or (kind == 'd' and c == 1):
                continue
            if result and result[-1][:3] == (kind, i, j):
                previous = result.pop()[3]
                c = previous+c if kind == 'e' else budget.multiply(previous, c)
            if (kind == 'e' and c) or (kind == 'd' and c != 1):
                result.append((kind, i, j, c))
    budget.word(result)
    return result


def ge_replay(atoms, ring, budget):
    result = eye(ring, 2)
    for kind, i, j, c in atoms:
        if kind == 'e':
            if i not in (0, 1) or j not in (0, 1) or i == j:
                raise ValueError('GE2 factor leaves the original two registers')
            for k in range(2):
                result[k, j] += budget.multiply(result[k, i], c)
        elif kind == 'd':
            if i not in (0, 1) or not c.is_unit:
                raise ValueError('GE2 diagonal is not an original-register Laurent unit')
            for k in range(2):
                result[k, i] = budget.multiply(result[k, i], c)
        else:
            raise ValueError('Unknown GE2 factor kind')
        budget.check(result)
    return result


def associate_unit(f, g, budget):
    if not f or not g or len(f.terms) != len(g.terms):
        return None
    ef, eg = min(f.terms), min(g.terms)
    c = f.terms[ef]*pow(g.terms[eg], -1, f.ring.p) % f.ring.p
    unit = f.ring.monomial(tuple(a-b for a, b in zip(ef, eg)), c)
    return unit if budget.multiply(unit, g) == f else None


def diagonal_pair(ring, unit):
    """Written four-factor identity diag(unit,unit^-1)."""
    return [('e', 0, 1, unit-1), ('e', 1, 0, ring.one()),
            ('e', 0, 1, unit**-1-1), ('e', 1, 0, -unit)]


def _power_word(word, exponent, budget):
    if type(exponent) is not int:
        raise ValueError('Exact integer translation power required')
    if len(word)*abs(exponent) > budget.max_factors:
        budget.stop('factor_limit', 'Monomial transport power exceeds the word cap')
    source = word if exponent >= 0 else invert_atoms(word)
    return join(budget, *([source]*abs(exponent)))


def translation_word(ring, exponent, budget):
    x, y = ring.variables
    Tx = [('d', 1, -1, x), ('e', 0, 1, 1-y)]
    Ty = [('d', 0, -1, y), ('e', 1, 0, 1-x)]
    return join(budget, _power_word(Tx, exponent[0], budget),
                _power_word(Ty, exponent[1], budget))


def _seed_word(ring):
    # P1=C(y-1,x-1)^-1. The seven-factor identity holds integrally and
    # involves no division by 2, including characteristic two.
    v, u = ring.variables
    C = [('e', 0, 1, v**-1), ('e', 1, 0, -v*u**-1),
         ('e', 0, 1, u), ('e', 1, 0, -v),
         ('e', 0, 1, u*v**-1-2*u), ('e', 1, 0, u**-1),
         ('e', 0, 1, -2*u)]
    return invert_atoms(C)


def transvection_word(f, budget):
    ring = f.ring
    seed, result = _seed_word(ring), []
    for exponent, coefficient in sorted(f.terms.items()):
        budget.check(f)
        count = min(coefficient, ring.p-coefficient)
        local = seed if coefficient <= ring.p-coefficient else invert_atoms(seed)
        if count*len(local) > budget.max_factors:
            budget.stop('factor_limit', 'Prime-field coefficient repetition exceeds the factor cap')
        T = translation_word(ring, exponent, budget)
        result = join(budget, result, T, *([local]*count), invert_atoms(T))
    x, y = ring.variables
    h = Matrix.from_rows(ring, [[x-1, y-1]])
    d = Matrix.from_rows(ring, [[1-y], [x-1]])
    expected = eye(ring, 2)+budget.mm(d, budget.scale(h, f))
    if ge_replay(result, ring, budget) != expected:
        raise ArithmeticError('Toric transvection elementary word failed exact replay')
    return result


def _divide_a_squared(value, budget):
    """Exact Laurent division by (x-1)^2, separately at each y exponent."""
    ring, groups, result = value.ring, {}, {}
    divisor = {0: 1, 2: 1}
    if (-2) % ring.p:
        divisor[1] = (-2) % ring.p
    for (a, b), c in value.terms.items():
        groups.setdefault(b, {})[a] = c
    for y_power, terms in groups.items():
        shift = min(terms)
        remainder = {a-shift: c for a, c in terms.items()}
        quotient = {}
        while remainder and max(remainder) >= 2:
            budget.check(terms=len(remainder)+len(quotient))
            power = max(remainder)-2
            coefficient = remainder[power+2]
            quotient[power] = coefficient
            budget.charge(len(divisor))
            for k, c in divisor.items():
                key = k+power
                new = (remainder.get(key, 0)-coefficient*c) % ring.p
                if new:
                    remainder[key] = new
                else:
                    remainder.pop(key, None)
        if remainder:
            raise ValueError('Target automorphism residual is not divisible by (x-1)^2')
        result.update({(power+shift, y_power): c for power, c in quotient.items()})
    answer = ring(result)
    divisor_poly = ring({(0, 0): 1, (1, 0): -2, (2, 0): 1})
    if budget.multiply(divisor_poly, answer) != value:
        raise ArithmeticError('Toric transvection exact division failed')
    return answer


def factor_target_automorphism(K, Ki, budget):
    ring = K.ring
    if K.shape != (2, 2) or Ki.shape != (2, 2) or Ki.ring != ring or ring.D != 2:
        raise ValueError('Expected a two-register prime-field target automorphism')
    I = eye(ring, 2)
    if budget.mm(K, Ki) != I or budget.mm(Ki, K) != I:
        raise ValueError('Target automorphism inverse failed')
    x, y = ring.variables
    h, d = Matrix.from_rows(ring, [[x-1, y-1]]), Matrix.from_rows(ring, [[1-y], [x-1]])
    hk, kd = budget.mm(h, K), budget.mm(K, d)
    lam, mu = associate_unit(hk[0, 0], x-1, budget), associate_unit(kd[0, 0], 1-y, budget)
    if lam is None or mu is None or hk != budget.scale(h, lam) or kd != budget.scale(d, mu):
        raise ValueError('Matrix does not preserve both standard toric endpoint modules')
    (le, lc), = lam.terms.items()
    (me, mc), = mu.terms.items()
    if lc != mc:
        raise ValueError('Toric endpoint augmentation scalars disagree')
    relative = tuple(a-b for a, b in zip(me, le))
    base = join(budget, [('d', 0, -1, lam), ('d', 1, -1, lam)],
                translation_word(ring, relative, budget))
    B, Bi = ge_replay(base, ring, budget), ge_replay(invert_atoms(base), ring, budget)
    if budget.mm(h, B) != budget.scale(h, lam) or budget.mm(B, d) != budget.scale(d, mu):
        raise ArithmeticError('Toric endpoint representative failed')
    residual = budget.mm(Bi, K)-I
    f = _divide_a_squared(residual[1, 0], budget)
    if residual != budget.mm(d, budget.scale(h, f)):
        raise ValueError('Target automorphism is not its extracted transvection')
    atoms = join(budget, base, transvection_word(f, budget))
    if ge_replay(atoms, ring, budget) != K or ge_replay(invert_atoms(atoms), ring, budget) != Ki:
        raise ArithmeticError('Complete target automorphism GE2 word failed')
    return atoms


def separate_atoms(U, V, atoms, budget, *, algorithm):
    """Normalize a written GE2 word into U=S F with chronological E2 records."""
    ring = U.ring
    units, elementary = [ring.one(), ring.one()], []
    for kind, i, j, c in atoms:
        if kind == 'd':
            units[i] = budget.multiply(units[i], c)
        else:
            coefficient = budget.multiply(budget.multiply(units[i], c), units[j]**-1)
            elementary = join(budget, elementary, [('e', i, j, coefficient)])
    determinant = budget.multiply(units[0], units[1])
    S_units = [ring.one(), determinant]
    word = []
    for _, i, j, c in elementary:
        c = budget.multiply(budget.multiply(S_units[i]**-1, c), S_units[j])
        word = join(budget, word, [('e', i, j, c)])
    word = join(budget, word, diagonal_pair(ring, units[0]))
    F, Fi = ge_replay(word, ring, budget), ge_replay(invert_atoms(word), ring, budget)
    S = Matrix(ring, 2, 2, {(i, i): u for i, u in enumerate(S_units)})
    Si = Matrix(ring, 2, 2, {(i, i): u**-1 for i, u in enumerate(S_units)})
    if (budget.mm(S, F) != U or budget.mm(Fi, Si) != V or
            budget.mm(F, Fi) != eye(ring, 2) or budget.mm(Fi, F) != eye(ring, 2)):
        raise ArithmeticError('Original rank-two U=S F identity failed')
    records = [{'row': i, 'column': j, 'coefficient': Matrix.from_rows(ring, [[c]])}
               for _, i, j, c in reversed(word)]
    transport = [next(iter(u.terms.items())) for u in S_units]
    return {'algorithm': algorithm, 'factors': records, 'added_ancillas': 0,
            'shift_exponents': [list(e) for e, _ in transport],
            'transport_coefficients': [c for _, c in transport],
            'ancilla_action': eye(ring, 0), 'ancilla_inverse': eye(ring, 0),
            'input_inverse': V, 'complete_general_suslin_algorithm': False,
            'statistics': budget.snapshot()}
