"""Exact common translation kernel, including mixed lattice relations.

Semisimple factors reduce to finite-field discrete logarithms; unipotent
factors reduce along a nilpotent-ideal filtration to linear algebra over
GF(p).  No finite translation group is enumerated.  The verifier does not
need the field decomposition or the discrete-logarithm search.
"""
from math import gcd, isqrt, lcm
from numbers import Integral

from sympy import Matrix as ZMatrix, Poly, factorint, isprime, symbols
from sympy.matrices.normalforms import hermite_normal_form
from sympy.ntheory.modular import crt

from css_decoupling.errors import ResourceLimit
from css_decoupling.lattice.retained import Lattice
from css_decoupling.lattice.finite_algebra import (
    FiniteFieldAlgebra, Span, add, sub, scale, identity, multiply, power,
    linear_combination, nullspace, modular_kernel, congruence_kernel,
)

SCHEMA = "css-decoupling-translation-kernel-v1"


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _actions(layers, D, *, max_linear_entries):
    D = _positive_integer(D, "D")
    result, p = [], None
    for degree, layer in sorted(layers.items()):
        if isinstance(degree, bool) or not isinstance(degree, Integral):
            raise ValueError("Layer degrees must be integers")
        actions = tuple(layer.actions)
        if len(actions) != D:
            raise ValueError("Layer needs one translation per lattice axis")
        if not actions:
            raise ValueError("No translation actions")
        ring, n = actions[0].ring, actions[0].nrows
        if p is not None and ring.p != p:
            raise ValueError("Layers have different characteristics")
        p = ring.p
        if ring.D != D:
            raise ValueError("Layer ring and lattice dimension differ")
        if n * n > max_linear_entries:
            raise ResourceLimit("Translation matrices exceed linear algebra entry cap")
        dense = []
        for action in actions:
            if action.ring != ring or (action.nrows, action.ncols) != (n, n):
                raise ValueError("Translation actions have inconsistent rings or dimensions")
            out = [0] * (n * n)
            for (i, j), polynomial in action.entries.items():
                for exponent, coefficient in polynomial.terms.items():
                    if any(exponent):
                        raise ValueError("Translation actions must be constant matrices")
                    out[i * n + j] = int(coefficient)
            dense.append(tuple(out))
        for i in range(D):
            for j in range(i):
                if multiply(dense[i], dense[j], n, p) != multiply(dense[j], dense[i], n, p):
                    raise ValueError("Translation actions do not commute")
        if n:
            result.append((int(degree), n, tuple(dense)))
    return p, result


def _split_actions(actions, n, p):
    """Multiplicative Jordan decomposition using characteristic polynomials."""
    one, semisimple, unipotent = identity(n), [], []
    t = symbols("t")
    ppower = 1
    while ppower < n:
        ppower *= p
    for action in actions:
        if action == one:
            semisimple.append(one)
            unipotent.append(one)
            continue
        characteristic = Poly(ZMatrix(n, n, action).charpoly(t).as_expr(), t, modulus=p)
        _, factors = characteristic.factor_list()
        if characteristic.eval(0) % p == 0:
            raise ValueError("Translation action is singular")
        multiple = lcm(*(p ** factor.degree() - 1 for factor, _ in factors))
        exponent = ppower * pow(ppower, -1, multiple) if multiple > 1 else 0
        s = power(action, exponent, n, p)
        u = multiply(action, power(s, multiple - 1, n, p), n, p)
        if power(s, multiple, n, p) != one or power(u, ppower, n, p) != one:
            raise ArithmeticError("Multiplicative Jordan decomposition failed")
        semisimple.append(s)
        unipotent.append(u)
    return tuple(semisimple), tuple(unipotent)


def _field_components(actions, n, p, *, max_linear_entries):
    one = identity(n)
    algebra = Span(p, n * n, max_entries=max_linear_entries)
    algebra.include(one)
    at = 0
    while at < len(algebra.vectors):
        for action in actions:
            algebra.include(multiply(action, algebra.vectors[at], n, p))
        at += 1
    dimension = len(algebra.vectors)
    if dimension > n:
        raise ArithmeticError("Semisimple commutative algebra exceeds faithful-module dimension")
    frobenius = [algebra.coordinates(power(b, p, n, p)) for b in algebra.vectors]
    fixed = nullspace([[frobenius[j][i] - int(i == j) for j in range(dimension)]
                       for i in range(dimension)], dimension, p)
    idempotents, t = [one], symbols("t")
    for coefficients in fixed:
        if len(idempotents) == len(fixed):
            break
        element = linear_combination(algebra.vectors, coefficients, p)
        _, factors = Poly(ZMatrix(n, n, element).charpoly(t).as_expr(), t, modulus=p).factor_list()
        if any(f.degree() != 1 for f, _ in factors):
            raise ArithmeticError("Frobenius-fixed element has a nonprime-field eigenvalue")
        roots = sorted({(-int(f.nth(0)) * pow(int(f.nth(1)), -1, p)) % p
                        for f, _ in factors})
        if len(roots) == 1:
            continue
        spectral = []
        for root in roots:
            projector = one
            for other in roots:
                if other != root:
                    factor = scale(sub(element, scale(one, other, p), p),
                                   pow((root - other) % p, -1, p), p)
                    projector = multiply(projector, factor, n, p)
            spectral.append(projector)
        idempotents = [product for old in idempotents for projector in spectral
                       if any(product := multiply(old, projector, n, p))]
    if len(idempotents) != len(fixed):
        raise ArithmeticError("Primitive-idempotent decomposition is incomplete")
    result = []
    for projector in idempotents:
        field = Span(p, n * n, max_entries=max_linear_entries)
        for b in algebra.vectors:
            field.include(multiply(projector, b, n, p))
        result.append((projector, FiniteFieldAlgebra(field, projector, n)))
    return result, dimension


def _factor_order(order, *, factor_limit):
    factors = {int(q): int(e) for q, e in factorint(order, limit=factor_limit).items()}
    if any(not isprime(q) for q in factors):
        raise ResourceLimit("Finite-field order factorization is incomplete within factor_limit")
    return factors


def _prime_certificate(q, *, factor_limit, cache):
    """Pratt/Lucas witness above SymPy's deterministic primality range.

    For q < 2**64, isprime uses deterministic tests.  A probable-prime result
    above that range is only a candidate; the recursive order witness below
    proves primality before it can enter an exact-order certificate.
    """
    if q < 2 ** 64:
        if not isprime(q):
            raise ArithmeticError("Nonprime cyclic-order factor")
        return None
    if q in cache:
        return cache[q]
    factors = _factor_order(q - 1, factor_limit=factor_limit)
    children = [_prime_certificate(r, factor_limit=factor_limit, cache=cache) for r in factors]
    for a in range(2, min(q, factor_limit + 2)):
        if pow(a, q - 1, q) == 1 and all(pow(a, (q - 1) // r, q) != 1 for r in factors):
            result = {"prime": q, "base": a,
                      "factors": [[r, e] for r, e in sorted(factors.items())],
                      "children": [child for child in children if child is not None]}
            cache[q] = result
            return result
    raise ResourceLimit("Exact large-prime order certificate exceeded witness-search cap")


class _PrimeLog:
    """Baby-step/giant-step in a prime-order cyclic subgroup."""

    def __init__(self, field, generator, order, *, max_dlog_steps, stats):
        self.field, self.order = field, order
        self.width = isqrt(order - 1) + 1
        if self.width > max_dlog_steps:
            raise ResourceLimit(f"Discrete logarithm needs {self.width} baby steps; cap is {max_dlog_steps}")
        stats["max_baby_steps"] = max(stats["max_baby_steps"], self.width)
        self.baby, current = {}, field.one
        for j in range(self.width):
            self.baby[current] = j
            current = field.mul(current, generator)
        self.stride = field.pow(generator, -self.width)

    def solve(self, value):
        current = value
        for j in range((self.order + self.width - 1) // self.width):
            if current in self.baby:
                answer = j * self.width + self.baby[current]
                if answer < self.order:
                    return answer
            current = self.field.mul(current, self.stride)
        raise ArithmeticError("Finite-field discrete logarithm has no solution")


def _cyclic_component(field, actions, *, max_dlog_steps, factor_limit, stats):
    if all(action == field.one for action in actions):
        return field.one, 1, {}, [0] * len(actions)
    factors = _factor_order(field.p ** field.d - 1, factor_limit=factor_limit)
    orders = []
    for action in actions:
        order = field.p ** field.d - 1
        for q, multiplicity in factors.items():
            for _ in range(multiplicity):
                if field.pow(action, order // q) != field.one:
                    break
                order //= q
        orders.append(order)
    order = lcm(*orders)
    relevant, generator = {}, field.one
    for q in factors:
        primary = max((q ** _valuation(value, q) for value in orders), default=1)
        if primary == 1:
            continue
        exponent = _valuation(primary, q)
        relevant[q] = exponent
        j = next(j for j, value in enumerate(orders) if value % primary == 0)
        generator = field.mul(generator, field.pow(actions[j], orders[j] // primary))
    engines = {q: _PrimeLog(field, field.pow(generator, order // q), q,
                           max_dlog_steps=max_dlog_steps, stats=stats) for q in relevant}
    logs = []
    for action in actions:
        residues, moduli = [], []
        for q, exponent in relevant.items():
            answer = 0
            for k in range(exponent):
                remainder = field.mul(action, field.pow(generator, -answer))
                digit = engines[q].solve(field.pow(remainder, order // q ** (k + 1)))
                answer += digit * q ** k
            residues.append(answer)
            moduli.append(q ** exponent)
        answer = int(crt(moduli, residues)[0]) if moduli else 0
        if field.pow(generator, answer) != action:
            raise ArithmeticError("Discrete-logarithm witness failed")
        logs.append(answer)
    return generator, order, relevant, logs


def _valuation(value, q):
    result = 0
    while value % q == 0:
        result += 1
        value //= q
    return result


def _intersect_character(B, row, order):
    if order == 1:
        return B
    transformed = [sum(int(row[i]) * int(B[i, j]) for i in range(B.rows))
                   for j in range(B.cols)]
    return hermite_normal_form(B * congruence_kernel(transformed, order))


def _check_index(B, max_index):
    if max_index is not None and abs(int(B.det())) > max_index:
        raise ResourceLimit(f"Exact translation-kernel index already exceeds {max_index}")


def _intersect_unipotent(B, actions, n, p, *, max_linear_entries, max_index=None):
    """Successive exact kernels of 1+J^k -> J^k/J^(k+1)."""
    one, D = identity(n), len(actions)
    nilpotents = [sub(action, one, p) for action in actions if action != one]
    if not nilpotents:
        return B, 0
    ideal = Span(p, n * n, max_entries=max_linear_entries)
    for nilpotent in nilpotents:
        if any(power(nilpotent, n, n, p)):
            raise ValueError("Claimed unipotent action is not unipotent")
        ideal.include(nilpotent)
    at = 0
    while at < len(ideal.vectors):
        for nilpotent in nilpotents:
            ideal.include(multiply(nilpotent, ideal.vectors[at], n, p))
        at += 1
    ppower = 1
    while ppower < n:
        ppower *= p
    for step in range(1, n + 1):
        if not ideal.vectors:
            return B, step - 1
        following = Span(p, n * n, max_entries=max_linear_entries)
        for vector in ideal.vectors:
            for nilpotent in nilpotents:
                following.include(multiply(nilpotent, vector, n, p))
        if len(following.vectors) >= len(ideal.vectors):
            raise ArithmeticError("Unipotent ideal filtration is not nilpotent")
        residuals = []
        for j in range(D):
            action = one
            for i, original in enumerate(actions):
                action = multiply(action, power(original, int(B[i, j]) % ppower, n, p), n, p)
            difference = sub(action, one, p)
            if any(ideal.reduce(difference)):
                raise ArithmeticError("Retained action escaped its nilpotent filtration step")
            residuals.append(following.reduce(difference))
        rows = list(zip(*residuals))
        B = hermite_normal_form(B * modular_kernel(rows, D, p))
        _check_index(B, max_index)
        ideal = following
    raise ArithmeticError("Unipotent ideal did not terminate within matrix dimension")


def _encode(matrix, n):
    return [[i // n, i % n, int(value)] for i, value in enumerate(matrix) if value]


def translation_kernel(layers, D, *, max_index=4096, max_dlog_steps=100_000,
                       factor_limit=100_000, max_linear_entries=2_000_000,
                       verify=True):
    """Return the exact Lambda_0 as a column-basis :class:`Lattice`.

    Each value of ``layers`` supplies ``D`` commuting, invertible constant
    Laurent ``Matrix`` objects through its ``actions`` attribute.  The result
    has JSON-compatible ``kernel_certificate`` and ``kernel_statistics``.
    ``max_index`` limits the actual common kernel, never a rectangular bound.
    ``max_dlog_steps`` bounds a BSGS table for each prime-order PH subproblem.
    """
    D = _positive_integer(D, "D")
    max_dlog_steps = _positive_integer(max_dlog_steps, "max_dlog_steps")
    factor_limit = _positive_integer(factor_limit, "factor_limit")
    max_linear_entries = _positive_integer(max_linear_entries, "max_linear_entries")
    if max_index is not None:
        max_index = _positive_integer(max_index, "max_index")
    p, source = _actions(layers, D, max_linear_entries=max_linear_entries)
    B, records, prime_cache = ZMatrix.eye(D), [], {}
    stats = {"max_baby_steps": 0, "semisimple_algebra_dimensions": [],
             "field_degrees": [], "unipotent_filtration_steps": [],
             "uses_full_image_enumeration": False}
    for degree, n, actions in source:
        semisimple, unipotent = _split_actions(actions, n, p)
        components, dimension = _field_components(semisimple, n, p,
                                                  max_linear_entries=max_linear_entries)
        stats["semisimple_algebra_dimensions"].append(dimension)
        records_for_layer = []
        for projector, field in components:
            values = [field.span.coordinates(multiply(projector, a, n, p)) for a in semisimple]
            generator, order, factors, logs = _cyclic_component(
                field, values, max_dlog_steps=max_dlog_steps, factor_limit=factor_limit, stats=stats)
            stats["field_degrees"].append(field.d)
            records_for_layer.append({"projector": _encode(projector, n),
                                      "generator": _encode(field.matrix(generator), n),
                                      "order": order,
                                      "prime_factors": [[q, e] for q, e in sorted(factors.items())],
                                      "prime_certificates": [proof for q in sorted(factors)
                                                             if (proof := _prime_certificate(q, factor_limit=factor_limit,
                                                                                            cache=prime_cache)) is not None],
                                      "logs": logs})
            B = _intersect_character(B, logs, order)
            _check_index(B, max_index)
        B, steps = _intersect_unipotent(B, unipotent, n, p,
                                        max_linear_entries=max_linear_entries,
                                        max_index=max_index)
        stats["unipotent_filtration_steps"].append(steps)
        records.append({"degree": degree, "dimension": n, "components": records_for_layer,
                        "unipotent_actions": [_encode(a, n) for a in unipotent]})
    result = Lattice([[int(B[i, j]) for j in range(D)] for i in range(D)], max_index=max_index)
    result.kernel_certificate = {"schema": SCHEMA, "D": D, "p": p,
                                 "basis_columns": [list(row) for row in result.basis],
                                 "index": result.index, "layers": records}
    result.kernel_statistics = stats
    if verify:
        from css_decoupling.certificates.translation_kernel import verify_translation_kernel
        verify_translation_kernel(layers, D, result.kernel_certificate,
                                  max_linear_entries=max_linear_entries)
    return result
