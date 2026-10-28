"""Constructive local Horrocks reduction and exact polynomial patching.

Rational functions occur only internally over the fraction field of the
coefficient polynomial ring. A local result is never passed off as a Laurent
completion: patching must remove every nonconstant denominator, in both
directions, before producing an ordinary Matrix.
"""
from dataclasses import dataclass
from time import perf_counter

from sympy import GF
from sympy.polys.fields import field

from css_decoupling.algebra.laurent import Matrix, Poly, eye
from css_decoupling.errors import ResourceLimit


class AlgebraBudget:
    """Cooperative bounds on actual rational-algebra work, not search degrees."""
    def __init__(self, *, timeout=120, max_terms=2000000, max_steps=1000000):
        self.deadline = None if timeout is None else perf_counter() + timeout
        self.max_terms, self.max_steps, self.steps = max_terms, max_steps, 0

    def check(self, *values):
        self.steps += 1
        if self.max_steps is not None and self.steps > self.max_steps:
            raise ResourceLimit("Local completion exceeded its algebra-operation cap")
        if self.deadline is not None and perf_counter() > self.deadline:
            raise ResourceLimit("Local completion exceeded its cooperative time limit")
        if self.max_terms is not None:
            for value in values:
                count = len(value.numer) + len(value.denom)
                if count > self.max_terms:
                    raise ResourceLimit("Local completion exceeded its rational-function term cap")


class RationalContext:
    """One exact GF(p)(x_1,...,x_D) context with explicit polynomial conversion."""
    def __init__(self, ring, budget=None):
        self.ring = ring
        self.field, *self.variables = field(','.join(f'q{i}' for i in range(ring.D)), GF(ring.p))
        self.budget = budget or AlgebraBudget()

    def convert(self, polynomial):
        polynomial = self.ring(polynomial)
        if any(e_i < 0 for e in polynomial.terms for e_i in e):
            raise ValueError("This stage requires ordinary polynomial inputs")
        return self.field(self.field.ring.from_dict(dict(polynomial.terms)))

    def polynomial(self, rational):
        self.budget.check(rational)
        if not rational.denom.is_ground:
            raise ArithmeticError("A nonconstant fraction denominator survived polynomial patching")
        denominator = int(rational.denom.get((0,) * self.ring.D, 0)) % self.ring.p
        factor = pow(denominator, -1, self.ring.p)
        return self.ring({e: int(c) * factor for e, c in rational.numer.items()})

    def matrix(self, entries):
        width = len(entries[0]) if entries else 0
        return Matrix(self.ring, len(entries), width,
                      {(i, j): self.polynomial(f) for i, row in enumerate(entries)
                       for j, f in enumerate(row) if f})

    def coefficients(self, value, axis):
        if any(e[axis] for e in value.denom):
            raise ArithmeticError("A local coefficient denominator depends on the main variable")
        pieces = {}
        for exponent, coefficient in value.numer.items():
            degree = exponent[axis]
            e = tuple(0 if i == axis else x for i, x in enumerate(exponent))
            pieces.setdefault(degree, {})[e] = coefficient
        return {d: self.field(self.field.ring.from_dict(data)) / self.field(value.denom)
                for d, data in pieces.items()}

    def degree(self, value, axis):
        return max(self.coefficients(value, axis), default=-1)

    def lead(self, value, axis):
        coefficients = self.coefficients(value, axis)
        return coefficients[max(coefficients)] if coefficients else self.field.zero

    def remainder(self, numerator, denominator, axis):
        quotient, remainder = self.field.zero, numerator
        d = self.degree(denominator, axis)
        if d < 0:
            raise ZeroDivisionError("Zero polynomial divisor")
        leading = self.lead(denominator, axis)
        while remainder and self.degree(remainder, axis) >= d:
            q = self.lead(remainder, axis) / leading * self.variables[axis] ** (self.degree(remainder, axis) - d)
            quotient, remainder = quotient + q, remainder - q * denominator
            self.budget.check(quotient, remainder)
        return quotient, remainder

    def substitute(self, value, axis, replacement):
        def polynomial_part(polynomial):
            answer = self.field.zero
            for exponent, coefficient in polynomial.items():
                e = tuple(0 if i == axis else a for i, a in enumerate(exponent))
                power = replacement ** exponent[axis] if exponent[axis] else self.field.one
                term = self.field(self.field.ring.from_dict({e: coefficient})) * power
                answer += term
                self.budget.check(answer)
            return answer
        answer = polynomial_part(value.numer) / polynomial_part(value.denom)
        self.budget.check(answer)
        return answer


def _identity(context, n):
    return [[context.field.one if i == j else context.field.zero for j in range(n)] for i in range(n)]


def _product(context, A, B):
    if not A or not B:
        return []
    result = []
    for row in A:
        new_row = []
        for column in zip(*B):
            value = context.field.zero
            for a, b in zip(row, column):
                if a and b:
                    value += a * b
                    context.budget.check(value)
            new_row.append(value)
        result.append(new_row)
    return result


def _equal(A, B):
    # SymPy can retain different nonzero constant scalings of numerator and
    # denominator over GF(p), e.g. 4/4. Equality here is field equality.
    return (len(A) == len(B) and all(len(a) == len(b) and
            all(not (x - y) for x, y in zip(a, b)) for a, b in zip(A, B)))


@dataclass
class LocalCompletion:
    original: Matrix
    axis: int
    context: RationalContext
    forward: list
    inverse: list
    denominator: Poly
    degrees: tuple


def local_completion(row, axis, point, *, context=None):
    """Complete a row whose first entry is monic in axis at a closed point.

    ``point`` is a homomorphism from the coefficient polynomial ring to an
    explicitly verified finite residue field. A coefficient is a local unit
    exactly when its image is nonzero. The main monic degree decreases at
    every outer step; Suslin's inner degree defect also strictly decreases.
    """
    if row.nrows != 1 or row.ncols < 3 or not 0 <= axis < row.ring.D:
        raise ValueError("Local Horrocks reduction expects at least three row entries and a valid axis")
    ctx = context or RationalContext(row.ring)
    if ctx.ring != row.ring:
        raise ValueError("Rational context has the wrong coefficient ring")
    n, F = row.ncols, ctx.field
    f = [ctx.convert(row[0, j]) for j in range(n)]
    if ctx.lead(f[0], axis) - F.one:
        raise ValueError("The first row entry must be monic in the main variable")
    U, V = _identity(ctx, n), _identity(ctx, n)

    def unit(coefficient):
        num = row.ring({e: int(c) for e, c in coefficient.numer.items()})
        den = row.ring({e: int(c) for e, c in coefficient.denom.items()})
        if point.evaluate(den) == point.field.zero:
            raise ArithmeticError("A supposedly local denominator vanishes at the chosen point")
        return point.evaluate(num) != point.field.zero

    def swap(i, j):
        if i == j:
            return
        f[i], f[j] = f[j], f[i]
        for k in range(n):
            U[k][i], U[k][j] = U[k][j], U[k][i]
        V[i], V[j] = V[j], V[i]

    def add(target, coefficients):
        if target in coefficients:
            raise ValueError("Elementary column addition cannot use its target as a source")
        f[target] += sum((f[i] * c for i, c in coefficients.items()), F.zero)
        for k in range(n):
            U[k][target] += sum((U[k][i] * c for i, c in coefficients.items()), F.zero)
        for i, c in coefficients.items():
            V[i] = [a - c * b for a, b in zip(V[i], V[target])]
        ctx.budget.check(f[target], *(U[k][target] for k in range(n)))

    def scale(j, coefficient):
        f[j] *= coefficient
        for k in range(n):
            U[k][j] *= coefficient
            V[j][k] /= coefficient
        ctx.budget.check(f[j])

    degrees = []
    while ctx.degree(f[0], axis) > 0:
        degree = ctx.degree(f[0], axis)
        if degrees and degree >= degrees[-1]:
            raise ArithmeticError("Horrocks main degree did not strictly decrease")
        degrees.append(degree)
        if not unit(ctx.lead(f[0], axis)):
            raise ArithmeticError("Main polynomial lost its local-unit leading coefficient")
        scale(0, 1 / ctx.lead(f[0], axis))
        for j in range(1, n):
            quotient, _ = ctx.remainder(f[j], f[0], axis)
            if quotient:
                add(j, {0: -quotient})
        candidates = [j for j in range(1, n) if any(unit(c) for c in ctx.coefficients(f[j], axis).values())]
        if not candidates:
            raise ValueError("The row is not unimodular over the chosen residue-field polynomial ring")
        swap(1, min(candidates, key=lambda j: (ctx.degree(f[j], axis), j)))
        if unit(ctx.lead(f[1], axis)):
            swap(0, 1)
            continue
        # Suslin step: g=alpha*f0+beta*f1. Modulo the maximal ideal,
        # multiplying by t^(degree-deg(g)) raises its nonzero residue degree,
        # while subtracting lc(g)*f0 cancels its actual leading term.
        g, alpha, beta = f[1], F.zero, F.one
        previous_defect = None
        while not unit(ctx.lead(g, axis)):
            d = ctx.degree(g, axis)
            residue_degree = max(k for k, c in ctx.coefficients(g, axis).items() if unit(c))
            defect = d - residue_degree
            if previous_defect is not None and defect >= previous_defect:
                raise ArithmeticError("Suslin degree defect did not strictly decrease")
            previous_defect = defect
            shift, leading = ctx.variables[axis] ** (degree - d), ctx.lead(g, axis)
            g, alpha, beta = shift * g - leading * f[0], shift * alpha - leading, shift * beta
            ctx.budget.check(g, alpha, beta)
            if not g or ctx.degree(g, axis) >= degree:
                raise ArithmeticError("Suslin reduction left its strict degree bound")
        leading = ctx.lead(g, axis)
        g, alpha, beta = g / leading, alpha / leading, beta / leading
        if alpha * f[0] + beta * f[1] - g:
            raise ArithmeticError("Suslin combination certificate failed")
        # Use a distinct third column: f2 -> rem(f2,g)+g is monic of
        # degree deg(g)<degree. This is an invertible elementary operation;
        # replacing f1 by g directly would in general not be invertible.
        quotient, _ = ctx.remainder(f[2], g, axis)
        add(2, {0: (1 - quotient) * alpha, 1: (1 - quotient) * beta})
        swap(0, 2)
    if not f[0] or not unit(f[0]):
        raise ArithmeticError("Local completion did not reach a unit constant")
    scale(0, 1 / f[0])
    for j in range(1, n):
        add(j, {0: -f[j]})
    identity = _identity(ctx, n)
    original = [[ctx.convert(row[0, j]) for j in range(n)]]
    if (not _equal(_product(ctx, U, V), identity) or not _equal(_product(ctx, V, U), identity) or
            not _equal(_product(ctx, original, U), [identity[0]])):
        raise ArithmeticError("Local completion inverse identities failed")
    # A product of the two common denominators clears U(t)V(t+s).
    # Unlike a minimal denominator it needs no extra dummy variable.
    denominators = []
    for entries in (U, V):
        common = F.ring.one
        for line in entries:
            for value in line:
                if any(e[axis] for e in value.denom):
                    raise ArithmeticError("Local matrix denominator depends on the main variable")
                common = common.lcm(value.denom)
        denominators.append(common)
    denominator = row.ring({e: int(c) for e, c in (denominators[0] * denominators[1]).items()})
    if point.evaluate(denominator) == point.field.zero:
        raise ArithmeticError("Local denominator does not cover its chosen point")
    return LocalCompletion(row.copy(), axis, ctx, U, V, denominator, tuple(degrees))


def patch_completions(row, axis, solutions, weights, *, context=None):
    """Patch local completions into W with row*W=row evaluated at axis=0.

    Polynomial weights satisfy sum(weights_i*denominator_i)=1. A telescoping
    sequence of substitutions differs by a multiple of each denominator,
    so each forward and inverse transition is polynomial. Both identities
    and every cleared denominator are checked explicitly.
    """
    if (not isinstance(row, Matrix) or row.nrows != 1 or
            not isinstance(weights, Matrix) or not 0 <= axis < row.ring.D or
            not solutions or weights.shape != (len(solutions), 1) or weights.ring != row.ring):
        raise ValueError("Patching requires matching local solutions and polynomial weights")
    if any(a < 0 for M in (row, weights) for f in M.entries.values() for e in f.terms for a in e):
        raise ValueError("Patching requires polynomial rows and weights")
    ctx = context or solutions[0].context
    n, F = row.ncols, ctx.field
    denominators = Matrix.from_rows(row.ring, [[s.denominator for s in solutions]])
    if denominators * weights != eye(row.ring, 1):
        raise ValueError("Local denominator weights do not sum to one")
    W, Winv = eye(row.ring, n), eye(row.ring, n)
    variable, current = ctx.variables[axis], ctx.variables[axis]
    for i, solution in enumerate(solutions):
        if solution.original != row or solution.axis != axis or solution.context.field != F:
            raise ValueError("Local solutions refer to different row-completion problems")
        following = current - variable * ctx.convert(weights[i, 0] * solution.denominator)
        Ua = [[ctx.substitute(f, axis, current) for f in line] for line in solution.forward]
        Ub = [[ctx.substitute(f, axis, following) for f in line] for line in solution.forward]
        Va = [[ctx.substitute(f, axis, current) for f in line] for line in solution.inverse]
        Vb = [[ctx.substitute(f, axis, following) for f in line] for line in solution.inverse]
        transition = ctx.matrix(_product(ctx, Ua, Vb))
        transition_inverse = ctx.matrix(_product(ctx, Ub, Va))
        if transition * transition_inverse != eye(row.ring, n) or transition_inverse * transition != eye(row.ring, n):
            raise ArithmeticError("Polynomial transition matrices are not inverse")
        W, Winv = W * transition, transition_inverse * Winv
        current = following
    if current:
        raise ArithmeticError("Patching failed to reach the zero specialization")
    specialized = Matrix(row.ring, 1, n,
                         {(i, j): Poly(row.ring, {e: c for e, c in f.terms.items() if not e[axis]})
                          for (i, j), f in row.entries.items()})
    if row * W != specialized or W * Winv != eye(row.ring, n) or Winv * W != eye(row.ring, n):
        raise ArithmeticError("Global patching identities failed")
    return W, Winv
