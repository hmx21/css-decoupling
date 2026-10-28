"""Local degree reduction, extension points and denominator-free patching."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from css_decoupling import Matrix, Ring, eye
from css_decoupling.algebra.local_completion import (
    AlgebraBudget, RationalContext, local_completion, patch_completions,
)
from css_decoupling.algebra.polynomial import solve_polynomial
from css_decoupling.algebra.residue_fields import PrimeField, ExtensionField
from css_decoupling.errors import ResourceLimit


def point_at(ring, domain, coordinates):
    def evaluate(f):
        total = domain.zero
        for exponent, coefficient in f.terms.items():
            term = domain.coerce(coefficient)
            for x, degree in zip(coordinates, exponent):
                term = domain.mul(term, domain.pow(x, degree))
            total = domain.add(total, term)
        return total
    return SimpleNamespace(field=domain, evaluate=evaluate)


@pytest.mark.parametrize('p', [2, 3, 5, 7])
def test_suslin_third_column_and_two_patch_polynomial_inverses(p):
    r = Ring(p, 2)
    x, t = r.variables
    row = Matrix.from_rows(r, [[t*t, x*t+1, x*x-x]])
    domain = PrimeField(p)
    solutions = [local_completion(row, 1, point_at(r, domain, (a, 0))) for a in (0, 1)]
    assert all(all(a > b for a, b in zip(s.degrees, s.degrees[1:])) for s in solutions)
    assert solutions[0].degrees == (2, 1)  # Includes the Suslin inner loop at x=0.
    weights = solve_polynomial(Matrix.from_rows(r, [[s.denominator for s in solutions]]), eye(r, 1))
    U, V = patch_completions(row, 1, solutions, weights)
    assert row*U == Matrix.from_rows(r, [[0, 1, x*x-x]])
    assert U*V == V*U == eye(r, 3)
    assert all(k >= 0 for A in (U, V) for f in A.entries.values() for e in f.terms for k in e)


def test_local_unit_test_at_extension_point_uses_field_zero_not_tuple_truth():
    r = Ring(2, 2)
    x, t = r.variables
    K = ExtensionField(PrimeField(2), (1, 1, 1))
    point = point_at(r, K, (K.generator, K.zero))
    c = x*x+x+1
    assert point.evaluate(c) == K.zero
    row = Matrix.from_rows(r, [[t*t, c*t+1, c*c-c]])
    result = local_completion(row, 1, point)
    assert result.degrees == (2, 1)
    assert point.evaluate(result.denominator) != K.zero


def test_patch_rejects_fake_denominator_and_wrong_cover():
    r = Ring(5, 2)
    x, t = r.variables
    row = Matrix.from_rows(r, [[t*t, x*t+1, x*x-x]])
    s = local_completion(row, 1, point_at(r, PrimeField(5), (0, 0)))
    with pytest.raises(ValueError, match='sum to one'):
        patch_completions(row, 1, [s], Matrix.from_rows(r, [[1]]))
    with pytest.raises(ArithmeticError, match='denominator'):
        patch_completions(row, 1, [replace(s, denominator=r.one())], Matrix.from_rows(r, [[1]]))
    with pytest.raises(ValueError, match='polynomial'):
        patch_completions(row, 1, [replace(s, denominator=x)], Matrix.from_rows(r, [[x**-1]]))


def test_bad_monic_input_and_cooperative_resource_failures():
    r = Ring(3, 2)
    x, t = r.variables
    point = point_at(r, PrimeField(3), (0, 0))
    with pytest.raises(ValueError, match='monic'):
        local_completion(Matrix.from_rows(r, [[x*t, 1, 0]]), 1, point)
    with pytest.raises(ResourceLimit):
        local_completion(Matrix.from_rows(r, [[t*t, x*t+1, x*x-x]]), 1, point,
                         context=RationalContext(r, AlgebraBudget(max_steps=0)))


def test_fraction_unit_scaling_and_zero_specialization_are_exact():
    r = Ring(5, 2)
    ctx = RationalContext(r)
    x, t = ctx.variables
    # SymPy may store 4/4 rather than 1; conversion is coefficient-field exact.
    assert ctx.polynomial(ctx.field.raw_new(ctx.field.ring(4), ctx.field.ring(4))) == 1
    assert ctx.substitute((t+1)/(x+1), 1, ctx.field.zero) - 1/(x+1) == ctx.field.zero
