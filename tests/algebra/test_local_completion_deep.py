"""Consecutive Suslin operations and extension-residue polynomial patching."""
import pytest

from css_decoupling.algebra.laurent import Matrix, Ring, eye
from css_decoupling.algebra.closed_points import closed_point
from css_decoupling.algebra.local_completion import local_completion, patch_completions
from css_decoupling.algebra.polynomial import solve_polynomial


def check_local_by_clearing_denominators(row, solution):
    """Recheck rational witnesses using independent sparse polynomial products."""
    ctx, denominator = solution.context, solution.denominator
    d = ctx.convert(denominator)
    U = Matrix(row.ring, row.ncols, row.ncols,
               {(i, j): ctx.polynomial(value * d) for i, line in enumerate(solution.forward)
                for j, value in enumerate(line) if value})
    V = Matrix(row.ring, row.ncols, row.ncols,
               {(i, j): ctx.polynomial(value * d) for i, line in enumerate(solution.inverse)
                for j, value in enumerate(line) if value})
    assert U * V == denominator**2 * eye(row.ring, row.ncols)
    assert V * U == denominator**2 * eye(row.ring, row.ncols)
    assert row * U == denominator * eye(row.ring, row.ncols)[:1, :]


@pytest.mark.parametrize("prime", [2, 5])
def test_two_consecutive_suslin_inner_steps_before_main_degree_drop(prime):
    ring = Ring(prime, 2)
    x, t = ring.variables
    # At x=0 the second entry has residue degree zero but actual degree 3.
    # The two inner combinations are x*t^3+t, then t^2. Thus this explicitly
    # exercises defect 3 -> 2 -> 0 before the outer degree drops 4 -> 2.
    row = Matrix.from_rows(ring, [[t**4, x*t**3 + x*t*t + 1, 0]])
    point = closed_point(ring, [x], axes=(0,))
    solution = local_completion(row, 1, point)
    assert solution.degrees == (4, 2)
    assert solution.denominator == 1
    check_local_by_clearing_denominators(row, solution)


def test_extension_and_prime_residue_patches_produce_global_polynomial_inverses():
    ring = Ring(2, 2)
    x, t = ring.variables
    c = x*x + x + 1
    row = Matrix.from_rows(ring, [[t*t, c*t + 1, c*c - c]])
    extension_point = closed_point(ring, [c], axes=(0,))
    prime_point = closed_point(ring, [x], axes=(0,))
    assert extension_point.field.order == 4 and prime_point.field.order == 2
    assert extension_point.evaluate(c) == extension_point.field.zero
    assert prime_point.evaluate(c) == prime_point.field.one
    solutions = [local_completion(row, 1, point) for point in (extension_point, prime_point)]
    for solution, point in zip(solutions, (extension_point, prime_point)):
        assert point.evaluate(solution.denominator) != point.field.zero
        check_local_by_clearing_denominators(row, solution)
    denominators = Matrix.from_rows(ring, [[s.denominator for s in solutions]])
    weights = solve_polynomial(denominators, eye(ring, 1))
    assert denominators * weights == eye(ring, 1)
    U, V = patch_completions(row, 1, solutions, weights)
    assert row * U == Matrix.from_rows(ring, [[0, 1, c*c - c]])
    assert U * V == V * U == eye(ring, 3)
    assert all(a >= 0 for matrix in (U, V) for f in matrix.entries.values()
               for exponent in f.terms for a in exponent)
