"""Exact arbitrary-length Euclidean Laurent row completion."""
from random import Random
import subprocess

import pytest

from css_decoupling.algebra.laurent import Matrix, Ring, eye
from css_decoupling.algebra.univariate import (
    complete_unimodular_row, UnivariateCompletionUnavailable, NoPolynomialSolution,
)
from css_decoupling.errors import NoLaurentSolution


def check(row, U, Uinv):
    expected = Matrix(row.ring, 1, row.ncols, {(0, 0): 1})
    assert row * U == expected
    assert U * Uinv == eye(row.ring, row.ncols)
    assert Uinv * U == eye(row.ring, row.ncols)
    assert Uinv[0:1, :] == row


@pytest.mark.parametrize("prime", [2, 3, 5, 2147483659])
def test_arbitrary_prime_laurent_bezout_with_zero_columns(prime):
    r = Ring(prime, 3)
    t = r.variable(1)
    row = Matrix.from_rows(r, [[0, t**-7 * (1 + t**3), 0, t**4 * (t**2 + t**3 + 1)]])
    U, Uinv = complete_unimodular_row(row, axis=1)
    check(row, U, Uinv)


@pytest.mark.parametrize("coefficient,exponent", [(1, 0), (2, -11), (4, 13)])
def test_one_entry_is_completed_exactly_when_it_is_a_unit(coefficient, exponent):
    r = Ring(5, 1)
    row = Matrix.from_rows(r, [[coefficient * r.variable(0)**exponent]])
    U, Uinv = complete_unimodular_row(row)
    check(row, U, Uinv)


@pytest.mark.parametrize("prime,length", [(2, 2), (3, 3), (5, 7), (101, 11)])
def test_known_unimodular_long_rows_via_elementary_transformations(prime, length):
    rng = Random(13001 + prime + length)
    r = Ring(prime, 2)
    t = r.variable(0)
    change = eye(r, length)
    for step in range(3 * length):
        source = step % length
        target = (source + 1 + (step // length) % (length - 1)) % length
        if target == source:
            target = (target + 1) % length
        coefficient = (1 + rng.randrange(prime - 1)) * t**rng.randrange(-4, 5) + t**rng.randrange(-4, 5)
        elementary = Matrix(r, length, length, {(source, target): coefficient})
        change = change * (eye(r, length) + elementary)
    row = change[0:1, :]
    U, Uinv = complete_unimodular_row(row)
    check(row, U, Uinv)


@pytest.mark.parametrize("entries", ["zero", "empty", "single_nonunit", "common_nonunit"])
def test_nonunimodular_row_rejected_by_exact_gcd(entries):
    r = Ring(3, 2)
    t = r.variable(0)
    values = {"zero": [0, 0, 0], "empty": [], "single_nonunit": [t - 1],
              "common_nonunit": [t**-3 * (t - 1), t**5 * (t*t - 1), 0]}[entries]
    row = Matrix(r, 1, len(values), {(0, j): f for j, f in enumerate(values)})
    with pytest.raises(NoLaurentSolution):
        complete_unimodular_row(row)


def test_multivariate_unimodular_row_is_unsupported_not_reported_nonexistent():
    r = Ring(5, 2)
    x, y = r.variables
    row = Matrix.from_rows(r, [[x, y]])  # Unimodular because x is a unit.
    with pytest.raises(UnivariateCompletionUnavailable, match="another axis"):
        complete_unimodular_row(row, axis=0)


@pytest.mark.parametrize("axis", [-1, 2, True, 0.5])
def test_bad_axis_rejected(axis):
    r = Ring(3, 2)
    with pytest.raises(ValueError, match="axis"):
        complete_unimodular_row(Matrix.from_rows(r, [[1, 0]]), axis=axis)


def test_bad_matrix_shape_rejected():
    with pytest.raises(ValueError, match="one-row"):
        complete_unimodular_row(eye(Ring(3, 2), 2))


def test_completion_needs_no_subprocess(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Univariate completion invoked an external process")
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    r = Ring(5, 4)
    t = r.variable(3)
    row = Matrix.from_rows(r, [[t**-2 * (t*t + t + 1), t**3 * (t*t + 1)]])
    check(row, *complete_unimodular_row(row, axis=3))


def test_polynomial_completion_preserves_the_polynomial_ring():
    r = Ring(5, 3)
    t = r.variable(1)
    row = Matrix.from_rows(r, [[t, 1 - t, t*t + 1]])
    U, Uinv = complete_unimodular_row(row, axis=1, polynomial=True)
    check(row, U, Uinv)
    assert all(e >= 0 for matrix in (U, Uinv) for f in matrix.entries.values()
               for exponent in f.terms for e in exponent)


def test_common_monomial_is_only_invertible_in_laurent_mode():
    r = Ring(3, 1)
    t = r.variable(0)
    row = Matrix.from_rows(r, [[t, t*t + t, t**4]])
    check(row, *complete_unimodular_row(row))
    with pytest.raises(NoPolynomialSolution, match="nonconstant polynomial gcd"):
        complete_unimodular_row(row, polynomial=True)


def test_polynomial_completion_rejects_negative_input_exponents():
    r = Ring(5, 1)
    row = Matrix.from_rows(r, [[r.variable(0)**-1, 1]])
    with pytest.raises(ValueError, match="negative exponents"):
        complete_unimodular_row(row, polynomial=True)
