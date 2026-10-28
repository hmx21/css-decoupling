"""Finite independent checks of the complete local-to-global QS route."""
import shutil
import subprocess

import pytest

from css_decoupling.algebra.laurent import Matrix, Ring, eye, zeros
from css_decoupling.algebra.polynomial import NoPolynomialSolution
from css_decoupling.algebra.quillen_suslin import complete_polynomial_row, complete_laurent_row
from css_decoupling.errors import NoLaurentSolution, ResourceLimit


def check_completion(row, U, V, *, polynomial):
    """Only independent sparse ring arithmetic, not the constructing checker."""
    expected = zeros(row.ring, 1, row.ncols)
    expected[0, 0] = 1
    assert row * U == expected
    assert U * V == eye(row.ring, row.ncols)
    assert V * U == eye(row.ring, row.ncols)
    assert V[:1, :] == row
    if polynomial:
        assert all(a >= 0 for matrix in (U, V) for f in matrix.entries.values()
                   for exponent in f.terms for a in exponent)


@pytest.mark.parametrize("prime", [2, 3, 5])
def test_two_patch_polynomial_completion_with_nonconstant_denominators(prime):
    ring = Ring(prime, 2)
    x, t = ring.variables
    row = Matrix.from_rows(ring, [[t*t, x*t + 1, x*x - x]])
    trace = []
    U, V = complete_polynomial_row(row, trace=trace, timeout=20, max_terms=100000)
    check_completion(row, U, V, polynomial=True)
    assert len(trace) == 1
    denominators = trace[0]["patch_denominators"]
    assert len(denominators) == 2
    assert all(not d.is_unit for d in denominators)
    assert Matrix.from_rows(ring, [denominators]) * trace[0]["cover_weights"] == eye(ring, 1)


def test_nontrivial_shear_and_genuine_three_variable_recursion():
    ring = Ring(3, 3)
    x, y, z = ring.variables
    row = Matrix.from_rows(ring, [[x, y, z, 1 - x - y - z]])
    trace = []
    U, V = complete_polynomial_row(row, trace=trace, timeout=20, max_terms=100000)
    check_completion(row, U, V, polynomial=True)
    assert [record["axes"] for record in trace] == [(0, 1, 2), (0, 1)]
    assert all(record["shear_weights"] for record in trace)
    assert any(len(record["patch_denominators"]) > 1 for record in trace)


def test_four_variable_recursion_visits_all_three_local_global_stages():
    ring = Ring(2, 4)
    x, y, z, w = ring.variables
    # Reverse axis order prevents avoidable high-power normalization while
    # still exercising three recursive QS stages and nontrivial patch covers.
    row = Matrix.from_rows(ring, [[w, z, y, x, 1 - x - y - z - w]])
    trace = []
    U, V = complete_polynomial_row(row, trace=trace, timeout=20, max_terms=100000)
    check_completion(row, U, V, polynomial=True)
    assert [record["axes"] for record in trace] == [(0, 1, 2, 3), (0, 1, 2), (0, 1)]
    assert [len(record["patch_denominators"]) for record in trace] == [4, 3, 2]


def test_negative_laurent_support_uses_normalization_then_full_qs():
    ring = Ring(5, 2)
    x, y = ring.variables
    row = Matrix.from_rows(ring, [[x**-1 * (x - 1), y**-1 * (y - 1),
                                  (3 - x - y) * x**-1 * y**-1]])
    assert not any(f.is_unit for f in row.entries.values())
    assert row * Matrix.from_rows(ring, [[x], [y], [x*y]]) == eye(ring, 1)
    trace = []
    U, V = complete_laurent_row(row, trace=trace, timeout=20, max_terms=100000)
    check_completion(row, U, V, polynomial=False)
    assert trace and any(len(record["axes"]) == 2 for record in trace)


@pytest.mark.parametrize("options", [{"max_patches": 1}, {"max_steps": 0}, {"timeout": 1e-12}])
def test_resource_stops_do_not_return_incomplete_completions(options):
    ring = Ring(3, 2)
    x, t = ring.variables
    row = Matrix.from_rows(ring, [[t*t, x*t + 1, x*x - x]])
    with pytest.raises(ResourceLimit):
        complete_polynomial_row(row, **options)


def test_polynomial_nonmembership_is_distinct_from_laurent_unimodularity():
    ring = Ring(3, 2)
    x, y = ring.variables
    row = Matrix.from_rows(ring, [[x, y, x + y]])
    with pytest.raises(NoPolynomialSolution):
        complete_polynomial_row(row)
    check_completion(row, *complete_laurent_row(row), polynomial=False)
    with pytest.raises(NoLaurentSolution):
        complete_laurent_row(Matrix.from_rows(ring, [[x - 1, y - 1, 0]]))


@pytest.mark.parametrize("kind", ["empty", "not_row", "negative", "zero"])
def test_bad_inputs_are_rejected(kind):
    ring = Ring(3, 2)
    if kind == "empty":
        row = zeros(ring, 1, 0)
    elif kind == "not_row":
        row = eye(ring, 2)
    elif kind == "negative":
        row = Matrix.from_rows(ring, [[ring.variable(0)**-1, 1, 0]])
    else:
        row = zeros(ring, 1, 3)
    with pytest.raises(ValueError):
        complete_polynomial_row(row)


def test_full_qs_uses_no_external_process_at_large_prime(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Native QS invoked external process or executable discovery")
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(shutil, "which", forbidden)
    ring = Ring(2147483659, 2)
    x, y = ring.variables
    row = Matrix.from_rows(ring, [[x, y, 1 - x - y]])
    trace = []
    U, V = complete_polynomial_row(row, trace=trace, timeout=20)
    check_completion(row, U, V, polynomial=True)
    assert trace
