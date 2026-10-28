"""Exactness and honest resource semantics of the C5 coefficient backend."""
from copy import deepcopy

import pytest

from css_decoupling.algebra.bounded_linear import BoundedLinearBackend, degree_bound
from css_decoupling.algebra.laurent import Matrix, Ring, eye, zeros
from css_decoupling.algebra.modules import using_backend, solve
from css_decoupling.certificates.bounded_linear import verify_bounded_linear
from css_decoupling.errors import BackendUnavailable, NoLaurentSolution, ResourceLimit


@pytest.mark.parametrize("p", [2, 3, 5, 101])
def test_nonunit_solution_and_negative_support(p):
    ring = Ring(p, 2)
    x, y = ring.variables
    A = Matrix.from_rows(ring, [[x - 1, y - 1]])
    B = Matrix.from_rows(ring, [[(x - 1) * x**-1 + (y - 1) * 2, x + y - 2]])
    backend = BoundedLinearBackend(ring)
    X = backend.solve(A, B)
    assert A * X == B
    assert backend.last_certificate["method"] == "bounded-coefficients"
    assert verify_bounded_linear(backend.last_certificate).valid
    assert backend.last_certificate["evidence"]["degree"] >= 1


@pytest.mark.parametrize("D", [1, 2, 4, 5])
def test_unit_kernel_is_complete_and_certified(D):
    ring = Ring(3, D)
    x = ring.variable(0)
    A = Matrix.from_rows(ring, [[x, x + 1, x**-2]])
    backend = BoundedLinearBackend(ring, max_degree=0)
    K = backend.kernel(A)
    assert K.ncols == 2
    assert A * K == zeros(ring, 1, 2)
    certificate = deepcopy(backend.last_certificate)
    assert verify_bounded_linear(certificate, max_degree=0).valid
    certificate["result"]["cols"] = 0
    certificate["result"]["entries"] = []
    assert not verify_bounded_linear(certificate).valid


def test_domain_and_empty_kernels():
    ring = Ring(5, 3)
    x = ring.variable(0)
    backend = BoundedLinearBackend(ring, max_degree=0)
    A = Matrix.from_rows(ring, [[x - 1], [0]])
    assert backend.kernel(A) == zeros(ring, 1, 0)
    assert verify_bounded_linear(backend.last_certificate).valid
    assert backend.kernel(zeros(ring, 0, 4)) == eye(ring, 4)
    assert verify_bounded_linear(backend.last_certificate).valid
    B = zeros(ring, 0, 2)
    assert backend.solve(zeros(ring, 0, 4), B) == zeros(ring, 4, 2)
    assert verify_bounded_linear(backend.last_certificate).valid


def test_specialization_is_a_finite_exact_nonmembership_proof():
    ring = Ring(3, 2)
    x, y = ring.variables
    A = Matrix.from_rows(ring, [[x - 1, y - 1]])
    backend = BoundedLinearBackend(ring, max_degree=0)
    with pytest.raises(NoLaurentSolution):
        backend.solve(A, eye(ring, 1))
    assert backend.last_certificate["method"] == "augmentation-obstruction"
    assert verify_bounded_linear(backend.last_certificate).valid
    data = deepcopy(backend.last_certificate)
    data["B"]["entries"] = []
    assert not verify_bounded_linear(data).valid


def test_insufficient_degree_is_not_nonmembership():
    ring = Ring(3, 2)
    x, y = ring.variables
    A = Matrix.from_rows(ring, [[x - 1, y - 1]])
    B = Matrix.from_rows(ring, [[(x - 1) * x**-1]])
    backend = BoundedLinearBackend(ring, max_degree=0)
    with pytest.raises(ResourceLimit):
        backend.solve(A, B)
    assert backend.last_certificate is None


def test_complete_kernel_cannot_be_claimed_from_low_degree_solutions():
    ring = Ring(3, 4)
    x, y = ring.variables[:2]
    backend = BoundedLinearBackend(ring, max_degree=2)
    A = Matrix.from_rows(ring, [[x - 1, y - 1]])
    with pytest.raises(ResourceLimit, match="full proven degree bound"):
        backend.kernel(A)
    assert backend.last_certificate is None
    metadata = backend.last_bound
    assert metadata["variables"] == 8
    assert metadata["base"] == 4
    assert metadata["exponent_power_of_two"] == 8


def test_rhs_degree_enters_theorem_bound():
    ring = Ring(5, 2)
    x, y = ring.variables
    A = Matrix.from_rows(ring, [[x - 1]])
    B = Matrix.from_rows(ring, [[x**-3 * y**4]])
    metadata = degree_bound(A, B)
    assert metadata["polynomial_degree"] == 7
    assert metadata["base"] == 14


def test_allocation_preflight_and_no_groebner_api():
    ring = Ring(3, 3)
    x, y = ring.variables[:2]
    A = Matrix.from_rows(ring, [[x - 1, y - 1]])
    B = Matrix.from_rows(ring, [[x + y - 2]])
    backend = BoundedLinearBackend(ring, max_matrix_entries=1)
    with pytest.raises(ResourceLimit):
        backend.solve(A, B)
    with pytest.raises(BackendUnavailable):
        backend.quotient_groebner(A)
    with pytest.raises(BackendUnavailable):
        backend.normal_form(A, B)


def test_context_selection_uses_only_bounded_backend(monkeypatch):
    import css_decoupling.algebra.modules as modules
    monkeypatch.setattr(modules, "sdm_groebner", lambda *a, **kw: pytest.fail("Groebner called"))
    ring = Ring(3, 2)
    x, y = ring.variables
    A = Matrix.from_rows(ring, [[x - 1, y - 1]])
    B = Matrix.from_rows(ring, [[x + y - 2]])
    backend = BoundedLinearBackend(ring)
    with using_backend(backend):
        assert A * solve(A, B) == B


def test_nontrivial_unit_reduction_nonmembership_child():
    ring = Ring(5, 2)
    x, y = ring.variables
    A = Matrix.from_rows(ring, [[1, x], [0, (x - 2) * (x - 1)]])
    B = Matrix.from_rows(ring, [[0], [x - 1]])
    # At x=1 the residual RHS vanishes, so no false negative follows there.
    # A small cap must leave this genuine nonmember undecided.
    backend = BoundedLinearBackend(ring, max_degree=1)
    with pytest.raises(ResourceLimit):
        backend.solve(A, B)


def test_bound_tamper_and_unknown_proof_method():
    ring = Ring(3, 2)
    backend = BoundedLinearBackend(ring)
    backend.kernel(zeros(ring, 2, 1))
    certificate = deepcopy(backend.last_certificate)
    certificate["bound"]["base"] += 1
    assert not verify_bounded_linear(certificate).valid
    certificate = deepcopy(backend.last_certificate)
    certificate["method"] = "trust-me"
    assert not verify_bounded_linear(certificate).valid
    certificate = deepcopy(backend.last_certificate)
    certificate["method"] = "exhausted-bound"
    certificate["evidence"] = {"degree": 0}
    assert not verify_bounded_linear(certificate).valid


@pytest.mark.parametrize("D", [1, 2, 3])
def test_independent_coefficient_assembly_agrees(D):
    from css_decoupling.algebra.bounded_linear import _polynomial_operator
    from css_decoupling.certificates.bounded_linear import _lifted_coefficients
    ring = Ring(5, D)
    x = ring.variable(0)
    A = Matrix.from_rows(ring, [[x - 1, 2 * x**-1], [0, x + 2]])
    B = Matrix.from_rows(ring, [[x**-2], [x - 1]])
    caps = {"max_degree": 2, "max_unknowns": 10000, "max_equations": 10000,
            "max_matrix_entries": 10_000_000}
    for degree in (0, 1, 2):
        assert _polynomial_operator(A, B, degree, caps) == _lifted_coefficients(A, B, degree, caps)


@pytest.mark.parametrize("version", [True, 1.0])
def test_linear_certificate_version_is_not_boolean_or_float(version):
    backend = BoundedLinearBackend(Ring(3, 2))
    backend.kernel(zeros(backend.ring, 0, 1))
    certificate = deepcopy(backend.last_certificate)
    certificate["version"] = version
    assert not verify_bounded_linear(certificate).valid
