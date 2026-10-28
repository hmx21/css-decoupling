"""Exact native image/kernel witnesses, independently compared with GB."""
from random import Random
from unittest.mock import patch

import pytest

from css_decoupling.algebra.laurent import Matrix, Ring, eye, zeros
from css_decoupling.algebra.modules import GeneralBackend, kernel, solve
from css_decoupling.algebra.resolutions import image_resolution
from css_decoupling.algebra.schreyer import (
    polynomial_groebner, polynomial_image_resolution, polynomial_kernel,
    verify_polynomial_kernel,
)
from css_decoupling.errors import ResourceLimit


@pytest.mark.parametrize("prime", [2, 3, 5, 2147483659])
def test_tracked_initial_basis_keeps_zero_columns_and_exact_scaling(prime):
    ring = Ring(prime, 3)
    x, y, z = ring.variables
    coefficient = ring(2) if prime != 2 else ring.one()
    A = Matrix.from_rows(ring, [
        [0, coefficient*(x-y), x*x-y*y, y-z, 0],
        [0, 0, 0, 0, 0],
    ])
    witness = polynomial_groebner(A)
    G, T, S = (witness[key] for key in (
        "groebner_generators", "basis_in_input", "input_in_basis"))
    assert A*T == G and G*S == A
    assert T.nrows == A.ncols and S.ncols == A.ncols
    assert not T[0:1, :] and not T[4:5, :]
    result = polynomial_image_resolution(A, return_image_witness=True)
    assert result["maps"] == polynomial_image_resolution(A)
    assert A*result["basis_in_input"] == result["maps"][0]
    assert result["maps"][0]*result["input_in_basis"] == A


def _independent_polynomial_kernel(A):
    """SymPy graph elimination, deliberately not GeneralBackend.kernel."""
    from sympy.polys.distributedmodules import sdm_groebner, sdm_nf_buchberger
    backend = GeneralBackend(A.ring)
    graph = []
    for column, values in enumerate(A.columns()):
        vector = backend.vector_sdm(values, localized=False, order=backend.elimination_order)
        vector.append(((A.nrows + column,) + (0,) * A.ring.D, backend.field.one))
        graph.append(sorted(vector, key=lambda term: backend.elimination_order(term[0]), reverse=True))
    basis = sdm_groebner(graph, sdm_nf_buchberger, backend.elimination_order, backend.field)
    columns = []
    for vector in basis:
        if any(monomial[0] < A.nrows for monomial, _ in vector):
            continue
        column = zeros(A.ring, A.ncols, 1)
        for (index, *exponent), coefficient in vector:
            column[index-A.nrows, 0] += A.ring.monomial(tuple(exponent), int(coefficient))
        if column:
            columns.append(column)
    return zeros(A.ring, A.ncols, 0).hstack(*columns)


def _polynomial_contains(generators, targets):
    """Ordinary polynomial membership, with no Laurent localization."""
    from sympy.polys.distributedmodules import sdm_groebner, sdm_nf_buchberger
    backend = GeneralBackend(generators.ring)
    raw = [backend.vector_sdm(c, localized=False) for c in generators.columns()]
    basis = sdm_groebner([g for g in raw if g], sdm_nf_buchberger, backend.order, backend.field)
    return all(not sdm_nf_buchberger(backend.vector_sdm(c, localized=False),
                                   basis, backend.order, backend.field)
               for c in targets.columns())


@pytest.mark.parametrize("seed", range(15))
def test_random_complete_kernel_agrees_with_independent_polynomial_graph(seed):
    rng = Random(410911 + seed)
    ring = Ring([2, 3, 5][seed % 3], 2)
    x, y = ring.variables
    monomials = [ring.one(), x, y, x*y, x*x]
    A = Matrix(ring, 2, 3)
    for i in range(A.nrows):
        for j in range(A.ncols):
            A[i, j] = sum((rng.randrange(ring.p)*rng.choice(monomials) for _ in range(2)), ring.zero())
    A = A.hstack(A[:, :1] + y*A[:, 1:2], zeros(ring, 2, 1))
    witness = polynomial_kernel(A, return_witness=True)
    assert verify_polynomial_kernel(A, witness)
    independent = _independent_polynomial_kernel(A)
    assert _polynomial_contains(independent, witness["kernel"])
    assert _polynomial_contains(witness["kernel"], independent)


@pytest.mark.parametrize("rows,cols", [(2, 3), (0, 3), (2, 0), (0, 0)])
def test_zero_and_empty_matrices_have_complete_kernel_witnesses(rows, cols):
    ring = Ring(3, 2)
    A = zeros(ring, rows, cols)
    witness = polynomial_kernel(A, return_witness=True)
    assert witness["kernel"] == eye(ring, cols)
    assert verify_polynomial_kernel(A, witness)
    result = polynomial_image_resolution(A, return_image_witness=True)
    assert result["maps"] == [zeros(ring, rows, 0)]
    assert A*result["basis_in_input"] == result["maps"][0]
    assert result["maps"][0]*result["input_in_basis"] == A


def test_kernel_verification_does_not_rerun_the_constructor_and_rejects_missing_syzygies():
    ring = Ring(3, 3)
    A = Matrix.from_rows(ring, [[x-1 for x in ring.variables]])
    witness = polynomial_kernel(A, return_witness=True)
    with patch('css_decoupling.algebra.schreyer._groebner', side_effect=AssertionError('constructor called')), \
         patch('css_decoupling.algebra.schreyer._syzygies', side_effect=AssertionError('constructor called')):
        assert verify_polynomial_kernel(A, witness)
    forged = dict(witness)
    forged["schreyer_kernel"] = zeros(ring, witness["groebner_generators"].ncols, 0)
    assert not witness["groebner_generators"]*forged["schreyer_kernel"]
    with pytest.raises(ArithmeticError, match="complete kernel"):
        verify_polynomial_kernel(A, forged)


@pytest.mark.parametrize("field", ["basis_in_input", "input_in_basis", "kernel", "selected_columns"])
def test_tampered_transfer_witnesses_are_rejected(field):
    ring = Ring(5, 2)
    x, y = ring.variables
    A = Matrix.from_rows(ring, [[x-1, y-1, (x-1)*(x+y), 0]])
    witness = polynomial_kernel(A, return_witness=True)
    forged = dict(witness)
    if field == "selected_columns":
        forged[field] = ()
    else:
        forged[field] = zeros(ring, *forged[field].shape)
    with pytest.raises((ArithmeticError, ValueError)):
        verify_polynomial_kernel(A, forged)


def test_incomplete_groebner_witness_is_rejected_even_if_image_identities_hold():
    ring = Ring(3, 2)
    x, y = ring.variables
    A = Matrix.from_rows(ring, [[x*x+y, x*y]])
    fake = dict(groebner_generators=A, basis_in_input=eye(ring, 2),
                input_in_basis=eye(ring, 2), schreyer_kernel=zeros(ring, 2, 0),
                kernel=zeros(ring, 2, 0), selected_columns=())
    with pytest.raises(ArithmeticError, match="Buchberger"):
        verify_polynomial_kernel(A, fake)


def test_laurent_initial_image_uses_witnesses_but_retains_full_exactness_checks():
    import css_decoupling.algebra.resolutions as resolutions
    ring = Ring(5, 3)
    x, y, z = ring.variables
    A = Matrix.from_rows(ring, [[x**-2*(x-1), y**-3*(y-1), z**-1*(z-1), 0]])
    with patch.object(resolutions, 'solve', wraps=solve) as solves, \
         patch.object(resolutions, 'kernel', wraps=kernel) as kernels:
        maps = image_resolution(A)
    # One solve per genuine subsequent kernel, none for either initial image.
    assert solves.call_count == len(maps)-1
    assert kernels.call_count == len(maps)
    assert _polynomial_contains(maps[0], Matrix.from_rows(ring, [[x-1, y-1, z-1, 0]]))


def test_resolution_rejects_corrupted_initial_image_witness_before_kernel_work():
    import css_decoupling.algebra.resolutions as resolutions
    import css_decoupling.algebra.schreyer as schreyer
    ring = Ring(3, 2)
    A = Matrix.from_rows(ring, [[x-1 for x in ring.variables]])
    original = schreyer.polynomial_image_resolution
    def corrupt(*args, **kwargs):
        result = original(*args, **kwargs)
        result["basis_in_input"] = zeros(ring, *result["basis_in_input"].shape)
        return result
    with patch.object(schreyer, 'polynomial_image_resolution', side_effect=corrupt), \
         patch.object(resolutions, 'kernel', side_effect=AssertionError('late validation')):
        with pytest.raises(ArithmeticError, match="image witnesses"):
            image_resolution(A)


def test_native_kernel_limits_and_laurent_input_are_not_negative_answers():
    ring = Ring(3, 3)
    A = Matrix.from_rows(ring, [[x-1 for x in ring.variables]])
    for limits in ({"max_basis": 1}, {"max_terms": 1}, {"max_pairs": 1}, {"timeout": 1e-12}):
        with pytest.raises(ResourceLimit):
            polynomial_kernel(A, **limits)
    with pytest.raises(ValueError, match="Laurent exponent"):
        polynomial_kernel(Matrix.from_rows(ring, [[ring.variable(0)**-1]]))
