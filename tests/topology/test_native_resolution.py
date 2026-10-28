"""Native Schreyer resolutions checked by independent graph elimination."""
from math import comb
from random import Random
import shutil
import subprocess

import pytest

from css_decoupling.algebra.laurent import Matrix, Ring, eye, zeros
from css_decoupling.algebra.modules import kernel, solve, using_backend
from css_decoupling.algebra.schreyer import polynomial_image_resolution
from css_decoupling.errors import ResourceLimit
from css_decoupling.models.examples import glued_css, toric_css, shear_css
from css_decoupling.topology.cohomology import profile
from css_decoupling.topology.resolutions import css_complex, image_resolution


def check_resolution(A, maps):
    """Different implementation: graph-elimination kernel and module lifts."""
    assert A * solve(A, maps[0]) == maps[0]
    assert maps[0] * solve(maps[0], A) == A
    for previous, following in zip(maps, maps[1:]):
        assert not previous * following
        K = kernel(previous)
        assert following * solve(following, K) == K
    assert not kernel(maps[-1])


def check_polynomial_resolution(A, maps):
    """Check before localization, using SymPy's independent distributed GB."""
    from sympy.polys.distributedmodules import sdm_groebner, sdm_nf_buchberger
    from css_decoupling.algebra.modules import GeneralBackend
    backend = GeneralBackend(A.ring)

    def contains(generators, targets):
        vectors = [backend.vector_sdm(c, localized=False) for c in generators.columns()]
        basis = sdm_groebner(
            [c for c in vectors if c],
            sdm_nf_buchberger, backend.order, backend.field)
        for c in targets.columns():
            assert not sdm_nf_buchberger(backend.vector_sdm(c, localized=False),
                                        basis, backend.order, backend.field)

    contains(A, maps[0])
    contains(maps[0], A)
    for previous, following in zip(maps, maps[1:]):
        assert not previous * following
        contains(following, backend._kernel_reduced(previous))
    assert not backend._kernel_reduced(maps[-1])


@pytest.mark.parametrize("dimension,prime", [(2, 2), (3, 3), (4, 5), (5, 3)])
def test_native_koszul_resolves_augmentation_with_exact_binomial_ranks(dimension, prime):
    ring = Ring(prime, dimension)
    A = Matrix.from_rows(ring, [[x - 1 for x in ring.variables]])
    with using_backend("sympy"):
        maps = image_resolution(A, method="native")
        assert [m.ncols for m in maps] == [comb(dimension, j) for j in range(1, dimension + 1)]
        check_resolution(A, maps)


def test_native_needs_no_subprocess_and_accepts_large_prime(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Native resolution invoked an external process")
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(shutil, "which", forbidden)
    ring = Ring(2147483659, 3)
    x, y, z = ring.variables
    A = Matrix.from_rows(ring, [[x - 1, y - 1, z - 1]])
    with using_backend("sympy"):
        maps = image_resolution(A)
        check_resolution(A, maps)
    assert [m.ncols for m in maps] == [3, 3, 1]


def test_laurent_redundancy_and_zero_ambient_rows_preserve_image():
    ring = Ring(5, 3)
    x, y, z = ring.variables
    A = Matrix.from_rows(ring, [
        [x**-3 * (x - 1), y - 1, z**-2 * (z - 1), (x - 1) * (y + z), 0],
        [0, 0, 0, 0, 0],
    ])
    maps = image_resolution(A)
    assert maps[0].nrows == 2
    assert len(maps) <= ring.D
    check_resolution(A, maps)


@pytest.mark.parametrize("seed", range(12))
def test_random_small_polynomial_modules_and_redundant_columns(seed):
    rng = Random(92401 + seed)
    ring = Ring([2, 3, 5][seed % 3], 2)
    x, y = ring.variables
    generators = Matrix(ring, 2, 3)
    monomials = [ring.one(), x, y, x * y, x * x, y * y]
    for i in range(2):
        for j in range(3):
            generators[i, j] = sum(rng.randrange(ring.p) * rng.choice(monomials) for _ in range(2))
    # Prescribed extra generators test a nonminimal presentation, not a
    # known answer: its actual syzygies are checked by graph elimination.
    A = generators.hstack(generators[:, 0:1] + x * generators[:, 1:2])
    maps = image_resolution(A, timeout=15)
    assert len(maps) <= ring.D
    check_resolution(A, maps)
    check_polynomial_resolution(A, maps)


def test_nonmonomial_inhomogeneous_schreyer_example():
    ring = Ring(5, 4)
    w, x, y, z = ring.variables
    A = Matrix.from_rows(ring, [[w*w-x*z, w*x-y*z, x*x-w*y, x*y-z*z, y*y-w*z]])
    maps = image_resolution(A)
    assert len(maps) <= 4
    check_resolution(A, maps)
    check_polynomial_resolution(A, maps)


@pytest.mark.parametrize("seed", range(4))
def test_nonhomogeneous_three_variable_modules_with_polynomial_verification(seed):
    rng = Random(72019 + seed)
    ring = Ring(5, 3)
    x, y, z = ring.variables
    monomials = [ring.one(), x, y, z, x*y, y*z]
    A = Matrix(ring, 2, 3)
    for i in range(2):
        for j in range(3):
            A[i, j] = sum(rng.randrange(5) * rng.choice(monomials) for _ in range(3))
    maps = image_resolution(A, timeout=15)
    assert len(maps) <= ring.D
    check_polynomial_resolution(A, maps)


@pytest.mark.parametrize("dimension,prime,lower,upper", [(3, 3, 1, 2), (4, 3, 1, 3), (5, 5, 2, 3)])
def test_native_completes_actual_gluing_css_and_retains_physical_checks(dimension, prime, lower, upper):
    ring = Ring(prime, dimension)
    HX, HZ = glued_css(ring, lower, upper)
    with using_backend("sympy"):
        C = css_complex(HX, HZ, complete_negative=False)
        assert C.d(0) == HX
        assert C.d(-1) == HZ.dagger()
        assert C.validate()
        assert {j: layer.dimension for j, layer in profile(C).items()} == {lower: 1, upper: 1}


def test_native_full_completion_handles_redundant_physical_presentation():
    ring = Ring(3, 3)
    HX, HZ = toric_css(ring, {1: 1, 2: 1})
    HX, HZ = shear_css(HX, HZ, 0, HX.ncols - 1, ring.variable(2)**-2)
    HX = HX.vstack(HX[0:1, :])
    HZ = HZ.vstack(HZ[0:1, :])
    C = css_complex(HX, HZ)
    assert C.d(0) == HX and C.d(-1) == HZ.dagger()
    assert C.validate()


def test_native_resource_failures_are_not_zero_kernels():
    ring = Ring(3, 3)
    A = Matrix.from_rows(ring, [[x - 1 for x in ring.variables]])
    for kwargs in ({"max_basis": 1}, {"max_terms": 1}, {"max_pairs": 1}, {"timeout": 1e-12}):
        with pytest.raises(ResourceLimit):
            image_resolution(A, **kwargs)
    with pytest.raises(ValueError, match="Laurent exponent"):
        polynomial_image_resolution(Matrix.from_rows(ring, [[ring.variable(0)**-1]]))
    with pytest.raises(ValueError, match="method"):
        image_resolution(eye(ring, 1), method="fraction-field")


def test_nonmonomial_redundancy_resolution_has_complete_polynomial_kernels():
    ring = Ring(3, 3)
    x, y, z = ring.variables
    A = Matrix.from_rows(ring, [[x*y - 1, y*z - 1, x*z - 1, (x*y - 1) * z]])
    maps = image_resolution(A, method="native", timeout=20)
    check_resolution(A, maps)
    check_polynomial_resolution(A, maps)
