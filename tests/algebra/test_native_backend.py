"""Analytic module controls retained when the external backend was removed."""
import pytest

from css_decoupling.algebra.laurent import Ring, Matrix, eye, zeros
from css_decoupling.algebra.modules import (
    GeneralBackend, NoLaurentSolution, get_backend, kernel, solve, using_backend,
)
from css_decoupling.algebra.resolutions import image_resolution


@pytest.mark.parametrize('p,D', [(3, 4), (5, 5)])
def test_nonbinary_laurent_solution_and_known_complete_syzygies(p, D):
    ring = Ring(p, D)
    backend = GeneralBackend(ring)
    x, y = ring.variables[:2]
    a, b = x**-3*(x-1), y**-2*(y-1)
    A = Matrix.from_rows(ring, [[a, b, a+b, 0]])
    rhs = A * Matrix.from_rows(ring, [[x+y], [2*x**-1], [y**-1], [1]])
    assert A * backend.solve(A, rhs) == rhs
    assert A * backend.solve(A, A) == A
    # a,b are unit multiples of a regular sequence. These three columns
    # generate the complete kernel, including redundancy and the zero input.
    expected = Matrix.from_rows(ring, [[b, -1, 0], [-a, -1, 0],
                                      [0, 1, 0], [0, 0, 1]])
    computed = backend.kernel(A)
    assert not A*computed
    assert computed*backend.solve(computed, expected) == expected
    assert expected*backend.solve(expected, computed) == computed
    with pytest.raises(NoLaurentSolution):
        backend.solve(Matrix.from_rows(ring, [[x-1]]), eye(ring, 1))


def test_localized_normal_form_has_known_point_values_and_complete_image_lift():
    ring = Ring(5, 4)
    x, y, z, w = ring.variables
    A = Matrix.from_rows(ring, [[x-2, y-3, z-4, w-2, 0],
                               [0, 0, 0, 0, x-y]])
    B = Matrix.from_rows(ring, [[x**-1+y**2, 2*z**-1], [w**2*x+y**-1, 0]])
    backend = GeneralBackend(ring)
    remainder = backend.normal_form(A, B)
    assert remainder[0, 0] == 2  # 2^-1 + 3^2 modulo 5.
    assert remainder[0, 1] == 3  # 2 * 4^-1 modulo 5.
    assert A*backend.solve(A, B-remainder) == B-remainder
    assert backend.normal_form(A, remainder) == remainder
    assert len(backend.reduce_quotient(A, B)) == B.ncols
    data = backend.quotient_groebner(A)
    for polynomial, lead in zip(data['basis'], data['leading_terms']):
        assert polynomial[0][0] == lead
        assert lead == max((m for m, _ in polynomial), key=backend.order)
    assert backend.kernel(eye(ring, 2)).shape == (2, 0)
    assert backend.kernel(zeros(ring, 2, 3)) == eye(ring, 3)
    assert backend.normal_form(zeros(ring, 2, 0), B) == B


def test_native_context_restores_custom_instance_and_checks_its_ring():
    ring = Ring(3, 4)
    original = get_backend(ring)

    class Custom(GeneralBackend):
        pass

    instance = Custom(ring)
    with using_backend(instance):
        assert get_backend(ring) is instance
        with using_backend('sympy'):
            assert type(get_backend(ring)) is GeneralBackend
        assert get_backend(ring) is instance
        with pytest.raises(ValueError, match='different'):
            get_backend(Ring(5, 4))
    assert get_backend(ring) is original
    with pytest.raises(ValueError, match='Unknown'):
        with using_backend('unavailable-name'):
            pass


def test_native_resolution_ends_at_injective_map_with_ambient_zero_row():
    ring = Ring(3, 4)
    x, y = ring.variables[:2]
    A = Matrix.from_rows(ring, [[x-1, y-1, 0], [0, 0, 1], [0, 0, 0]])
    maps = image_resolution(A, timeout=20)
    assert len(maps) == 2
    assert maps[0].nrows == 3
    assert maps[1].nrows == maps[0].ncols
    assert not maps[0]*maps[1]
    assert not kernel(maps[-1])
    assert A*solve(A, maps[0]) == maps[0]
    assert maps[0]*solve(maps[0], A) == A
    assert maps[1]*solve(maps[1], kernel(maps[0])) == kernel(maps[0])


def test_polynomial_nonmembership_does_not_block_exact_laurent_lift():
    ring = Ring(3, 4)
    x, y = ring.variables[:2]
    backend = GeneralBackend(ring)
    A, B = Matrix.from_rows(ring, [[x*(y-1)]]), Matrix.from_rows(ring, [[y-1]])
    assert backend.solve(A, B)[0, 0] == x**-1
