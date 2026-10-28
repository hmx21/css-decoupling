"""Laurent reduction, coordinate faces, exact inverses and polynomial unity."""
from copy import deepcopy
from dataclasses import replace

import pytest

from css_decoupling import Matrix, Ring, eye
from css_decoupling.algebra.normalization import normalize_unimodular_row, monomial_substitution
from css_decoupling.certificates.normalization import verify_normalization
from css_decoupling.errors import NoLaurentSolution, ResourceLimit


@pytest.mark.parametrize('p,D,axis', [(2, 2, 0), (3, 3, 2), (5, 5, 3), (2147483659, 2, 1)])
def test_normalization_from_known_multivariate_unimodular_rows(p, D, axis, monkeypatch):
    import subprocess
    def forbidden(*args, **kwargs):
        raise AssertionError('No external process is used by support normalization')
    monkeypatch.setattr(subprocess, 'Popen', forbidden)
    r = Ring(p, D)
    x, y = r.variables[:2]
    U, V = eye(r, 4), eye(r, 4)
    for i, j, f in [(0, 1, x**-2+y), (1, 2, x+y**-1), (2, 0, x**-1), (0, 3, r.variable(axis)+1)]:
        T = eye(r, 4) + Matrix(r, 4, 4, {(i, j): f})
        Tinv = eye(r, 4) - Matrix(r, 4, 4, {(i, j): f})
        U, V = U*T, Tinv*V
    f, a = U[:1, :], V[:, :1]
    result = normalize_unimodular_row(f, right_inverse=a, axis=axis)
    assert verify_normalization(result)
    assert result.row * result.saturation_bezout == Matrix.from_rows(
        r, [[r.monomial((result.saturation_power,)*D)]])
    assert monomial_substitution(monomial_substitution(U, result.exponents), result.inverse_exponents) == U
    # Completion pullback depends only on actual inverse identities, not on
    # the choice of algorithm used to obtain the normalized completion.
    local = result.change_inverse * monomial_substitution(V, result.exponents)
    local_inverse = monomial_substitution(U, result.exponents) * result.change
    actual, inverse = result.lift_completion(local, local_inverse)
    assert actual == V and inverse == U


@pytest.mark.parametrize('p', [2, 3, 5])
def test_strict_face_positivity_avoids_spurious_polynomial_zeros(p):
    r = Ring(p, 3)
    x, y, z = r.variables
    # This is Laurent unimodular, but has polynomial common zeros at x=0,
    # yz=-1. Merely nonnegative exponents would not establish polynomial unity.
    f = Matrix.from_rows(r, [[1+y*z, x*z]])
    a = Matrix.from_rows(r, [[0], [(x*z)**-1]])
    result = normalize_unimodular_row(f, right_inverse=a, axis=2)
    assert verify_normalization(result)
    poly_inverse = result.polynomial_bezout()
    assert all(e_i >= 0 for f_i in poly_inverse.entries.values() for e in f_i.terms for e_i in e)
    assert result.row * poly_inverse == eye(r, 1)
    g, h = result.row[0, 0], result.row[0, 1]
    a, b = poly_inverse[0, 0], poly_inverse[1, 0]
    U = Matrix.from_rows(r, [[a, -h], [b, g]])
    Uinv = Matrix.from_rows(r, [[g, h], [-b, a]])
    V, Vinv = result.lift_completion(U, Uinv)
    assert f*V == Matrix.from_rows(r, [[1, 0]])
    assert V*Vinv == eye(r, 2)


@pytest.mark.parametrize('entries', [lambda x,y: [0, x**-3*y**7, 0],
                                      lambda x,y: [x**-12*y**19],
                                      lambda x,y: [1, 0]])
def test_zero_entries_units_and_single_columns(entries):
    r = Ring(5, 2)
    f = Matrix.from_rows(r, [entries(*r.variables)])
    result = normalize_unimodular_row(f)
    assert verify_normalization(result)
    assert result.row*result.polynomial_bezout() == eye(r, 1)


def test_non_unimodular_and_bad_inputs_are_rejected():
    r = Ring(3, 2)
    x, y = r.variables
    with pytest.raises(NoLaurentSolution):
        normalize_unimodular_row(Matrix.from_rows(r, [[x-1, y-1]]))
    with pytest.raises(ValueError, match='Bezout'):
        normalize_unimodular_row(Matrix.from_rows(r, [[1, 0]]), right_inverse=Matrix.from_rows(r, [[0], [1]]))
    with pytest.raises(ValueError, match='axis'):
        normalize_unimodular_row(Matrix.from_rows(r, [[1, 0]]), axis=True)


@pytest.mark.parametrize('mutation', ['map', 'inverse', 'row', 'bezout', 'power', 'negative', 'face'])
def test_normalization_certificate_detects_tampering(mutation):
    r = Ring(5, 2)
    x, y = r.variables
    w = normalize_unimodular_row(Matrix.from_rows(r, [[1+x, y]]))
    if mutation == 'map':
        w = replace(w, exponents=((1, 2), (0, 2)))
    elif mutation == 'inverse':
        w.change_inverse[0, 0] += 1
    elif mutation == 'row':
        w.row[0, 0] += 1
    elif mutation == 'bezout':
        w.saturation_bezout[0, 0] += 1
    elif mutation == 'power':
        w = replace(w, saturation_power=True)
    elif mutation == 'negative':
        w = replace(w, face_remainder=x**-1)
    else:
        w = replace(w, face_remainder=w.face_remainder+1)
    with pytest.raises(ValueError):
        verify_normalization(w)


def test_polynomial_bezout_expansion_has_explicit_resource_cap():
    r = Ring(5, 2)
    x, y = r.variables
    w = normalize_unimodular_row(Matrix.from_rows(r, [[1+x+x*x, y**-3]]))
    with pytest.raises(ResourceLimit):
        w.polynomial_bezout(max_terms=1)
    assert verify_normalization(w)  # A refused expansion does not refute the compact witness.


def test_monomial_map_accumulates_terms_when_not_injective():
    r = Ring(3, 2)
    x, y = r.variables
    assert monomial_substitution(x+2*y, ((1, 1), (0, 0))) == 0
