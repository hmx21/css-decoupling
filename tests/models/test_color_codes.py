"""Geometric color-code checks, independent stars, and small finite tori."""
from hashlib import sha256
from itertools import product
import json
from math import comb, factorial

import pytest

from css_decoupling import coarse_grain, verify_coarse_graining
from css_decoupling.errors import ResourceLimit
from css_decoupling.models.color_codes import (
    bcc_color_css, fcc_color_css, freudenthal_color_css, freudenthal_color_geometry,
)
from css_decoupling.topology.periodic import periodic_code


def _geometric_stars(geometry, faces):
    """Find stars by inclusion in neighboring simplices, not face extraction.

    A canonical face contains zero, so a containing simplex's cube origin
    has coordinates in {-1,0}. Enumerate those neighboring cubes and test
    ordinary vertex-set containment. This also fixes the Laurent sign.
    """
    neighbors = []
    for offset in product((-1, 0), repeat=geometry.dimension):
        for qubit, simplex in enumerate(geometry.simplices):
            vertices = frozenset(tuple(a + b for a, b in zip(v, offset)) for v in simplex)
            neighbors.append((qubit, offset, vertices))
    return [[(q, cell) for q, cell, vertices in neighbors if set(face) <= vertices]
            for face in faces]


def _gf2_rank(rows):
    pivots = {}
    for word in rows:
        while word:
            pivot = word.bit_length() - 1
            if pivot in pivots:
                word ^= pivots[pivot]
            else:
                pivots[pivot] = word
                break
    return len(pivots)


def _finite_star_rows(stars, qudits, cells):
    # Quotient Z^D / {sum(v)=0 mod(D+1)}; cell representatives are c*e_D.
    result = []
    for c in range(cells):
        for support in stars:
            word = 0
            for q, offset in support:
                word ^= 1 << (((c + sum(offset)) % cells) * qudits + q)
            result.append(word)
    return result


@pytest.mark.parametrize("D,j,xrows,zrows,xterms,zterms", [
    (3, 1, 1, 7, 24, 36), (4, 1, 1, 50, 120, 240),
    (4, 2, 15, 15, 240, 240), (4, 3, 50, 1, 240, 120),
])
def test_actual_colored_triangulation_and_infinite_laurent_commutation(D, j, xrows, zrows, xterms, zterms):
    geometry = freudenthal_color_geometry(D, j)
    HX, HZ = freudenthal_color_css(D, j)
    assert len(geometry.simplices) == factorial(D)
    for simplex in geometry.simplices:
        assert {sum(vertex) % (D + 1) for vertex in simplex} == set(range(D + 1))
    assert HX.shape == (xrows, factorial(D)) and HZ.shape == (zrows, factorial(D))
    assert sum(len(f.terms) for f in HX.entries.values()) == xterms
    assert sum(len(f.terms) for f in HZ.entries.values()) == zterms
    assert not HX * HZ.dagger()
    for checks, faces in ((HX, geometry.x_faces), (HZ, geometry.z_faces)):
        stars = _geometric_stars(geometry, faces)
        for row, support in enumerate(stars):
            actual = {(q, tuple(-a for a in exponent))
                      for (i, q), f in checks.entries.items() if i == row for exponent in f.terms}
            assert actual == set(support)


@pytest.mark.parametrize("D,j", [(3, 1), (4, 1), (4, 2)])
def test_color_preserving_torus_from_independent_geometric_stars(D, j):
    geometry = freudenthal_color_geometry(D, j)
    HX, HZ = freudenthal_color_css(D, j)
    xrows = _finite_star_rows(_geometric_stars(geometry, geometry.x_faces), HX.ncols, D + 1)
    zrows = _finite_star_rows(_geometric_stars(geometry, geometry.z_faces), HZ.ncols, D + 1)
    assert all((x & z).bit_count() % 2 == 0 for x in xrows for z in zrows)
    k = HX.ncols * (D + 1) - _gf2_rank(xrows) - _gf2_rank(zrows)
    # Unfolding gives binom(D,j) toric codes; each has b_j(T^D)=binom(D,j).
    assert k == comb(D, j) ** 2
    finite = periodic_code(HX, HZ, geometry.color_preserving_basis)
    assert finite.logical_qudits == k


def _row_digest(A):
    rows = [tuple((j, tuple(sorted(f.terms.items())))
                  for (row, j), f in sorted(A.entries.items()) if row == i)
            for i in range(A.nrows)]
    return sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@pytest.mark.parametrize("factory,expected", [
    # BCC: source JSON HX unchanged; new Z rows correspond to source rows
    # [2,1,5,0,4,3,6]. Source SHA256:
    # 19144596df8e6b3a2839a83d5dd68f88894cbe6c8c444da7b9cb614a7d2858d4.
    (bcc_color_css, ("2b25357ae8d845f48b50426d0248132048c95b58c635c140e97db8e9a56f7050",
                     "941abc7819fcd2fda38be3d4dc6cc7df54324e4878de855b69bd8f0246f34499")),
    # FCC is entry-for-entry identical to the independently saved source;
    # digest pins the Laurent signs, species order, and exact coefficients.
    (fcc_color_css, ("24fe6a7cde00379aa012000a0202832d09996ca2a02d3e9439388c4875a8f1fa",
                     "58af74fb3a49eebb72c8415cdb37a1415b4ea3232cf6564f9cda838c36d3a28a")),
])
def test_installed_models_pin_source_checks_and_dual_only_swaps_paulis(factory, expected):
    HX, HZ = factory()
    assert (_row_digest(HX), _row_digest(HZ)) == expected
    assert factory(dual=True) == (HZ, HX)
    assert not HX * HZ.dagger()
    result = coarse_grain(HX, HZ, backend="sympy", resolution_method="native")
    assert result.decouplable and result.multiplicities == {1: 3}
    assert result.lattice.index == 4
    assert verify_coarse_graining(result).valid


def test_four_dimensional_symmetric_color_code_is_self_dual():
    HX, HZ = freudenthal_color_css(4, 2)
    assert HX == HZ
    assert HX is not HZ
    A, B = freudenthal_color_css(4, 1)
    assert freudenthal_color_css(4, 3) == (B, A)


@pytest.mark.parametrize("kwargs", [
    {"dimension": 2}, {"dimension": True}, {"dimension": 4.0},
    {"degree": 0}, {"degree": 4}, {"degree": True},
    {"max_qudits": 0}, {"max_incidence_terms": False},
])
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        freudenthal_color_css(**kwargs)


def test_limits_are_checked_before_permutation_or_face_allocation(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("allocation reached before resource check")
    monkeypatch.setattr("css_decoupling.models.color_codes.permutations", forbidden)
    with pytest.raises(ResourceLimit, match="qudits"):
        freudenthal_color_css(100000, 1)
    with pytest.raises(ResourceLimit, match="terms"):
        freudenthal_color_css(4, 1, max_incidence_terms=359)


@pytest.mark.parametrize("factory", [bcc_color_css, fcc_color_css])
def test_dual_flag_is_strict(factory):
    with pytest.raises(ValueError, match="dual"):
        factory(dual=1)
