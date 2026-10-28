"""Raw-check integration tests, including genuine gluing and skew lattices."""
from copy import deepcopy

import pytest

from css_decoupling.algebra.laurent import Matrix, Ring, zeros
from css_decoupling.models.examples import (
    css_checks, glued_css, glued_toric_chain, glued_toric_complex, shear_css, toric_css,
)
from css_decoupling.lattice.retained import ResourceLimit
from css_decoupling.algebra.modules import NoLaurentSolution
from css_decoupling.decoupling.pipeline import decouple
from css_decoupling.certificates.decoupling import verify_decoupling
from css_decoupling.topology.cohomology import InfiniteLayer


@pytest.mark.parametrize('prime', [2, 3, 5])
def test_glued_three_dimensional_input_matches_manuscript_entry_by_entry(prime):
    ring = Ring(prime, 3)
    a, b, c = (x - 1 for x in ring.variables)
    HX, HZ = glued_css(ring)
    expected_X = Matrix.from_rows(ring, [
        [a, b, c, -1, 0, 0],
        [0, 0, 0, -b, -c, 0],
        [0, 0, 0, a, 0, -c],
        [0, 0, 0, 0, a, b],
    ])
    expected_Zdagger = Matrix.from_rows(ring, [
        [-b, -c, 0, 0],
        [a, 0, -c, 0],
        [0, a, b, 1],
        [0, 0, 0, c],
        [0, 0, 0, -b],
        [0, 0, 0, a],
    ])
    assert HX == expected_X
    assert HZ.dagger() == expected_Zdagger


@pytest.mark.parametrize('dimension,prime', [(3, 2), (4, 3), (5, 5)])
def test_higher_gluing_all_layer_gaps_has_a_signed_complex(dimension, prime):
    ring = Ring(prime, dimension)
    for lower in range(1, dimension - 1):
        for upper in range(lower + 1, dimension):
            C = glued_toric_complex(ring, lower, upper)
            assert C.validate()
            assert C.d(0) * C.d(-1) == zeros(ring, C.rank(1), C.rank(-1))


@pytest.mark.parametrize('prime', [2, 3])
def test_nonzero_gluing_rejected_on_original_lattice_then_decoupled(prime):
    ring = Ring(prime, 3)
    HX, HZ = glued_css(ring)
    with pytest.raises(NoLaurentSolution):
        decouple(HX, HZ, lattice_basis=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], backend='sympy', resolution_timeout=20)
    result = decouple(HX, HZ, lattice_basis=[[prime, 0, 0], [0, 1, 0], [0, 0, 1]],
                      backend='sympy', resolution_timeout=20)
    assert result.multiplicities == {1: 1, 2: 1}
    assert result.lattice.index == prime
    assert verify_decoupling(result.certificate()).valid


@pytest.mark.integration
def test_automatic_coarse_graining_kills_nonzero_gluing_without_lattice_search():
    result = decouple(*glued_css(Ring(2, 3)), backend='sympy', resolution_timeout=20)
    assert result.lattice.index == 2
    assert result.multiplicities == {1: 1, 2: 1}
    assert [a['split'] for a in result.trace['attempts']] == [False, True]
    assert result.trace['translation_lattice_kind'] == 'exact-common-kernel'
    assert result.profile_analysis.translation_kernel.index == 1


def test_odd_prime_higher_dimensional_multilayer_scrambled_raw_checks():
    ring = Ring(3, 4)
    HX, HZ = toric_css(ring, {1: 1, 2: 1})
    HX, HZ = shear_css(HX, HZ, 0, HX.ncols - 1,
                      ring.variable(2) ** -1 + 2 * ring.variable(3))
    result = decouple(HX, HZ, backend='sympy', resolution_timeout=20)
    assert result.multiplicities == {1: 1, 2: 1}
    assert result.lattice.index == 1
    assert verify_decoupling(result.certificate()).valid
    forged = deepcopy(result.certificate())
    forged['multiplicities'] = [[1, 2]]
    assert not verify_decoupling(forged).valid


@pytest.mark.integration
def test_four_dimensional_ext3_gluing():
    ring = Ring(2, 4)
    result = decouple(*glued_css(ring, 1, 3),
                      lattice_basis=[[2, 0, 0, 0], [0, 1, 0, 0],
                                     [0, 0, 1, 0], [0, 0, 0, 1]],
                      backend='sympy', resolution_timeout=20)
    assert result.multiplicities == {1: 1, 3: 1}
    assert result.lattice.index == 2


def test_three_layers_with_successive_nonzero_gluing():
    ring = Ring(2, 4)
    HX, HZ = css_checks(glued_toric_chain(ring, [1, 2, 3]))
    with pytest.raises(NoLaurentSolution):
        decouple(HX, HZ, lattice_basis=[[1, 0, 0, 0], [0, 1, 0, 0],
                                       [0, 0, 1, 0], [0, 0, 0, 1]],
                 backend='sympy', resolution_timeout=20)
    result = decouple(HX, HZ, lattice_basis=[[2, 0, 0, 0], [0, 1, 0, 0],
                                            [0, 0, 1, 0], [0, 0, 0, 1]],
                      backend='sympy', resolution_timeout=20)
    assert result.multiplicities == {1: 1, 2: 1, 3: 1}
    assert result.lattice.index == 2


@pytest.mark.integration
def test_five_dimensional_prime_five_raw_input():
    ring = Ring(5, 5)
    HX, HZ = toric_css(ring, {2: 1})
    HX, HZ = shear_css(HX, HZ, 2, 6,
                      ring.variable(4) ** -2 + 3 * ring.variable(0))
    result = decouple(HX, HZ, backend='sympy', resolution_timeout=20)
    assert result.multiplicities == {2: 1}
    assert result.lattice.index == 1


def test_nontrivial_translation_and_skew_retained_lattice():
    ring = Ring(3, 2)
    x, y = ring.variables
    HX = Matrix.from_rows(ring, [[x + 1, y + 1]])
    HZ = Matrix.from_rows(ring, [[-(y + 1)], [x + 1]]).dagger()
    with pytest.raises(ValueError, match='translations act nontrivially'):
        decouple(HX, HZ, lattice_basis=[[1, 0], [0, 1]], backend='sympy', resolution_timeout=20)
    # Columns (1,1),(0,2) form an index-two kernel lattice.  Both old
    # translations act as -1 on the one-dimensional charge module.
    result = decouple(HX, HZ, lattice_basis=[[1, 0], [1, 2]], backend='sympy', resolution_timeout=20)
    assert result.lattice.index == 2
    assert result.multiplicities == {1: 1}
    assert verify_decoupling(result.certificate()).valid


def test_infinite_layer_and_resource_exhaustion_are_distinct():
    ring = Ring(3, 3)
    x, y, _ = ring.variables
    # Exact two-variable toric complex extended by a free third translation.
    HX = Matrix.from_rows(ring, [[x - 1, y - 1]])
    HZ = Matrix.from_rows(ring, [[-(y - 1)], [x - 1]]).dagger()
    with pytest.raises(InfiniteLayer):
        decouple(HX, HZ, backend='sympy', resolution_timeout=20)
    with pytest.raises(ResourceLimit):
        decouple(*glued_css(Ring(2, 3)), max_cell_index=1, backend='sympy', resolution_timeout=20)


def test_commuting_but_nontopological_input_is_not_accepted():
    ring = Ring(3, 2)
    with pytest.raises(NoLaurentSolution):
        decouple(zeros(ring, 0, 1), zeros(ring, 0, 1), backend='sympy', resolution_timeout=20)
