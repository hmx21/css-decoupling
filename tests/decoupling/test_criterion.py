"""Specified-cell decisions exercise ordinary inputs, not example dispatch."""
import pytest

from css_decoupling import Matrix, Ring
from css_decoupling.algebra.laurent import eye
from css_decoupling.certificates.criterion import verify_lattice_criterion
from css_decoupling.decoupling.criterion import check_lattice
from css_decoupling.errors import ResourceLimit
from css_decoupling.lattice.retained import Lattice
from css_decoupling.models.examples import css_checks, glued_css, glued_toric_chain, glued_toric_complex, toric_css
from css_decoupling.topology.complexes import direct_sum_complexes, toric_window


def identity(D):
    return [[int(i == j) for j in range(D)] for i in range(D)]


def shifted_css(p=3):
    ring = Ring(p, 2)
    x, y = ring.variables
    HX = Matrix.from_rows(ring, [[x + 1, y + 1]])
    HZ = Matrix.from_rows(ring, [[-(y + 1)], [x + 1]]).dagger()
    return HX, HZ


@pytest.mark.parametrize("p,D,degree", [(2, 2, 1), (3, 3, 2), (5, 4, 2), (3, 5, 2)])
def test_full_general_input_positive_profile(p, D, degree):
    result = check_lattice(*toric_css(Ring(p, D), {degree: 1}), identity(D))
    assert result.decouplable and result.reason == "split-profile"
    assert result.multiplicities == {degree: 1}
    assert result.phi.validate() and result.lattice.index == 1
    report = verify_lattice_criterion(result)
    assert report.valid, report.errors


@pytest.mark.parametrize("sector", ["X", "Z", "auto"])
def test_sector_backend_and_arbitrary_tilted_negative_basis(sector):
    basis = [[-2, 2], [0, 2]]
    result = check_lattice(*shifted_css(), basis, sector=sector, backend="sympy")
    assert result.decouplable and result.lattice.index == 4
    assert result.sector == ("X" if sector == "auto" else sector)
    assert verify_lattice_criterion(result, backend="sympy").valid


@pytest.mark.parametrize("p", [2, 3, 5])
def test_nonzero_gluing_and_actual_killing_lattice(p):
    checks = glued_css(Ring(p, 3))
    no = check_lattice(*checks, identity(3))
    assert not no.decouplable and no.reason == "nonzero-gluing"
    assert no.obstruction == {"degree": 2} and len(no.gluing) == 1
    assert no.phi is None and not no.gluing[-1].is_zero
    assert verify_lattice_criterion(no).valid
    basis = [[p, -1, 0], [0, 1, 0], [0, 0, 1]]
    yes = check_lattice(*checks, basis)
    assert yes.decouplable and yes.lattice.index == p
    assert yes.gluing[0].is_zero and verify_lattice_criterion(yes).valid


def test_three_layers_stop_at_first_unsplit_attachment():
    C = glued_toric_chain(Ring(2, 4), [1, 2, 3])
    no = check_lattice(*css_checks(C), identity(4))
    assert no.multiplicities == {1: 1, 2: 1, 3: 1}
    assert no.obstruction == {"degree": 2}
    assert [s.upper for s in no.gluing] == [2]
    assert verify_lattice_criterion(no).valid
    basis = identity(4)
    basis[0][0] = 2
    yes = check_lattice(*css_checks(C), basis)
    assert yes.decouplable and [s.upper for s in yes.gluing] == [2, 3]
    assert verify_lattice_criterion(yes).valid


def test_first_gluing_obstruction_above_an_already_split_two_layer_tower():
    ring = Ring(3, 4)
    C = direct_sum_complexes(toric_window(ring, {1: 1}),
                            glued_toric_complex(ring, 2, 3, axes=(1, 2)))
    no = check_lattice(*css_checks(C), identity(4))
    assert no.obstruction == {"degree": 3}
    assert [s.is_zero for s in no.gluing] == [True, False]
    assert [c.lower for c in no.gluing[-1].classes] == [1, 2]
    assert verify_lattice_criterion(no).valid


def test_huge_translation_obstruction_never_expands_or_finds_common_kernel(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("negative action test must not expand the cell or find Lambda_triv")
    import css_decoupling.lattice.kernel as kernel_module
    monkeypatch.setattr(kernel_module, "translation_kernel", forbidden)
    monkeypatch.setattr(Lattice, "restrict", forbidden)
    basis = [[3**60, 0], [0, 1]]
    result = check_lattice(*shifted_css(), basis, max_cell_index=1)
    assert result.reason == "nontrivial-layer-translation"
    assert result.lattice._representatives is None and result.lattice._lookup is None
    assert result.obstruction["action"] != eye(result.HX.ring, 1)
    assert verify_lattice_criterion(result, max_cell_index=1).valid


@pytest.mark.parametrize("sector", ["X", "Z", "auto"])
def test_infinite_standard_ray_is_an_actual_obstruction(sector):
    ring = Ring(5, 3)
    x, y, _ = ring.variables
    HX = Matrix.from_rows(ring, [[x - 1, y - 1]])
    HZ = Matrix.from_rows(ring, [[-(y - 1)], [x - 1]]).dagger()
    result = check_lattice(HX, HZ, identity(3), sector=sector)
    assert not result.decouplable and result.reason == "infinite-layer"
    assert result.layers == {} and result.obstruction["degree"] == 1
    assert result.lattice._representatives is None
    assert verify_lattice_criterion(result).valid


def test_resource_failure_never_turns_into_no():
    ring = Ring(3, 2)
    x, y = ring.variables
    HX = Matrix.from_rows(ring, [[(x - 1)**2, y - 1]])
    HZ = Matrix.from_rows(ring, [[-(y - 1)], [(x - 1)**2]]).dagger()
    with pytest.raises(ResourceLimit, match="dimension"):
        check_lattice(HX, HZ, identity(2), max_layer_dimension=1)
    no = check_lattice(HX, HZ, identity(2), max_layer_dimension=None)
    assert no.reason == "nontrivial-layer-translation"
    with pytest.raises(ResourceLimit, match="dimension"):
        verify_lattice_criterion(no, max_layer_dimension=1)
    checks = toric_css(Ring(3, 3), {1: 1})
    with pytest.raises(ResourceLimit, match="index"):
        check_lattice(*checks, [[3**30, 0, 0], [0, 1, 0], [0, 0, 1]], max_cell_index=1)
    with pytest.raises(ResourceLimit, match="unknowns"):
        check_lattice(*checks, identity(3), max_unknowns=1)
    yes = check_lattice(*checks, identity(3))
    with pytest.raises(ResourceLimit, match="entries"):
        verify_lattice_criterion(yes, max_unknowns=1)


@pytest.mark.parametrize("basis", [[[1, 0], [0, 0]], [[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[True, 0], [0, 1]]])
def test_invalid_lattices_rejected_before_building_the_model(basis, monkeypatch):
    import css_decoupling.decoupling.criterion as module
    monkeypatch.setattr(module, "css_complex", lambda *a, **k: pytest.fail("invalid lattice reached model construction"))
    with pytest.raises(ValueError):
        check_lattice(*shifted_css(), basis)


def test_bad_css_is_an_input_error_and_not_a_negative_decision():
    ring = Ring(3, 2)
    with pytest.raises(ValueError, match="commute"):
        check_lattice(Matrix.from_rows(ring, [[1]]), Matrix.from_rows(ring, [[1]]), identity(2))


def test_zero_profile_has_explicit_empty_toric_splitting():
    ring = Ring(3, 2)
    HX = Matrix.from_rows(ring, [[1]])
    HZ = Matrix(ring, 0, 1, {})
    result = check_lattice(HX, HZ, identity(2))
    assert result.decouplable and result.layers == {} and result.phi.source.ranks == {}
    assert verify_lattice_criterion(result).valid
