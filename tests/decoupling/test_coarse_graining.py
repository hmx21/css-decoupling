"""Algorithm 1 produces an actual uniform restriction without splitting."""
import pytest

from css_decoupling import Matrix, Ring
from css_decoupling.decoupling.coarse_graining import coarse_grain
from css_decoupling.certificates.coarse_graining import verify_coarse_graining
from css_decoupling.errors import ResourceLimit
from css_decoupling.lattice.retained import Lattice
from css_decoupling.models.examples import glued_css, toric_css


def shifted_css(p=3):
    ring = Ring(p, 2)
    x, y = ring.variables
    f, g = x ** -1 + 1, y + 1
    return Matrix.from_rows(ring, [[f, g]]), Matrix.from_rows(ring, [[-g], [f]]).dagger()


@pytest.mark.parametrize("p,D,degree", [(2, 2, 1), (3, 3, 2), (5, 4, 1), (3, 5, 2)])
def test_complete_bounded_model_single_layer(p, D, degree):
    result = coarse_grain(*toric_css(Ring(p, D), {degree: 1}))
    assert result.decouplable and result.reason == "uniform-sufficient-lattice"
    assert result.multiplicities == {degree: 1}
    assert result.lattice.index == result.translation_kernel.index == 1
    assert result.complex.rank(degree - D) == 1
    assert result.complex.d(0) == result.HX
    assert result.complex.d(-1) == result.HZ.dagger()
    assert result.restricted.ranks == result.complex.ranks
    assert verify_coarse_graining(result).valid


def test_zero_profile_has_identity_lattice_and_actual_empty_profile_complex():
    ring = Ring(5, 4)
    HX, HZ = Matrix.from_rows(ring, [[1]]), Matrix(ring, 0, 1, {})
    result = coarse_grain(HX, HZ)
    assert result.multiplicities == {} and result.lattice.index == 1
    assert result.translation_kernel.kernel_certificate["p"] is None
    assert result.restricted.d(0) == HX
    assert verify_coarse_graining(result).valid


@pytest.mark.parametrize("p", [2, 3, 5])
def test_two_layers_use_uniform_formula_without_gluing_or_splitting(p, monkeypatch):
    import css_decoupling.decoupling.analysis as analysis
    import css_decoupling.decoupling.gluing as gluing
    import css_decoupling.decoupling.splitting as splitting
    def forbidden(*args, **kwargs):
        pytest.fail("Algorithm 1 must not solve gluing or splitting")
    monkeypatch.setattr(analysis, "analyze_profile", forbidden)
    monkeypatch.setattr(gluing, "gluing_classes", forbidden)
    monkeypatch.setattr(splitting, "splitting_map", forbidden)
    result = coarse_grain(*glued_css(Ring(p, 3)))
    assert result.multiplicities == {1: 1, 2: 1}
    assert result.lattice.basis == ((p, 0, 0), (0, p, 0), (0, 0, p))
    assert result.lattice.index == p ** 3
    assert result.restricted.rank(0) == result.complex.rank(0) * p ** 3
    assert min(result.restricted.degrees) < -1
    assert verify_coarse_graining(result).valid


@pytest.mark.parametrize("sector", ["X", "Z", "auto"])
def test_nontrivial_exact_kernel_negative_laurent_exponents_and_sector(sector):
    HX, HZ = shifted_css()
    result = coarse_grain(HX, HZ, sector=sector, backend="sympy")
    assert result.sector == ("X" if sector == "auto" else sector)
    # Both variables act by -1: the common kernel is a mixed index-two
    # lattice, not the rectangular index-four sufficient bound.
    assert result.translation_kernel.index == result.lattice.index == 2
    assert result.lattice.basis == ((2, 1), (0, 1))
    active_X, active_Z = (HX, HZ) if result.sector == "X" else (HZ, HX)
    assert result.restricted.d(0) == result.lattice.restrict(active_X)
    assert result.restricted.d(-1) == result.lattice.restrict(active_Z.dagger())
    assert verify_coarse_graining(result, backend="sympy").valid


@pytest.mark.parametrize("sector", ["X", "Z", "auto"])
def test_infinite_layer_certifies_no_lattice_without_any_restriction(sector, monkeypatch):
    ring = Ring(5, 3)
    x, y, _ = ring.variables
    HX = Matrix.from_rows(ring, [[x - 1, y - 1]])
    HZ = Matrix.from_rows(ring, [[-(y - 1)], [x - 1]]).dagger()
    def forbidden(*args, **kwargs):
        pytest.fail("Infinite-layer obstruction must not expand a cell")
    monkeypatch.setattr(Lattice, "restrict", forbidden)
    result = coarse_grain(HX, HZ, max_cell_index=1, sector=sector)
    assert not result.decouplable and result.reason == "infinite-layer"
    assert result.translation_kernel is result.lattice is result.restricted is None
    assert result.obstruction["degree"] == 1
    assert verify_coarse_graining(result, max_cell_index=1).valid


def test_three_layer_uniform_scale_is_checked_before_coset_expansion(monkeypatch):
    checks = toric_css(Ring(2, 4), {1: 1, 2: 1, 3: 1})
    def forbidden(*args, **kwargs):
        pytest.fail("Cell cap must be checked before a restriction is allocated")
    monkeypatch.setattr(Lattice, "restrict", forbidden)
    with pytest.raises(ResourceLimit, match="index 256") as caught:
        coarse_grain(*checks, max_cell_index=255)
    partial = caught.value.coarse_graining_partial
    assert partial["status"] == "resource-limited" and partial["restricted"] is None
    assert partial["multiplicities"] == {1: 1, 2: 1, 3: 1}
    assert partial["sufficient_basis"] == [[4, 0, 0, 0], [0, 4, 0, 0],
                                            [0, 0, 4, 0], [0, 0, 0, 4]]
    assert partial["translation_kernel"]["index"] == 1
    assert "decouplable" not in partial


def test_layer_cap_and_bad_input_are_not_negative_decisions():
    ring = Ring(3, 2)
    x, y = ring.variables
    f, g = (x - 1) ** 2, y - 1
    checks = Matrix.from_rows(ring, [[f, g]]), Matrix.from_rows(ring, [[-g], [f]]).dagger()
    with pytest.raises(ResourceLimit, match="dimension"):
        coarse_grain(*checks, max_layer_dimension=1)
    result = coarse_grain(*checks, max_layer_dimension=None)
    assert result.multiplicities == {1: 2}
    assert result.lattice.index == 3
    with pytest.raises(ResourceLimit, match="dimension"):
        verify_coarse_graining(result, max_layer_dimension=1)
    with pytest.raises(ValueError, match="commute"):
        coarse_grain(Matrix.from_rows(ring, [[1]]), Matrix.from_rows(ring, [[1]]))


@pytest.mark.parametrize("options", [
    {"sector": "other"}, {"verify": 1}, {"max_cell_index": 0},
    {"max_layer_dimension": True}, {"max_dlog_steps": None},
    {"max_linear_entries": 0}, {"factor_limit": False},
])
def test_invalid_options_fail_before_free_model_construction(options, monkeypatch):
    import css_decoupling.decoupling.coarse_graining as module
    monkeypatch.setattr(module, "css_complex",
                        lambda *a, **k: pytest.fail("bad options reached module computations"))
    with pytest.raises(ValueError):
        coarse_grain(*shifted_css(), **options)
