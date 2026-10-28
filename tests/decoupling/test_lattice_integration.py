from copy import deepcopy

import pytest

from css_decoupling import Ring, Matrix, decouple_general as decouple
from css_decoupling.models.examples import glued_css, glued_toric_complex
from css_decoupling.topology.complexes import CochainComplex, koszul
from css_decoupling.decoupling.analysis import analyze_profile
from css_decoupling.decoupling.lattice_query import check_two_layer_lattice
from css_decoupling.certificates.lattice_query import verify_lattice_decision
from css_decoupling.certificates.analysis import verify_analysis
from css_decoupling.certificates.decoupling import verify_decoupling
from css_decoupling.errors import BackendUnavailable


def three_plaquette_css(ring):
    """Ordinary matrix input, scalar form with three nonzero supports."""
    base = glued_toric_complex(ring, coefficient=0)
    pieces = [glued_toric_complex(ring, axes=axes) for axes in ((0, 1), (0, 2), (1, 2))]
    differentials = {}
    for n in base.degrees:
        d = base.d(n)
        for piece in pieces:
            d = d + piece.d(n) - base.d(n)
        differentials[n] = d
    C = CochainComplex(ring, base.ranks, differentials)
    return C.d(0), C.d(-1).dagger()


@pytest.mark.parametrize("p", [2, 3, 5])
def test_slanted_optimum_beats_coordinate_and_v1_stays_readable(p):
    HX, HZ = three_plaquette_css(Ring(p, 3))
    result = analyze_profile(HX, HZ)
    assert result.lattice.index == p
    assert result.lattice_guarantee == "minimum-cell"
    data = result.certificate()
    assert data["version"] == 2 and verify_analysis(data).valid
    coordinate = analyze_profile(HX, HZ, lattice_strategy="coordinate")
    assert coordinate.lattice.index == p * p
    assert coordinate.lattice_guarantee == "sufficient-cell"
    assert coordinate.certificate()["version"] == 1
    assert verify_analysis(coordinate.certificate()).valid
    for mutate in (
        lambda d: d["attachments"][0]["selection"].update(relative_index=1),
        lambda d: d["attachments"][0]["selection"]["relative_basis"][0].__setitem__(0, True),
        lambda d: d.update(version=1),
        lambda d: d["attachments"][1].update(selection=d["attachments"][0]["selection"]),
        lambda d: d["attachments"][0].update(kill_axes=[0, 1, 2]),
    ):
        bad = deepcopy(data)
        mutate(bad)
        assert not verify_analysis(bad).valid


def test_full_pipeline_uses_the_proven_small_slanted_cell():
    result = decouple(*three_plaquette_css(Ring(2, 3)))
    assert result.lattice.index == 2
    assert result.profile_analysis.lattice_guarantee == "minimum-cell"
    assert verify_decoupling(result).valid


@pytest.mark.parametrize("sector", ["X", "Z"])
def test_lattice_strategy_propagates_through_backend_and_sector(sector):
    result = analyze_profile(*glued_css(Ring(3, 3)), backend="sympy", sector=sector,
                             lattice_strategy="coordinate")
    assert result.certificate()["version"] == 1
    assert verify_analysis(result.certificate()).valid


def test_lightweight_query_handles_huge_and_non_killing_periods():
    analysis = analyze_profile(*glued_css(Ring(3, 3)))
    huge = [[1, 0, 0], [0, 3**20, 0], [0, 0, 1]]
    yes = check_two_layer_lattice(analysis, huge)
    assert yes.decouplable and yes.lattice.index == 3**20
    assert verify_lattice_decision(yes).valid
    no = check_two_layer_lattice(analysis, [[1, 0, 0], [0, 1, 0], [0, 0, 3]])
    assert not no.decouplable and no.reason == "nonzero-gluing"
    assert verify_lattice_decision(no).valid
    bad = no.certificate()
    bad["decouplable"] = True
    assert not verify_lattice_decision(bad).valid


def test_query_translation_obstruction_and_single_layer():
    ring = Ring(3, 2)
    x, y = ring.variables
    HX = Matrix.from_rows(ring, [[x + 1, y + 1]])
    HZ = Matrix.from_rows(ring, [[-(y + 1)], [x + 1]]).dagger()
    analysis = analyze_profile(HX, HZ)
    no = check_two_layer_lattice(analysis, [[1, 0], [0, 1]])
    assert not no.decouplable and no.reason == "nontrivial-layer-translation"
    assert verify_lattice_decision(no).valid
    yes = check_two_layer_lattice(analysis, analysis.translation_kernel.basis)
    assert yes.decouplable and verify_lattice_decision(yes).valid
    T = koszul(Ring(5, 4), 2)
    split = analyze_profile(T.d(0), T.d(-1).dagger())
    assert split.lattice_guarantee == "minimum-cell"


def test_multilayer_query_does_not_pretend_pairwise_criterion():
    from css_decoupling.models.examples import glued_toric_chain
    C = glued_toric_chain(Ring(2, 4), [1, 2, 3])
    analysis = analyze_profile(C.d(0), C.d(-1).dagger())
    assert analysis.lattice_guarantee == "sufficient-cell"
    with pytest.raises(BackendUnavailable, match="two"):
        check_two_layer_lattice(analysis, analysis.lattice.basis)


def test_bad_strategy_is_not_hidden_by_empty_profile():
    T = koszul(Ring(2, 2), 1)
    with pytest.raises(ValueError, match="lattice_strategy"):
        analyze_profile(T.d(0), T.d(-1).dagger(), lattice_strategy="magic")
