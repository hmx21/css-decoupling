"""Current manuscript Algorithm 2 and its independent composed certificate."""
from copy import deepcopy

import pytest

from css_decoupling import (Ring, coarse_grain, decouple_general as decouple, decouple_complex,
                           verify_construction, ResourceLimit)
from css_decoupling.topology.complexes import koszul, disk, direct_sum_complexes, koszul_window
from css_decoupling.models.examples import glued_css


@pytest.mark.parametrize("D,p,j", [(2, 3, 1), (3, 5, 1), (4, 2, 2), (5, 3, 1)])
def test_full_complex_circuit_and_schanuel_directions(D, p, j):
    result = decouple_complex(koszul(Ring(p, D), j))
    assert result.multiplicities == {j: 1}
    comp = result.comparison
    cancelled = len(comp["unit_cancellation"]["directions"])
    assert comp["schanuel"]["odd_free_rank"] > 0
    # A single toric layer: every extra Schanuel direction has a unit row.
    assert cancelled == comp["schanuel"]["odd_free_rank"]
    assert comp["free_T"] == result.complex.rank(2)
    assert comp["free_C"] == comp["schanuel"]["even_free_rank"] - cancelled
    assert verify_construction(result).valid


def test_strict_bounded_complete_path_never_uses_groebner(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Strict bounded path called a Groebner method")
    from css_decoupling.algebra.modules import GeneralBackend
    for name in ("solve", "kernel", "quotient_groebner", "reduce_quotient"):
        monkeypatch.setattr(GeneralBackend, name, forbidden)
    C = direct_sum_complexes(disk(Ring(3, 2), -1), disk(Ring(3, 2), 1))
    result = decouple_complex(C, linear_backend="bounded")
    assert not result.multiplicities
    assert verify_construction(result, linear_backend="bounded").valid


def test_nonunit_complete_kernel_budget_is_not_a_negative_decision():
    with pytest.raises(ResourceLimit):
        decouple_complex(koszul(Ring(3, 2), 1), linear_backend="bounded",
                         bounded_options={"max_degree": 0})


def test_default_raw_pipeline_does_not_call_quillen_suslin(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("The current paper route must not compute a Q basis")
    monkeypatch.setattr("css_decoupling.algebra.projective.free_basis", forbidden)
    result = decouple(*glued_css(Ring(2, 3)))
    assert result.trace["comparison_method"] == "schanuel"
    assert verify_construction(result.trace["construction"]).valid


def test_uniform_nonzero_gluing_algorithms_compose():
    coarse = coarse_grain(*glued_css(Ring(2, 3)))
    assert coarse.lattice.index == 8
    built = decouple_complex(coarse.restricted)
    built.coarse_graining = coarse.certificate()
    assert built.multiplicities == {1: 1, 2: 1}
    assert verify_construction(built).valid


def test_bounded_backend_cannot_be_silently_ignored_in_free_basis_mode():
    C = koszul(Ring(3, 2), 1)
    with pytest.raises(ValueError, match="requires comparison_method"):
        decouple(C.d(0), C.d(-1).dagger(), comparison_method="free-basis", linear_backend="bounded")


def test_input_must_have_complete_negative_resolution():
    with pytest.raises(ValueError):
        decouple_complex(koszul_window(Ring(3, 4), 1))


def test_free_model_degree_range_follows_lemma_four():
    R = Ring(3, 2)
    edge = direct_sum_complexes(koszul(R, 1), disk(R, -4), disk(R, 3))
    assert verify_construction(decouple_complex(edge)).valid
    for outside in (disk(R, 4), disk(R, -5)):
        with pytest.raises(ValueError, match=r"D\+2"):
            decouple_complex(direct_sum_complexes(koszul(R, 1), outside))


def test_rectangular_block_permutation_does_not_claim_one_swap_layer():
    from css_decoupling.circuits.gates import Circuit, SWAP, compile_theta
    from css_decoupling import eye, zeros
    R = Ring(3, 2)
    theta = compile_theta(zeros(R, 2, 1), zeros(R, 1, 2))
    swaps = tuple(g for g in theta.gates if isinstance(g, SWAP))
    permutation = Circuit(R, theta.qudits, swaps).matrix()
    # A disjoint single layer of SWAPs is an involution; this legitimate
    # rectangular block permutation is not, even though it is on-site.
    assert len(swaps) >= 2
    assert permutation * permutation != eye(R, theta.qudits)


@pytest.mark.parametrize("sector", ["X", "Z"])
def test_algorithm_one_two_composition_binds_original_code(sector):
    C = koszul(Ring(3, 2), 1)
    coarse = coarse_grain(C.d(0), C.d(-1).dagger(), sector=sector)
    result = decouple_complex(coarse.restricted)
    result.coarse_graining = coarse.certificate()
    assert verify_construction(result).valid
    wrong = result.certificate()
    wrong["coarse_graining"]["basis"][0][0] = 2
    assert not verify_construction(wrong).valid


@pytest.mark.parametrize("change", ["dense-size", "boolean-resource", "missing-layer", "comparison",
                                   "circuit-source", "coarse-without-source", "cocycles"])
def test_algorithm_two_rejects_cross_component_tampering(change):
    result = decouple_complex(koszul(Ring(3, 2), 1), verify=False)
    data = deepcopy(result.certificate())
    if change == "dense-size": data["input_size"]["dense_coefficients"] += 1
    elif change == "boolean-resource": data["resources"]["interaction_radius"] = True
    elif change == "missing-layer": data["cohomology"] = []
    elif change == "comparison": data["comparison"]["ranks"]["odd"] += 1
    elif change == "circuit-source": data["decoupling"]["input"]["HZ"]["entries"].pop()
    elif change == "coarse-without-source": data["physical_decoupling"] = data["decoupling"]
    else: data["cohomology"][0]["representatives"]["entries"] = []
    assert not verify_construction(data).valid
