"""Signed wedge sums and equivalent presentations of artificial CSS inputs."""
from copy import deepcopy
import shutil
import subprocess

import pytest

from css_decoupling.algebra.laurent import Matrix, Ring, zeros
from css_decoupling.certificates.criterion import verify_lattice_criterion
from css_decoupling.decoupling.criterion import check_lattice
from css_decoupling.models.stress import (
    STRESS_CASES, glued_stress_complex, glued_stress_css, lattice_relabel_css,
    rank_four_glued_css, redundant_css_checks, stress_case,
)
from css_decoupling.topology.cohomology import profile
from css_decoupling.topology.resolutions import css_complex


@pytest.fixture(autouse=True)
def forbid_external_algebra(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Stress models must run with the native Python backend")
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(shutil, "which", forbidden)


def diagonal(values):
    return [[value if i == j else 0 for j in range(len(values))]
            for i, value in enumerate(values)]


@pytest.mark.parametrize("name", tuple(STRESS_CASES))
def test_named_stress_inputs_are_actual_signed_css_matrices(name):
    HX, HZ = stress_case(name)
    D, p, _ = STRESS_CASES[name]
    assert HX.ring == HZ.ring == Ring(p, D)
    assert HX * HZ.dagger() == zeros(HX.ring, HX.nrows, HZ.nrows)


@pytest.mark.parametrize("D,p,degrees", [(4, 3, (1, 3)), (4, 3, (1, 2, 3)),
                                       (5, 5, (1, 3)), (5, 5, (1, 2, 3))])
def test_native_raw_input_resolution_recovers_the_expected_layers(D, p, degrees):
    HX, HZ = glued_stress_css(Ring(p, D), degrees)
    resolved = css_complex(HX, HZ, complete_negative=False, resolution_method="native")
    assert resolved.d(0) == HX and resolved.d(-1) == HZ.dagger()
    assert {j: layer.dimension for j, layer in profile(resolved).items()} == {
        j: 1 for j in degrees}


def test_rank_four_form_has_both_planes_on_one_low_layer_and_exact_obstruction():
    ring = Ring(3, 4)
    result = check_lattice(*rank_four_glued_css(ring), diagonal([1] * 4), backend="sympy")
    assert result.reason == "nonzero-gluing"
    assert result.multiplicities == {1: 1, 2: 1}
    assert len(result.gluing[-1].classes) == 1
    kappa = result.gluing[-1].classes[0]
    assert kappa.source_dimension == kappa.target_dimension == 1
    assert kappa.ext_degree == 2
    support = {axes for axes, _, _, _ in kappa.terms()}
    assert support == {(0, 1), (2, 3)}
    assert kappa.coefficients[0, 0] == kappa.coefficients[0, 5] != ring.zero()
    assert not kappa.restrict(diagonal([3, 1, 1, 1])).is_zero
    assert kappa.restrict(diagonal([3, 1, 3, 1])).is_zero
    assert verify_lattice_criterion(result, backend="sympy").valid
    forged = deepcopy(result.certificate())
    forged["attachments"][0]["classes"][0]["coefficients"]["entries"] = []
    assert not verify_lattice_criterion(forged, backend="sympy").valid


@pytest.mark.parametrize("p", [3, 5])
def test_exterior_sum_is_linear_and_keeps_odd_characteristic_signs(p):
    ring = Ring(p, 4)
    first = glued_stress_complex(ring, couplings=(((0, 1), 1),))
    second = glued_stress_complex(ring, couplings=(((2, 3), 2),))
    split = glued_stress_complex(ring, coefficient=0)
    summed = glued_stress_complex(ring, couplings=(((0, 1), 1), ((2, 3), 2)))
    for n in summed.degrees:
        assert summed.d(n) == first.d(n) + second.d(n) - split.d(n)
    # Both terms act between the same two layers; no doubled toric ranks.
    assert summed.ranks == first.ranks == second.ranks
    assert summed.validate()


def test_unimodular_relabelling_is_exact_and_commutes_with_adjoint():
    ring = Ring(5, 4)
    HX, HZ = glued_stress_css(ring, (1, 3), shear=True)
    basis = [[1, 2, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    inverse = [[1, -2, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    changed = lattice_relabel_css(HX, HZ, basis)
    assert lattice_relabel_css(*changed, inverse) == (HX, HZ)
    assert not changed[0] * changed[1].dagger()
    assert changed != (HX, HZ)


def test_redundant_checks_keep_original_rows_and_exact_laurent_span():
    HX, HZ = stress_case("gluing_d4_p3_ext3")
    extended = redundant_css_checks(HX, HZ)
    for original, enlarged in zip((HX, HZ), extended):
        assert enlarged[:-1, :] == original
        coefficients = Matrix.from_rows(original.ring, [[
            (original.ring.variable(0)**-1 if i == 0 else 0)
            + (2 if i == original.nrows - 1 else 0) for i in range(original.nrows)]])
        assert coefficients * original == enlarged[-1:, :]
    assert not extended[0] * extended[1].dagger()


def test_scrambled_redundant_presentation_retains_its_inter_layer_obstruction():
    HX, HZ = stress_case("gluing_d4_p3_ext3_scrambled")
    result = check_lattice(HX, HZ, diagonal([1] * 4), backend="sympy")
    assert result.multiplicities == {1: 1, 3: 1}
    assert result.reason == "nonzero-gluing"
    assert result.gluing[-1].classes[0].ext_degree == 3
    assert verify_lattice_criterion(result, backend="sympy").valid


@pytest.mark.parametrize("options", [
    {"degrees": (1, 1)}, {"degrees": (True, 2)},
    {"couplings": (((0, 0), 1),)}, {"couplings": (((1, 0), 1),)},
    {"couplings": (((0, 4), 1),)},
    {"couplings": (((0, 1), 1),), "degrees": (1, 2, 3)},
])
def test_malformed_exterior_inputs_are_rejected(options):
    with pytest.raises(ValueError):
        glued_stress_complex(Ring(3, 4), **options)


@pytest.mark.parametrize("basis", [diagonal([2, 1, 1, 1]), [[1]],
                                   diagonal([True, 1, 1, 1])])
def test_relabelling_rejects_coarse_cells_and_noninteger_bases(basis):
    with pytest.raises(ValueError):
        lattice_relabel_css(*rank_four_glued_css(Ring(3, 4)), basis)
