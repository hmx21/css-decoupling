"""Independent verification of periodic layer contributions and v2 profiles."""
from copy import deepcopy
import json

import pytest

from css_decoupling import Matrix, Ring, eye, zeros
from css_decoupling.certificates.periodic import verify_periodic, verify_periodic_layer
from css_decoupling.decoupling.analysis import analyze_profile
from css_decoupling.errors import ResourceLimit
from css_decoupling.lattice.retained import Lattice
from css_decoupling.models.examples import glued_css
from css_decoupling.topology.periodic import periodic_code, periodic_layer, periodic_profile


def diagonal(values):
    return [[v if i == j else 0 for j in range(len(values))] for i, v in enumerate(values)]


@pytest.fixture(scope="module")
def scalar_analysis():
    ring = Ring(3, 2)
    x, y = ring.variables
    HX = Matrix.from_rows(ring, [[x + 1, y + 1]])
    HZ = Matrix.from_rows(ring, [[-(y + 1)], [x + 1]]).dagger()
    return analyze_profile(HX, HZ)


@pytest.fixture(scope="module")
def glued_analysis():
    return analyze_profile(*glued_css(Ring(3, 3)))


@pytest.mark.parametrize("p,D,j", [(2, 3, 1), (3, 4, 2), (5, 5, 3)])
def test_layer_certificates_independent_recomputation_and_tampering(p, D, j, monkeypatch):
    ring = Ring(p, D)
    J = Matrix.from_rows(ring, [[1, 1], [0, 1]])
    result = periodic_layer([J] + [eye(ring, 2)] * (D - 1), j,
                            diagonal([-1] + [1] * (D - 1)))
    data = json.loads(json.dumps(result.certificate()))
    import css_decoupling.topology.periodic as constructor

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Verifier must not reuse the counting constructor")

    for name in ("periodic_layer", "_layer_period_actions", "_layer_differential", "finite_rank"):
        monkeypatch.setattr(constructor, name, forbidden)
    assert verify_periodic_layer(data).valid
    for change in (
        lambda d: d["contribution"].update(logical_qudits=123),
        lambda d: d["contribution"].update(incoming_rank=0),
        lambda d: d["contribution"].update(dimension=True),
        lambda d: d["contribution"].update(translation_trivial=True),
        lambda d: d["period_actions"][0].update(entries=[]),
        lambda d: d["actions"][0].update(entries=[]),
        lambda d: d["lattice_basis"][0].__setitem__(0, p),
        lambda d: d.update(version=True),
        lambda d: d.update(success=True),
    ):
        bad = deepcopy(data)
        change(bad)
        assert not verify_periodic_layer(bad).valid
    result.logical_qudits += 1
    assert not verify_periodic_layer(result).valid


def test_layer_verifier_rejects_noncommuting_nonconstant_and_wrong_shapes():
    ring = Ring(3, 2)
    A = Matrix.from_rows(ring, [[1, 1], [0, 1]])
    data = periodic_layer([A, eye(ring, 2)], 1, diagonal([1, 1])).certificate()
    from css_decoupling.certificates.comparison import _matrix_dict
    for invalid in (Matrix.from_rows(ring, [[1, 0], [1, 1]]), eye(ring, 1),
                    Matrix.from_rows(ring, [[ring.variable(0), 0], [0, 1]])):
        bad = deepcopy(data)
        bad["actions"][1] = _matrix_dict(invalid)
        assert not verify_periodic_layer(bad).valid


def test_single_layer_profile_v2_caps_and_current_result_claims(scalar_analysis):
    result = periodic_profile(scalar_analysis, diagonal([10**60 + 1] * 2), max_qudits=1)
    data = json.loads(json.dumps(result.certificate()))
    assert data["version"] == 2 and data["method"] == "single-layer"
    assert verify_periodic(data, max_qudits=1).valid
    for change in (
        lambda d: d["diagnostics"].update(translation_loss=0),
        lambda d: d["diagnostics"].update(gluing_loss=1),
        lambda d: d["diagnostics"]["layers"][0].update(logical_qudits=2),
        lambda d: d["diagnostics"]["layers"][0].update(degree=2),
        lambda d: d["diagnostics"]["layers"][0].update(translation_trivial=0),
        lambda d: d["diagnostics"].update(layers=[]),
        lambda d: d.update(profile_analysis=None),
        lambda d: d["profile_analysis"]["layers"][0]["actions"][0].update(entries=[]),
        lambda d: d["count"].update(logical_qudits=2),
    ):
        bad = deepcopy(data)
        change(bad)
        assert not verify_periodic(bad, max_qudits=1).valid
    result.translation_loss = 0
    assert not verify_periodic(result, max_qudits=1).valid
    result.translation_loss = 2
    result.layer_contributions[0]["incoming_rank"] = 0
    assert not verify_periodic(result, max_qudits=1).valid
    with pytest.raises(ResourceLimit):
        verify_periodic(data, max_exterior_entries=1)
    with pytest.raises(ResourceLimit):
        verify_periodic(data, max_rank_work=1)


@pytest.mark.parametrize("prefer_fast", [False, True])
def test_v2_gluing_loss_and_original_v1_semantics(glued_analysis, prefer_fast):
    data = periodic_profile(glued_analysis, diagonal([1, 1, 1]), prefer_fast=prefer_fast).certificate()
    assert verify_periodic(data).valid
    assert data["diagnostics"]["gluing_loss"] == 2
    bad = deepcopy(data)
    bad["diagnostics"]["gluing_loss"] = 0
    assert not verify_periodic(bad).valid
    legacy = deepcopy(data)
    legacy["version"] = 1
    del legacy["diagnostics"]
    assert verify_periodic(legacy).valid
    raw = periodic_code(*glued_css(Ring(3, 3)), diagonal([1, 1, 1])).certificate()
    assert raw["version"] == 1 and verify_periodic(raw).valid


def test_large_period_verifier_never_allocates_period_cells(scalar_analysis, monkeypatch):
    data = periodic_profile(scalar_analysis, [[10**80 + 1, -10**80], [0, 1]], max_qudits=1).certificate()
    original = Lattice.representatives.fget

    def guarded(lattice):
        if lattice.index > 100:
            raise AssertionError("Period-volume expansion is forbidden")
        return original(lattice)

    monkeypatch.setattr(Lattice, "representatives", property(guarded))
    assert verify_periodic(data, max_qudits=1).valid


def test_zero_profile_certificate_has_no_layer_or_loss():
    ring = Ring(3, 2)
    analysis = analyze_profile(eye(ring, 1), zeros(ring, 0, 1))
    data = periodic_profile(analysis, diagonal([10**30, 1]), max_qudits=1).certificate()
    assert data["diagnostics"] == {"layers": [], "translation_loss": 0, "gluing_loss": 0}
    assert verify_periodic(data, max_qudits=1).valid


def test_layer_resource_limits_and_malformed_certificates():
    ring = Ring(5, 4)
    J = Matrix.from_rows(ring, [[1, 1], [0, 1]])
    result = periodic_layer([J] + [eye(ring, 2)] * 3, 2, diagonal([1] * 4))
    for options in (dict(max_matrix_entries=1), dict(max_exterior_entries=1), dict(max_rank_work=1)):
        with pytest.raises(ResourceLimit):
            verify_periodic_layer(result, **options)
    for bad in (None, [], {}, {"schema": "unknown"}):
        assert not verify_periodic_layer(bad).valid
