from copy import deepcopy
import json

import pytest

from css_decoupling import Ring, Matrix
from css_decoupling.certificates.periodic import verify_periodic
from css_decoupling.decoupling.analysis import analyze_profile
from css_decoupling.decoupling.gluing import GluingClass
from css_decoupling.models.examples import glued_css
from css_decoupling.topology.periodic import periodic_code, periodic_from_gluing, periodic_profile
from css_decoupling.errors import ResourceLimit


def diagonal(values):
    return [[v if i == j else 0 for j in range(len(values))] for i, v in enumerate(values)]


@pytest.mark.parametrize("p", [2, 3, 5])
def test_finite_count_certificate_and_rank_input_period_tampering(p):
    result = periodic_code(*glued_css(Ring(p, 3)), diagonal([p, 1, 1]))
    data = json.loads(json.dumps(result.certificate()))
    assert verify_periodic(result).valid
    assert verify_periodic(data).valid
    for change in (
        lambda d: d["count"].update(rank_x=d["count"]["rank_x"] + 1),
        lambda d: d["count"].update(logical_qudits=0),
        lambda d: d["count"].update(index=True),
        lambda d: d["count"].update(upper_bound=6, decouplable=True),
        lambda d: d["lattice_basis"][0].__setitem__(0, 1),
        lambda d: d["source"]["HX"].update(entries=[]),
        lambda d: d.update(success=True),
    ):
        bad = deepcopy(data)
        change(bad)
        assert not verify_periodic(bad).valid


def test_abstract_model_cannot_claim_physical_decoupling():
    ring = Ring(5, 3)
    cls = GluingClass(2, 1, 2, 1, Matrix(ring, 1, 6, {(0, 0): 1, (0, 3): 1}))
    result = periodic_from_gluing([cls], {1: 1, 2: 2}, [[1, 0, 0], [0, 1, 0], [0, 1, 5]])
    data = result.certificate()
    assert result.logical_qudits == 7 and verify_periodic(data).valid
    for change in (
        lambda d: d["count"].update(incoming_rank=2),
        lambda d: d["count"].update(upper_bound=9, decouplable=False),
        lambda d: d["model"]["classes"][0]["coefficients"].update(entries=[]),
        lambda d: d["model"].update(multiplicities=[[1, 1], [2, 1]]),
    ):
        bad = deepcopy(data)
        change(bad)
        assert not verify_periodic(bad).valid


@pytest.mark.parametrize("force_finite", [False, True])
def test_actual_profile_binding_and_tampering(force_finite):
    analysis = analyze_profile(*glued_css(Ring(3, 3)))
    result = periodic_profile(analysis, diagonal([1, 1, 1]), prefer_fast=not force_finite)
    assert result.logical_qudits == 4 and result.upper_bound == 6 and result.decouplable is False
    data = json.loads(json.dumps(result.certificate()))
    assert verify_periodic(data).valid
    for change in (
        lambda d: d["count"].update(upper_bound=4, decouplable=True),
        lambda d: d["profile_analysis"]["input"]["HX"].update(entries=[]),
        lambda d: d["source"]["HX"].update(entries=[]),
        lambda d: d.update(profile_analysis=None),
    ):
        bad = deepcopy(data)
        change(bad)
        assert not verify_periodic(bad).valid
    if not force_finite:
        bad = deepcopy(data)
        term = bad["model"]["classes"][0]["coefficients"]["entries"][0][2][0]
        term[1] = 2 * term[1] % 3
        # A nonzero scalar multiple preserves both ranks: only the physical
        # source binding detects replacing the actual class with another one.
        assert not verify_periodic(bad).valid


def test_large_index_certificate_fast_verification_has_no_cell_expansion():
    analysis = analyze_profile(*glued_css(Ring(2, 3)))
    data = periodic_profile(analysis, diagonal([2 * 10**30, 1, 1]), max_qudits=1).certificate()
    assert verify_periodic(data, max_qudits=1).valid


def test_nontrivial_actions_profile_certificate_and_caps():
    ring = Ring(3, 2)
    x, y = ring.variables
    HX = Matrix.from_rows(ring, [[x + 1, y + 1]])
    HZ = Matrix.from_rows(ring, [[-(y + 1)], [x + 1]]).dagger()
    analysis = analyze_profile(HX, HZ)
    for basis in ([[1, 0], [0, 1]], [[1, 0], [1, 2]]):
        data = periodic_profile(analysis, basis).certificate()
        assert verify_periodic(data).valid
    data = periodic_code(HX, HZ, diagonal([3, 3])).certificate()
    with pytest.raises(ResourceLimit):
        verify_periodic(data, max_qudits=1)
    with pytest.raises(ResourceLimit):
        verify_periodic(data, max_matrix_entries=1)


@pytest.mark.parametrize("malformed", [None, [], {}, {"schema": "bad"}])
def test_malformed_certificate_is_a_verification_failure(malformed):
    assert not verify_periodic(malformed).valid
