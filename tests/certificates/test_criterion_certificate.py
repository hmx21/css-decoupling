"""Tamper tests bind every kind of decision to source, lattice, and maps."""
from copy import deepcopy
import json

import pytest

from css_decoupling import Matrix, Ring
from css_decoupling.algebra.laurent import eye
from css_decoupling.certificates.comparison import _matrix_dict
from css_decoupling.certificates.criterion import verify_lattice_criterion
from css_decoupling.decoupling.criterion import check_lattice
from css_decoupling.models.examples import css_checks, glued_css, glued_toric_complex, toric_css
from css_decoupling.topology.complexes import direct_sum_complexes, toric_window


@pytest.fixture(scope="module")
def split():
    return check_lattice(*glued_css(Ring(3, 3)), [[3, 0, 0], [0, 1, 0], [0, 0, 1]])


@pytest.fixture(scope="module")
def obstructed():
    ring = Ring(3, 4)
    C = direct_sum_complexes(toric_window(ring, {1: 1}),
                            glued_toric_complex(ring, 2, 3, axes=(1, 2)))
    return check_lattice(*css_checks(C), [[int(i == j) for j in range(4)] for i in range(4)])


@pytest.fixture(scope="module")
def infinite():
    ring = Ring(3, 3)
    x, y, _ = ring.variables
    HX = Matrix.from_rows(ring, [[x - 1, y - 1]])
    HZ = Matrix.from_rows(ring, [[-(y - 1)], [x - 1]]).dagger()
    return check_lattice(HX, HZ, [[int(i == j) for j in range(3)] for i in range(3)])


def test_verifier_uses_maps_and_algebra_not_the_constructor(split, monkeypatch):
    import css_decoupling.decoupling.criterion as constructor
    import css_decoupling.decoupling.analysis as analysis
    def forbidden(*args, **kwargs):
        pytest.fail("verifier must not trust or repeat the high-level constructor")
    monkeypatch.setattr(constructor, "check_lattice", forbidden)
    monkeypatch.setattr(analysis, "analyze_profile", forbidden)
    report = verify_lattice_criterion(json.loads(json.dumps(split.certificate())))
    assert report.valid, report.errors


@pytest.mark.parametrize("mutation", [
    "source", "model", "basis", "claim", "boolean-claim", "sector", "omit-layer", "action",
    "restriction", "split-map", "lower-split", "cone-lift", "coefficient", "extra-stage", "scope",
])
def test_positive_tampering_rejected(split, mutation):
    data = deepcopy(split.certificate())
    if mutation == "source":
        data["input"]["HZ"]["entries"].pop()
    elif mutation == "model":
        data["model"]["differentials"][0][1]["entries"].pop()
    elif mutation == "basis":
        data["basis"][0][0] = 1
    elif mutation == "claim":
        data["decouplable"] = False
    elif mutation == "boolean-claim":
        data["decouplable"] = 1
    elif mutation == "sector":
        data["sector"] = "auto"
    elif mutation == "omit-layer":
        data["layers"].pop()
    elif mutation == "action":
        data["layers"][0]["actions"][0]["entries"] = []
    elif mutation == "restriction":
        data["restricted"]["differentials"][0][1]["entries"].pop()
    elif mutation == "split-map":
        for _, matrix in data["splitting"]:
            if matrix["entries"]:
                matrix["entries"].pop()
                break
    elif mutation in ("lower-split", "cone-lift"):
        key = "lower_splitting" if mutation == "lower-split" else "cone_lift"
        for _, matrix in data["attachments"][0][key]:
            if matrix["entries"]:
                matrix["entries"].pop()
                break
    elif mutation == "coefficient":
        data["attachments"][0]["classes"][0]["coefficients"]["entries"] = [[0, 0, [[[0, 0, 0], 1]]]]
    elif mutation == "extra-stage":
        data["attachments"].append(data["attachments"][0])
    elif mutation == "scope":
        data["scope"] = "minimum-lattice"
    report = verify_lattice_criterion(data)
    assert not report.valid, mutation


@pytest.mark.parametrize("mutation", ["claim", "skip-lower", "higher-stage", "zero-class", "obstruction-degree", "change-basis"])
def test_multilayer_negative_tampering_rejected(obstructed, mutation):
    data = deepcopy(obstructed.certificate())
    if mutation == "claim":
        data["decouplable"] = True
        data["reason"] = "split-profile"
    elif mutation == "skip-lower":
        data["attachments"].pop(0)
    elif mutation == "higher-stage":
        data["attachments"].append(data["attachments"][-1])
    elif mutation == "zero-class":
        for cls in data["attachments"][-1]["classes"]:
            cls["coefficients"]["entries"] = []
    elif mutation == "obstruction-degree":
        data["obstruction"]["degree"] = 2
    elif mutation == "change-basis":
        data["basis"][1][1] = 3
    assert not verify_lattice_criterion(data).valid


@pytest.mark.parametrize("mutation", ["degree", "ray", "boundary", "presentation", "cycles", "claim", "spurious-attachment"])
def test_infinite_ray_witness_tampering_rejected(infinite, mutation):
    data = deepcopy(infinite.certificate())
    if mutation == "degree":
        data["obstruction"]["degree"] = 9
    elif mutation == "ray":
        data["obstruction"]["ray"]["axis"] = 0
    elif mutation == "boundary":
        data["obstruction"]["boundary_coordinates"]["entries"] = []
    elif mutation == "presentation":
        data["obstruction"]["presentation"]["entries"] = []
    elif mutation == "cycles":
        data["obstruction"]["cycles"]["entries"] = []
    elif mutation == "claim":
        data["decouplable"] = True
    elif mutation == "spurious-attachment":
        data["attachments"] = [{}]
    assert not verify_lattice_criterion(data).valid


def test_translation_witness_binds_candidate_action():
    ring = Ring(3, 2)
    x, y = ring.variables
    HX = Matrix.from_rows(ring, [[x + 1, y - 1]])
    HZ = Matrix.from_rows(ring, [[-(y - 1)], [x + 1]]).dagger()
    result = check_lattice(HX, HZ, [[1, 0], [0, 1]])
    assert result.reason == "nontrivial-layer-translation" and verify_lattice_criterion(result).valid
    for mutation in ("action", "column", "basis"):
        data = result.certificate()
        if mutation == "action":
            data["obstruction"]["action"] = _matrix_dict(eye(ring, 1))
        elif mutation == "column":
            data["obstruction"]["column"] = 1
        else:
            data["basis"][0][0] = 2
        assert not verify_lattice_criterion(data).valid


def test_mutating_public_result_claim_does_not_reuse_a_success_snapshot():
    result = check_lattice(*toric_css(Ring(2, 2), {1: 1}), [[1, 0], [0, 1]])
    assert verify_lattice_criterion(result).valid
    result.decouplable = False
    assert result.certificate()["decouplable"] is False
    assert not verify_lattice_criterion(result).valid


@pytest.mark.parametrize("data", [None, [], {}, {"schema": "css-decoupling.lattice-criterion"}])
def test_malformed_certificates_are_invalid(data):
    assert not verify_lattice_criterion(data).valid
