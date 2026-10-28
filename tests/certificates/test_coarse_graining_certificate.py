"""The Algorithm 1 verifier checks source mathematics, not construction flags."""
from copy import copy, deepcopy

import pytest

from css_decoupling import Matrix, Ring
from css_decoupling.certificates.analysis import complex_dict
from css_decoupling.certificates.coarse_graining import verify_coarse_graining
from css_decoupling.decoupling.coarse_graining import coarse_grain
from css_decoupling.decoupling.splitting import restrict_complex
from css_decoupling.errors import ResourceLimit
from css_decoupling.lattice.retained import Lattice
from css_decoupling.models.examples import toric_css


@pytest.fixture(scope="module")
def positive():
    return coarse_grain(*toric_css(Ring(3, 3), {1: 1, 2: 1}), verify=False)


def test_independent_verifier_never_calls_constructor_or_kernel_search(positive, monkeypatch):
    import css_decoupling.decoupling.coarse_graining as module
    import css_decoupling.topology.resolutions as resolutions
    import css_decoupling.lattice.kernel as kernel_module
    def forbidden(*args, **kwargs):
        pytest.fail("Independent verification called a construction algorithm")
    monkeypatch.setattr(module, "coarse_grain", forbidden)
    monkeypatch.setattr(module, "css_complex", forbidden)
    monkeypatch.setattr(resolutions, "css_complex", forbidden)
    monkeypatch.setattr(kernel_module, "translation_kernel", forbidden)
    report = verify_coarse_graining(positive)
    assert report.valid, report.errors


def _tamper(data, kind):
    if kind == "flag":
        data["decouplable"] = False
    elif kind == "reason":
        data["reason"] = "split-profile"
    elif kind == "input":
        term = data["input"]["HX"]["entries"][0][2][0]
        term[1] = 3 - term[1]
    elif kind == "missing-negative-resolution":
        data["model"]["ranks"] = [pair for pair in data["model"]["ranks"] if pair[0] >= -1]
        data["model"]["differentials"] = [pair for pair in data["model"]["differentials"] if pair[0] >= -1]
    elif kind == "missing-layer":
        data["layers"].pop()
    elif kind == "false-action":
        data["layers"][0]["actions"][0]["entries"][0][2][0][1] = 2
    elif kind == "false-kernel":
        data["translation_kernel"]["basis_columns"][0][0] = 3
        data["translation_kernel"]["index"] = 3
    elif kind == "wrong-cover":
        data["basis"][0][0] = 6
        data["index"] *= 2
    elif kind == "wrong-index":
        data["index"] += 1
    elif kind == "wrong-restriction":
        term = data["restricted"]["differentials"][0][1]["entries"][0][2][0]
        term[1] = 3 - term[1]
    elif kind == "extra-restricted-map":
        data["restricted"]["differentials"].append([99, {"rows": 0, "cols": 0, "entries": []}])
    elif kind == "obstruction":
        data["obstruction"] = {}
    elif kind == "sector":
        data["sector"] = "auto"
    elif kind == "version":
        data["version"] = 2
    else:
        raise AssertionError(kind)


@pytest.mark.parametrize("kind", [
    "flag", "reason", "input", "missing-negative-resolution", "missing-layer",
    "false-action", "false-kernel", "wrong-cover", "wrong-index",
    "wrong-restriction", "extra-restricted-map", "obstruction", "sector", "version",
])
def test_finite_certificate_tampering_rejected(positive, kind):
    data = deepcopy(positive.certificate())
    _tamper(data, kind)
    report = verify_coarse_graining(data)
    assert not report.valid, kind


def test_current_result_mutations_are_not_hidden_by_cached_certificates(positive):
    result = copy(positive)
    result.translation_kernel = copy(positive.translation_kernel)
    result.translation_kernel.index *= 2
    assert not verify_coarse_graining(result).valid
    result = copy(positive)
    result.lattice = copy(positive.lattice)
    result.lattice.index *= 2
    assert not verify_coarse_graining(result).valid
    result = copy(positive)
    result.decouplable = False
    assert not verify_coarse_graining(result).valid


def test_signed_sheared_basis_checked_in_its_actual_coordinates(positive):
    data = positive.certificate()
    lattice = Lattice([[-3, 3, 0], [0, 3, 0], [0, 0, 3]])
    data["basis"] = [list(row) for row in lattice.basis]
    data["restricted"] = complex_dict(restrict_complex(positive.complex, lattice))
    assert verify_coarse_graining(data).valid
    # Equivalent lattice geometry does not allow a stale restriction matrix.
    data["restricted"] = positive.certificate()["restricted"]
    assert not verify_coarse_graining(data).valid


def test_verifier_cell_cap_precedes_coset_materialization(positive, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Verification expanded a cell before enforcing the cap")
    monkeypatch.setattr(Lattice, "restrict", forbidden)
    with pytest.raises(ResourceLimit, match="index"):
        verify_coarse_graining(positive, max_cell_index=1)


@pytest.fixture(scope="module")
def infinite():
    ring = Ring(3, 3)
    x, y, _ = ring.variables
    HX = Matrix.from_rows(ring, [[x - 1, y - 1]])
    HZ = Matrix.from_rows(ring, [[-(y - 1)], [x - 1]]).dagger()
    return coarse_grain(HX, HZ, verify=False)


@pytest.mark.parametrize("kind", ["ray", "presentation", "degree", "output"])
def test_infinite_certificate_tampering_rejected(infinite, kind):
    data = infinite.certificate()
    if kind == "ray":
        data["obstruction"]["ray"]["axis"] = 0
    elif kind == "presentation":
        data["obstruction"]["presentation"]["entries"] = []
    elif kind == "degree":
        data["obstruction"]["degree"] = 2
    else:
        data["basis"] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    assert not verify_coarse_graining(data).valid


@pytest.mark.parametrize("data", [None, [], {}, {"schema": "css-decoupling.coarse-graining"}])
def test_malformed_certificate_is_invalid(data):
    assert not verify_coarse_graining(data).valid
