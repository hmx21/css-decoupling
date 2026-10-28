"""Strict source binding and independent validation of path-comparison data."""
from copy import deepcopy
import importlib
import json

import pytest

from css_decoupling.algebra.laurent import Ring
from css_decoupling.certificates.koszul_comparison import verify_koszul_comparison
from css_decoupling.errors import ResourceLimit
from css_decoupling.lattice.koszul_comparison import koszul_comparison


def _result():
    return koszul_comparison(Ring(5, 3), [[2, 1, 0], [-1, 2, 1], [0, -1, 2]], degrees=(1, 2, 3))


def test_json_roundtrip_and_verifier_does_not_reuse_construction(monkeypatch):
    result = _result()
    data = json.loads(json.dumps(result.certificate()))
    construction = importlib.import_module("css_decoupling.lattice.koszul_comparison")

    def forbidden(*args, **kwargs):
        raise AssertionError("Verifier called the construction algorithm")

    for name in ("koszul_comparison", "_path_matrix", "_determinant"):
        monkeypatch.setattr(construction, name, forbidden)
    report = verify_koszul_comparison(data)
    assert report.valid and len(report.checks) == 4


@pytest.mark.parametrize("change", [
    lambda d: d.update(version=2),
    lambda d: d.update(scope="all-lattice-degrees"),
    lambda d: d.update(extra=True),
    lambda d: d["basis"][0].__setitem__(0, 3),
    lambda d: d["basis"].__setitem__(0, [0, 0, 0]),
    lambda d: d["basis"][0].__setitem__(0, True),
    lambda d: d["ring"].update(prime=7),
    lambda d: d["conventions"].update(paths="arbitrary-path"),
    lambda d: d["requested_degrees"].append(3),
    lambda d: d["requested_degrees"].append(4),
    lambda d: d["maps"].pop(),
    lambda d: d["maps"].reverse(),
    lambda d: d["maps"][0]["matrix"]["entries"][0][2][0].__setitem__(1, 2),
    lambda d: d["maps"][2]["matrix"]["entries"][0][2][0].__setitem__(1, 4),
    lambda d: d["maps"][3]["matrix"].update(rows=2),
    lambda d: d["path_map"]["entries"][0][2][0][0].__setitem__(1, 1),
    lambda d: d["path_map"]["entries"][0][2][0].__setitem__(1, 2),
    lambda d: d["path_map"]["entries"].append(deepcopy(d["path_map"]["entries"][0])),
])
def test_mutations_of_source_degree_path_exterior_or_format_are_rejected(change):
    data = _result().certificate()
    change(data)
    report = verify_koszul_comparison(data)
    assert not report.valid, data


def test_mutated_result_fields_are_not_hidden_by_a_saved_certificate():
    result = _result()
    assert verify_koszul_comparison(result).valid
    result.maps[3][0, 0] = result.maps[3][0, 0] + 1
    assert not verify_koszul_comparison(result).valid


def test_resource_exhaustion_propagates_and_bad_input_returns_invalid():
    data = _result().certificate()
    for kwargs in ({"max_terms": 10}, {"max_entries": 10}, {"max_products": 10}):
        with pytest.raises(ResourceLimit):
            verify_koszul_comparison(data, **kwargs)
    malicious = deepcopy(data)
    malicious["basis"][0][0] = 10 ** 100
    with pytest.raises(ResourceLimit, match="max_terms"):
        verify_koszul_comparison(malicious, max_terms=1000)
    for value in (None, [], "not a certificate", {}):
        assert not verify_koszul_comparison(value).valid


def test_wrong_chain_identity_detected_even_when_exterior_computation_is_tampered(monkeypatch):
    # Check the actual chain equation independently of the minor engine.
    module = importlib.import_module("css_decoupling.certificates.koszul_comparison")
    result = koszul_comparison(Ring(3, 2), [[1, 0], [0, 1]], degrees=(2,))
    result.maps[2][0, 0] += result.ring.variable(0) - 1  # same augmentation
    monkeypatch.setattr(module, "_bareiss", lambda rows, ring, limits: ring.variable(0))
    report = verify_koszul_comparison(result)
    assert not report.valid and "Chain equation" in report.errors[0]
