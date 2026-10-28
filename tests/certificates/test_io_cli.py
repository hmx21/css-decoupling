import copy
import json

import pytest

from css_decoupling.general_cli import main
from css_decoupling import Ring, eye, zeros, decouple_general as decouple, save_input, load_decoupling
from css_decoupling.topology.complexes import koszul
from css_decoupling.certificates.io import input_dict, parse_input, save_decoupling
from css_decoupling.certificates.decoupling import verify_decoupling


@pytest.mark.parametrize("command", ["run", "general"])
def test_cli_roundtrip(tmp_path, capsys, command):
    T = koszul(Ring(3, 2), 1)
    HX, HZ = T.d(0), T.d(-1).dagger()
    inp, out = tmp_path / "input.json", tmp_path / "result.json"
    save_input(HX, HZ, inp)
    assert parse_input(json.loads(inp.read_text()))[:2] == (HX, HZ)
    assert main([command, str(inp), "--backend", "sympy", "--output", str(out)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["profile"] == {"1": 1}
    assert summary["prime"] == 3
    analysis_path = tmp_path / "result.analysis.json"
    assert summary["analysis_certificate"] == str(analysis_path)
    assert summary["Lambda0_index"] == 1
    assert main(["verify", str(analysis_path)]) == 0
    assert json.loads(capsys.readouterr().out)["valid"]
    assert load_decoupling(out)["schema"] == "css-decoupling.general.complete"
    assert main(["verify", str(out)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["valid"]
    altered = json.loads(out.read_text())
    altered["multiplicities"][0][1] = 2
    out.write_text(json.dumps(altered))
    assert main(["verify", str(out)]) == 1
    assert not json.loads(capsys.readouterr().out)["valid"]


@pytest.mark.parametrize("sector", ["X", "Z"])
def test_cli_native_analysis_exports_classes_without_external_process(tmp_path, capsys, monkeypatch, sector):
    import subprocess
    from css_decoupling.models.examples import glued_css
    def forbidden(*args, **kwargs):
        raise AssertionError("Native analysis must not launch an external process")
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    HX, HZ = glued_css(Ring(2, 3))
    inp, out = tmp_path / "input.json", tmp_path / "analysis.json"
    save_input(HX, HZ, inp)
    assert main(["analyze", str(inp), "--sector", sector, "--output", str(out)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert (summary["Lambda0_index"], summary["final_cell_index"]) == (1, 2)
    assert summary["gluing_stages"] == 2
    assert summary["sector"] == sector
    assert main(["verify", str(out)]) == 0
    assert json.loads(capsys.readouterr().out)["valid"]


@pytest.mark.parametrize("sector", ["X", "Z"])
@pytest.mark.parametrize("error_name", ["ProjectiveBasisUnavailable", "ResourceLimit"])
@pytest.mark.parametrize("method", ["free-basis", "schanuel"])
def test_complement_failure_preserves_analysis_without_changing_method(tmp_path, capsys, monkeypatch, sector, error_name, method):
    from css_decoupling.algebra.projective import ProjectiveBasisUnavailable
    from css_decoupling.errors import ResourceLimit
    from css_decoupling.decoupling import pipeline
    def unavailable(*args, **kwargs):
        error_type = {"ProjectiveBasisUnavailable": ProjectiveBasisUnavailable, "ResourceLimit": ResourceLimit}[error_name]
        raise error_type("Injected complement failure")
    if method == "free-basis":
        monkeypatch.setattr(pipeline, "stable_cokernel_isomorphism", unavailable)
    else:
        monkeypatch.setattr("css_decoupling.decoupling.comparison.paper_stable_cokernel_isomorphism", unavailable)
    T = koszul(Ring(3, 3), 1)
    inp, out = tmp_path / "input.json", tmp_path / "circuit.json"
    save_input(T.d(0), T.d(-1).dagger(), inp)
    assert main(["run", str(inp), "--sector", sector, "--comparison-method", method, "--output", str(out)]) == 2
    error = json.loads(capsys.readouterr().err)
    assert error["error"] == error_name
    assert not out.exists()
    analysis_path = tmp_path / "circuit.analysis.json"
    assert json.loads(analysis_path.read_text())["sector"] == sector
    assert main(["verify", str(analysis_path)]) == 0
    assert json.loads(capsys.readouterr().out)["valid"]


def test_strict_input_and_complete_certificate(tmp_path):
    T = koszul(Ring(5, 2), 1)
    data = input_dict(T.d(0), T.d(-1).dagger())
    invalid = copy.deepcopy(data)
    invalid["ring"]["prime"] = True
    with pytest.raises(ValueError):
        parse_input(invalid)
    invalid = copy.deepcopy(data)
    invalid["HX"]["entries"].append(invalid["HX"]["entries"][0])
    with pytest.raises(ValueError):
        parse_input(invalid)
    result = decouple(T.d(0), T.d(-1).dagger())
    path = save_decoupling(result, tmp_path / "nested" / "certificate.json")
    assert verify_decoupling(load_decoupling(path)).valid
    for mutate in (
        lambda d: d["identity_checks"].update(source=True),
        lambda d: d["lattice"].update(index=True),
        lambda d: d["input"]["HZ"]["entries"].clear(),
        lambda d: d["lattice"]["basis_columns"][0].__setitem__(0, 2),
    ):
        broken = copy.deepcopy(result.certificate())
        mutate(broken)
        assert not verify_decoupling(broken).valid


@pytest.mark.parametrize("x_fixed,z_fixed", [(2, 0), (0, 2), (0, 0)])
def test_zero_layer_product_states(x_fixed, z_fixed):
    ring = Ring(3, 4)
    HX = eye(ring, x_fixed).hstack(zeros(ring, x_fixed, z_fixed))
    HZ = zeros(ring, z_fixed, x_fixed).hstack(eye(ring, z_fixed))
    result = decouple(HX, HZ)
    assert result.multiplicities == {}
    assert verify_decoupling(result.certificate()).valid


def test_multiple_copies_and_interlayer_physical_mixing():
    from css_decoupling.models.examples import toric_css, shear_css
    ring = Ring(5, 3)
    HX, HZ = toric_css(ring, {1: 2, 2: 1})
    HX, HZ = shear_css(HX, HZ, 1, 7, 2 * ring.variable(0) ** -2 + ring.variable(2))
    result = decouple(HX, HZ, backend="sympy", resolution_timeout=20)
    assert result.multiplicities == {1: 2, 2: 1}
    assert result.lattice.index == 1
