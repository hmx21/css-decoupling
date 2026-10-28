import json

import pytest

from css_decoupling import Ring, analyze_profile, save_analysis, save_input
from css_decoupling.general_cli import main
from css_decoupling.models.examples import glued_css


def test_cli_lattice_query_and_periodic_certificates(tmp_path, capsys):
    HX, HZ = glued_css(Ring(3, 3))
    inp, evidence = tmp_path / "code.json", tmp_path / "analysis.json"
    save_input(HX, HZ, inp)
    save_analysis(analyze_profile(HX, HZ), evidence)
    candidate = '[[1,0,0],[0,3,0],[0,0,1]]'
    query, torus = tmp_path / "query.json", tmp_path / "torus.json"
    assert main(["lattice-check", str(evidence), "--basis", candidate, "--output", str(query)]) == 0
    assert json.loads(capsys.readouterr().out)["decouplable"]
    assert main(["verify", str(query)]) == 0
    assert json.loads(capsys.readouterr().out)["valid"]
    for extra, method in (([], "finite"), (["--analysis", str(evidence)], "two-layer")):
        assert main(["torus", str(inp), "--basis", candidate, "--output", str(torus), *extra]) == 0
        summary = json.loads(capsys.readouterr().out)
        assert summary["logical_qudits"] == 6 and summary["method"] == method
        assert summary["decouplable"] == (True if extra else None)
        assert main(["verify", str(torus)]) == 0
        assert json.loads(capsys.readouterr().out)["valid"]
    assert main(["torus", str(inp), "--basis", candidate, "--output", str(torus),
                 "--analysis", str(evidence), "--finite"]) == 0
    assert json.loads(capsys.readouterr().out)["method"] == "finite"


def test_periodic_cli_resource_failure_does_not_write_success(tmp_path, capsys):
    inp, out = tmp_path / "input.json", tmp_path / "result.json"
    save_input(*glued_css(Ring(3, 3)), inp)
    assert main(["torus", str(inp), "--basis", '[[1000000,0,0],[0,1,0],[0,0,1]]',
                 "--output", str(out), "--max-qudits", "100"]) == 2
    assert json.loads(capsys.readouterr().err)["error"] == "ResourceLimit"
    assert not out.exists()


def test_periodic_cli_rejects_evidence_from_another_code(tmp_path):
    HX, HZ = glued_css(Ring(2, 3))
    inp, evidence = tmp_path / "code.json", tmp_path / "wrong-analysis.json"
    save_input(HX, HZ, inp)
    zero = glued_css(Ring(2, 3), coefficient=0)
    save_analysis(analyze_profile(*zero), evidence)
    with pytest.raises(ValueError, match="different CSS"):
        main(["torus", str(inp), "--basis", '[[1,0,0],[0,1,0],[0,0,1]]',
              "--analysis", str(evidence), "--output", str(tmp_path / "bad.json")])


def test_periodic_verifies_current_result_claims_not_a_stale_snapshot():
    from css_decoupling import periodic_code, verify_periodic
    result = periodic_code(*glued_css(Ring(3, 3)), [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert verify_periodic(result).valid
    result.logical_qudits = 999
    assert result.certificate()["count"]["logical_qudits"] == 999
    assert not verify_periodic(result).valid
