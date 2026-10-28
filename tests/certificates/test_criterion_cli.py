"""Public command integration: mathematical negatives are certified results."""
import json

import pytest

from css_decoupling import Ring, Matrix, analyze_profile, save_analysis, save_input
from css_decoupling.general_cli import main
from css_decoupling.models.examples import glued_css, glued_toric_chain


@pytest.mark.parametrize("p", [2, 3, 5])
def test_lattice_check_raw_input_positive_and_negative(tmp_path, capsys, p):
    path, output = tmp_path / "input.json", tmp_path / "criterion.json"
    save_input(*glued_css(Ring(p, 3)), path)
    for factor, decouplable in [(1, False), (p, True)]:
        basis = json.dumps([[1, 0, 0], [0, factor, 0], [0, 0, 1]])
        assert main(["lattice-check", str(path), "--basis", basis,
                     "--output", str(output)]) == 0
        result = json.loads(capsys.readouterr().out)
        assert result["decouplable"] is decouplable
        assert result["reason"] == ("split-profile" if decouplable else "nonzero-gluing")
        data = json.loads(output.read_text())
        assert data["schema"] == "css-decoupling.lattice-criterion"
        assert main(["verify", str(output)]) == 0
        assert json.loads(capsys.readouterr().out)["valid"]


def test_lattice_check_multilayer_analysis_dispatch(tmp_path, capsys):
    C = glued_toric_chain(Ring(2, 4), [1, 2, 3])
    analysis = analyze_profile(C.d(0), C.d(-1).dagger())
    path, output = tmp_path / "analysis.json", tmp_path / "criterion.json"
    save_analysis(analysis, path)
    for basis, decouplable in [([[int(i == j) for j in range(4)] for i in range(4)], False),
                               (analysis.lattice.basis, True)]:
        assert main(["lattice-check", str(path), "--basis", json.dumps(basis),
                     "--output", str(output)]) == 0
        assert json.loads(capsys.readouterr().out)["decouplable"] is decouplable
        assert main(["verify", str(output)]) == 0
        assert json.loads(capsys.readouterr().out)["valid"]


def test_criterion_resource_failure_and_large_translation_obstruction(tmp_path, capsys):
    path, output = tmp_path / "code.json", tmp_path / "result.json"
    save_input(*glued_css(Ring(3, 3)), path)
    assert main(["lattice-check", str(path), "--basis", "[[1,0,0],[0,3,0],[0,0,1]]",
                 "--output", str(output), "--max-cell-index", "1"]) == 2
    assert json.loads(capsys.readouterr().err)["error"] == "ResourceLimit"
    assert not output.exists()
    ring = Ring(3, 2)
    x, y = ring.variables
    HX = Matrix.from_rows(ring, [[x + 1, y + 1]])
    HZ = Matrix.from_rows(ring, [[-y - 1], [x + 1]]).dagger()
    save_input(HX, HZ, path)
    huge = json.dumps([[10**40 + 1, 0], [0, 1]])
    assert main(["lattice-check", str(path), "--basis", huge, "--output", str(output),
                 "--max-cell-index", "1"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["reason"] == "nontrivial-layer-translation" and not result["decouplable"]
    assert main(["verify", str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["valid"]


def test_koszul_map_cli_and_resource_failure(tmp_path, capsys):
    output = tmp_path / "chi.json"
    args = ["koszul-map", "--prime", "5", "--dimension", "3",
            "--basis", "[[1,-1,0],[0,2,1],[0,0,1]]", "--degrees", "2",
            "--output", str(output)]
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out)["materialized_degrees"] == [0, 1, 2]
    assert main(["verify", str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["valid"]
    unchanged = output.read_bytes()
    assert main(["verify", str(output), "--max-entries", "1"]) == 2
    assert json.loads(capsys.readouterr().err)["error"] == "ResourceLimit"
    assert output.read_bytes() == unchanged
    limited = tmp_path / "limited.json"
    assert main([*args[:-1], str(limited), "--max-entries", "1"]) == 2
    assert json.loads(capsys.readouterr().err)["error"] == "ResourceLimit"
    assert not limited.exists()


def test_torus_cli_reports_single_layer_losses_without_volume_expansion(tmp_path, capsys):
    ring = Ring(3, 2)
    x, y = ring.variables
    HX = Matrix.from_rows(ring, [[x + 1, y + 1]])
    HZ = Matrix.from_rows(ring, [[-y - 1], [x + 1]]).dagger()
    path, evidence, output = tmp_path / "input.json", tmp_path / "analysis.json", tmp_path / "torus.json"
    save_input(HX, HZ, path)
    save_analysis(analyze_profile(HX, HZ), evidence)
    assert main(["torus", str(path), "--analysis", str(evidence),
                 "--basis", json.dumps([[10**30 + 1, 0], [0, 1]]),
                 "--max-qudits", "1", "--output", str(output)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["method"] == "single-layer" and result["logical_qudits"] == 0
    assert (result["translation_loss"], result["gluing_loss"]) == (2, 0)
    assert len(result["layer_contributions"]) == 1
    assert main(["verify", str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["valid"]
