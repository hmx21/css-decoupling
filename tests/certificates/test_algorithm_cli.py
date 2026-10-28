import json

from css_decoupling import Ring, save_input
from css_decoupling.general_cli import main
from css_decoupling.topology.complexes import koszul


def test_separate_algorithms_cli_and_independent_verification(tmp_path, capsys):
    C = koszul(Ring(3, 2), 1)
    inp, coarse, circuit = (tmp_path / name for name in ("input.json", "coarse.json", "circuit.json"))
    save_input(C.d(0), C.d(-1).dagger(), inp)
    assert main(["coarse", str(inp), "--output", str(coarse)]) == 0
    assert json.loads(capsys.readouterr().out)["sufficient_index"] == 1
    assert main(["verify", str(coarse)]) == 0
    assert json.loads(capsys.readouterr().out)["valid"]
    assert main(["construct", str(coarse), "--output", str(circuit)]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["input_size"]["dense_coefficients"] > 0
    assert data["resources"]["qudits_per_cell"] > 0
    assert main(["verify", str(circuit)]) == 0
    assert json.loads(capsys.readouterr().out)["valid"]


def test_strict_construct_cli_resource_failure_does_not_write_certificate(tmp_path, capsys):
    C = koszul(Ring(3, 2), 1)
    inp, coarse, out = (tmp_path / name for name in ("input.json", "coarse.json", "failed.json"))
    save_input(C.d(0), C.d(-1).dagger(), inp)
    assert main(["coarse", str(inp), "--output", str(coarse)]) == 0
    capsys.readouterr()
    assert main(["construct", str(coarse), "--linear-backend", "bounded",
                 "--bounded-max-degree", "0", "--output", str(out)]) == 2
    assert json.loads(capsys.readouterr().err)["error"] == "ResourceLimit"
    assert not out.exists()
