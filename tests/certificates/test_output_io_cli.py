"""Two physical output views, safe publication, and independent CLI re-reading."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from css_decoupling import (
    Ring, Matrix, QCA, eye, direct_sum, decouple_general as decouple, coarse_grain, decouple_complex,
    save_input, save_decoupling, save_decoupling_outputs, verify_output_pair,
)
from css_decoupling.certificates.output_io import check_new_output_paths, write_new_json_files
from css_decoupling.general_cli import main
from css_decoupling.topology.complexes import koszul


def source_result(sector="X"):
    C = koszul(Ring(3, 2), 1)
    return decouple(C.d(0), C.d(-1).dagger(), sector=sector)


@pytest.mark.parametrize("prime", (2, 3, 5))
@pytest.mark.parametrize("dimension", (2, 3, 4, 5))
def test_qca_local_action_negative_powers_pairing_and_inverse(prime, dimension):
    R = Ring(prime, dimension)
    f = R.variable(0) ** -2 + 2 * R.variable(dimension - 1)
    Z = Matrix.from_rows(R, [[1, f], [0, 1]])
    inverse = Matrix.from_rows(R, [[1, -f], [0, 1]])
    qca = QCA(Z, inverse)
    assert qca.validate()
    assert qca.matrix("X").dagger() * qca.matrix("Z") == eye(R, 2)
    assert qca.inverse().symplectic_matrix() * qca.symplectic_matrix() == eye(R, 4)
    assert qca.symplectic_matrix() == direct_sum(inverse.dagger(), Z)
    assert qca.fourier_dual().fourier_dual() == qca
    bad = QCA(Z, Z)
    if prime != 2:
        with pytest.raises(ValueError, match="two-sided"):
            bad.validate()
    with pytest.raises(ValueError, match="X or Z"):
        qca.matrix("V")


@pytest.mark.parametrize("sector", ("X", "Z"))
def test_result_views_and_three_saved_files_agree(tmp_path, sector):
    result = source_result(sector)
    assert result.qca.matrix("Z") == result.circuit.matrix("Z")
    assert result.qca.matrix("X") == result.circuit.matrix("X")
    assert result.qca.inverse().matrix("Z") == result.circuit.inverse().matrix("Z")
    paths = save_decoupling_outputs(result, tmp_path / "result.json")
    saved = {k: json.loads(p.read_text(encoding="utf-8")) for k, p in paths.items()}
    report = verify_output_pair(saved["qca"], saved["circuit"], source_certificate=saved["certificate"])
    assert report.valid, report.errors
    for path in paths.values():
        assert b"\r\n" not in path.read_bytes()


def test_algorithm_two_input_and_physical_views_keep_dual_scope_distinct():
    C = koszul(Ring(3, 2), 1)
    built = decouple_complex(coarse_grain(C.d(0), C.d(-1).dagger(), sector="Z"))
    assert built.qca.matrix("Z") == built.circuit.matrix("Z")
    assert built.physical_qca.matrix("Z") == built.physical_circuit.matrix("Z")
    assert built.physical_qca.matrix("X") == built.physical_circuit.matrix("X")
    assert built.physical_qca == built.qca.fourier_dual()


@pytest.mark.parametrize("command", ("run", "general"))
@pytest.mark.parametrize("sector", ("X", "Z"))
def test_cli_success_publishes_two_outputs_and_verifies_saved_bytes(tmp_path, capsys, command, sector):
    C = koszul(Ring(3, 2), 1)
    source, out = tmp_path / "input.json", tmp_path / "result.json"
    save_input(C.d(0), C.d(-1).dagger(), source)
    assert main([command, str(source), "--output", str(out), "--sector", sector]) == 0
    summary = json.loads(capsys.readouterr().out)
    qca, circuit = Path(summary["qca_output"]), Path(summary["circuit_output"])
    assert qca == tmp_path / "result.qca.json" and circuit == tmp_path / "result.circuit.json"
    assert main(["verify-outputs", str(out), "--qca", str(qca), "--circuit", str(circuit)]) == 0
    assert json.loads(capsys.readouterr().out)["valid"]
    assert main(["verify", str(qca), "--source-certificate", str(out)]) == 0
    capsys.readouterr()
    assert main(["verify", str(circuit), "--source-certificate", str(out)]) == 0
    capsys.readouterr()
    assert main(["verify", str(qca)]) == 2
    assert json.loads(capsys.readouterr().err)["error"] == "MissingSource"


def test_construct_cli_uses_outer_physical_output_when_coarse_source_is_dual(tmp_path, capsys):
    C = koszul(Ring(3, 2), 1)
    coarse = coarse_grain(C.d(0), C.d(-1).dagger(), sector="Z")
    source, out = tmp_path / "coarse.json", tmp_path / "construction.json"
    source.write_text(json.dumps(coarse.certificate()), encoding="utf-8")
    assert main(["construct", str(source), "--output", str(out)]) == 0
    summary = json.loads(capsys.readouterr().out)
    data = json.loads(Path(summary["qca_output"]).read_text(encoding="utf-8"))
    assert data["source"]["physical_certificate"] == "physical_decoupling"
    assert data["context"]["target"]["basis"] == "dual-of-inner-canonical-toric-stack"
    assert main(["verify-outputs", str(out), "--qca", summary["qca_output"],
                 "--circuit", summary["circuit_output"]]) == 0
    capsys.readouterr()


def test_export_existing_certificate_is_read_only_and_supports_custom_paths(tmp_path, capsys):
    source = tmp_path / "old.json"
    save_decoupling(source_result(), source)
    original = source.read_bytes()
    qca, circuit = tmp_path / "qca/view.json", tmp_path / "gates/view.json"
    assert main(["export-outputs", str(source), "--qca-output", str(qca), "--circuit-output", str(circuit)]) == 0
    capsys.readouterr()
    assert source.read_bytes() == original
    assert main(["verify-outputs", str(source), "--qca", str(qca), "--circuit", str(circuit)]) == 0
    capsys.readouterr()
    assert main(["export-outputs", str(source), "--qca-output", str(qca), "--circuit-output", str(circuit)]) == 2
    capsys.readouterr()
    assert source.read_bytes() == original


@pytest.mark.parametrize("collision", ("input", "certificate", "circuit", "stage", "existing"))
def test_cli_rejects_output_collisions_before_running_algebra(tmp_path, capsys, monkeypatch, collision):
    import css_decoupling.general_cli as cli
    C = koszul(Ring(3, 2), 1)
    source, out = tmp_path / "input.json", tmp_path / "result.json"
    save_input(C.d(0), C.d(-1).dagger(), source)
    targets = {"input": source, "certificate": out, "circuit": tmp_path / "result.circuit.json",
               "stage": tmp_path / "result.analysis.json", "existing": tmp_path / "protected.json"}
    if collision == "existing":
        targets[collision].write_bytes(b"keep these bytes")
    def forbidden(*args, **kwargs):
        pytest.fail("unsafe output path reached algebra")
    monkeypatch.setattr(cli, "decouple", forbidden)
    original = source.read_bytes()
    assert main(["run", str(source), "--output", str(out), "--qca-output", str(targets[collision])]) == 2
    capsys.readouterr()
    assert source.read_bytes() == original and not out.exists()
    if collision == "existing":
        assert targets[collision].read_bytes() == b"keep these bytes"


def test_ordinary_write_failure_rolls_back_only_this_call_files(tmp_path, monkeypatch):
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    original_open = Path.open
    def failing_open(path, *args, **kwargs):
        if path == second and args and args[0] == "x":
            raise OSError("injected disk write failure")
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", failing_open)
    with pytest.raises(OSError, match="injected"):
        write_new_json_files({"a": {"a": 1}, "b": {"b": 2}}, {"a": first, "b": second})
    assert not first.exists() and not second.exists()


def test_invalid_source_does_not_publish_success_outputs(tmp_path):
    bad = deepcopy(source_result().certificate())
    bad["identity_checks"]["source"] += 1
    with pytest.raises(ValueError, match="Source certificate"):
        save_decoupling_outputs(bad, tmp_path / "bad.json")
    assert not list(tmp_path.glob("*.json"))


def test_path_aliases_and_null_source_are_not_accepted(tmp_path, capsys):
    with pytest.raises(ValueError, match="distinct"):
        check_new_output_paths({"a": tmp_path / "same.json", "b": tmp_path / "nested/../same.json"})
    result = source_result()
    paths = save_decoupling_outputs(result, tmp_path / "result.json")
    null = tmp_path / "null.json"
    null.write_text("null", encoding="utf-8")
    assert main(["verify", str(paths["qca"]), "--source-certificate", str(null)]) == 1
    assert not json.loads(capsys.readouterr().out)["valid"]


def test_dangling_output_link_is_rejected_before_resolving_its_target(tmp_path, monkeypatch):
    link, target = tmp_path / "link.json", tmp_path / "missing-target.json"
    real_resolve, real_is_symlink = Path.resolve, Path.is_symlink
    monkeypatch.setattr(Path, "resolve", lambda self, *a, **k:
                        target if self == link else real_resolve(self, *a, **k))
    monkeypatch.setattr(Path, "is_symlink", lambda self:
                        True if self == link else real_is_symlink(self))
    with pytest.raises(FileExistsError, match="symlink"):
        check_new_output_paths({"qca": link})
    assert not target.exists()


def test_rollback_does_not_follow_a_replacement_symlink(tmp_path, monkeypatch):
    import stat
    from types import SimpleNamespace
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    real_open, real_lstat = Path.open, Path.lstat
    replaced = False
    def open_file(path, *args, **kwargs):
        nonlocal replaced
        if path == second and args and args[0] == "x":
            replaced = True
            raise OSError("injected failure after path replacement")
        return real_open(path, *args, **kwargs)
    def lstat_file(path, *args, **kwargs):
        original = real_lstat(path, *args, **kwargs)
        if path == first and replaced:
            return SimpleNamespace(st_mode=stat.S_IFLNK, st_dev=original.st_dev, st_ino=original.st_ino)
        return original
    monkeypatch.setattr(Path, "open", open_file)
    monkeypatch.setattr(Path, "lstat", lstat_file)
    with pytest.raises(OSError, match="path replacement"):
        write_new_json_files({"a": {}, "b": {}}, {"a": first, "b": second})
    assert first.exists() and not second.exists()


def test_run_export_restores_raw_backend_and_timeout_not_construction_backend(tmp_path, capsys, monkeypatch):
    from contextlib import contextmanager
    import css_decoupling.general_cli as cli
    result = source_result()
    source, output = tmp_path / "input.json", tmp_path / "out.json"
    save_input(result.HX, result.HZ, source)
    monkeypatch.setattr(cli, "decouple", lambda *a, **k: result)
    entered = []
    @contextmanager
    def selected(backend, **options):
        entered.append((backend, options.get("timeout")))
        yield
    monkeypatch.setattr(cli, "using_backend", selected)
    assert main(["run", str(source), "--output", str(output), "--backend", "sympy",
                 "--linear-backend", "bounded", "--timeout", "17"]) == 0
    capsys.readouterr()
    assert entered == [("sympy", 17.0)]


def test_bounded_construct_keeps_backend_for_composed_certificate_and_exports(tmp_path, capsys, monkeypatch):
    from css_decoupling.algebra.bounded_linear import BoundedLinearBackend
    from css_decoupling.algebra.modules import get_backend
    from css_decoupling.topology.complexes import direct_sum_complexes, disk
    import css_decoupling.certificates.construction as certificates
    R = Ring(3, 2)
    C = direct_sum_complexes(disk(R, -1), disk(R, 1))
    source = tmp_path / "coarse.json"
    source.write_text(json.dumps(coarse_grain(C.d(0), C.d(-1).dagger()).certificate()), encoding="utf-8")
    original = certificates.verify_construction
    seen = []
    def verify(data, **options):
        has_origin = (data.get("coarse_graining") if isinstance(data, dict)
                      else data.coarse_graining) is not None
        if has_origin:
            selected = options.get("linear_backend")
            assert selected == "bounded" or isinstance(get_backend(R), BoundedLinearBackend)
            seen.append(selected)
        return original(data, **options)
    monkeypatch.setattr(certificates, "verify_construction", verify)
    out = tmp_path / "result.json"
    assert main(["construct", str(source), "--output", str(out), "--linear-backend", "bounded"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert seen and Path(summary["qca_output"]).exists() and Path(summary["circuit_output"]).exists()


def test_bounded_composition_separates_original_provenance_without_skipping_it(monkeypatch):
    from css_decoupling.algebra.bounded_linear import BoundedLinearBackend
    from css_decoupling.algebra.modules import GeneralBackend, get_backend
    from css_decoupling.certificates.comparison import VerificationReport
    from css_decoupling.topology.complexes import direct_sum_complexes, disk
    import css_decoupling.certificates.construction as construction
    import css_decoupling.certificates.coarse_graining as provenance
    R = Ring(3, 2)
    raw = direct_sum_complexes(disk(R, -1), disk(R, 1))
    coarse = coarse_grain(raw.d(0), raw.d(-1).dagger())
    built = decouple_complex(coarse.restricted, linear_backend="bounded", verify=False)
    built.coarse_graining = coarse.certificate()
    data = built.certificate()
    physical, original = construction.verify_decoupling, provenance.verify_coarse_graining
    seen = []
    def check_physical(value):
        if value is data["decoupling"]:
            assert isinstance(get_backend(R), BoundedLinearBackend)
            seen.append("bounded-input-C")
        elif value is data["physical_decoupling"]:
            assert isinstance(get_backend(R), GeneralBackend)
            seen.append("native-original-code")
        else:
            pytest.fail("unexpected physical source")
        return physical(value)
    def check_origin(value):
        assert isinstance(get_backend(R), GeneralBackend)
        seen.append("native-Algorithm1")
        return original(value)
    monkeypatch.setattr(construction, "verify_decoupling", check_physical)
    monkeypatch.setattr(provenance, "verify_coarse_graining", check_origin)
    report = construction.verify_construction(data, linear_backend="bounded")
    assert report.valid, report.errors
    assert seen == ["bounded-input-C", "native-Algorithm1", "native-original-code"]
    monkeypatch.setattr(provenance, "verify_coarse_graining", lambda *a, **k:
                        VerificationReport(False, (), ("rejected original provenance",), "test"))
    assert not construction.verify_construction(data, linear_backend="bounded").valid
