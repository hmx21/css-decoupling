"""Exercise deliverable certificates and native-runtime guards on real inputs."""
import importlib.util
import json
import gzip
import hashlib
from pathlib import Path
import subprocess

import pytest

from css_decoupling import verify_decoupling, verify_coarse_graining


@pytest.fixture(scope="module")
def benchmark():
    path = Path(__file__).resolve().parents[1] / "support" / "native_examples.py"
    spec = importlib.util.spec_from_file_location("native_examples_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_raw_odd_prime_circuit_deliverable_without_external_programs(benchmark, tmp_path):
    benchmark.worker("toric-d4-p3-degree2", "decouple", tmp_path, 16, 30)
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["status"] == "verified"
    assert result["multiplicities"] == {"2": 1}
    assert result["native_guard"] == {"subprocess_attempts": 0, "executable_discovery_attempts": 0}
    assert verify_decoupling(json.loads((tmp_path / "certificate.json").read_text())).valid


def test_infinite_layer_is_a_verifiable_negative_not_a_timeout(benchmark, tmp_path):
    benchmark.worker("haah", "coarse", tmp_path, 16, 30)
    row = json.loads((tmp_path / "result.json").read_text())
    assert row["status"] == "certified-obstruction" and row["reason"] == "infinite-layer"
    assert verify_coarse_graining(json.loads((tmp_path / "certificate.json").read_text())).valid


def test_cell_budget_does_not_write_a_success_certificate(benchmark, tmp_path):
    benchmark.worker("glued-d3-p3", "coarse", tmp_path, 1, 30)
    row = json.loads((tmp_path / "result.json").read_text())
    assert row["status"] == "incomplete" and row["reason"] == "resource-limit"
    assert not (tmp_path / "certificate.json").exists()
    partial = json.loads((tmp_path / "partial.json").read_text())
    assert partial["sufficient_index"] == 27 and partial["restricted"] is None


def test_external_attempt_cannot_be_suppressed(benchmark):
    with pytest.raises(RuntimeError, match="suppressed"):
        with benchmark.native_only():
            with pytest.raises(RuntimeError, match="external program"):
                subprocess.Popen(["unavailable-cas"])


def test_saved_evidence_is_not_overwritten(benchmark, tmp_path):
    saved = tmp_path / "certificate.json"
    saved.write_text("protected evidence")
    with pytest.raises(FileExistsError):
        benchmark.run(tmp_path, ["haah"], ["coarse"], 10, 16)
    assert saved.read_text() == "protected evidence"
    with pytest.raises(FileExistsError):
        benchmark.worker("haah", "coarse", tmp_path, 16, 10)
    assert saved.read_text() == "protected evidence"


def test_compressed_certificate_preserves_verified_json(benchmark, tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, "COMPRESS_CERTIFICATE_BYTES", 1)
    benchmark.worker("toric-d4-p3-degree2", "decouple", tmp_path, 16, 30,
                     construction_certificate=True)
    row = json.loads((tmp_path / "result.json").read_text())
    certificate = row["certificate"]
    assert row["status"] == "verified" and certificate["encoding"] == "gzip"
    saved = (tmp_path / certificate["file"]).read_bytes()
    decoded = gzip.decompress(saved)
    assert certificate["sha256"] == hashlib.sha256(saved).hexdigest()
    assert certificate["canonical_json_sha256"] == hashlib.sha256(decoded).hexdigest()
    assert verify_decoupling(json.loads(decoded)).valid
    assert row["construction_certificate"]["scope"] == "input-complex-to-circuit-Algorithm-2"
    assert (tmp_path / row["construction_certificate"]["file"]).exists()
