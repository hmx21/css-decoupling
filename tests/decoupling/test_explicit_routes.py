"""Independent coarse-graining/construction selections retain complete evidence."""
from copy import deepcopy
import json

import pytest

from css_decoupling import Ring, decouple_general as decouple, decouple_complex
from css_decoupling.certificates.coarse_graining import verify_coarse_graining
from css_decoupling.certificates.construction import verify_construction
from css_decoupling.certificates.decoupling import verify_decoupling
from css_decoupling.certificates.io import save_input
from css_decoupling.errors import ResourceLimit
from css_decoupling.models.examples import glued_css
from css_decoupling.topology.complexes import direct_sum_complexes, disk, koszul


@pytest.mark.parametrize("dimension", (2, 3, 4, 5))
@pytest.mark.parametrize("prime", (2, 3, 5))
def test_sections_keep_nonzero_Q_signed_disks_and_serialized_frame(dimension, prime):
    R = Ring(prime, dimension)
    C = direct_sum_complexes(disk(R, -1), disk(R, 2))
    result = decouple_complex(C, contraction_strategy="sections")
    assert result.contraction_strategy == "sections"
    assert result.comparison["schanuel"]["idempotent"]
    data = json.loads(json.dumps(result.certificate()))
    assert verify_construction(data).valid
    bad = deepcopy(data)
    bad["decoupling"]["identity_checks"]["source"] += 1
    assert not verify_construction(bad).valid


@pytest.mark.parametrize("coarse", ("adaptive", "uniform"))
@pytest.mark.parametrize("strategy", ("normalized", "sections"))
@pytest.mark.parametrize("sector", ("X", "Z"))
def test_raw_routes_are_independent_and_dual_preserves_their_evidence(coarse, strategy, sector):
    C = koszul(Ring(3, 2), 1)
    result = decouple(C.d(0), C.d(-1).dagger(), backend="sympy", sector=sector,
                      coarse_graining=coarse, contraction_strategy=strategy)
    trace = result.trace.get("inner", result.trace)
    assert trace["coarse_graining_mode"] == coarse
    assert trace["contraction_strategy"] == strategy
    assert verify_decoupling(json.loads(json.dumps(result.certificate()))).valid
    assert verify_construction(trace["construction"]).valid
    if coarse == "uniform":
        origin = result.coarse_graining_result
        assert origin.sector == sector
        assert origin.HX == C.d(0) and origin.HZ == C.d(-1).dagger()
        assert verify_coarse_graining(json.loads(json.dumps(origin.certificate()))).valid
    else:
        assert result.coarse_graining_result is None
        assert result.profile_analysis is not None


def test_uniform_gluing_uses_sufficient_formula_without_adaptive_analysis(monkeypatch):
    import css_decoupling.decoupling.pipeline as pipeline
    HX, HZ = glued_css(Ring(2, 3))
    adaptive = decouple(HX, HZ)
    monkeypatch.setattr(pipeline, "analyze_profile", lambda *a, **k: pytest.fail("uniform ran adaptive analysis"))
    uniform = decouple(HX, HZ, coarse_graining="uniform")
    origin = uniform.coarse_graining_result
    assert adaptive.lattice.index == 2 and uniform.lattice.index == 8
    assert uniform.multiplicities == adaptive.multiplicities == {1: 1, 2: 1}
    assert origin.lattice.index == origin.translation_kernel.index * 2 ** 3
    assert verify_coarse_graining(origin).valid
    assert verify_construction(uniform.trace["construction"]).valid
    assert verify_decoupling(uniform).valid
    with pytest.raises(AttributeError, match="coarse_graining_result"):
        _ = uniform.profile_analysis


@pytest.mark.parametrize("options", [
    {"coarse_graining": "minimum"},
    {"coarse_graining": "uniform", "lattice_basis": [[1, 0], [0, 1]]},
    {"coarse_graining": "uniform", "lattice_strategy": "optimal"},
    {"coarse_graining": "uniform", "comparison_method": "free-basis"},
    {"contraction_strategy": "paper"},
    {"contraction_strategy": "normalized", "structural_search": True},
])
def test_conflicting_options_reject_before_raw_analysis(options, monkeypatch):
    import css_decoupling.decoupling.pipeline as pipeline
    monkeypatch.setattr(pipeline, "analyze_profile", lambda *a, **k: pytest.fail("invalid policy reached algebra"))
    C = koszul(Ring(3, 2), 1)
    with pytest.raises(ValueError):
        decouple(C.d(0), C.d(-1).dagger(), **options)


@pytest.mark.parametrize("sector", ("X", "Z"))
def test_budget_stop_preserves_uniform_source_and_does_not_fallback(sector):
    C = koszul(Ring(3, 2), 1)
    with pytest.raises(ResourceLimit) as caught:
        decouple(C.d(0), C.d(-1).dagger(), sector=sector, coarse_graining="uniform",
                  contraction_strategy="sections", max_unknowns=0)
    origin = caught.value.coarse_graining_result
    assert origin.decouplable and origin.sector == sector
    assert verify_coarse_graining(origin).valid


def test_bounded_sections_keep_the_strict_backend(monkeypatch):
    from css_decoupling.algebra.modules import GeneralBackend
    def forbidden(*args, **kwargs):
        pytest.fail("strict sections construction called Groebner backend")
    for name in ("solve", "kernel", "quotient_groebner", "reduce_quotient"):
        monkeypatch.setattr(GeneralBackend, name, forbidden)
    R = Ring(3, 2)
    C = direct_sum_complexes(disk(R, -1), disk(R, 1))
    result = decouple_complex(C, linear_backend="bounded", contraction_strategy="sections")
    assert verify_construction(result, linear_backend="bounded").valid


def test_uniform_sections_cli_writes_actual_coarse_source_and_resource_failure(tmp_path, capsys):
    from css_decoupling.general_cli import main
    C = koszul(Ring(3, 2), 1)
    source, output = tmp_path / "input.json", tmp_path / "output.json"
    save_input(C.d(0), C.d(-1).dagger(), source)
    args = ["run", str(source), "--output", str(output), "--coarse-graining", "uniform",
            "--contraction-strategy", "sections", "--sector", "Z"]
    assert main(args) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["coarse_graining"] == "uniform" and summary["contraction_strategy"] == "sections"
    assert summary["analysis_certificate"] is None
    origin = json.loads((tmp_path / "output.coarse.json").read_text(encoding="utf-8"))
    assert verify_coarse_graining(origin).valid
    assert verify_decoupling(json.loads(output.read_text(encoding="utf-8"))).valid
    failed = tmp_path / "failed.json"
    args[args.index(str(output))] = str(failed)
    assert main([*args, "--max-unknowns", "0"]) == 2
    failure = json.loads(capsys.readouterr().err)
    assert failure["error"] == "ResourceLimit" and failure["coarse_graining_certificate"]
    assert not failed.exists()
    assert verify_coarse_graining(json.loads((tmp_path / "failed.coarse.json").read_text(encoding="utf-8"))).valid


@pytest.mark.parametrize("sector", ("X", "Z"))
def test_uniform_cli_retains_pre_expansion_diagnostics(tmp_path, capsys, sector):
    from css_decoupling.general_cli import main
    source, output = tmp_path / "input.json", tmp_path / "out.json"
    save_input(*glued_css(Ring(2, 3)), source)
    assert main(["run", str(source), "--output", str(output), "--sector", sector,
                 "--coarse-graining", "uniform", "--max-cell-index", "1"]) == 2
    diagnostic = json.loads(capsys.readouterr().err)
    assert diagnostic["status"] == "incomplete"
    assert diagnostic["partial"]["sector"] == sector
    assert diagnostic["partial"]["sufficient_index"] == 8
    assert diagnostic["partial"]["restricted"] is None
    assert diagnostic["coarse_graining_certificate"] is None
    assert not output.exists()


@pytest.mark.parametrize("sector", ("X", "Z"))
def test_uniform_public_verify_cannot_skip_independent_construction_check(monkeypatch, sector):
    from css_decoupling.certificates.comparison import VerificationReport
    import css_decoupling.certificates.construction as certificate
    def reject(*args, **kwargs):
        return VerificationReport(False, (), ("injected complete certificate rejection",), "test")
    monkeypatch.setattr(certificate, "verify_construction", reject)
    C = koszul(Ring(3, 2), 1)
    with pytest.raises(ArithmeticError, match="injected complete certificate rejection"):
        decouple(C.d(0), C.d(-1).dagger(), sector=sector, coarse_graining="uniform")


@pytest.mark.parametrize("sector,where", [("X", "physical"), ("Z", "physical"), ("Z", "construction")])
def test_uniform_checker_resource_stop_keeps_completed_source(monkeypatch, sector, where):
    import css_decoupling.decoupling.pipeline as pipeline
    import css_decoupling.decoupling.dual as dual
    import css_decoupling.certificates.construction as construction
    def exhausted(*args, **kwargs):
        raise ResourceLimit("independent checker resource ceiling")
    if where == "construction":
        monkeypatch.setattr(construction, "verify_construction", exhausted)
    elif sector == "X":
        monkeypatch.setattr(pipeline, "verify_decoupling", exhausted)
    else:
        monkeypatch.setattr(dual, "verify_dual_certificate", exhausted)
    C = koszul(Ring(3, 2), 1)
    with pytest.raises(ResourceLimit) as caught:
        decouple(C.d(0), C.d(-1).dagger(), sector=sector, coarse_graining="uniform")
    assert caught.value.coarse_graining_result.sector == sector
    assert verify_coarse_graining(caught.value.coarse_graining_result).valid


def test_cli_does_not_label_an_exception_without_a_certificate_as_certified(tmp_path, capsys, monkeypatch):
    import css_decoupling.general_cli as cli
    from css_decoupling.errors import InfiniteLayer
    def reported(*args, **kwargs):
        raise InfiniteLayer("reported obstruction without exported witness")
    monkeypatch.setattr(cli, "decouple", reported)
    C = koszul(Ring(3, 2), 1)
    source, output = tmp_path / "input.json", tmp_path / "out.json"
    save_input(C.d(0), C.d(-1).dagger(), source)
    assert cli.main(["run", str(source), "--output", str(output)]) == 2
    diagnostic = json.loads(capsys.readouterr().err)
    assert diagnostic["status"] == "obstruction-without-certificate"
    assert diagnostic["coarse_graining_certificate"] is None and not output.exists()


def test_removed_strategy_names_are_rejected():
    C = koszul(Ring(3, 2), 1)
    assert decouple_complex(C, contraction_strategy="sections").contraction_strategy == "sections"
    with pytest.raises(ValueError, match="auto, normalized or sections"):
        decouple_complex(C, contraction_strategy="direct")
    with pytest.raises(ValueError, match="was removed"):
        decouple_complex(C, contraction_strategy="paper")
    with pytest.raises(TypeError):
        decouple(C.d(0), C.d(-1).dagger(), paper_options={})
