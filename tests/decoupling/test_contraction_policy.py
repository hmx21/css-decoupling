"""Public policy selection retains full source/circuit acceptance and isolation."""
from copy import deepcopy
import json

import pytest

from css_decoupling.algebra.laurent import Ring
from css_decoupling.certificates.construction import verify_construction
from css_decoupling.decoupling.construction import decouple_complex
from css_decoupling.decoupling.pipeline import decouple
from css_decoupling.errors import ResourceLimit
from css_decoupling.topology.complexes import direct_sum_complexes, disk, koszul
from css_decoupling.topology.direct_splitting import structural_sections


@pytest.mark.parametrize("prime", (2, 3, 5))
@pytest.mark.parametrize("dimension", (2, 3, 4, 5))
def test_native_default_and_sections_keep_source_profile_and_full_serialized_checks(prime, dimension):
    C = koszul(Ring(prime, dimension), 1)
    normalized = decouple_complex(C)
    direct = decouple_complex(C, contraction_strategy="sections")
    assert normalized.contraction_strategy == "normalized"
    assert direct.contraction_strategy == "sections"
    assert not normalized.structural_search and not direct.structural_search
    assert normalized.multiplicities == direct.multiplicities == {1: 1}
    assert normalized.complex == direct.complex == C
    assert normalized.decoupling.free_C == direct.decoupling.free_C
    assert normalized.decoupling.free_T == direct.decoupling.free_T
    for result in (normalized, direct):
        data = json.loads(json.dumps(result.certificate()))
        assert data["configuration"] == {"linear_backend": "sympy"}  # v1 stays readable.
        report = verify_construction(data)
        assert report.valid, report.errors
        bad = deepcopy(data)
        bad["decoupling"]["identity_checks"]["source"] += 1
        assert not verify_construction(bad).valid


@pytest.mark.parametrize("options", [
    {"contraction_strategy": "model-specific"}, {"contraction_strategy": None},
    {"structural_search": 1}, {"structural_search": True},
    {"contraction_strategy": "normalized", "linear_backend": "bounded"},
    {"contraction_strategy": "normalized", "linear_backend": "singular"},
    {"contraction_strategy": "sections", "structural_search": True, "linear_backend": "bounded"},
])
def test_bad_or_unmeasured_policy_combinations_reject_before_construction(options, monkeypatch):
    import css_decoupling.decoupling.construction as module
    monkeypatch.setattr(module, "_construct", lambda *a, **k: pytest.fail("bad policy reached algebra"))
    with pytest.raises(ValueError):
        decouple_complex(koszul(Ring(3, 2), 1), **options)


def test_disabled_probes_override_and_restore_outer_context(monkeypatch):
    import css_decoupling.algebra.structural_c4 as structural
    import css_decoupling.topology.direct_splitting as direct
    monkeypatch.setattr(structural, "solve_structural_section",
                        lambda *a, **k: pytest.fail("disabled probe was invoked"))
    with structural_sections(max_terms=7):
        assert direct._structural_options.get() == {"max_terms": 7}
        result = decouple_complex(koszul(Ring(3, 2), 1), contraction_strategy="sections")
        assert verify_construction(result).valid
        assert direct._structural_options.get() == {"max_terms": 7}
    assert direct._structural_options.get() is None


def test_complete_weak_fallback_after_local_shortcut_resource_failure(monkeypatch):
    import css_decoupling.decoupling.toric_comparison as toric
    def limited(*args, **kwargs):
        raise ResourceLimit("local shortcut retention limit")
    monkeypatch.setattr(toric, "toric_comparison_inverse", limited)
    result = decouple_complex(koszul(Ring(5, 3), 1))
    assert result.contraction_strategy == "normalized"
    assert verify_construction(json.loads(json.dumps(result.certificate()))).valid


@pytest.mark.parametrize("sector", ("X", "Z"))
@pytest.mark.parametrize("strategy", ("normalized", "sections"))
def test_raw_api_propagates_policy_through_backend_and_dual_wrappers(sector, strategy):
    C = koszul(Ring(3, 2), 1)
    HX, HZ = C.d(0), C.d(-1).dagger()
    result = decouple(HX, HZ, sector=sector, backend="sympy", contraction_strategy=strategy)
    trace = result.trace.get("inner", result.trace)
    assert trace["contraction_strategy"] == strategy
    assert trace["structural_search"] is False


def test_bounded_auto_retains_sections_and_explicit_native_probe_is_available():
    C = koszul(Ring(3, 2), 1)
    bounded_source = direct_sum_complexes(disk(C.ring, -1), disk(C.ring, 1))
    bounded = decouple_complex(bounded_source, linear_backend="bounded")
    assert bounded.contraction_strategy == "sections"
    assert verify_construction(bounded, linear_backend="bounded").valid
    probed = decouple_complex(C, contraction_strategy="sections", structural_search=True)
    assert probed.structural_search is True
    assert verify_construction(probed).valid


@pytest.mark.parametrize("strategy", ("auto", "sections"))
def test_construct_cli_exposes_the_effective_policy_without_changing_certificate(tmp_path, capsys, strategy):
    from css_decoupling.general_cli import main
    from css_decoupling.certificates.cone_decoupling import _complex_dict
    C = koszul(Ring(3, 2), 1)
    source, output = tmp_path / "source.json", tmp_path / "circuit.json"
    source.write_text(json.dumps({"schema": "css-decoupling.free-complex", "version": 1,
        "ring": {"prime": 3, "dimension": 2}, "model": _complex_dict(C)}), encoding="utf-8")
    assert main(["construct", str(source), "--output", str(output),
                 "--contraction-strategy", strategy]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["contraction_strategy"] == ("normalized" if strategy == "auto" else "sections")
    assert summary["structural_search"] is False
    assert verify_construction(json.loads(output.read_text(encoding="utf-8"))).valid
