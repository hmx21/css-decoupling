"""The manuscript's Sec. V claims, recomputed by examples/paper_section_v.py."""
import importlib.util
from pathlib import Path

import pytest

from css_decoupling import Ring
from css_decoupling.models.examples import glued_css


def _example():
    path = Path(__file__).resolve().parents[2] / "examples" / "paper_section_v.py"
    spec = importlib.util.spec_from_file_location("paper_section_v", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("p", [2, 3, 5])
def test_equation_sixteen_is_the_glued_model(p):
    R = Ring(p, 3)
    assert _example().paper_glued(R) == glued_css(R)


@pytest.mark.parametrize("p,uniform", [(2, True), (3, False)])
def test_every_section_v_claim_holds(p, uniform):
    report = _example().reproduce(p, uniform=uniform, torus_sizes=range(1, 4))
    failed = [claim for claim, ok in report["claims"].items() if not ok]
    assert not failed
    assert report["circuits"]["adaptive"]["lattice_columns"] == [[1, 0, 0], [0, p, 0], [0, 0, 1]]
