"""Literature fidelity, finite quotients, and infinite-lattice TT checks."""
import shutil
import subprocess

import pytest

from css_decoupling.algebra.laurent import Matrix, Ring, eye, zeros
from css_decoupling.certificates.criterion import verify_lattice_criterion
from css_decoupling.decoupling.criterion import check_lattice
from css_decoupling.errors import InfiniteLayer, NoLaurentSolution, ResourceLimit
from css_decoupling.lattice.kernel import translation_kernel
from css_decoupling.models.tricycle import (
    TRICYCLE_CASES, tricycle_case, tricycle_complex, tricycle_css,
)
from css_decoupling.topology.cohomology import FiniteQuotient
from css_decoupling.topology.periodic import periodic_code
from css_decoupling.topology.resolutions import css_complex


@pytest.fixture(autouse=True)
def forbid_external_algebra(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("TT model tests must run without external algebra processes")
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(shutil, "which", forbidden)


def diagonal(values):
    return [[value if i == j else 0 for j in range(len(values))]
            for i, value in enumerate(values)]


def test_table_one_polynomials_and_paper_check_order():
    ring = Ring(2, 3)
    x, y, z = ring.variables
    literal = {
        "tt72": (1 + y + x*y**2, 1 + y*z + x**2*y**2, 1 + x*y**2*z + x**2*y),
        "tt180": (1 + x**2*y**3*z + x**4*y, 1 + x**3 + x**4*z**2,
                  1 + x**3*y**3 + x**4*y*z**2),
        "tt432": (1 + x*y*z**3 + x**3*y**4*z**2,
                  1 + x**3*y*z**2 + x**3*y**2*z**3,
                  1 + x**4*y**3*z**3 + x**5*z**2),
    }
    for name, (a, b, c) in literal.items():
        assert TRICYCLE_CASES[name].polynomials(ring) == (a, b, c)
        HX, HZ = tricycle_case(name, ring=ring)
        assert HX == Matrix.from_rows(ring, [[a, b, c]])
        assert HZ.dagger() == Matrix.from_rows(ring, [[0, c, b], [c, 0, a], [b, a, 0]])
        full = tricycle_complex(ring, (a, b, c))
        assert full.d(-1) * full.d(-2) == zeros(ring, 3, 1)
        assert HX * HZ.dagger() == zeros(ring, 1, 3)


@pytest.mark.parametrize("name", tuple(TRICYCLE_CASES))
def test_published_finite_n_and_k_are_recomputed_from_raw_checks(name):
    case = TRICYCLE_CASES[name]
    result = periodic_code(*case.checks(), diagonal(case.periodic_lengths))
    assert (result.physical_qudits, result.logical_qudits) == case.published_nkd[:2]
    # No distance search, no infinite-lattice conclusion from this finite count.
    assert result.rank_x + result.rank_z + result.logical_qudits == result.physical_qudits


def test_tt72_has_actual_nontrivial_translation_and_complete_index_three_kernel():
    HX, HZ = tricycle_case("tt72")
    quotient = FiniteQuotient(HX)
    assert quotient.dimension == 2
    identity = eye(HX.ring, 2)
    tx, ty, tz = quotient.actions
    assert tx == tz == identity
    assert ty != identity and ty * ty * ty == identity
    kernel = translation_kernel({1: quotient}, 3, max_index=3)
    assert kernel.index == 3
    assert kernel.basis == ((1, 0, 0), (0, 3, 0), (0, 0, 1))
    no = check_lattice(HX, HZ, diagonal([1, 1, 1]), backend="sympy")
    assert no.reason == "nontrivial-layer-translation"
    assert no.multiplicities == {1: 2}
    assert verify_lattice_criterion(no, backend="sympy").valid
    with pytest.raises(ResourceLimit):
        translation_kernel({1: quotient}, 3, max_index=2)


def test_a_commuting_tricycle_triple_need_not_be_topological_or_finite():
    ring = Ring(2, 3)
    x, y, _ = ring.variables
    # Synthetic degenerate triple, deliberately not a published TT instance.
    HX, HZ = tricycle_css(ring, (x + 1, y + 1, 0))
    assert not HX * HZ.dagger()
    with pytest.raises(InfiniteLayer):
        FiniteQuotient(HX)
    with pytest.raises(NoLaurentSolution):
        css_complex(HX, HZ, resolution_method="native")


@pytest.mark.parametrize("ring", [Ring(3, 3), Ring(2, 2), Ring(2, 4)])
def test_published_tt_constructor_does_not_claim_prime_qudit_generalization(ring):
    with pytest.raises(ValueError, match="Ring"):
        tricycle_case("tt72", ring=ring)


def test_bad_name_and_polynomial_count():
    with pytest.raises(ValueError, match="Unknown TT"):
        tricycle_case("not-a-literature-example")
    with pytest.raises(ValueError, match="exactly three"):
        tricycle_css(Ring(2, 3), (1, 1))


@pytest.mark.integration
def test_tt72_decouples_from_literal_raw_checks_with_independent_verification():
    from css_decoupling.certificates.decoupling import verify_decoupling
    from css_decoupling.decoupling.pipeline import decouple
    result = decouple(*tricycle_case("tt72"), backend="sympy",
                      resolution_method="native", max_cell_index=3,
                      resolution_timeout=120)
    assert result.multiplicities == {1: 2}
    assert result.lattice.index == 3
    report = verify_decoupling(result.certificate())
    assert report.valid, report.errors
