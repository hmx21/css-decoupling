"""Certificate-selected lattice acceleration must preserve every equation."""
from copy import deepcopy
from unittest.mock import patch

import pytest

from css_decoupling import decouple_general as decouple
from css_decoupling.algebra.bounded_linear import BoundedLinearBackend
from css_decoupling.algebra.laurent import Ring, eye, zeros
from css_decoupling.algebra.modules import GeneralBackend, using_backend
from css_decoupling.certificates.decoupling import verify_decoupling
from css_decoupling.errors import ResourceLimit
from css_decoupling.lattice.retained import Lattice
from css_decoupling.lattice.restriction_backend import RestrictionBackend
from css_decoupling.models.examples import glued_css


@pytest.fixture(scope='module')
def coarse_result():
    result = decouple(*glued_css(Ring(2, 3)), verify=False)
    assert result.lattice.index > 1
    return result


def test_verifier_uses_certificate_lattice_but_still_checks_both_stabilizer_directions(coarse_result):
    data = deepcopy(coarse_result.certificate())
    selected, equations = [], []

    def accelerator(base, lattice):
        backend = RestrictionBackend(base, lattice)
        selected.append(backend)
        original = backend.solve

        def checked(A, B):
            X = original(A, B)
            assert A * X == B
            equations.append((A.shape, B.shape))
            return X

        backend.solve = checked
        return backend

    with using_backend(GeneralBackend(coarse_result.HX.ring)), patch(
            'css_decoupling.lattice.restriction_backend.RestrictionBackend', side_effect=accelerator):
        report = verify_decoupling(data)
    assert report.valid, report.errors
    assert len(selected) == 1
    assert selected[0].lattice.basis == coarse_result.lattice.basis
    assert len(equations) >= 4  # two original exactness equations and both Z-module inclusions
    assert 'original CSS commutation and both exactness conditions' in report.checks
    assert 'Z stabilizer generators mapped onto one another in both directions' in report.checks


def test_consistent_but_different_lattice_metadata_cannot_rebind_old_matrices(coarse_result):
    data = deepcopy(coarse_result.certificate())
    basis = [list(row) for row in coarse_result.lattice.basis]
    for row in basis:
        row[0] *= -1
    changed = Lattice(basis)
    assert changed.index == coarse_result.lattice.index
    assert changed.restrict(coarse_result.HX) != coarse_result.lattice.restrict(coarse_result.HX)
    data['lattice'] = changed.metadata()
    report = verify_decoupling(data)
    assert not report.valid
    assert any('Source differential' in error for error in report.errors)


def test_unknown_custom_backend_is_preserved_by_the_verifier(coarse_result):
    class Custom(GeneralBackend):
        def __init__(self, ring):
            super().__init__(ring)
            self.calls = []

        def kernel(self, A):
            self.calls.append('kernel')
            return super().kernel(A)

        def solve(self, A, B):
            self.calls.append('solve')
            return super().solve(A, B)

    backend = Custom(coarse_result.HX.ring)
    with using_backend(backend), patch(
            'css_decoupling.lattice.restriction_backend.RestrictionBackend',
            side_effect=AssertionError('Custom algebra semantics must not be replaced')):
        report = verify_decoupling(coarse_result.certificate())
    assert report.valid, report.errors
    assert backend.calls.count('kernel') >= 2
    assert backend.calls.count('solve') >= 4


def test_strict_bounded_backend_is_not_replaced_for_a_coarse_certificate():
    R = Ring(3, 2)
    result = decouple(eye(R, 1), zeros(R, 0, 1),
                      lattice_basis=[[2, 0], [0, 1]], verify=False)
    with using_backend(BoundedLinearBackend(R)), patch(
            'css_decoupling.lattice.restriction_backend.RestrictionBackend',
            side_effect=AssertionError('Bounded arithmetic must remain selected')):
        report = verify_decoupling(result.certificate())
    assert report.valid, report.errors


def test_verifier_budget_failure_is_not_an_invalid_certificate_report(coarse_result):
    class Limited(GeneralBackend):
        def kernel(self, A):
            raise ResourceLimit('independent verification budget')

    with using_backend(Limited(coarse_result.HX.ring)):
        with pytest.raises(ResourceLimit, match='independent verification budget'):
            verify_decoupling(coarse_result.certificate())
