"""Serialized, independent C4 verification and adversarial witness changes."""
from copy import deepcopy
import json

import pytest

from css_decoupling.algebra.laurent import Matrix, Ring, direct_sum, eye, zeros
from css_decoupling.algebra.modules import solve
from css_decoupling.certificates.comparison import _matrix_dict, _matrix_from_dict
from css_decoupling.certificates.schanuel import schanuel_certificate_dict, verify_schanuel
from css_decoupling.decoupling.comparison import paper_stable_cokernel_isomorphism
from css_decoupling.topology.complexes import CochainMap, direct_sum_complexes, disk, koszul


@pytest.fixture
def result():
    R = Ring(3, 3)
    T, A = koszul(R, 1), disk(R, 2)
    C = direct_sum_complexes(T, A)
    phi = CochainMap(T, C, {n: eye(R, T.rank(n)).vstack(zeros(R, A.rank(n), T.rank(n)))
                            for n in C.degrees})
    psi = CochainMap(C, T, {n: eye(R, T.rank(n)).hstack(zeros(R, T.rank(n), A.rank(n)))
                            for n in C.degrees})
    hc = {n: direct_sum(zeros(R, T.rank(n - 1), T.rank(n)),
                       eye(R, 1) if n == 3 else zeros(R, A.rank(n - 1), A.rank(n)))
          for n in C.degrees}
    return paper_stable_cokernel_isomorphism(phi, psi, hc, {}, solve)


def bump_matrix(data):
    R = Ring(3, 3)
    m = _matrix_from_dict(data, R, 'test')
    assert m.nrows and m.ncols
    return _matrix_dict(m + Matrix(R, m.nrows, m.ncols, {(0, 0): R.one()}))


def bump_degree(matrices, degree=None):
    entry = next(pair for pair in matrices if (degree is None or pair[0] == degree)
                 and pair[1]['rows'] and pair[1]['cols'])
    entry[1] = bump_matrix(entry[1])


def test_json_roundtrip_and_scope(result):
    data = json.loads(json.dumps(schanuel_certificate_dict(result)))
    report = verify_schanuel(data)
    assert report.valid, report.errors
    assert report.scope == 'bounded-comparison-to-stable-check-cokernel'
    assert len(report.checks) == 6


def test_verifier_uses_no_constructor_solver_or_qs(result, monkeypatch):
    import css_decoupling.decoupling.schanuel as constructor
    import css_decoupling.decoupling.comparison as comparison
    import css_decoupling.algebra.modules as modules
    import css_decoupling.algebra.projective as projective
    import css_decoupling.algebra.resolutions as resolutions
    data = schanuel_certificate_dict(result)
    def forbidden(*args, **kwargs):
        raise AssertionError('Verifier must not call algebraic construction')
    monkeypatch.setattr(constructor, 'stable_complement_from_cone', forbidden)
    monkeypatch.setattr(comparison, 'paper_stable_cokernel_isomorphism', forbidden)
    monkeypatch.setattr(modules, 'solve', forbidden)
    monkeypatch.setattr(projective, 'free_basis', forbidden)
    monkeypatch.setattr(resolutions, 'image_resolution', forbidden)
    assert verify_schanuel(data).valid


@pytest.mark.parametrize('name', ['idempotent', 'section', 'projection', 'to_free', 'from_free'])
def test_modified_frame_rejected(result, name):
    data = schanuel_certificate_dict(result)
    data['frame'][name] = bump_matrix(data['frame'][name])
    assert not verify_schanuel(data).valid


@pytest.mark.parametrize('name', ['even', 'odd', 'free_C', 'free_T'])
def test_rank_and_ancilla_claim_tampering_rejected(result, name):
    data = schanuel_certificate_dict(result)
    data['ranks'][name] += 1
    assert not verify_schanuel(data).valid


@pytest.mark.parametrize('name', ['phi0', 'phi1', 'psi0', 'psi1', 'etaC', 'etaT', 'dC', 'dT'])
def test_modified_comparison_lifts_rejected(result, name):
    data = schanuel_certificate_dict(result)
    data['lifts'][name] = bump_matrix(data['lifts'][name])
    assert not verify_schanuel(data).valid


@pytest.mark.parametrize('name', ['phi', 'psi', 'hC'])
def test_input_comparison_tampering_rejected(result, name):
    data = schanuel_certificate_dict(result)
    bump_degree(data['input'][name])
    assert not verify_schanuel(data).valid


@pytest.mark.parametrize('part', ['rho', 'contraction', 'cone', 'negative_term'])
def test_negative_mapping_cone_and_splitting_tampering_rejected(result, part):
    data = schanuel_certificate_dict(result)
    if part in ('rho', 'contraction'):
        bump_degree(data[part])
    elif part == 'cone':
        bump_degree(data['cone']['differentials'])
    else:
        data['cone']['ranks'].pop(0)
        data['cone']['differentials'].pop(0)
    assert not verify_schanuel(data).valid


def test_mutable_result_serializes_current_claims(result):
    data = schanuel_certificate_dict(result)
    assert verify_schanuel(data).valid
    result['free_T'] += 1
    assert not verify_schanuel(result).valid
    assert verify_schanuel(data).valid


@pytest.mark.parametrize('data', [None, [], {}, {'schema': 'wrong'}])
def test_malformed_certificates_return_failure(data):
    assert not verify_schanuel(data).valid


def test_boolean_integer_and_extraneous_claims_rejected(result):
    data = schanuel_certificate_dict(result)
    bad = deepcopy(data)
    bad['ranks']['odd'] = True
    assert not verify_schanuel(bad).valid
    bad = deepcopy(data)
    bad['success'] = True
    assert not verify_schanuel(bad).valid
