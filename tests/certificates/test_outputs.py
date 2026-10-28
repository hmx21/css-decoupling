"""Physical output views remain bound to complete source mathematics."""
from copy import deepcopy
import hashlib
import json

import pytest

from css_decoupling import coarse_grain, decouple_general as decouple, decouple_complex
from css_decoupling.algebra.laurent import Ring, eye, zeros
from css_decoupling.algebra.modules import GeneralBackend
from css_decoupling.certificates.comparison import (
    CertificateFormatError, certificate_from_dict, circuit_from_dict, _matrix_from_dict,
)
from css_decoupling.certificates.outputs import (
    export_outputs, verify_qca_output, verify_circuit_output, verify_output_pair,
)
from css_decoupling.circuits.gates import Circuit
from css_decoupling.errors import ResourceLimit
from css_decoupling.models.examples import toric_css
from css_decoupling.topology.complexes import direct_sum_complexes, disk, koszul


@pytest.fixture(scope='module')
def complete():
    R = Ring(3, 2)
    HX, HZ = toric_css(R, {1: 1})
    shear = zeros(R, HX.ncols, HX.ncols)
    shear[0, 1] = R.variable(0) ** -1 + 2 * R.variable(1)
    A, inverse = eye(R, HX.ncols) + shear, eye(R, HX.ncols) - shear
    return decouple(HX * A, HZ * inverse.dagger(), backend='sympy').certificate()


@pytest.fixture(scope='module')
def pair(complete):
    return export_outputs(complete)


def _actions(qca):
    R = Ring(qca['ring']['prime'], qca['ring']['dimension'])
    return {name: _matrix_from_dict(raw, R, name) for name, raw in qca['actions'].items()}


def _assert_pair(source, result):
    qca, circuit = json.loads(json.dumps(result['qca'])), json.loads(json.dumps(result['circuit']))
    report = verify_output_pair(qca, circuit, source_certificate=source)
    assert report.valid, report.errors
    for name in ('Z', 'X', 'Zinv', 'Xinv'):
        assert 'exported gate ' + name + ' product' in report.checks
    assert qca['source'] == circuit['source']
    assert qca['context'] == circuit['context']
    for registers in (qca['context']['source_registers'], qca['context']['target_registers']):
        assert sum(item['qudits'] for item in registers) == qca['qudits']
        assert [item['offset'] for item in registers] == [0, registers[0]['qudits'],
                                                       registers[0]['qudits'] + registers[1]['qudits']]


def test_complete_exports_actual_physical_maps_and_canonical_source(complete, pair):
    _assert_pair(complete, pair)
    assert verify_qca_output(pair['qca'], source_certificate=complete).valid
    assert verify_circuit_output(pair['circuit'], source_certificate=complete).valid
    stable = certificate_from_dict(complete['stabilization'])
    actions = _actions(pair['qca'])
    assert actions == dict(Z=stable.U, Zinv=stable.Uinv, X=stable.Uinv.dagger(), Xinv=stable.U.dagger())
    assert stable.U.shape != stable.V.shape
    assert any(any(v < 0 for v in exp) for A in actions.values()
               for polynomial in A.entries.values() for exp in polynomial.terms)
    raw = (json.dumps(complete, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False) + '\n').encode('utf-8')
    assert pair['qca']['source']['canonical_sha256'] == hashlib.sha256(raw).hexdigest()
    assert pair['qca']['source']['physical_certificate'] == 'self'
    assert pair['qca']['context']['input_scope'] == 'original-CSS-input'
    reordered = dict(reversed(list(complete.items())))
    assert verify_output_pair(pair['qca'], pair['circuit'], source_certificate=reordered).valid


@pytest.mark.parametrize('dimension,prime', [(2, 2), (3, 3), (4, 5), (5, 3)])
def test_algorithm_two_sources_keep_original_stable_registers(dimension, prime):
    R = Ring(prime, dimension)
    C = direct_sum_complexes(koszul(R, 1), disk(R, -1), disk(R, 2))
    source = decouple_complex(C).certificate()
    assert source['comparison']['frame']['idempotent']['entries']
    result = export_outputs(source)
    _assert_pair(source, result)
    assert result['qca']['source']['physical_certificate'] == 'decoupling'
    assert result['qca']['context']['input_scope'] == 'Algorithm2-input-complex'
    assert result['qca']['context']['target']['basis'] == 'canonical-toric-stack'


def test_dual_uses_outer_gates_and_inverse_dagger_not_inner_U():
    source = decouple(*toric_css(Ring(5, 3), {1: 1}), sector='Z', backend='sympy').certificate()
    result = export_outputs(source)
    _assert_pair(source, result)
    stable = certificate_from_dict(source['inner']['stabilization'])
    assert _actions(result['qca']) == dict(Z=stable.Uinv.dagger(), Zinv=stable.U.dagger(),
                                          X=stable.U, Xinv=stable.Uinv)
    assert result['circuit']['circuit'] == source['circuit']
    context = result['qca']['context']
    assert context['sector'] == 'Z'
    assert context['target'] == source['target']
    assert context['source_registers'][1]['state'] == 'Z-fixed'
    assert context['source_registers'][2]['state'] == 'X-fixed'
    assert context['ancillas'] == source['ancillas']


def test_slanted_lattice_keeps_exact_coset_order():
    R = Ring(3, 2)
    source = decouple(*toric_css(R, {1: 1}), lattice_basis=[[1, 0], [1, 2]],
                      backend='sympy').certificate()
    result = export_outputs(source)
    _assert_pair(source, result)
    context = result['qca']['context']
    assert context['lattice'] == source['lattice']
    assert context['lattice']['index'] == 2
    assert context['source_registers'][0]['qudits'] == source['input']['HX']['cols'] * 2
    assert context['source_physical_order'] == (
        'index = species*lattice.index + coset; coset follows lattice.representatives')


@pytest.mark.parametrize('sector', ['X', 'Z'])
def test_algorithm_two_prefers_bound_original_physical_sector(sector):
    coarse = coarse_grain(*toric_css(Ring(3, 3), {1: 1}), sector=sector, backend='sympy')
    source = decouple_complex(coarse).certificate()
    result = export_outputs(source)
    _assert_pair(source, result)
    assert result['qca']['source']['physical_certificate'] == 'physical_decoupling'
    assert result['qca']['context']['input_scope'] == 'original-CSS-input'
    physical = source['physical_decoupling']
    actual = physical['circuit'] if sector == 'Z' else physical['stabilization']['circuit']
    assert result['circuit']['circuit'] == actual
    assert result['qca']['source']['schema'] == 'css-decoupling.algorithm-two'


def test_bounded_source_keeps_bounded_verification_without_override(monkeypatch):
    source = decouple_complex(disk(Ring(3, 2), -1), linear_backend='bounded').certificate()

    def forbidden(*args, **kwargs):
        raise AssertionError('Bounded source silently selected native module arithmetic')

    monkeypatch.setattr(GeneralBackend, 'solve', forbidden)
    monkeypatch.setattr(GeneralBackend, 'kernel', forbidden)
    result = export_outputs(source)
    _assert_pair(source, result)


@pytest.mark.parametrize('member', ['qca', 'circuit'])
@pytest.mark.parametrize('field', ['lattice', 'basis', 'profile', 'offset', 'ancilla-state', 'ancilla-count',
                                   'identity-checks', 'qudits', 'source-hash', 'source-selector', 'phase-type',
                                   'extra', 'version-bool', 'ring'])
def test_metadata_tampering_rejected(complete, pair, member, field):
    result = deepcopy(pair)
    data = result[member]
    context = data['context']
    if field == 'lattice':
        context['lattice']['basis_columns'][0][0] *= -1
    elif field == 'basis':
        context['target']['basis'] = 'different-output-basis'
    elif field == 'profile':
        context['target']['multiplicities'][0][1] += 1
    elif field == 'offset':
        context['source_registers'][1]['offset'] += 1
    elif field == 'ancilla-state':
        context['target_registers'][1]['state'] = 'Z-fixed'
    elif field == 'ancilla-count':
        context['ancillas']['source_X_fixed'] += 1
    elif field == 'identity-checks':
        context['identity_checks']['source'] += 1
    elif field == 'qudits':
        data['qudits'] += 1
    elif field == 'source-hash':
        data['source']['canonical_sha256'] = '0' * 64
    elif field == 'source-selector':
        data['source']['physical_certificate'] = 'decoupling'
    elif field == 'phase-type':
        data['conventions']['single_generator_phases'] = False
    elif field == 'extra':
        data['source_path'] = 'silently-loaded.json'
    elif field == 'version-bool':
        data['version'] = True
    elif field == 'ring':
        data['ring']['prime'] = 5
    assert not verify_output_pair(result['qca'], result['circuit'], source_certificate=complete).valid


@pytest.mark.parametrize('action', ['Z', 'Zinv', 'X', 'Xinv'])
def test_each_action_is_checked(complete, pair, action):
    result = deepcopy(pair)
    raw = result['qca']['actions'][action]
    raw['entries'][0][2][0][1] = 1 + raw['entries'][0][2][0][1] % 2
    assert not verify_output_pair(result['qca'], result['circuit'], source_certificate=complete).valid


def test_check_generator_matrix_V_is_not_a_physical_output(complete, pair):
    result = deepcopy(pair)
    result['qca']['actions']['Z'] = deepcopy(complete['stabilization']['stabilization']['V'])
    assert not verify_qca_output(result['qca'], source_certificate=complete).valid


def test_equivalent_but_different_gate_sequence_is_not_the_source_export(complete, pair):
    result = deepcopy(pair)
    result['circuit']['circuit']['gates'].extend([
        dict(kind='SWAP', left=0, right=1), dict(kind='SWAP', left=0, right=1)])
    assert not verify_output_pair(result['qca'], result['circuit'], source_certificate=complete).valid


def test_pair_independently_replays_inverse_X(complete, pair, monkeypatch):
    R = Ring(3, 2)
    forward = circuit_from_dict(pair['circuit']['circuit'], R)
    inverse = forward.inverse()
    assert forward != inverse
    original = Circuit.matrix

    def corrupt_inverse_x(self, action='Z'):
        if action == 'X' and self == inverse:
            return zeros(R, self.qudits, self.qudits)
        return original(self, action)

    monkeypatch.setattr(Circuit, 'matrix', corrupt_inverse_x)
    report = verify_output_pair(pair['qca'], pair['circuit'], source_certificate=complete)
    assert not report.valid
    assert any('Xinv' in error for error in report.errors)


def test_mathematically_equal_but_different_source_cannot_rebind_old_outputs(complete, pair):
    source = deepcopy(complete)
    source['stabilization']['stabilization']['U']['entries'].reverse()
    report = verify_output_pair(pair['qca'], pair['circuit'], source_certificate=source)
    assert not report.valid
    assert any('source' in error for error in report.errors)


def test_invalid_full_source_is_never_accepted_or_exported(complete, pair):
    source = deepcopy(complete)
    source['identity_checks']['source'] += 1
    with pytest.raises(CertificateFormatError, match='Source certificate failed'):
        export_outputs(source)
    assert not verify_output_pair(pair['qca'], pair['circuit'], source_certificate=source).valid


def test_intermediate_certificate_is_not_enough(complete, pair):
    partial = complete['stabilization']
    with pytest.raises(CertificateFormatError, match='complete, dual, or Algorithm 2'):
        export_outputs(partial)
    assert not verify_qca_output(pair['qca'], source_certificate=partial).valid


@pytest.mark.parametrize('function,member', [(verify_qca_output, 'qca'), (verify_circuit_output, 'circuit')])
def test_source_is_required_and_paths_are_not_read(pair, tmp_path, function, member):
    with pytest.raises(TypeError):
        function(pair[member])
    path = tmp_path / 'source.json'
    for source in (str(path), path, None):
        assert not function(pair[member], source_certificate=source).valid
    assert not path.exists()


@pytest.mark.parametrize('missing', [None, object(), {}, []])
def test_missing_output_is_not_a_successful_source_only_check(complete, pair, missing):
    assert not verify_qca_output(missing, source_certificate=complete).valid
    assert not verify_circuit_output(missing, source_certificate=complete).valid
    assert not verify_output_pair(missing, pair['circuit'], source_certificate=complete).valid
    assert not verify_output_pair(pair['qca'], missing, source_certificate=complete).valid


def test_bounded_options_cannot_silently_select_native_verification(complete):
    source = decouple_complex(disk(Ring(3, 2), -1)).certificate()
    for value in (complete, source):
        with pytest.raises(CertificateFormatError, match='bounded_options require'):
            export_outputs(value, linear_backend='sympy', bounded_options={})


def test_source_verification_resource_limit_propagates(complete, pair):
    class Limited(GeneralBackend):
        def kernel(self, A):
            raise ResourceLimit('output source verification budget')

    backend = Limited(Ring(3, 2))
    with pytest.raises(ResourceLimit, match='output source verification budget'):
        export_outputs(complete, linear_backend=backend)
    with pytest.raises(ResourceLimit, match='output source verification budget'):
        verify_output_pair(pair['qca'], pair['circuit'], source_certificate=complete, linear_backend=backend)
