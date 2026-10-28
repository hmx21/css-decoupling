"""Productized fixed-cell q=2 fallback, exact initial-U binding and budgets."""
from unittest.mock import patch

import pytest

from css_decoupling.two_d.algebra import Matrix, Ring, eye
from css_decoupling.two_d.models import standard_code
from css_decoupling.two_d.pipeline import decouple
from css_decoupling.two_d.code_circuits import compile_qca_no_ancilla, verify_no_ancilla_result
from css_decoupling.two_d.rank_two import factor_rank_two, parse_rank_two_options
from css_decoupling.two_d.rank_two_budget import RankTwoIncomplete, RankTwoResourceLimit
from css_decoupling.two_d.serialization import encode, decode


def source(prime, kind, *, lattice=None):
    R = Ring(prime, 2)
    x, y = R.variables
    a, b = x-1, y-1

    def E(i, j, c):
        result = eye(R, 2)
        result[i, j] = c
        return result

    pairs = {'cohn-1': (a, b), 'cohn-2': (a+b, b), 'cohn-3': (a, y),
             'cohn-4': (a+b*b, b+a*a)}
    if kind in pairs:
        u, v = pairs[kind]
        C = Matrix.from_rows(R, [[1+u*v, u*u], [-v*v, 1-u*v]])
    elif kind == 'elementary':
        C = E(0, 1, x+y**-1)*E(1, 0, y-x**-1)*E(0, 1, x*y+1)
    elif kind == 'univariate':
        C = E(0, 1, x**3+x**-2)*E(1, 0, x*x+1)*E(0, 1, x**-1)
    elif kind in ('target-auto', 'hidden-target-auto'):
        h = Matrix.from_rows(R, [[a, b]])
        d = Matrix.from_rows(R, [[-b], [a]])
        C = eye(R, 2)+d*((x+y**-1)*h)
        if kind == 'hidden-target-auto':
            C = C*E(0, 1, x+y**-1)*E(1, 0, y-x**-1)*E(0, 1, x*y+1)
    else:
        raise AssertionError(kind)
    Ci = Matrix.from_rows(R, [[C[1, 1], -C[0, 1]], [-C[1, 0], C[0, 0]]])
    assert C*Ci == Ci*C == eye(R, 2)
    HX, HZ = standard_code(ring=R)
    return decouple(HX*C, HZ*Ci.dagger(), compile_circuit=False,
                    lattice_mode='fixed', lattice_basis=lattice or [[1, 0], [0, 1]])


def replay_initial(result, data):
    U, V = result['qca']['psi'][1], result['qca']['psi_inverse'][1]
    R = U.ring
    F, Fi = eye(R, 2), eye(R, 2)
    for factor in data['factors']:
        i, j = factor['row'], factor['column']
        assert 0 <= i < 2 and 0 <= j < 2 and i != j
        E, Ei = eye(R, 2), eye(R, 2)
        E[i, j], Ei[i, j] = factor['coefficient'][0, 0], -factor['coefficient'][0, 0]
        F, Fi = E*F, Fi*Ei
    units = [R.monomial(tuple(e), c) for e, c in zip(data['shift_exponents'], data['transport_coefficients'])]
    S, Si = (Matrix(R, 2, 2, {(i, i): u**power for i, u in enumerate(units)}) for power in (1, -1))
    assert S*F == U and Fi*Si == V
    assert F*Fi == Fi*F == eye(R, 2)
    assert data['added_ancillas'] == 0
    assert not data['complete_general_suslin_algorithm']


@pytest.mark.parametrize('prime', [2, 3, 5, 7])
@pytest.mark.parametrize('kind', ['cohn-1', 'cohn-2', 'cohn-3', 'elementary',
                                  'univariate', 'target-auto', 'hidden-target-auto'])
def test_default_auto_compiles_verified_same_cell_research_families(prime, kind):
    result = source(prime, kind)
    original = encode(result)
    artifact = compile_qca_no_ancilla(result, timeout=30)
    assert encode(result) == original
    report = verify_no_ancilla_result(decode(encode(artifact)),
        expected_HX=result['original']['HX'], expected_HZ=result['original']['HZ'])
    assert report['verified'] and report['physical_qudits'] == 2 and report['input_ancillas'] == 0
    assert artifact['code_binding']['retained_lattice'] == [[1, 0], [0, 1]]
    initial = artifact['initial_factorization']
    if initial is not None:
        assert initial['original_qca'] == result['qca']['psi'][1]
        assert initial['original_inverse'] == result['qca']['psi_inverse'][1]
        assert initial['ancillas']['count'] == 0
    else:
        assert artifact['actual_circuit']['z_action'] == eye(result['original']['HX'].ring, 2)


def test_row_route_binds_original_U_and_cannot_call_stable_or_external_compiler(monkeypatch):
    result = source(3, 'cohn-1')

    def forbidden(*args, **kwargs):
        pytest.fail('Rank-two route attempted stabilization or an external process')

    monkeypatch.setattr('css_decoupling.two_d.suslin.factor_suslin', forbidden)
    monkeypatch.setattr('css_decoupling.two_d.circuits.compile_qca', forbidden)
    monkeypatch.setattr('subprocess.Popen', forbidden)
    monkeypatch.setattr('shutil.which', forbidden)
    events = []
    data = factor_rank_two(result, timeout=20, progress=lambda stage, details: events.append((stage, details)))
    replay_initial(result, data)
    assert data['statistics']['route'] == 'row-orbit-target-GE2'
    assert data['statistics']['search']['generated_nodes'] > 0
    assert events[-1][0] == 'rank_two_complete'


def test_fixed_index_one_oriented_basis_is_preserved():
    basis = [[0, 1], [-1, 0]]
    result = source(3, 'cohn-1', lattice=basis)
    artifact = compile_qca_no_ancilla(result, timeout=30)
    assert artifact['code_binding']['retained_lattice'] == basis
    assert artifact['resources']['physical_qudits'] == 2
    assert verify_no_ancilla_result(artifact)['verified']


def test_redundant_original_checks_use_the_verified_independent_pair():
    old = source(3, 'cohn-1')
    HX, HZ = old['original']['HX'], old['original']['HZ']
    x, y = HX.ring.variables
    redundant_X = HX.vstack((x+1)*HX)
    redundant_Z = HZ.vstack(2*y*HZ)
    result = decouple(redundant_X, redundant_Z, compile_circuit=False,
                      lattice_mode='fixed', lattice_basis=[[1, 0], [0, 1]])
    assert result['qca']['input_HX'].shape == (1, 2)
    artifact = compile_qca_no_ancilla(result, timeout=30)
    assert artifact['code_binding']['source_HX'] == redundant_X
    assert artifact['code_binding']['source_HZ'] == redundant_Z
    assert verify_no_ancilla_result(artifact, expected_HX=redundant_X, expected_HZ=redundant_Z)['verified']


@pytest.mark.parametrize('options', [{'search_depth': 0}, {'max_nodes': 1}])
def test_search_limits_have_structured_inconclusive_reasons(options):
    result = source(2, 'cohn-4')
    with patch('css_decoupling.two_d.rank_two._fast_atoms', return_value=None):
        with pytest.raises(RankTwoIncomplete) as error:
            factor_rank_two(result, rank_two_options=options)
    assert error.value.reason == ('depth_limit' if 'search_depth' in options else 'node_limit')
    assert error.value.phase == 'rank_two_row_search'
    assert error.value.details['non_elementarity_proved'] is False
    assert error.value.details['general_suslin_rank_requirement'] == 'q >= 3'


def test_global_budget_caps_do_not_restart_for_search_or_bridge():
    result = source(2, 'cohn-1')
    data = factor_rank_two(result, timeout=20)
    steps = data['statistics']['operations']
    exact = factor_rank_two(result, timeout=20, max_steps=steps)
    replay_initial(result, exact)
    with pytest.raises(RankTwoResourceLimit) as error:
        factor_rank_two(result, timeout=20, max_steps=steps-1)
    assert error.value.reason == 'step_limit'
    assert error.value.details['budget']['operations'] > steps-1
    for option, reason in (({'max_products': 1}, 'product_limit'),
                           ({'max_terms': 1}, 'term_limit'),
                           ({'max_factors': 1}, 'factor_limit')):
        with pytest.raises(RankTwoResourceLimit) as error:
            factor_rank_two(result, **option)
        assert error.value.reason == reason


def test_timeout_is_shared_and_remains_cooperative(monkeypatch):
    result = source(2, 'cohn-1')
    ticks = iter(range(100000))
    monkeypatch.setattr('css_decoupling.two_d.rank_two_budget.perf_counter', lambda: next(ticks))
    with pytest.raises(RankTwoResourceLimit) as error:
        factor_rank_two(result, timeout=0.5)
    assert error.value.reason == 'timeout'


def test_rank_two_gate_budget_failure_retains_scope_and_the_verified_qca():
    result = source(3, 'cohn-1')
    before = encode(result)
    with pytest.raises(RankTwoResourceLimit) as error:
        compile_qca_no_ancilla(result, timeout=20, max_families=0)
    assert error.value.reason == 'circuit_resource_limit'
    assert error.value.phase == 'rank_two_gate_compilation'
    assert error.value.details['general_suslin_rank_requirement'] == 'q >= 3'
    assert error.value.details['non_elementarity_proved'] is False
    assert encode(result) == before


def test_unsupported_target_does_not_enter_row_search():
    HX, HZ = standard_code(0, product_x=1, product_z=1, ring=Ring(3, 2))
    result = decouple(HX, HZ, compile_circuit=False)
    with patch('css_decoupling.two_d.rank_two._fast_atoms', return_value=None), patch(
            'css_decoupling.two_d.rank_two._search', side_effect=AssertionError('unsupported target searched')):
        with pytest.raises(RankTwoIncomplete) as error:
            factor_rank_two(result)
    assert error.value.reason == 'unsupported_target'


@pytest.mark.parametrize('options', [{'unknown': 1}, {'search_depth': True}, {'search_width': 0},
                                    {'max_nodes': -1}, [], {'max_quotient_terms': 1.5}])
def test_rank_two_options_are_strict(options):
    with pytest.raises(ValueError):
        parse_rank_two_options(options)


def test_certificate_tampering_still_rejects_original_U_and_actual_gate_changes():
    result = source(3, 'cohn-1')
    artifact = compile_qca_no_ancilla(result, timeout=20)
    for mode in ('factor', 'input', 'gate', 'witness'):
        bad = decode(encode(artifact))
        if mode == 'factor':
            bad['initial_factorization']['factors'][0]['coefficient'][0, 0] += 1
        elif mode == 'input':
            bad['initial_factorization']['original_qca'][0, 0] += 1
        elif mode == 'gate':
            gate = next(g for layer in bad['actual_circuit']['layers'] for g in layer if g['kind'] == 'SUM')
            gate['coefficient'] = -gate['coefficient'] % 3
        else:
            bad['code_binding']['stabilizers']['Z']['source_in_target'][0, 0] += 1
        with pytest.raises((ValueError, ArithmeticError)):
            verify_no_ancilla_result(bad)
