"""Common Matrix, dispatch, source binding and honest partial-result contracts."""
from copy import deepcopy
import json
from unittest.mock import patch

import pytest

from css_decoupling import (Ring, Matrix, decouple, decouple_general, UnifiedResult,
    save_result, load_result, verify_result, save_input, save_decoupling, load_decoupling)
from css_decoupling.models.examples import toric_css
from css_decoupling.result import to_public, to_two_d, RESULT_SCHEMA
from css_decoupling.raw_input import read_json


@pytest.mark.parametrize('p', [2,3,5])
def test_default_two_d_uses_public_matrices_and_keeps_native_evidence(p):
    HX,HZ=toric_css(Ring(p,2),{1:1})
    with patch('css_decoupling.two_d.code_circuits.compile_qca_no_ancilla',
               side_effect=AssertionError('QCA-only must not compile')):
        result=decouple(HX,HZ,dimension=2,p=p)
    assert isinstance(result,UnifiedResult)
    assert result.algorithm=='two_d' and result.status=='qca_ready' and result.circuit is None
    assert isinstance(result.qca.z_action,Matrix) and result.qca.ring==HX.ring
    assert result.qca.qudits==2
    assert result.ancillas=={'kind':'none','input_count':0,'output_count':0}
    assert result.metadata['input_physical_qudits']==2
    assert result.target['toric_multiplicities']==[[1,1]]
    assert verify_result(result,expected_HX=HX,expected_HZ=HZ).valid
    # The two-dimensional algorithm shares the public types; the adapter copies.
    X=to_two_d(HX)
    assert type(X) is type(HX) and X==HX and X is not HX and to_public(X)==HX


@pytest.mark.parametrize('p,D', [(2,2),(3,3),(5,4)])
def test_general_qca_honestly_records_existing_stabilized_circuit(p,D):
    HX,HZ=toric_css(Ring(p,D),{1:1})
    result=decouple(HX,HZ,algorithm='general')
    assert result.algorithm=='general' and result.status=='qca_ready' and result.circuit is None
    assert result.metadata['general_constructed_circuit_internally'] is True
    assert result.ancillas['kind']=='C4-disk-stabilization'
    assert result.qca.qudits>result.metadata['input_physical_qudits']
    assert result.backend_evidence['qca']['stabilization']['circuit'] is not None
    assert result.backend_evidence['construction'] is not None
    assert verify_result(result).valid


@pytest.mark.parametrize('algorithm', ['two_d','general'])
def test_fixed_oblique_lattice_and_actual_circuit_views(algorithm):
    HX,HZ=toric_css(Ring(3,2),{1:1});basis=[[1,1],[0,1]]
    result=decouple(HX,HZ,algorithm=algorithm,output='circuit',lattice_mode='fixed',lattice_basis=basis)
    assert result.status=='circuit_ready' and result.lattice['basis_columns']==basis
    circuit=result.circuit
    assert circuit.matrix('Z')==circuit.action.z_action
    assert circuit.matrix('X')==circuit.action.x_action
    same=circuit.action.z_action==result.qca.z_action
    assert circuit.relation_to_qca==('exact' if same else 'code_equivalent')
    assert verify_result(result).valid


def test_auto_chooses_general_above_two_dimensions():
    HX,HZ=toric_css(Ring(2,3),{1:1})
    result=decouple(HX,HZ)
    assert result.algorithm=='general'
    assert verify_result(result).valid


@pytest.mark.parametrize('kwargs', [
    {'dimension':3}, {'p':5}, {'dimension':True}, {'algorithm':'unknown'},
    {'output':'matrix'}, {'lattice_mode':'fixed'}, {'lattice_basis':[[1,0],[0,1]]},
    {'options':{'max_index':4}}, {'options':{'backend':'sympy'}},
    {'options':{'circuit_options':{}}},
    {'output':'circuit','options':{'circuit_options':{'factor_data':{}}}},
    {'output':'circuit','options':{'circuit_options':{'max_factors':0}}},
    {'output':'circuit','options':{'circuit_options':{'max_matrix_terms':0}}},
    {'output':'circuit','options':{'circuit_options':{'rank_two_options':{'unknown':1}}}},
    {'algorithm':'general','options':{'backend':'singular'}},
    {'algorithm':'general','options':{'resolution_method':'singular'}},
    {'algorithm':'general','options':{'bounded_options':{}}},
    {'algorithm':'general','options':{'qs_options':{'trace':[]},'comparison_method':'free-basis'}},
    {'algorithm':'general','lattice_mode':'fixed','lattice_basis':[[1,0],[0,1]],
     'options':{'coarse_graining':'uniform'}},
])
def test_conflicting_domain_policy_and_unknown_options_fail_before_construction(kwargs):
    HX,HZ=toric_css(Ring(3,2),{1:1})
    with patch('css_decoupling.two_d.pipeline.decouple',side_effect=AssertionError('must not construct')), \
         patch('css_decoupling.decoupling.pipeline.decouple',side_effect=AssertionError('must not construct')):
        with pytest.raises((ValueError,TypeError)):decouple(HX,HZ,**kwargs)


def test_two_d_selection_rejects_higher_dimension_and_non_matrix_input():
    HX,HZ=toric_css(Ring(3,3),{1:1})
    with pytest.raises(ValueError):decouple(HX,HZ,algorithm='two_d')
    HX,HZ=toric_css(Ring(3,2),{1:1})
    with pytest.raises(TypeError):to_two_d(dict(HX.entries))
    with pytest.raises(TypeError):to_public(dict(HX.entries))


def test_circuit_budget_stop_retains_a_verified_qca_and_structured_reason():
    R=Ring(3,2);x,y=R.variables;a=x-1;b=y-1;u=a+b*b;v=b+a*a
    HX=Matrix.from_rows(R,[[a,b]])
    HZ=Matrix.from_rows(R,[[1-y**-1,x**-1-1]])
    C=Matrix.from_rows(R,[[1+u*v,u*u],[-v*v,1-u*v]])
    Ci=Matrix.from_rows(R,[[1-u*v,-u*u],[v*v,1+u*v]])
    X,Z=HX*C,HZ*Ci.dagger()
    result=decouple(X,Z,output='circuit',lattice_mode='fixed',lattice_basis=[[1,0],[0,1]],
                    options={'circuit_options':{'max_steps':1}})
    assert result.status=='circuit_incomplete' and result.circuit is None
    assert result.qca.validate()
    assert result.diagnostics['error']['reason']
    assert result.diagnostics['error']['message']
    assert result.diagnostics['error']['details']['non_elementarity_proved'] is False
    assert result.diagnostics['error']['details']['general_suslin_rank_requirement']=='q >= 3'
    assert verify_result(result,expected_HX=X,expected_HZ=Z).valid
    for name,value in [('reason',None),('reason',''),('phase',False),('phase',''),('details',[])]:
        damaged=result.certificate();damaged['diagnostics']['error'][name]=value
        assert not verify_result(damaged).valid


@pytest.mark.parametrize('algorithm', ['two_d','general'])
def test_fixed_lattice_cannot_be_relabelled_as_automatic_full_kernel(algorithm):
    HX,HZ=toric_css(Ring(3,2),{1:1})
    result=decouple(HX,HZ,algorithm=algorithm,lattice_mode='fixed',lattice_basis=[[2,0],[0,1]])
    damaged=result.certificate()
    damaged['input']['lattice_mode']='auto';damaged['input']['lattice_basis']=None
    report=verify_result(damaged)
    assert not report.valid and any('lattice' in e.lower() or 'kernel' in e.lower() for e in report.errors)


def test_two_d_automatic_evidence_cannot_be_relabelled_as_fixed():
    HX,HZ=toric_css(Ring(3,2),{1:1});result=decouple(HX,HZ)
    damaged=result.certificate();damaged['input']['lattice_mode']='fixed'
    damaged['input']['lattice_basis']=result.lattice['basis_columns']
    assert not verify_result(damaged).valid


@pytest.mark.parametrize('options', [
    {'sector':'Z'}, {'coarse_graining':'uniform'},
    {'coarse_graining':'uniform','sector':'Z'},
    {'comparison_method':'free-basis'},
    {'linear_backend':'bounded','bounded_options':{}},
])
def test_general_alternative_routes_retain_independent_provenance(options):
    from css_decoupling.algebra.laurent import eye,zeros
    # Acyclic source keeps the strict bounded-linear theorem's system small.
    ring=Ring(3,2);HX=eye(ring,1);HZ=zeros(ring,0,1)
    result=decouple(HX,HZ,algorithm='general',options=options)
    assert verify_result(result,expected_HX=HX,expected_HZ=HZ).valid
    selection=result.backend_evidence['lattice_selection']
    if options.get('coarse_graining')=='uniform':
        assert selection is None
        assert result.backend_evidence['construction']['coarse_graining'] is not None
    else:
        assert selection['schema']=='css-decoupling.profile-analysis'
        damaged=result.certificate();damaged['backend_evidence']['lattice_selection']=None
        assert not verify_result(damaged).valid


def test_reader_does_not_call_any_constructor_and_rejects_tampered_views():
    HX,HZ=toric_css(Ring(3,2),{1:1})
    result=decouple(HX,HZ,output='circuit')
    with patch('css_decoupling.two_d.pipeline.decouple',side_effect=AssertionError('constructor forbidden')), \
         patch('css_decoupling.two_d.code_circuits.compile_qca_no_ancilla',side_effect=AssertionError('compiler forbidden')), \
         patch('css_decoupling.decoupling.pipeline.decouple',side_effect=AssertionError('constructor forbidden')):
        assert verify_result(result).valid
    data=result.certificate()
    mutations=[
        lambda d:d.__setitem__('unknown',1),
        lambda d:d.__setitem__('version',True),
        lambda d:d.__setitem__('algorithm','general'),
        lambda d:d.__setitem__('status','qca_ready'),
        lambda d:d['metadata'].__setitem__('qca_registers',999),
        lambda d:d['diagnostics']['scope'].__setitem__('qca_input_ancillas',1),
        lambda d:d['qca'].__setitem__('qudits',999),
        lambda d:d['circuit'].__setitem__('relation_to_qca','exact' if d['circuit']['relation_to_qca']!='exact' else 'code_equivalent'),
        lambda d:d['backend_evidence'].__setitem__('kind','general'),
        lambda d:d['request']['options'].__setitem__('not_supported',1),
    ]
    for mutation in mutations:
        damaged=deepcopy(data);mutation(damaged)
        assert not verify_result(damaged).valid
    assert not verify_result(data,expected_HX=HX*HX.ring.variable(0)).valid


def test_strict_no_overwrite_roundtrip_and_duplicate_keys(tmp_path):
    HX,HZ=toric_css(Ring(5,2),{1:1});result=decouple(HX,HZ)
    path=save_result(result,tmp_path/'nested'/'result.json')
    assert read_json(path)['schema']==RESULT_SCHEMA
    restored=load_result(path)
    assert isinstance(restored,UnifiedResult) and restored.qca.z_action==result.qca.z_action
    assert verify_result(restored,expected_HX=HX,expected_HZ=HZ).valid
    before=path.read_bytes()
    with pytest.raises(FileExistsError):save_result(result,path)
    assert path.read_bytes()==before
    duplicate=tmp_path/'duplicate.json';duplicate.write_text('{"schema":1,"schema":2}')
    with pytest.raises(ValueError,match='Duplicate'):load_result(duplicate)


def test_legacy_general_alias_and_certificate_io_keep_their_meaning(tmp_path):
    HX,HZ=toric_css(Ring(3,2),{1:1})
    old=decouple_general(HX,HZ,backend='sympy')
    assert hasattr(old,'stabilization') and not isinstance(old,UnifiedResult)
    path=save_decoupling(old,tmp_path/'old.json')
    assert load_decoupling(path)['schema']=='css-decoupling.general.complete'
    assert save_input(HX,HZ,tmp_path/'old-input.json').exists()
    with pytest.raises(TypeError):decouple(HX,HZ,backend='sympy')
