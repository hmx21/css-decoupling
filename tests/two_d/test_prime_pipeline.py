"""Prime-field raw checks, lattice constraints and independent certificates."""
from unittest.mock import patch

import pytest

from css_decoupling.two_d import Ring, Matrix, decouple, verify, LatticeObstruction
from css_decoupling.two_d.algebra import eye
from css_decoupling.two_d.models import standard_code, two_generator_css
from css_decoupling.two_d.pipeline import SCHEMA, PRIME_SCHEMA
from css_decoupling.two_d.adapter import dispatch_contract
from css_decoupling.two_d.serialization import encode, decode, save, load


@pytest.mark.parametrize('p', [2, 3, 5, 7])
@pytest.mark.parametrize('counts', [(0,0,0), (0,2,1), (1,0,0), (2,1,2)])
def test_prime_raw_qca_roundtrip(p, counts, tmp_path):
    t, px, pz = counts
    HX,HZ = standard_code(t, product_x=px, product_z=pz, ring=Ring(p,2))
    r = decouple(HX,HZ,compile_circuit=False,lattice_mode='auto')
    path=tmp_path/'qca.json'; save(r,path)
    report=verify(load(path),expected_HX=HX,expected_HZ=HZ)
    assert report['qca'] and not report['circuit']
    assert report['toric_copies']==t
    assert r['qca']['product_x']==px and r['qca']['product_z']==pz
    assert r['original']['p']==p and r['schema']==PRIME_SCHEMA


@pytest.mark.parametrize('p,index', [(3,2), (5,4), (7,3)])
def test_prime_scalar_translation_auto_and_fixed(p,index):
    R=Ring(p,2);x,y=R.variables
    HX,HZ=two_generator_css(x-2,y-1)
    r=decouple(HX,HZ,compile_circuit=False,lattice_mode='auto')
    assert verify(r)['lattice_index']==index
    assert r['qca']['toric_copies']==1
    basis=[[index,0],[1,1]]
    with patch('css_decoupling.two_d.pipeline.translation_kernel',side_effect=AssertionError('fixed mode searched')):
        fixed=decouple(HX,HZ,compile_circuit=False,lattice_mode='fixed',lattice_basis=basis)
        assert verify(fixed)['qca']
        assert fixed['translation_kernel'] is None
        assert fixed['lattice']['basis_columns']==basis
        with pytest.raises(LatticeObstruction) as caught:
            decouple(HX,HZ,compile_circuit=False,lattice_mode='fixed',lattice_basis=[[1,0],[0,1]])
        assert caught.value.action != eye(R,1)


@pytest.mark.parametrize('p', [3,5])
def test_redundant_prime_checks_and_negative_exponents(p):
    R=Ring(p,2);x,y=R.variables
    HX,HZ=two_generator_css(x**-1-1,y**-1-1)
    zero=Matrix(R,1,2)
    # These pairs generate the unit ideal; no one polynomial row suffices.
    HX=((x+1)*HX).vstack((x+2)*HX).vstack(zero)
    HZ=((y+1)*HZ).vstack((y+2)*HZ).vstack(zero)
    r=decouple(HX,HZ,compile_circuit=False,lattice_mode='fixed',lattice_basis=[[1,-1],[1,1]])
    assert verify(r,expected_HX=HX,expected_HZ=HZ)['qca']
    assert r['reduced_HX'].nrows==r['reduced_HZ'].nrows==1


def test_lattice_policy_and_version_tampering():
    HX,HZ=standard_code()
    for options in ({'lattice_mode':'fixed'}, {'lattice_mode':'auto','lattice_basis':[[1,0],[0,1]]},
                    {'lattice_mode':'guess'}, {'lattice_mode':'fixed','lattice_basis':[[1]]}):
        with pytest.raises(ValueError):decouple(HX,HZ,compile_circuit=False,**options)
    legacy=decouple(HX,HZ,compile_circuit=False)
    assert legacy['schema']==SCHEMA and verify(legacy)['qca']
    HX,HZ=standard_code(ring=Ring(3,2))
    fixed=decouple(HX,HZ,compile_circuit=False,lattice_mode='fixed',lattice_basis=[[1,1],[0,1]])
    for change in ('schema','request','scope','p','phase'):
        bad=decode(encode(fixed))
        if change=='schema':bad['schema']=SCHEMA
        elif change=='request':bad['lattice_request']['basis_columns'][0][1]=2
        elif change=='scope':bad['lattice_request']['kernel_scope']='complete'
        elif change=='p':bad['original']['p']=2
        else:bad['conventions']['stabilizer_eigenvalue']='-1'
        with pytest.raises(ValueError):verify(bad)


def test_prime_adapter_checks_domain_and_keeps_fixed_mode():
    HX,HZ=standard_code(ring=Ring(5,2))
    payload={'algorithm':'two_d','p':5,'D':2,'HX':encode(HX),'HZ':encode(HZ),
             'lattice_mode':'fixed','lattice_basis':[[1,1],[0,1]],'options':{'compile_circuit':False}}
    r=dispatch_contract(payload)
    assert r['lattice_request']['mode']=='fixed' and verify(r)['qca']
    for delta in ({'p':3},{'algorithm':'arxiv-2608.09915v1-native-stabilized'},{'D':3},{'p':True}):
        with pytest.raises(ValueError):dispatch_contract({**payload,**delta})


def test_symbolic_integer_fixed_basis_is_losslessly_serializable(tmp_path):
    from sympy import Integer
    HX,HZ=standard_code(ring=Ring(3,2))
    basis=[[Integer(1),Integer(-1)],[Integer(0),Integer(1)]]
    r=decouple(HX,HZ,compile_circuit=False,lattice_mode='fixed',lattice_basis=basis)
    target=tmp_path/'qca.json';save(r,target)
    decoded=load(target)
    assert decoded['lattice_request']['basis_columns']==[[1,-1],[0,1]]
    assert verify(decoded)['qca']
