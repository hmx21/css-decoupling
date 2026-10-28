"""Public CLI contracts: dispatch, original-input binding, and honest failures."""
import json
from pathlib import Path
import pytest

from css_decoupling import Ring, Matrix
from css_decoupling.cli import main
from css_decoupling.models.examples import toric_css
from css_decoupling.raw_input import css_input_dict, parse_css_input, save_css_input, load_css_input


def test_canonical_input_and_historical_imports(tmp_path):
    R=Ring(3,2);HX,HZ=toric_css(R,{1:1})
    path=save_css_input(HX,HZ,tmp_path/'input.json',lattice_mode='fixed',lattice_basis=[[1,1],[0,1]])
    restored=load_css_input(path)
    assert restored['HX']==HX and restored['HZ']==HZ and restored['lattice_basis']==[[1,1],[0,1]]
    with pytest.raises(FileExistsError):save_css_input(HX,HZ,path)
    from css_decoupling.certificates.io import input_dict
    old=parse_css_input(input_dict(HX,HZ))
    assert old['HX']==HX and old['lattice_mode']=='auto'
    from css_decoupling.two_d.models import standard_code
    from css_decoupling.two_d.algebra import Ring as TwoRing
    from css_decoupling.two_d.serialization import encode
    X,Z=standard_code(ring=TwoRing(5,2))
    imported=parse_css_input(encode({'HX':X,'HZ':Z,'p':5,'D':2}))
    assert imported['p']==5 and isinstance(imported['HX'],Matrix)


def test_strict_sparse_input_and_no_silent_lattice_change(tmp_path):
    HX,HZ=toric_css(Ring(3,2),{1:1})
    data=css_input_dict(HX,HZ)
    for extra in ({'unknown':True},{'version':True},{'ring':{'prime':3,'dimension':3}}):
        with pytest.raises((ValueError,TypeError)):parse_css_input({**data,**extra})
    with pytest.raises(ValueError):css_input_dict(HX,HZ,lattice_mode='auto',lattice_basis=[[1,0],[0,1]])
    with pytest.raises(ValueError):css_input_dict(HX,HZ,lattice_mode='fixed')
    with pytest.raises(ValueError):css_input_dict(HX,HZ,lattice_basis=[[1,0],[0,0]])
    path=tmp_path/'duplicate.json';path.write_text('{"HX":1,"HX":2}')
    with pytest.raises(ValueError,match='Duplicate'):load_css_input(path)


@pytest.mark.parametrize('dimension,algorithm,expected',[(2,'auto','two_d'),(2,'general','general'),(3,'auto','general')])
def test_run_qca_default_and_input_bound_verify(tmp_path,capsys,dimension,algorithm,expected):
    from css_decoupling.result import load_result
    HX,HZ=toric_css(Ring(3,dimension),{1:1})
    path=save_css_input(HX,HZ,tmp_path/'input.json')
    out=tmp_path/'result.json'
    assert main(['run',str(path),'--output',str(out),'--algorithm',algorithm])==0
    result=load_result(out)
    assert result.algorithm==expected and result.status=='qca_ready' and result.circuit is None
    assert main(['verify',str(out),'--input',str(path)])==0
    capsys.readouterr()
    before=out.read_bytes()
    assert main(['run',str(path),'--output',str(out)])==2
    assert out.read_bytes()==before


def test_domain_mismatch_rejected_before_output(tmp_path,capsys):
    HX,HZ=toric_css(Ring(3,2),{1:1})
    path=save_css_input(HX,HZ,tmp_path/'input.json');out=tmp_path/'result.json'
    assert main(['run',str(path),'--output',str(out),'--prime','5'])==2
    assert not out.exists()
    assert 'metadata' in capsys.readouterr().err
    assert main(['run',str(path),'--output',str(path)])==2


def test_real_rank_two_incomplete_keeps_verified_qca(tmp_path,capsys):
    from css_decoupling.result import load_result
    R=Ring(3,2);x,y=R.variables;a=x-1;b=y-1;u=a+b*b;v=b+a*a
    HX=Matrix.from_rows(R,[[a,b]])
    HZ=Matrix.from_rows(R,[[1-y**-1,x**-1-1]])
    C=Matrix.from_rows(R,[[1+u*v,u*u],[-v*v,1-u*v]])
    Ci=Matrix.from_rows(R,[[1-u*v,-u*u],[v*v,1+u*v]])
    path=save_css_input(HX*C,HZ*Ci.dagger(),tmp_path/'input.json',
        lattice_mode='fixed',lattice_basis=[[1,0],[0,1]])
    out=tmp_path/'result.json'
    code=main(['run',str(path),'--output',str(out),'--format','circuit',
        '--circuit-options','{"max_steps":1}'])
    assert code==3,capsys.readouterr()
    result=load_result(out)
    assert result.status=='circuit_incomplete' and result.circuit is None
    assert result.qca.validate()
    assert 'reason' in json.dumps(result.diagnostics).lower()
    assert main(['verify',str(out),'--input',str(path)])==0


def test_examples_and_options_not_silently_ignored(tmp_path,capsys):
    path=tmp_path/'input.json';out=tmp_path/'result.json'
    assert main(['example','toric','--dimension','2','--prime','5','--output',str(path)])==0
    assert load_css_input(path)['p']==5
    assert main(['run',str(path),'--output',str(out),'--options','{"backend":"singular"}'])==2
    assert not out.exists()
    assert main(['run',str(path),'--output',str(out),'--circuit-options','{"timeout":1}'])==2
    assert not out.exists()


def test_glued_example_is_the_manuscript_code_and_decouples(tmp_path):
    from css_decoupling.models.examples import glued_css
    path=tmp_path/'glued.json';out=tmp_path/'circuit.json'
    assert main(['example','glued','--prime','3','--output',str(path)])==0
    source=load_css_input(path)
    assert (source['HX'],source['HZ'])==glued_css(Ring(3,3))
    assert main(['run',str(path),'--format','circuit','--output',str(out)])==0
    assert main(['verify',str(out),'--input',str(path)])==0
    assert main(['example','glued','--dimension','2','--output',str(tmp_path/'bad.json')])!=0
    assert not (tmp_path/'bad.json').exists()
