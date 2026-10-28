"""Installed CLI and public default for the original-register circuit route."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from css_decoupling.two_d import Ring,Matrix,decouple,load,save
from css_decoupling.two_d.models import standard_code


def cli(*args):
    return subprocess.run([sys.executable,'-I','-B','-m','css_decoupling.two_d.no_ancilla_cli',
                           *map(str,args)],capture_output=True,text=True,timeout=45)


def test_public_decouple_defaults_to_qca_without_stabilizing_registers():
    result=decouple(*standard_code())
    assert result['circuit'] is None and result['circuit_status']=='not_requested'


def test_cli_saved_qca_continuation_and_separate_reader(tmp_path):
    first=tmp_path/'qca';second=tmp_path/'compiled'
    run=cli('example','666','--output',first,'--qca-only')
    assert run.returncode==0,run.stdout+run.stderr
    before=(first/'qca.json').read_bytes()
    run=cli('compile',first/'qca.json','--output',second,'--factorization','fast')
    assert run.returncode==0,run.stdout+run.stderr
    status=json.loads(run.stdout)
    assert status['verified'] and status['input_ancillas']==0 and status['physical_qudits']==6
    assert status['target_product_X']==status['target_product_Z']==1
    assert (first/'qca.json').read_bytes()==before
    checked=cli('verify',second/'circuit-result.json','--input',first/'input.json')
    assert checked.returncode==0,checked.stdout+checked.stderr
    assert json.loads(checked.stdout)['stabilizer_equivalent']


def test_cli_preserves_existing_directory_and_rejects_stable_strategy(tmp_path):
    marker=tmp_path/'keep';marker.write_text('unchanged')
    run=cli('example','toric','--output',tmp_path)
    assert run.returncode==2 and sorted(p.name for p in tmp_path.iterdir())==['keep']
    run=cli('example','toric','--output',tmp_path/'missing','--factorization','higman')
    assert run.returncode==2 and not (tmp_path/'missing').exists()


def test_cli_odd_prime_fixed_lattice_keeps_original_register(tmp_path):
    run=cli('example','toric','--prime','3','--lattice-mode','fixed',
            '--lattice','[[1,1],[0,1]]','--output',tmp_path/'run')
    assert run.returncode==0,run.stdout+run.stderr
    status=json.loads(run.stdout)
    assert status['input_ancillas']==0 and status['physical_qudits']==2
    artifact=load(tmp_path/'run/circuit-result.json')
    assert artifact['qca_result']['lattice']['basis_columns']==[[1,1],[0,1]]


def test_cli_rank_two_unfinished_keeps_verified_qca_and_reports_truthfully(tmp_path):
    R=Ring(2,2);x,y=R.variables;a,b=x-1,y-1
    C=Matrix.from_rows(R,[[1+a*b,a*a],[-b*b,1-a*b]])
    Ci=Matrix.from_rows(R,[[1-a*b,-a*a],[b*b,1+a*b]])
    HX,HZ=standard_code(ring=R)
    raw=tmp_path/'input.json';save({'HX':HX*C,'HZ':HZ*Ci.dagger(),'p':2,'D':2},raw)
    out=tmp_path/'run'
    run=cli('run',raw,'--output',out,'--factorization','fast')
    assert run.returncode==2
    status=json.loads(run.stderr)
    assert status['status']=='factorization_incomplete' and status['qca_verified']
    assert load(out/'remaining-equation.json')['qca_verified']
    assert (out/'qca-verification.json').exists() and not (out/'circuit-result.json').exists()


def test_cli_invalid_saved_qca_is_not_marked_ready_for_factorization(tmp_path):
    qca=decouple(*standard_code())
    qca['qca']['psi'][1][0,0]+=1
    path=tmp_path/'bad-qca.json';save(qca,path)
    out=tmp_path/'run'
    run=cli('compile',path,'--output',out)
    assert run.returncode==2
    status=json.loads(run.stderr)
    assert not status['qca_verified']
    remaining=load(out/'remaining-equation.json')
    assert not remaining['qca_verified'] and remaining['remaining_task'].startswith('validate')
    assert not (out/'qca-verification.json').exists()
