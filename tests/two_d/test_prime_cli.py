"""CLI readers bind complete raw-input metadata and preserve existing paths."""
import json
import subprocess
import sys

import pytest

from css_decoupling.two_d import Ring, decouple
from css_decoupling.two_d.models import standard_code
from css_decoupling.two_d.serialization import save


@pytest.mark.parametrize('change',[{'p':5},{'D':True},{'unknown':1}])
def test_qca_cli_external_input_domain_is_not_ignored(change,tmp_path):
    HX,HZ=standard_code(ring=Ring(3,2))
    result=decouple(HX,HZ,compile_circuit=False,lattice_mode='fixed',lattice_basis=[[1,0],[0,1]])
    artifact=tmp_path/'qca.json';save(result,artifact)
    raw=tmp_path/'input.json';save({'HX':HX,'HZ':HZ,'p':3,'D':2,**change},raw)
    proc=subprocess.run([sys.executable,'-I','-B','-m','css_decoupling.two_d.cli','verify',
                         str(artifact),'--input',str(raw)],capture_output=True,text=True,timeout=30)
    assert proc.returncode==2,proc.stdout+proc.stderr
    assert 'verified' not in proc.stdout
    assert json.loads(proc.stderr)['exception']=='ValueError'


def test_existing_cli_output_directory_stays_unmodified(tmp_path):
    marker=tmp_path/'keep.txt';marker.write_text('keep',encoding='utf-8')
    proc=subprocess.run([sys.executable,'-I','-B','-m','css_decoupling.two_d.cli','example','toric',
                         '--output',str(tmp_path)],capture_output=True,text=True,timeout=30)
    assert proc.returncode==2
    assert sorted(p.name for p in tmp_path.iterdir())==['keep.txt']
    assert marker.read_text(encoding='utf-8')=='keep'
