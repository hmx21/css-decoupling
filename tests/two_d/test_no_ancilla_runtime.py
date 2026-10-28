"""The installed original-register route cannot call CAS or stable compilers."""
import os
import subprocess
import sys


SCRIPT=r'''
import builtins, importlib, json, os, subprocess, shutil
from contextlib import ExitStack
from unittest.mock import patch
os.environ['SYMPY_GROUND_TYPES']='python'
original=builtins.__import__
attempts=[]
forbidden={'sage','sageall','sageconf','sagemath','julia','juliacall','oscar','singular','pysingular','flint','symengine'}
def guarded(name,globals=None,locals=None,fromlist=(),level=0):
    qualified=name
    if level and globals and globals.get('__package__'):
        qualified=importlib.util.resolve_name('.'*level+name,globals['__package__'])
    if qualified.split('.')[0] in forbidden:
        attempts.append(qualified);raise AssertionError(qualified)
    return original(name,globals,locals,fromlist,level)
def fail(*a,**kw):
    attempts.append('external-or-stabilized');raise AssertionError('forbidden dependency')
with ExitStack() as stack:
    stack.enter_context(patch('builtins.__import__',guarded))
    for name in ('subprocess.Popen','shutil.which','os.system','os.popen'):
        stack.enter_context(patch(name,fail))
    from css_decoupling.two_d import Ring,Matrix,decouple,compile_qca_no_ancilla,verify_no_ancilla_result,factor_unstabilized
    from css_decoupling.two_d.algebra import eye
    from css_decoupling.two_d.models import COLOR_CODE_666
    from css_decoupling.two_d.direct_circuits import elementary_product
    from css_decoupling.two_d.serialization import encode,decode
    stack.enter_context(patch('css_decoupling.two_d.suslin.factor_suslin',fail))
    stack.enter_context(patch('css_decoupling.two_d.circuits.compile_qca',fail))
    R=Ring(2,2);x,y=R.variables;a,b=x-1,y-1
    U,V=eye(R,3),eye(R,3)
    U[:2,:2]=Matrix.from_rows(R,[[1+a*b,a*a],[-b*b,1-a*b]])
    V[:2,:2]=Matrix.from_rows(R,[[1-a*b,-a*a],[b*b,1+a*b]])
    data=factor_unstabilized(U,V,method='suslin')
    S=Matrix(R,3,3,{(i,i):R.monomial(e,c) for i,(e,c) in enumerate(zip(data['shift_exponents'],data['transport_coefficients']))})
    assert data['added_ancillas']==0
    assert all(f['row']<3 and f['column']<3 for f in data['factors'])
    assert S*elementary_product(R,3,data['factors'])==U
    qca=decouple(*COLOR_CODE_666.checks())
    assert qca['circuit'] is None
    circuit=compile_qca_no_ancilla(qca)
    report=verify_no_ancilla_result(decode(json.loads(json.dumps(encode(circuit)))))
    assert report['input_ancillas']==0 and report['physical_qudits']==6
    assert report['stabilizer_equivalent'] and not attempts
print(json.dumps({'verified':True,'external_or_stable_attempts':attempts,'physical_qudits':6,'factor_registers':3}))
'''


def test_installed_original_register_route_is_native_and_never_stabilizes():
    environment=dict(os.environ);environment['SYMPY_GROUND_TYPES']='python'
    run=subprocess.run([sys.executable,'-I','-B','-c',SCRIPT],env=environment,
                       capture_output=True,text=True,timeout=45)
    assert run.returncode==0,run.stdout+run.stderr
    assert '"verified": true' in run.stdout
