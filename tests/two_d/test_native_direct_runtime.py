"""Fresh-process native QS/Suslin execution with external CAS use forbidden.

The child process is ordinary CPython. Its external-process/import guards are
installed BEFORE importing any native mathematics; the mathematics itself is
not mocked. A separate child prevents previously imported optional backends
from hiding a dependency in a larger test-suite process.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest


_RUNTIME_SCRIPT = r'''
import builtins
from contextlib import ExitStack
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

# Force SymPy's pure-Python coefficient backend even if optional native CAS
# packages happen to be installed elsewhere on the host.
os.environ['SYMPY_GROUND_TYPES'] = 'python'
forbidden = {'sage', 'sageall', 'sageconf', 'julia', 'juliacall', 'singular',
             'pysingular', 'oscar', 'pyoscar', 'cypari', 'cypari2', 'pari',
             'gap', 'libgap', 'magma', 'macaulay2', 'symengine', 'flint'}
attempts = []

def denied_name(name):
    return str(name).split('.', 1)[0].lower() in forbidden

def deny_process(name):
    def denied(*args, **kwargs):
        attempts.append('process:' + name)
        raise AssertionError('External process use is forbidden: ' + name)
    return denied

original_import = builtins.__import__
def checked_import(name, globals=None, locals=None, fromlist=(), level=0):
    # A relative SymPy printer named ".julia" is pure Python code generation,
    # not the external top-level Julia runtime package. Resolve real identity.
    absolute = importlib.util.resolve_name('.'*level + name, globals['__package__']) if level else name
    if denied_name(absolute):
        attempts.append('import:' + absolute)
        raise AssertionError('External CAS import is forbidden: ' + absolute)
    return original_import(name, globals, locals, fromlist, level)

class NoExternalCAS(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if denied_name(fullname):
            attempts.append('find_spec:' + fullname)
            raise AssertionError('External CAS import is forbidden: ' + fullname)
        return None

assert not any(denied_name(name) for name in sys.modules)
finder = NoExternalCAS()
sys.meta_path.insert(0, finder)
try:
    with ExitStack() as stack:
        stack.enter_context(patch.object(builtins, '__import__', checked_import))
        stack.enter_context(patch.object(subprocess, 'Popen', deny_process('subprocess.Popen')))
        stack.enter_context(patch.object(os, 'system', deny_process('os.system')))
        # The listed launch APIs are not needed by the actual computation.
        for name in ('popen', 'posix_spawn', 'posix_spawnp', 'spawnv', 'spawnve',
                     'spawnvp', 'spawnvpe', 'execl', 'execle', 'execlp', 'execlpe',
                     'execv', 'execve', 'execvp', 'execvpe', 'startfile'):
            if hasattr(os, name):
                stack.enter_context(patch.object(os, name, deny_process('os.' + name)))

        from css_decoupling.two_d.algebra import Matrix, Ring, eye, block_diag, complete_laurent_row
        from css_decoupling.two_d.suslin import factor_suslin
        from css_decoupling.two_d.direct_circuits import compile_direct_circuit, verify_direct_circuit
        from css_decoupling.two_d.circuits import circuit_product
        from css_decoupling.two_d.serialization import save, load, digest
        from sympy.external.gmpy import GROUND_TYPES
        assert GROUND_TYPES == 'python'

        ring = Ring(2, 2)
        x, y = ring.variables
        a, b = 1+x, 1+y
        U = Matrix.from_rows(ring, [[1+a*b, a*a], [b*b, 1+a*b]])
        assert all(not f.is_unit for f in U.entries.values())
        assert U*U == eye(ring, 2)
        original_digest = digest(U)
        data = factor_suslin(U, U, transport='balance', timeout=30,
                             max_linearized_size=16, max_terms=100000,
                             max_factors=20000, max_products=2000000)
        assert data['algorithm'] == 'higman-pid-f2-laurent-v1'
        assert data['complete'] is True and data['complete_general_suslin_algorithm'] is True
        assert data['statistics']['identity_ancillas'] > 0
        assert data['statistics']['linearized_rank'] > U.nrows
        C = compile_direct_circuit(U, data['input_inverse'], data['factors'],
                added_ancillas=data['added_ancillas'], ancilla_action=data['ancilla_action'],
                ancilla_inverse=data['ancilla_inverse'],
                shift_exponents=[list(e) for e in data['shift_exponents']],
                max_factors=20000, max_families=100000, max_matrix_terms=100000)
        assert C['mode'] == 'exact'
        assert C['ancilla_action'] == eye(ring, data['added_ancillas'])
        assert C['z_action'] == block_diag(U, eye(ring, data['added_ancillas']))
        assert C['equivalence']['original_qca_implemented_by_circuit'] is True
        assert C['equivalence']['extended_qca_implemented_by_circuit'] is True
        assert digest(U) == original_digest
        with tempfile.TemporaryDirectory(prefix='native-direct-runtime-') as directory:
            path = Path(directory)/'actual-circuit.json'
            save(C, path)
            before = path.read_bytes()
            reread = load(path)
            assert verify_direct_circuit(reread, max_factors=20000,
                                        max_families=100000, max_matrix_terms=100000)
            for pauli, reverse, key in [('Z',False,'z_action'), ('Z',True,'z_inverse'),
                                        ('X',False,'x_action'), ('X',True,'x_inverse')]:
                assert circuit_product(ring, reread['z_action'].nrows, reread['layers'],
                    pauli=pauli, reverse=reverse, max_families=100000,
                    max_matrix_terms=100000) == reread[key]
            assert path.read_bytes() == before

        # A nontrivial polynomial length-three unimodular row: no unit entry,
        # two genuine variables, with the explicit checked Bezout column ones.
        row = Matrix.from_rows(ring, [[1+x, 1+y, 1+x+y]])
        bezout = Matrix.from_rows(ring, [[1], [1], [1]])
        assert row*bezout == eye(ring, 1)
        assert all(not f.is_unit for f in row.entries.values())
        trace = []
        Q, Qinv = complete_laurent_row(row, timeout=30, max_terms=100000,
                                       max_steps=100000, trace=trace)
        assert row*Q == Matrix.from_rows(ring, [[1,0,0]])
        assert Q*Qinv == eye(ring, 3) and Qinv*Q == eye(ring, 3)
        assert all(e >= 0 for A in (Q,Qinv) for f in A.entries.values()
                   for exponent in f.terms for e in exponent)
        assert trace, 'Expected the actual polynomial QS local/patch route'
        assert not attempts, attempts
        assert not any(denied_name(name) for name in sys.modules)
        print(json.dumps({'native_only': True, 'ground_types': GROUND_TYPES,
            'external_attempts': attempts, 'suslin_algorithm': data['algorithm'],
            'cohn_rank': 2, 'identity_ancillas': data['statistics']['identity_ancillas'],
            'gate_families': C['resources']['gate_family_count'],
            'four_actual_gate_streams_checked': True, 'serialization_readback_checked': True,
            'qs_row_length': 3, 'qs_trace_steps': len(trace), 'qs_two_sided_inverse_checked': True}))
finally:
    sys.meta_path.remove(finder)
'''


class NativeDirectRuntimeTests(unittest.TestCase):
    def test_actual_suslin_gates_readback_and_qs_without_external_cas(self):
        environment = dict(os.environ)
        environment['SYMPY_GROUND_TYPES'] = 'python'
        process = subprocess.run([sys.executable,'-I','-B','-c',_RUNTIME_SCRIPT],
                                 env=environment,capture_output=True,text=True,timeout=90)
        self.assertEqual(process.returncode,0,process.stdout+'\n'+process.stderr)
        result = json.loads(process.stdout.splitlines()[-1])
        self.assertTrue(result['native_only'])
        self.assertEqual(result['external_attempts'],[])
        self.assertEqual(result['ground_types'],'python')
        self.assertEqual(result['suslin_algorithm'],'higman-pid-f2-laurent-v1')
        self.assertGreater(result['identity_ancillas'],0)
        self.assertGreater(result['gate_families'],0)
        self.assertTrue(result['four_actual_gate_streams_checked'])
        self.assertTrue(result['serialization_readback_checked'])
        self.assertTrue(result['qs_two_sided_inverse_checked'])
        self.assertEqual(result['qs_row_length'],3)
        self.assertGreater(result['qs_trace_steps'],0)


if __name__ == '__main__':
    unittest.main()
