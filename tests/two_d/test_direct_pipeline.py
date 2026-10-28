"""Raw-code binding and separate-process direct-circuit CLI verification."""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from css_decoupling.two_d.algebra import Matrix, Ring, eye
from css_decoupling.two_d.direct_cli import main as direct_cli
from css_decoupling.two_d.direct_pipeline import (
    compile_qca_direct, verify_direct_binding, verify_direct_result,
)
from css_decoupling.two_d.models import COLOR_CODE_666, COLOR_CODE_488, standard_code
from css_decoupling.two_d.pipeline import decouple
from css_decoupling.two_d.serialization import decode, digest, encode, load, save
from css_decoupling.two_d.verify import verify_generation
from css_decoupling.errors import ResourceLimit


def clone(value):
    return decode(encode(value))


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class DirectPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = {'666':COLOR_CODE_666.checks(), '488':COLOR_CODE_488.checks(),
                   'toric':standard_code(1), 'empty':standard_code(0)}
        cls.qca = {name:decouple(HX,HZ,compile_circuit=False)
                   for name,(HX,HZ) in cls.raw.items()}

    def test_constructor_core_verified_once_and_full_reader_checks_again(self):
        from css_decoupling.two_d.direct_circuits import verify_direct_circuit
        with patch('css_decoupling.two_d.direct_circuits.verify_direct_circuit',
                   wraps=verify_direct_circuit) as actual_check:
            artifact=compile_qca_direct(self.qca['666'])
            self.assertEqual(actual_check.call_count,1)
            self.assertTrue(verify_direct_result(artifact)['verified'])
            self.assertEqual(actual_check.call_count,2)
        bad=clone(artifact)
        bad['direct_circuit']['z_inverse'][0,0]+=1
        with self.assertRaises(ValueError):verify_direct_result(bad)

    def test_constructor_checks_new_code_binding_after_core(self):
        invalid=clone(self.qca['toric'])
        invalid['qca_stabilizers']['Z']['source_in_target'][0,0]+=1
        with self.assertRaises(ValueError):compile_qca_direct(invalid,verify_source=False)

    def test_forced_higman_honors_api_and_cli_operation_budget(self):
        with self.assertRaises(ResourceLimit):
            compile_qca_direct(self.qca['666'],factorization='higman',max_steps=1)
        for value in (True,0,1.0):
            with self.subTest(value=value),self.assertRaises(ValueError):
                compile_qca_direct(self.qca['toric'],factorization='higman',max_steps=value)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'qca.json';target=root/'result'
            save(self.qca['toric'],source)
            before=file_sha(source)
            with redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):
                code=direct_cli(['compile',str(source),'--output',str(target),
                                 '--factorization','higman','--max-steps','1'])
            self.assertEqual(code,2)
            self.assertEqual(file_sha(source),before)
            self.assertEqual(load(target/'status.json')['status'],'resource_incomplete')
            self.assertTrue(load(target/'qca-verification.json')['verified'])
            self.assertFalse((target/'direct-result.json').exists())

    def test_raw_color_toric_empty_full_checks_and_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            for name,result in self.qca.items():
                with self.subTest(name=name):
                    before=digest(result)
                    artifact=compile_qca_direct(result,transport='balance')
                    self.assertEqual(digest(result),before)
                    self.assertIsNone(result['circuit'])
                    C=artifact['direct_circuit']
                    self.assertEqual(C['mode'],'exact')
                    self.assertLessEqual(C['ancillas']['count'],1)
                    self.assertEqual(C['resources']['circuit_qubits_per_cell'],
                                     result['qca']['psi'][1].nrows+C['ancillas']['count'])
                    path=Path(directory)/(name+'.json')
                    save(artifact,path)
                    self.assertNotIn(b'\r',path.read_bytes())
                    saved_hash=file_sha(path)
                    answer=verify_direct_result(load(path),expected_HX=self.raw[name][0],expected_HZ=self.raw[name][1])
                    self.assertTrue(answer['qca'] and answer['direct_circuit'])
                    self.assertEqual(answer['direct_mode'],'exact')
                    self.assertEqual(file_sha(path),saved_hash)
                    self.assertEqual(digest(result),before)
                    if name=='empty':
                        self.assertEqual(C['layers'],[])
                        self.assertEqual(C['ancillas']['count'],0)

    def test_shift_separated_actual_targets_and_two_way_generation(self):
        result=self.qca['666']
        before=digest(result)
        artifact=compile_qca_direct(result,transport='separate')
        C,B=artifact['direct_circuit'],artifact['code_binding']
        S,Sinv=C['transport']['z_action'],C['transport']['z_inverse']
        self.assertEqual(C['mode'],'shift_separated')
        self.assertFalse(C['equivalence']['original_qca_implemented_by_circuit'])
        self.assertEqual(C['ancillas']['count'],0)
        self.assertEqual(B['actual_target_HX'],B['standard_HX']*S)
        self.assertEqual(B['actual_target_HZ'],B['standard_HZ']*Sinv.dagger())
        self.assertNotEqual(S,eye(S.ring,S.nrows))
        self.assertFalse(B['actual_target_HX']*B['actual_target_HZ'].dagger())
        verify_generation(C['z_action']*B['source_HZ'].dagger(),B['actual_target_HZ'].dagger(),B['stabilizers']['Z'])
        verify_generation(C['x_action']*B['source_HX'].dagger(),B['actual_target_HX'].dagger(),B['stabilizers']['X'])
        report=verify_direct_result(artifact,expected_HX=self.raw['666'][0],expected_HZ=self.raw['666'][1])
        self.assertEqual(report['direct_mode'],'shift_separated')
        self.assertFalse(report['original_qca_implemented_by_circuit'])
        self.assertEqual(digest(result),before)

    def test_original_input_qca_lattice_target_and_witness_tampering(self):
        good=compile_qca_direct(self.qca['666'],transport='separate')
        def input_check(a):a['qca_result']['original']['HX'][0,0]+=1
        def original_qca(a):a['qca_result']['qca']['psi'][1][0,0]+=1
        def copied_qca(a):a['direct_circuit']['original_qca'][0,0]+=1
        def lattice(a):a['code_binding']['retained_lattice'][0][0]+=1
        def target_x(a):a['code_binding']['actual_target_HX'][0,0]+=1
        def target_z(a):a['code_binding']['actual_target_HZ'][0,0]+=1
        def standard(a):a['code_binding']['standard_HZ'][0,0]+=1
        def product_count(a):a['code_binding']['original_product_Z']+=1
        def count_type(a):a['code_binding']['added_Z_zero_ancillas']=False
        for change in [input_check,original_qca,copied_qca,lattice,target_x,target_z,standard,product_count,count_type]:
            with self.subTest(change=change.__name__):
                bad=clone(good);change(bad)
                with self.assertRaises((ValueError,ArithmeticError)):
                    verify_direct_result(bad)
        for sector in ('X','Z'):
            for direction in ('source_in_target','target_in_source'):
                with self.subTest(sector=sector,direction=direction):
                    bad=clone(good)
                    bad['code_binding']['stabilizers'][sector][direction][0,0]+=1
                    with self.assertRaises(ValueError):verify_direct_binding(bad)
        bad=clone(good);del bad['code_binding']['stabilizers']['Z']
        with self.assertRaises(ValueError):verify_direct_binding(bad)

    def test_verifier_does_not_call_constructing_or_factorizing_functions(self):
        good=compile_qca_direct(self.qca['toric'])
        with patch('css_decoupling.two_d.direct_pipeline.factor_unit_reduction',side_effect=AssertionError('factorizer called')):
            with patch('css_decoupling.two_d.pipeline.construct_qca',side_effect=AssertionError('QCA constructor called')):
                self.assertTrue(verify_direct_result(good)['direct_circuit'])

    def test_wrong_expected_raw_input_and_scope_tampering_reject(self):
        good=compile_qca_direct(self.qca['666'],transport='separate')
        with self.assertRaises(ValueError):
            verify_direct_result(good,expected_HX=self.raw['toric'][0])
        bad=clone(good);bad['construction']['transport_policy']='balance'
        with self.assertRaises(ValueError):verify_direct_binding(bad)
        bad=clone(good);bad['construction']['general_suslin_complete']=not bad['construction']['general_suslin_complete']
        with self.assertRaises(ValueError):verify_direct_binding(bad)
        bad=clone(good);bad['direct_circuit']['equivalence']['original_qca_implemented_by_circuit']=True
        with self.assertRaises(ValueError):verify_direct_binding(bad)

    def run_cli(self,*args):
        process=subprocess.run([sys.executable,'-I','-B','-m','css_decoupling.two_d.direct_cli',*map(str,args)],
                               capture_output=True,text=True,timeout=30)
        if process.returncode:
            self.fail(f'Direct CLI failed ({process.returncode}): {process.stderr}')
        return json.loads(process.stdout.splitlines()[-1])

    def test_cli_raw_run_compile_and_separate_reader_do_not_mutate_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            HX,HZ=self.raw['666']
            raw=root/'input.json';save({'HX':HX,'HZ':HZ,'p':2,'D':2},raw)
            raw_hash=file_sha(raw)
            run=root/'raw-run'
            status=self.run_cli('run',raw,'--output',run,'--transport','balance')
            self.assertEqual(status['status'],'complete')
            self.assertEqual(status['mode'],'exact')
            self.assertEqual(file_sha(raw),raw_hash)
            qca=run/'qca.json';qca_hash=file_sha(qca)
            compiled=root/'compiled'
            separated=self.run_cli('compile',qca,'--output',compiled,'--transport','separate')
            self.assertEqual(separated['mode'],'shift_separated')
            self.assertEqual(separated['status'],'complete_shift_separated')
            self.assertFalse(separated['original_qca_implemented_by_circuit'])
            self.assertEqual(file_sha(qca),qca_hash)
            artifact=compiled/'direct-result.json';artifact_hash=file_sha(artifact)
            report_path=root/'independent-report.json'
            report=self.run_cli('verify',artifact,'--input',raw,'--report',report_path)
            self.assertTrue(report['direct_circuit'] and report['qca'])
            self.assertFalse(report['original_qca_implemented_by_circuit'])
            self.assertEqual(load(report_path)['direct_mode'],'shift_separated')
            self.assertEqual(file_sha(artifact),artifact_hash)
            self.assertEqual(file_sha(raw),raw_hash)
            self.assertEqual(file_sha(qca),qca_hash)

    def test_cli_rejects_existing_output_without_writing_failure_into_it(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'existing';output.mkdir()
            sentinel=output/'keep.txt';sentinel.write_bytes(b'keep\n')
            before={p.name:p.read_bytes() for p in output.iterdir()}
            with redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):
                code=direct_cli(['example','empty','--output',str(output)])
            self.assertEqual(code,2)
            self.assertEqual({p.name:p.read_bytes() for p in output.iterdir()},before)

    def test_cli_verify_rejects_expected_input_metadata_and_unknown_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);artifact=root/'direct.json'
            save(compile_qca_direct(self.qca['toric']),artifact)
            HX,HZ=self.raw['toric']
            for i,extra in enumerate(({'p':3},{'D':True},{'ignored_option':1})):
                raw=root/f'bad-input-{i}.json'
                save({'HX':HX,'HZ':HZ,'p':2,'D':2,**extra},raw)
                before=file_sha(artifact)
                with redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):
                    code=direct_cli(['verify',str(artifact),'--input',str(raw)])
                self.assertEqual(code,2)
                self.assertEqual(file_sha(artifact),before)

    def test_factor_budget_failure_retains_verified_qca(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'limited'
            with redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):
                code=direct_cli(['example','666','--output',str(output),'--max-factors','1'])
            self.assertEqual(code,2)
            self.assertTrue((output/'qca.json').is_file())
            self.assertTrue(load(output/'qca-verification.json')['qca'])
            self.assertFalse((output/'direct-result.json').exists())
            self.assertEqual(load(output/'status.json')['status'],'resource_incomplete')


class PrimeDirectPipelineTests(unittest.TestCase):
    def transformed_code(self, prime):
        R=Ring(prime,2);x,y=R.variables
        HX,HZ=standard_code(1,product_x=1,product_z=1,ring=R)
        A,Ainv,S,Sinv=(eye(R,4) for _ in range(4))
        A[0,2]=2*x**-1*y;Ainv[0,2]=-A[0,2]
        S[0,0]=2*x;Sinv[0,0]=(2*x)**-1
        U,V=S*A,Ainv*Sinv
        return HX*U,HZ*V.dagger()

    def test_prime_raw_codes_auto_fixed_skew_and_both_transport_policies(self):
        for prime in (3,5):
            HX,HZ=self.transformed_code(prime)
            for options in ({'lattice_mode':'auto'},
                            {'lattice_mode':'fixed','lattice_basis':[[2,1],[0,1]]}):
                source=decouple(HX,HZ,compile_circuit=False,**options)
                before=digest(source)
                for policy in ('balance','separate'):
                    with self.subTest(prime=prime,options=options,policy=policy):
                        artifact=compile_qca_direct(source,transport=policy)
                        self.assertEqual(artifact['schema'],'css-decoupling-2d/direct-decoupling/v2')
                        report=verify_direct_result(clone(artifact),expected_HX=HX,expected_HZ=HZ)
                        self.assertTrue(report['verified'])
                        C=artifact['direct_circuit'];B=artifact['code_binding']
                        self.assertEqual(report['remaining_transport_coefficients'],C['transport']['coefficients'])
                        self.assertEqual(B['actual_target_HX'],B['standard_HX']*C['transport']['z_action'])
                        self.assertEqual(B['actual_target_HZ'],B['standard_HZ']*C['transport']['z_inverse'].dagger())
                        if policy=='balance':
                            self.assertEqual(C['mode'],'exact')
                            self.assertTrue(report['original_qca_implemented_by_circuit'])
                        self.assertEqual(digest(source),before)
                        bad=clone(artifact);bad['schema']='css-decoupling-2d/direct-decoupling/v1'
                        with self.assertRaises(ValueError):verify_direct_result(bad)
                        bad=clone(artifact);bad['construction']['method']='unit-pivots-and-decreasing-monomial-cancellations-v1'
                        bad['construction']['general_suslin_complete']=False
                        with self.assertRaises(ValueError):verify_direct_binding(bad)

    def test_binary_direct_wrapper_accepts_modern_binary_source_schema(self):
        HX,HZ=standard_code(1)
        source=decouple(HX,HZ,compile_circuit=False,lattice_mode='fixed',lattice_basis=[[1,1],[0,1]])
        self.assertEqual(source['schema'],'css-decoupling-2d/qca-circuit/v2')
        result=compile_qca_direct(source)
        self.assertEqual(result['schema'],'css-decoupling-2d/direct-decoupling/v1')
        self.assertTrue(verify_direct_result(result)['verified'])


if __name__=='__main__':
    unittest.main()
