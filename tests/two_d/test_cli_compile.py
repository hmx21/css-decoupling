"""Compilation budgets survive the CLI through construction and verification."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from css_decoupling.two_d import cli
from css_decoupling.two_d.models import standard_code
from css_decoupling.two_d.pipeline import decouple
from css_decoupling.two_d.serialization import load, save
from css_decoupling.two_d.verify import verify


class CompileCliTests(unittest.TestCase):
    def run_compile(self, extra):
        directory = tempfile.TemporaryDirectory(prefix='css-2d-compile-cli-')
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        source, output = root / 'qca.json', root / 'result.json'
        HX, HZ = standard_code(1)
        save(decouple(HX, HZ, compile_circuit=False), source)
        original_bytes = source.read_bytes()
        with patch.object(cli, 'verify', wraps=cli.verify) as checked, \
             patch.object(cli, 'compile_result_circuit', wraps=cli.compile_result_circuit) as compiled, \
             redirect_stdout(io.StringIO()):
            code = cli.main(['compile', str(source), '--output', str(output), *extra])
        self.assertEqual(code, 0)
        self.assertEqual(source.read_bytes(), original_bytes)
        self.assertEqual(checked.call_count, 2)
        self.assertEqual(compiled.call_count, 1)
        return load(output), checked.call_args_list, compiled.call_args

    def test_explicit_budgets_reach_source_compiler_and_result_checks(self):
        result, checks, compilation = self.run_compile(
            ['--max-families', '37', '--max-terms', '1234'])
        expected = {'max_families': 37, 'max_matrix_terms': 1234}
        self.assertEqual(compilation.kwargs, expected)
        self.assertTrue(all(call.kwargs == expected for call in checks))
        report = verify(result, max_families=37, max_matrix_terms=1234)
        self.assertTrue(report['verified'])
        self.assertTrue(report['circuit'])

    def test_legacy_default_budgets_remain_compatible(self):
        result, checks, compilation = self.run_compile([])
        expected = {'max_families': 200000, 'max_matrix_terms': 1000000}
        self.assertEqual(compilation.kwargs, expected)
        self.assertTrue(all(call.kwargs == expected for call in checks))
        self.assertTrue(verify(result)['circuit'])

    def test_small_requested_budget_keeps_verified_qca_without_false_circuit(self):
        result, _, compilation = self.run_compile(
            ['--max-families', '1', '--max-terms', '1234'])
        self.assertEqual(compilation.kwargs['max_families'], 1)
        self.assertEqual(result['circuit_status'], 'resource_incomplete')
        self.assertIsNone(result['circuit'])
        report = verify(result, max_families=1, max_matrix_terms=1234)
        self.assertTrue(report['verified'])
        self.assertTrue(report['qca'])
        self.assertFalse(report['circuit'])


if __name__ == '__main__':
    unittest.main()
