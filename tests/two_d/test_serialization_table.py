import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from css_decoupling.two_d import Ring,Matrix,save,load,verify,decouple
from css_decoupling.two_d.models import standard_code
from css_decoupling.two_d.serialization import decode,digest


class MatrixTableTests(unittest.TestCase):
    def test_roundtrip_sharing_and_legacy_transport(self):
        R=Ring(2,2);A=Matrix.from_rows(R,[[R.variable(0)**-2,1]])
        value={"a":A,"b":A.copy(),"degree":{0:A},"gates":[{"offset":[-1,2]}]}
        with tempfile.TemporaryDirectory() as directory:
            compact=Path(directory)/'compact.json';legacy=Path(directory)/'legacy.json'
            save(value,compact);save(value,legacy,compact=False)
            raw=json.loads(compact.read_text());self.assertEqual(len(raw['matrices']),1)
            first,second=load(compact),load(legacy)
            self.assertEqual(first,second);self.assertEqual(digest(first),digest(value))
            self.assertIs(first['a'],first['b'])
            self.assertFalse(compact.with_name('compact.json.partial').exists())
            with self.assertRaises(FileExistsError):save(value,compact)

    def test_reference_and_matrix_tampering(self):
        R=Ring(2,2);A=Matrix.from_rows(R,[[1]])
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'x.json';save({'a':A},path);original=json.loads(path.read_text())
        for ref in (-1,1,True,0.0,'0'):
            data=copy.deepcopy(original);data['payload']['a']['__matrix_ref__']=ref
            with self.subTest(ref=ref),self.assertRaises(ValueError):decode(data)
        data=copy.deepcopy(original);data['matrices'].append(data['matrices'][0])
        with self.assertRaises(ValueError):decode(data)
        data=copy.deepcopy(original);data['matrices'][0]['entries'][0][2][0][-1]=2
        with self.assertRaises(ValueError):decode(data)
        data=copy.deepcopy(original);data['__css_matrix_table__']=True
        with self.assertRaises(ValueError):decode(data)
        with self.assertRaises(ValueError):decode({'__matrix_ref__':0})

    def test_failure_never_publishes_partial_certificate(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'bad.json'
            with self.assertRaises(ValueError):save({'nan':float('nan')},path)
            self.assertFalse(path.exists());self.assertTrue(path.with_name('bad.json.partial').exists())
            with self.assertRaises(FileExistsError):save({'good':1},path)

    def test_full_certificate_table_math_validation(self):
        result=decouple(*standard_code(2,product_x=1,product_z=1),compile_circuit=True)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'result.json';save(result,path)
            self.assertTrue(verify(load(path))['circuit'])

    def test_decimal_overflow_is_not_finite_json(self):
        with tempfile.TemporaryDirectory() as directory:
            for i,text in enumerate(('1e999','-1e999','NaN','Infinity')):
                path=Path(directory)/f'{i}.json';path.write_text('{"time":'+text+'}',encoding='utf8')
                with self.subTest(text=text),self.assertRaises(ValueError):load(path)


if __name__ == '__main__':unittest.main()
