import unittest
from css_decoupling.two_d.catalog import table_records, additional_inputs, checks, identifier
from css_decoupling.two_d.models import PUBLISHED_BB_CODE_INSTANCES


class CatalogTests(unittest.TestCase):
    def test_all_published_rows_and_completed_crosswalk(self):
        records=table_records()
        self.assertEqual(len(records),153)
        self.assertEqual(len(additional_inputs()),93)
        self.assertEqual(len({identifier(r) for r in records}),153)
        known={case.identifier:case for case in PUBLISHED_BB_CODE_INSTANCES}
        self.assertEqual(len(known),60)
        for record in records:
            HX,HZ=checks(record)
            self.assertFalse(HX*HZ.dagger())
            if record["upstream_completed"]:
                self.assertEqual((HX,HZ),known[identifier(record)].checks())


if __name__ == "__main__":
    unittest.main()
