"""All 153 pinned Julia table rows, including 93 without upstream completion.

Data from the upstream MIT-licensed repository; see LICENSE and notices.
Raw source rows and source file hashes remain available for provenance.
"""
from importlib.resources import files
import json
from .algebra import Ring, Matrix


def table_records():
    data = json.loads(files("css_decoupling.two_d").joinpath("data/upstream-bb-tables.json").read_text(encoding="utf8"))
    return data["records"]


def identifier(record):
    if record["native_identifiers"]:
        return record["native_identifiers"][0]
    prefix = "julia-additional-" if record["source_file"].startswith("additional_") else "julia-main-"
    return prefix+record["source_case_id"].replace("_","-")


def additional_inputs():
    return {identifier(row):row for row in table_records() if not row["upstream_completed"]}


def checks(record):
    R=Ring(2,2)
    f=sum((R.monomial(e) for e in record["f_exponents"]),R.zero())
    g=sum((R.monomial(e) for e in record["g_exponents"]),R.zero())
    return Matrix.from_rows(R,[[f,g]]),Matrix.from_rows(R,[[g.involution(),f.involution()]])
