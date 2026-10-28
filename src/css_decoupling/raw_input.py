"""Portable, strict sparse check-matrix input for both decoupling algorithms."""
from pathlib import Path
import json

from .algebra.laurent import Ring, Matrix
from .certificates.comparison import _matrix_dict, _matrix_from_dict
from .certificates.output_io import write_new_json_files

INPUT_SCHEMA = "css-decoupling.input"


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("Duplicate JSON key: " + key)
        result[key] = value
    return result


def read_json(path):
    def invalid(value):
        raise ValueError("Non-finite JSON number: " + value)
    return json.loads(Path(path).read_text(encoding="utf-8-sig"),
                      object_pairs_hook=_pairs, parse_constant=invalid)


def _lattice(mode, basis, dimension):
    if mode is None:
        mode = "fixed" if basis is not None else "auto"
    if mode not in ("auto", "fixed"):
        raise ValueError("lattice_mode must be auto or fixed")
    if (mode == "fixed") != (basis is not None):
        raise ValueError("fixed requires lattice_basis; auto does not accept one")
    if basis is not None:
        if (not isinstance(basis, (list, tuple)) or len(basis) != dimension or
                any(not isinstance(row, (list, tuple)) or len(row) != dimension or
                    any(type(value) is not int for value in row) for row in basis)):
            raise ValueError("lattice_basis must be a square integer matrix; basis vectors are columns")
        from sympy import Matrix as IntegerMatrix
        if IntegerMatrix(basis).det() == 0:
            raise ValueError("lattice_basis must have full rank")
        basis = [list(row) for row in basis]
    return mode, basis


def css_input_dict(HX, HZ, *, lattice_mode=None, lattice_basis=None):
    if not isinstance(HX, Matrix) or not isinstance(HZ, Matrix):
        raise TypeError("Use the public css_decoupling.Matrix type")
    if HX.ring != HZ.ring or HX.ncols != HZ.ncols or HX.ring.D < 2:
        raise ValueError("Checks must share a prime Laurent ring, dimension >= 2, and column count")
    mode, basis = _lattice(lattice_mode, lattice_basis, HX.ring.D)
    return {"schema": INPUT_SCHEMA, "version": 1,
            "ring": {"prime": HX.ring.p, "dimension": HX.ring.D},
            "HX": _matrix_dict(HX), "HZ": _matrix_dict(HZ),
            "lattice_mode": mode, "lattice_basis": basis}


def parse_css_input(data):
    """Return public matrices plus domain/lattice metadata; reject unknown fields.

    The former general input and the former two-dimensional raw input are
    accepted as import formats. New exports always use INPUT_SCHEMA.
    """
    if not isinstance(data, dict):
        raise ValueError("CSS input must be a JSON object")
    schema = data.get("schema")
    if schema == "css-decoupling.general.input":
        from .certificates.io import parse_input
        HX, HZ, basis = parse_input(data)
        mode = None
    elif schema == INPUT_SCHEMA:
        required = {"schema", "version", "ring", "HX", "HZ"}
        if not required <= set(data) or set(data) - required - {"lattice_mode", "lattice_basis"}:
            raise ValueError("Unknown or missing unified input field")
        if type(data["version"]) is not int or data["version"] != 1:
            raise ValueError("Unsupported input version")
        ring = data["ring"]
        if (not isinstance(ring, dict) or set(ring) != {"prime", "dimension"} or
                type(ring["prime"]) is not int or type(ring["dimension"]) is not int or ring["dimension"] < 2):
            raise ValueError("ring requires integer prime and dimension >= 2")
        R = Ring(ring["prime"], ring["dimension"])
        HX = _matrix_from_dict(data["HX"], R, "HX")
        HZ = _matrix_from_dict(data["HZ"], R, "HZ")
        mode, basis = data.get("lattice_mode"), data.get("lattice_basis")
    elif schema is None:
        from .two_d.serialization import decode
        from .two_d.cli import raw_input
        X, Z = raw_input(decode(data))
        R = Ring(X.ring.p, X.ring.D)
        def convert(A):
            return Matrix(R, A.nrows, A.ncols, {ij: R(dict(f.terms)) for ij, f in A.entries.items()})
        HX, HZ = convert(X), convert(Z)
        mode, basis = None, None
    else:
        raise ValueError("Unsupported CSS input schema")
    # Reuse the same complete validation as the canonical exporter.
    checked = css_input_dict(HX, HZ, lattice_mode=mode, lattice_basis=basis)
    return {"HX": HX, "HZ": HZ, "dimension": HX.ring.D, "p": HX.ring.p,
            "lattice_mode": checked["lattice_mode"], "lattice_basis": checked["lattice_basis"]}


def load_css_input(path):
    return parse_css_input(read_json(path))


def save_css_input(HX, HZ, path, *, lattice_mode=None, lattice_basis=None):
    data = css_input_dict(HX, HZ, lattice_mode=lattice_mode, lattice_basis=lattice_basis)
    return write_new_json_files({"input": data}, {"input": path})["input"]
