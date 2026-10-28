"""Public JSON input/output for the general algorithm, without eval parsing."""
import json
from pathlib import Path

from css_decoupling.algebra.laurent import Ring
from css_decoupling.certificates.comparison import _keys, _int, _matrix_dict, _matrix_from_dict
from css_decoupling.certificates.decoupling import verify_decoupling


def input_dict(HX, HZ, *, lattice_basis=None):
    if HX.ring != HZ.ring:
        raise ValueError("Checks have different coefficient rings")
    return {"schema": "css-decoupling.general.input", "version": 1,
            "ring": {"prime": HX.ring.p, "dimension": HX.ring.D},
            "HX": _matrix_dict(HX), "HZ": _matrix_dict(HZ),
            "lattice_basis": lattice_basis}


def parse_input(data):
    _keys(data, ("schema", "version", "ring", "HX", "HZ", "lattice_basis"), "input")
    if data["schema"] != "css-decoupling.general.input" or _int(data["version"], "version") != 1:
        raise ValueError("Unsupported input schema")
    _keys(data["ring"], ("prime", "dimension"), "ring")
    ring = Ring(_int(data["ring"]["prime"], "prime", 2), _int(data["ring"]["dimension"], "dimension", 2))
    return (_matrix_from_dict(data["HX"], ring, "HX"),
            _matrix_from_dict(data["HZ"], ring, "HZ"), data["lattice_basis"])


def save_input(HX, HZ, path, *, lattice_basis=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(input_dict(HX, HZ, lattice_basis=lattice_basis), indent=2) + "\n")
    return path


def save_decoupling(result, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.certificate(), indent=2) + "\n")
    return path


def load_decoupling(path, *, verify=True):
    data = json.loads(Path(path).read_text())
    if verify:
        report = verify_decoupling(data)
        if not report.valid:
            raise ValueError("Invalid decoupling certificate: " + "; ".join(report.errors))
    return data
