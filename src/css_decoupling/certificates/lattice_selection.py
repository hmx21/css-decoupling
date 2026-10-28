"""Independent verification of a killing lattice and its stated optimum.

The verifier checks actual coefficients, subspace equations and integer
indices. It does not invoke select_lattice or trust a saved success flag.
"""
from css_decoupling.algebra.exterior import integer_rows
from css_decoupling.algebra.laurent import Ring
from css_decoupling.lattice.retained import Lattice
from css_decoupling.lattice.finite_algebra import rref
from .comparison import _int, _keys, _matrix_dict, _matrix_from_dict, VerificationReport

SCHEMA = "css-decoupling.lattice-selection"
SCOPE = "given-translation-trivial-class-family"


def selection_dict(selection):
    return {"schema": SCHEMA, "version": 1,
            "ring": {"prime": selection.p, "dimension": selection.D},
            "scope": SCOPE,
            "classes": [{"upper": int(c.upper), "lower": int(c.lower),
                         "source_dimension": int(c.source_dimension), "target_dimension": int(c.target_dimension),
                         "coefficients": _matrix_dict(c.coefficients)} for c in selection.classes],
            "relative_basis": [list(row) for row in selection.relative_basis],
            "relative_index": selection.relative_index,
            "method": selection.method, "optimality": selection.optimality,
            "witness": {k: [list(row) if isinstance(row, (tuple, list)) else row for row in v]
                        if isinstance(v, (tuple, list)) else v for k, v in selection.witness.items()}}


def verify_lattice_selection(classes, certificate, *, max_exterior_entries=2_000_000):
    """Verify coefficients, annihilation, preimage lattice and any optimum."""
    checks, errors = [], []
    try:
        from css_decoupling.decoupling.gluing import GluingClass
        data = _keys(certificate, ("schema", "version", "ring", "scope", "classes", "relative_basis",
                                  "relative_index", "method", "optimality", "witness"), "lattice selection")
        if data["schema"] != SCHEMA or _int(data["version"], "version") != 1 or data["scope"] != SCOPE:
            raise ValueError("Unsupported lattice-selection certificate")
        _keys(data["ring"], ("prime", "dimension"), "ring")
        D, p = _int(data["ring"]["dimension"], "dimension", 1), _int(data["ring"]["prime"], "prime", 2)
        ring, classes = Ring(p, D), tuple(classes)
        if not isinstance(data["classes"], list) or len(data["classes"]) != len(classes):
            raise ValueError("Certificate omits or adds a class")
        for cls, raw in zip(classes, data["classes"]):
            if not isinstance(cls, GluingClass) or cls.coefficients.ring != ring:
                raise ValueError("Certificate ring does not match the class family")
            cls.__post_init__()
            _keys(raw, ("upper", "lower", "source_dimension", "target_dimension", "coefficients"), "class")
            for name in ("upper", "lower", "source_dimension", "target_dimension"):
                if _int(raw[name], name, 0) != getattr(cls, name):
                    raise ValueError("Certificate changed class degrees or dimensions")
            if _matrix_from_dict(raw["coefficients"], ring, "coefficients") != cls.coefficients:
                raise ValueError("Certificate coefficients are not the supplied classes")
        checks.append("exact coefficient family, degrees, prime and dimensions")
        witness = data["witness"]
        method = data["method"]
        fields = {"zero-family": ("subspace_basis",),
                  "d3-two-form": ("subspace_basis", "coefficient_rank", "kernel_vector"),
                  "coordinate-cover": ("subspace_basis", "axes")}
        if method not in fields:
            raise ValueError("Unknown lattice-selection method")
        _keys(witness, fields[method], "selection witness")
        W = integer_rows(witness["subspace_basis"], nrows=D, name="subspace basis")
        dimension = len(rref(list(zip(*W)), D, p)[1])
        if len(W[0]) != dimension:
            raise ValueError("Subspace basis has dependent columns")
        lattice = Lattice(data["relative_basis"], max_index=None)
        if lattice.D != D or _int(data["relative_index"], "relative index", 1) != lattice.index:
            raise ValueError("Wrong lattice dimension or claimed index")
        combined = list(zip(*W)) + list(zip(*lattice.basis))
        if len(rref(combined, D, p)[1]) != dimension or lattice.index != p ** (D - dimension):
            raise ValueError("Integer lattice is not the full preimage of the stated subspace")
        for cls in classes:
            if not cls.vanishes_on(W, max_exterior_entries=max_exterior_entries):
                raise ValueError("The stated subspace leaves nonzero gluing")
        checks.append("exterior annihilation and exact integer preimage lattice")
        nonzero = any(not c.is_zero for c in classes)
        if method == "zero-family":
            if nonzero or dimension != D or data["optimality"] != "family-optimal":
                raise ValueError("Invalid zero-family optimum")
        elif method == "d3-two-form":
            if D != 3 or not nonzero or any(not c.is_zero and c.ext_degree != 2 for c in classes):
                raise ValueError("Three-dimensional two-form theorem does not apply")
            rows = []
            for cls in classes:
                if cls.ext_degree != 2:
                    continue
                slots = {}
                for (i, j), f in cls.coefficients.entries.items():
                    block, source = divmod(j, cls.source_dimension)
                    slots.setdefault((i, source), [0, 0, 0])[block] = f.constant()
                rows.extend(slots.values())
            rank = len(rref(rows, 3, p)[1])
            if _int(witness["coefficient_rank"], "coefficient rank", 1) != rank:
                raise ValueError("Incorrect coefficient rank")
            expected_dimension = 1 if rank == 3 else 2
            if dimension != expected_dimension or data["optimality"] != "family-optimal":
                raise ValueError("Index does not attain the three-dimensional lower bound")
            vector = witness["kernel_vector"]
            if not isinstance(vector, list) or len(vector) != (0 if rank == 3 else 3):
                raise ValueError("Incorrect kernel witness size")
            vector = [_int(x, "kernel coefficient", 0) for x in vector]
            if any(x >= p for x in vector):
                raise ValueError("Noncanonical prime-field kernel coefficient")
            if vector:
                if not any(vector) or any(sum(a * b for a, b in zip(row, vector)) % p for row in rows):
                    raise ValueError("Vector is not a nonzero coefficient relation")
                normal = vector[2], -vector[1], vector[0]
                if any(sum(normal[i] * W[i][j] for i in range(3)) % p for j in range(dimension)):
                    raise ValueError("Plane does not correspond to the relation bivector")
        else:
            axes = witness["axes"]
            if not isinstance(axes, list):
                raise ValueError("Coordinate axes must be a list")
            axes = [_int(a, "coordinate axis", 0) for a in axes]
            if axes != sorted(set(axes)) or any(a >= D for a in axes):
                raise ValueError("Invalid coordinate axes")
            expected = tuple(tuple(int(i == j) for j in range(D) if j not in axes) for i in range(D))
            if W != expected or not nonzero or data["optimality"] != "sufficient":
                raise ValueError("Invalid coordinate-cover witness or unsupported optimality claim")
        checks.append("family optimum proved" if data["optimality"] == "family-optimal" else "sufficiency only; no minimum-index claim")
    except (ValueError, TypeError, KeyError, AttributeError, IndexError) as exc:
        errors.append(str(exc))
    return VerificationReport(not errors, tuple(checks), tuple(errors), SCOPE)
