"""Exact certificates for Postnikov gluing relative to recorded lower bases."""
from css_decoupling.algebra.laurent import Ring, eye
from css_decoupling.algebra.modules import solve
from css_decoupling.topology.cohomology import cohomology_layer
from css_decoupling.topology.complexes import CochainComplex, CochainMap, koszul_window, toric_window, mapping_cone
from .comparison import (_int, _keys, _matrix_dict, _matrix_from_dict, VerificationReport)


def gluing_dict(step):
    C = step.lower_splitting.target
    return {
        "schema": "css-decoupling.gluing", "version": 1,
        "ring": {"prime": C.ring.p, "dimension": C.ring.D},
        "upper": step.upper,
        "complex": {"ranks": [[n, r] for n, r in sorted(C.ranks.items())],
                    "differentials": [[n, _matrix_dict(d)] for n, d in sorted(C.differentials.items())]},
        "representatives": [[n, _matrix_dict(a)] for n, a in sorted(step.representatives.items())],
        "lower_splitting": [[n, _matrix_dict(a)] for n, a in sorted(step.lower_splitting.components.items())],
        "cone_lift": [[n, _matrix_dict(a)] for n, a in sorted(step.cone_lift.components.items())],
        "classes": [{"lower": k.lower, "ext_degree": k.ext_degree,
                     "exterior_basis": [list(a) for a in k.exterior_basis],
                     "coefficients": _matrix_dict(k.coefficients)} for k in step.classes],
        "convention": "Cone(Phi)^n=C^n+T^(n+1); projection to T[1]; lexicographic exterior axes, then source copy",
    }


def _pairs(values, read, what):
    if not isinstance(values, list):
        raise ValueError(f"{what} must be an ordered pair list")
    result = {}
    for pair in values:
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError(f"Malformed {what} entry")
        degree = _int(pair[0], what + " degree", 0)
        if degree in result:
            raise ValueError(f"Duplicate {what} degree")
        result[degree] = read(pair[1])
    return result


def verify_gluing(data, *, max_layer_dimension=10000):
    """Recheck bases, lower splitting, cone lift and all augmentation classes."""
    checks, errors = [], []
    try:
        _keys(data, ("schema", "version", "ring", "upper", "complex", "representatives",
                     "lower_splitting", "cone_lift", "classes", "convention"), "gluing certificate")
        if data["schema"] != "css-decoupling.gluing" or _int(data["version"], "version") != 1:
            raise ValueError("Unsupported gluing certificate")
        _keys(data["ring"], ("prime", "dimension"), "ring")
        ring = Ring(_int(data["ring"]["prime"], "prime", 2), _int(data["ring"]["dimension"], "dimension", 2))
        upper = _int(data["upper"], "upper", 2)
        read = lambda a: _matrix_from_dict(a, ring, "matrix")
        _keys(data["complex"], ("ranks", "differentials"), "complex")
        C = CochainComplex(ring,
            _pairs(data["complex"]["ranks"], lambda r: _int(r, "rank", 0), "ranks"),
            _pairs(data["complex"]["differentials"], read, "differentials"))
        reps = _pairs(data["representatives"], read, "representatives")
        if upper not in reps or any(i < 1 or i > upper for i in reps) or len(reps) < 2:
            raise ValueError("Invalid represented layer degrees")
        for j in range(1, upper + 1):
            layer = cohomology_layer(C, j, max_dimension=max_layer_dimension)
            if not layer.dimension:
                if j in reps:
                    raise ValueError("A zero layer was recorded as nonzero")
                continue
            if j not in reps or reps[j].shape != (C.rank(j), layer.dimension):
                raise ValueError("Missing or incorrectly sized cohomology basis")
            if C.d(j) * reps[j]:
                raise ValueError("Representatives are not cycles")
            coordinates = layer.quotient.coordinates(solve(layer.cycles, reps[j]))
            solve(coordinates, eye(ring, layer.dimension))
            if any(a != eye(ring, layer.dimension) for a in layer.actions):
                raise ValueError("Layers must have trivial translation action")
        checks.append("actual cohomology bases and trivial translation actions through upper degree")
        lower = {i: r.ncols for i, r in reps.items() if i < upper}
        T = toric_window(ring, lower, lowest_degree=0)
        phi = CochainMap(T, C, _pairs(data["lower_splitting"], read, "lower splitting"))
        for i in sorted(lower):
            offset = sum(koszul_window(ring, k, lower[k], lowest_degree=0).rank(i)
                         for k in lower if k < i)
            if phi.at(i)[:, offset:offset + lower[i]] != reps[i]:
                raise ValueError("Lower splitting does not induce the recorded bases")
        source = koszul_window(ring, upper, reps[upper].ncols, lowest_degree=0)
        lift = CochainMap(source, mapping_cone(phi), _pairs(data["cone_lift"], read, "cone lift"))
        if lift.at(upper) != reps[upper]:
            raise ValueError("Cone lift does not induce the recorded upper basis")
        checks.append("lower splitting and connecting-map Koszul lift equations")
        # Reconstruct only the coordinate projection; no constructor or solve
        # for a splitting is called by the verifier.
        from itertools import combinations
        classes = data["classes"]
        if not isinstance(classes, list) or len(classes) != len(lower):
            raise ValueError("Wrong number of gluing components")
        for expected_i, item in zip(sorted(lower), classes):
            _keys(item, ("lower", "ext_degree", "exterior_basis", "coefficients"), "class")
            i = _int(item["lower"], "lower", 1)
            order = upper - i + 1
            if i != expected_i or _int(item["ext_degree"], "Ext degree", 1) != order:
                raise ValueError("Wrong gluing component degree")
            if item["exterior_basis"] != [list(a) for a in combinations(range(ring.D), order)]:
                raise ValueError("Wrong exterior basis convention")
            for axes in item["exterior_basis"]:
                for axis in axes:
                    _int(axis, "exterior axis", 0)
            offset = C.rank(i - 1) + sum(koszul_window(ring, k, lower[k], lowest_degree=0).rank(i)
                                       for k in lower if k < i)
            projected = lift.at(i - 1)[offset:offset + lower[i], :]
            coefficients = read(item["coefficients"])
            if coefficients.shape != projected.shape:
                raise ValueError("Wrong gluing coefficient shape")
            for row in range(projected.nrows):
                for column in range(projected.ncols):
                    if coefficients[row, column] != ring(sum(projected[row, column].terms.values())):
                        raise ValueError("Gluing coefficients do not equal the connecting cocycle")
        if data["convention"] != "Cone(Phi)^n=C^n+T^(n+1); projection to T[1]; lexicographic exterior axes, then source copy":
            raise ValueError("Unsupported gluing convention")
        checks.append("all kappa components in the canonical Koszul exterior coordinate basis")
    except (ValueError, TypeError, KeyError, AttributeError, IndexError) as exc:
        errors.append(str(exc))
    return VerificationReport(not errors, tuple(checks), tuple(errors), "Postnikov-gluing-relative-to-lower-splitting")
