"""Independent source-bound witnesses for the specified-lattice criterion.

Verification uses the supplied free model and maps, exact module membership,
actual cohomology, and the candidate's action. It never calls the constructor,
selects a lattice, or computes a complete common translation kernel.
"""
from types import SimpleNamespace

from css_decoupling.algebra.laurent import Ring, eye
from css_decoupling.algebra.modules import get_backend, kernel, solve, using_backend
from css_decoupling.errors import ResourceLimit
from css_decoupling.lattice.retained import Lattice, matrix_power
from css_decoupling.topology.cohomology import cohomology_layer
from css_decoupling.topology.complexes import CochainComplex, CochainMap, koszul_window, toric_window
from .analysis import complex_dict, _model
from .comparison import _keys, _int, _matrix_dict, _matrix_from_dict, VerificationReport
from .gluing import _pairs, verify_gluing

SCHEMA = "css-decoupling.lattice-criterion"
SCOPE = "original-CSS-to-specified-lattice-profile-splitting-criterion"


def criterion_dict(result):
    """Serialize current public claims, never a cached successful snapshot."""
    obstruction = result.obstruction
    if obstruction is not None:
        obstruction = dict(obstruction)
        if result.reason == "infinite-layer":
            for name in ("cycles", "boundary_coordinates", "cycle_relations", "presentation"):
                obstruction[name] = _matrix_dict(obstruction[name])
            obstruction["ray"] = dict(obstruction["ray"])
        elif result.reason == "nontrivial-layer-translation":
            obstruction["action"] = _matrix_dict(obstruction["action"])
    return {
        "schema": SCHEMA, "version": 1, "scope": SCOPE,
        "ring": {"prime": result.HX.ring.p, "dimension": result.HX.ring.D},
        "input": {"HX": _matrix_dict(result.HX), "HZ": _matrix_dict(result.HZ)},
        "sector": result.sector,
        "basis": [list(row) for row in result.lattice.basis],
        "decouplable": result.decouplable, "reason": result.reason,
        "model": complex_dict(result.complex),
        "layers": [{"degree": j, "representatives": _matrix_dict(layer.representatives),
                    "actions": [_matrix_dict(a) for a in layer.actions]}
                   for j, layer in sorted(result.layers.items())],
        "obstruction": obstruction,
        "restricted": None if result.restricted is None else complex_dict(result.restricted),
        "attachments": [step.certificate() for step in result.gluing],
        "splitting": (None if result.phi is None else
                      [[n, _matrix_dict(a)] for n, a in sorted(result.phi.components.items())]),
    }


def _validate_cap(value, name):
    if value is not None:
        _int(value, name, 1)


def _check_splitting_cap(C, dimensions, max_unknowns):
    if max_unknowns is None:
        return
    for degree, dimension in dimensions.items():
        T = koszul_window(C.ring, degree, dimension, lowest_degree=0)
        size = sum(C.rank(n) * T.rank(n) for n in T.degrees if n < degree)
        if size > max_unknowns:
            raise ResourceLimit(f"Splitting witness has {size} Laurent entries; cap is {max_unknowns}")


def _check_infinite(C, raw, read):
    """Verify a full quotient presentation and an infinite standard ray."""
    _keys(raw, ("degree", "cycles", "boundary_coordinates", "cycle_relations", "presentation", "ray"),
          "infinite-layer witness")
    degree = _int(raw["degree"], "infinite layer degree", 1)
    if degree not in C.degrees:
        raise ValueError("Infinite layer degree is outside the source model")
    cycles, boundary, relations, presentation = (
        read(raw[name]) for name in ("cycles", "boundary_coordinates", "cycle_relations", "presentation"))
    if cycles.nrows != C.rank(degree) or C.d(degree) * cycles:
        raise ValueError("Infinite-layer cycle generators are not cycles")
    solve(cycles, kernel(C.d(degree)))
    if cycles * boundary != C.d(degree - 1):
        raise ValueError("Infinite-layer presentation does not encode the actual boundaries")
    if cycles * relations:
        raise ValueError("Infinite-layer cycle relations are false")
    solve(relations, kernel(cycles))
    if presentation != boundary.hstack(relations):
        raise ValueError("Infinite-layer presentation changed the actual cohomology quotient")
    _keys(raw["ray"], ("component", "axis"), "standard monomial ray")
    component = _int(raw["ray"]["component"], "ray component", 0)
    axis = _int(raw["ray"]["axis"], "ray axis", 0)
    if component >= presentation.nrows or axis > C.ring.D:
        raise ValueError("Standard monomial ray is outside the localized free module")
    leading = get_backend(C.ring).quotient_groebner(presentation)["leading_terms"]
    for term in leading:
        if term[0] == component and all(not term[k + 1] for k in range(C.ring.D + 1) if k != axis):
            raise ValueError("A leading relation intersects the claimed infinite standard ray")
    return degree


def verify_lattice_criterion(data, *, max_cell_index=4096,
                             max_layer_dimension=10000, max_unknowns=250000,
                             backend=None, resolution_timeout=120):
    """Check positive or negative witnesses; resource failures propagate.

    ``max_cell_index`` only applies when the witness needs cell expansion.
    Infinite modules and nontrivial action matrices are checked in the
    original cell, even for a candidate whose index exceeds that cap.
    """
    if backend is not None:
        with using_backend(backend, timeout=resolution_timeout):
            return verify_lattice_criterion(data, max_cell_index=max_cell_index,
                    max_layer_dimension=max_layer_dimension, max_unknowns=max_unknowns,
                    resolution_timeout=resolution_timeout)
    checks, errors = [], []
    try:
        for cap, name in ((max_cell_index, "max_cell_index"),
                          (max_layer_dimension, "max_layer_dimension"),
                          (max_unknowns, "max_unknowns")):
            _validate_cap(cap, name)
        if not isinstance(data, dict) and hasattr(data, "certificate"):
            data = data.certificate()
        _keys(data, ("schema", "version", "scope", "ring", "input", "sector", "basis",
                     "decouplable", "reason", "model", "layers", "obstruction", "restricted",
                     "attachments", "splitting"), "lattice criterion")
        if data["schema"] != SCHEMA or _int(data["version"], "version") != 1 or data["scope"] != SCOPE:
            raise ValueError("Unsupported lattice criterion schema, version, or scope")
        _keys(data["ring"], ("prime", "dimension"), "ring")
        ring = Ring(_int(data["ring"]["prime"], "prime", 2),
                    _int(data["ring"]["dimension"], "dimension", 2))
        lattice = Lattice(data["basis"], max_index=None)
        if lattice.D != ring.D:
            raise ValueError("Lattice and ring dimensions differ")
        read = lambda a: _matrix_from_dict(a, ring, "matrix")
        _keys(data["input"], ("HX", "HZ"), "input")
        HX, HZ = read(data["input"]["HX"]), read(data["input"]["HZ"])
        if data["sector"] not in ("X", "Z"):
            raise ValueError("Criterion sector must be X or Z")
        if data["sector"] == "Z":
            HX, HZ = HZ, HX
        if HX.ncols != HZ.ncols or HX * HZ.dagger():
            raise ValueError("Original CSS checks do not commute or have inconsistent columns")
        C = _model(data["model"], ring)
        if C.d(0) != HX or C.d(-1) != HZ.dagger():
            raise ValueError("Source model changed the original CSS window")
        solve(HZ.dagger(), kernel(HX))
        solve(HX.dagger(), kernel(HZ))
        for n in range(1, max(C.degrees) + 1):
            solve(C.d(n).dagger(), kernel(C.d(n - 1).dagger()))
        checks.append("source-bound CSS commutation, physical exactness, and finite free model")
        reason = data["reason"]
        reasons = ("split-profile", "infinite-layer", "nontrivial-layer-translation", "nonzero-gluing")
        if reason not in reasons or type(data["decouplable"]) is not bool or data["decouplable"] != (reason == "split-profile"):
            raise ValueError("Decision and obstruction kind disagree")
        if not isinstance(data["attachments"], list):
            raise ValueError("Attachments must be an ordered list")
        bound = None
        if reason == "infinite-layer":
            bound = _check_infinite(C, data["obstruction"], read)
        actual = {}
        for degree in C.degrees:
            if degree <= 0 or (bound is not None and degree >= bound):
                continue
            layer = cohomology_layer(C, degree, max_dimension=max_layer_dimension)
            if layer.dimension:
                if degree >= ring.D:
                    raise ValueError("Nonzero layer outside the topological CSS degree bound")
                actual[degree] = layer
        if not isinstance(data["layers"], list):
            raise ValueError("Layers must be an ordered list")
        layers = {}
        for raw in data["layers"]:
            _keys(raw, ("degree", "representatives", "actions"), "layer")
            degree = _int(raw["degree"], "layer degree", 1)
            if degree not in actual or degree in layers:
                raise ValueError("Duplicate or spurious cohomology layer")
            layer, reps = actual[degree], read(raw["representatives"])
            if reps.shape != (C.rank(degree), layer.dimension) or C.d(degree) * reps:
                raise ValueError("Layer representatives are not cycles of the actual dimension")
            change = layer.quotient.coordinates(solve(layer.cycles, reps))
            solve(change, eye(ring, layer.dimension))
            if not isinstance(raw["actions"], list) or len(raw["actions"]) != ring.D:
                raise ValueError("Missing original translation matrices")
            actions = tuple(read(a) for a in raw["actions"])
            for given, computed in zip(actions, layer.actions):
                if given.shape != (layer.dimension, layer.dimension) or any(not f.is_constant for f in given.entries.values()):
                    raise ValueError("Layer translation matrices must be constant and correctly sized")
                if change * given != computed * change:
                    raise ValueError("Recorded translations are not the actual action on the recorded basis")
            layers[degree] = SimpleNamespace(dimension=layer.dimension, representatives=reps, actions=actions)
        if set(layers) != set(actual):
            raise ValueError("Criterion omitted a nonzero layer in the required profile")
        checks.append("actual finite cohomology bases and translation actions")
        if reason == "infinite-layer":
            if data["restricted"] is not None or data["splitting"] is not None or data["attachments"]:
                raise ValueError("An infinite-layer obstruction cannot assert later gluing or a splitting")
            checks.append("full cohomology presentation and independently recomputed infinite standard-monomial ray")
            return VerificationReport(True, tuple(checks), (), SCOPE)
        actions = {}
        for degree, layer in sorted(layers.items()):
            identity = eye(ring, layer.dimension)
            for column, vector in enumerate(zip(*lattice.basis)):
                action = identity
                for matrix, exponent in zip(layer.actions, vector):
                    action = action * matrix_power(matrix, exponent)
                actions[degree, column] = action
        if reason == "nontrivial-layer-translation":
            raw = data["obstruction"]
            _keys(raw, ("degree", "column", "action"), "translation obstruction")
            degree = _int(raw["degree"], "obstructed degree", 1)
            column = _int(raw["column"], "candidate column", 0)
            if (degree, column) not in actions:
                raise ValueError("Translation obstruction is outside the profile or candidate basis")
            action = read(raw["action"])
            if action != actions[degree, column] or action == eye(ring, layers[degree].dimension):
                raise ValueError("Claimed translation obstruction is not the actual nonidentity action")
            if data["restricted"] is not None or data["splitting"] is not None or data["attachments"]:
                raise ValueError("Translation obstruction cannot assert later gluing or a splitting")
            checks.append("candidate lattice has a verified nonidentity action before cell expansion")
            return VerificationReport(True, tuple(checks), (), SCOPE)
        if any(action != eye(ring, layers[degree].dimension) for (degree, _), action in actions.items()):
            raise ValueError("Gluing criterion requires trivial candidate translations on every layer")
        if max_cell_index is not None and lattice.index > max_cell_index:
            raise ResourceLimit(f"Coarse unit cell has index {lattice.index}; cap is {max_cell_index}")
        restricted = CochainComplex(ring,
            {n: rank * lattice.index for n, rank in C.ranks.items()},
            {n: lattice.restrict(d) for n, d in C.differentials.items()})
        if data["restricted"] != complex_dict(restricted):
            raise ValueError("Recorded restriction is not the actual source on the specified lattice")
        window = CochainComplex(ring, {n: rank for n, rank in restricted.ranks.items() if n >= 0},
                               {n: d for n, d in restricted.differentials.items() if n >= 0})
        reps = {j: lattice.pack(layer.representatives) for j, layer in layers.items()}
        degrees = sorted(layers)[1:]
        required = degrees
        if reason == "nonzero-gluing":
            _keys(data["obstruction"], ("degree",), "gluing obstruction")
            upper = _int(data["obstruction"]["degree"], "first unsplit degree", 2)
            if upper not in degrees:
                raise ValueError("Gluing obstruction is not an attachment stage")
            required = degrees[:degrees.index(upper) + 1]
            if data["splitting"] is not None:
                raise ValueError("Nonzero gluing cannot carry a final splitting")
        elif data["obstruction"] is not None:
            raise ValueError("A split profile must not contain an obstruction")
        if len(data["attachments"]) != len(required):
            raise ValueError("Missing or extraneous stages before the first gluing obstruction")
        for degree, witness in zip(required, data["attachments"]):
            if not isinstance(witness, dict) or witness.get("upper") != degree:
                raise ValueError("Gluing witnesses must follow increasing nonzero layer degrees")
            if witness.get("complex") != complex_dict(window):
                raise ValueError("Gluing witness changed the actually restricted source complex")
            expected = [[j, _matrix_dict(reps[j])] for j in sorted(reps) if j <= degree]
            if witness.get("representatives") != expected:
                raise ValueError("Gluing witness changed the actual restricted layer bases")
            lower = {j: layer.dimension for j, layer in layers.items() if j < degree}
            _check_splitting_cap(window, lower, max_unknowns)
            if max_unknowns is not None:
                T = toric_window(ring, lower, lowest_degree=0)
                source = koszul_window(ring, degree, layers[degree].dimension, lowest_degree=0)
                size = sum((window.rank(n) + T.rank(n + 1)) * source.rank(n) for n in source.degrees)
                if size > max_unknowns:
                    raise ResourceLimit(f"Gluing witness has {size} Laurent entries; cap is {max_unknowns}")
            report = verify_gluing(witness, max_layer_dimension=max_layer_dimension)
            if not report.valid:
                raise ValueError("Invalid source-bound gluing witness: " + "; ".join(report.errors))
            nonzero = any(bool(read(c["coefficients"])) for c in witness["classes"])
            expected_nonzero = reason == "nonzero-gluing" and degree == required[-1]
            if nonzero != expected_nonzero:
                raise ValueError("The attachment sequence does not identify the first nonzero gluing")
        checks.append("successive source-bound lower splittings, cone lifts, and actual conditional gluing classes")
        if reason == "nonzero-gluing":
            return VerificationReport(True, tuple(checks), (), SCOPE)
        dimensions = {j: layer.dimension for j, layer in layers.items()}
        _check_splitting_cap(window, dimensions, max_unknowns)
        T = toric_window(ring, dimensions, lowest_degree=0)
        phi = CochainMap(T, window, _pairs(data["splitting"], read, "final splitting"))
        for degree in sorted(layers):
            offset = sum(koszul_window(ring, j, dimensions[j], lowest_degree=0).rank(degree)
                         for j in dimensions if j < degree)
            if phi.at(degree)[:, offset:offset + dimensions[degree]] != reps[degree]:
                raise ValueError("Final splitting fails to induce the recorded layer basis")
        checks.append("complete toric-profile cochain splitting inducing every actual cohomology basis")
    except (ValueError, TypeError, KeyError, AttributeError, IndexError) as exc:
        errors.append(str(exc))
    return VerificationReport(not errors, tuple(checks), tuple(errors), SCOPE)
