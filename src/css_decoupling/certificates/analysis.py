"""Persist and verify exact Lambda0 and conditional gluing diagnostics."""
import json
from pathlib import Path
from types import SimpleNamespace

from css_decoupling.algebra.laurent import Ring, eye, zeros
from css_decoupling.algebra.modules import kernel, solve
from css_decoupling.lattice.retained import Lattice, matrix_power
from css_decoupling.topology.cohomology import profile
from css_decoupling.topology.complexes import CochainComplex, CochainMap, toric_window
from .comparison import _keys, _int, _matrix_dict, _matrix_from_dict, VerificationReport
from .gluing import verify_gluing, _pairs
from .translation_kernel import verify_translation_kernel


def complex_dict(C):
    return {"ranks": [[n, r] for n, r in sorted(C.ranks.items())],
            "differentials": [[n, _matrix_dict(d)] for n, d in sorted(C.differentials.items())]}


def analysis_dict(result):
    version = 2 if any(r.get("selection") is not None for r in result.gluing) else 1
    attachments = []
    for r in result.gluing:
        record = {"basis": [list(row) for row in r["lattice"].basis],
                  "kill_axes": list(r["killing_axes"]), "gluing": r["step"].certificate()}
        if version == 2:
            record["selection"] = r["selection"].certificate() if r.get("selection") is not None else None
        attachments.append(record)
    return {
        "schema": "css-decoupling.profile-analysis", "version": version,
        "ring": {"prime": result.HX.ring.p, "dimension": result.HX.ring.D},
        "input": {"HX": _matrix_dict(result.HX), "HZ": _matrix_dict(result.HZ)},
        "sector": result.sector,
        "model": complex_dict(result.complex),
        "layers": [{"degree": j, "representatives": _matrix_dict(layer.representatives),
                    "actions": [_matrix_dict(a) for a in layer.actions]}
                   for j, layer in sorted(result.layers.items())],
        "translation_kernel": result.translation_kernel.kernel_certificate,
        "initial_basis": [list(row) for row in result.initial_lattice.basis],
        "final_basis": [list(row) for row in result.lattice.basis],
        "attachments": attachments,
        "final_splitting": [[n, _matrix_dict(a)] for n, a in sorted(result.phi.components.items())],
    }


def save_analysis(result, path):
    """Save actual construction diagnostics, separate from the circuit format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.certificate(), indent=2) + "\n")
    return path


def _model(data, ring):
    _keys(data, ("ranks", "differentials"), "model")
    def values(items, read):
        if not isinstance(items, list):
            raise ValueError("Model data must be pair lists")
        result = {}
        for pair in items:
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError("Malformed model pair")
            n = _int(pair[0], "degree", -1)
            if n in result:
                raise ValueError("Duplicate model degree")
            result[n] = read(pair[1])
        return result
    return CochainComplex(ring, values(data["ranks"], lambda r: _int(r, "rank", 0)),
                          values(data["differentials"], lambda a: _matrix_from_dict(a, ring, "model map")))


def verify_analysis(data):
    """Bind all diagnostics to the original CSS model and actual restrictions."""
    checks, errors = [], []
    try:
        _keys(data, ("schema", "version", "ring", "input", "sector", "model", "layers",
                     "translation_kernel", "initial_basis", "final_basis", "attachments",
                     "final_splitting"), "profile analysis")
        version = _int(data["version"], "version")
        if data["schema"] != "css-decoupling.profile-analysis" or version not in (1, 2):
            raise ValueError("Unsupported profile analysis schema")
        _keys(data["ring"], ("prime", "dimension"), "ring")
        ring = Ring(_int(data["ring"]["prime"], "prime", 2), _int(data["ring"]["dimension"], "dimension", 2))
        read = lambda a: _matrix_from_dict(a, ring, "matrix")
        _keys(data["input"], ("HX", "HZ"), "input")
        HX, HZ = read(data["input"]["HX"]), read(data["input"]["HZ"])
        if data["sector"] not in ("X", "Z"):
            raise ValueError("Unknown analysis sector")
        if data["sector"] == "Z":
            HX, HZ = HZ, HX
        C = _model(data["model"], ring)
        if C.d(0) != HX or C.d(-1) != HZ.dagger():
            raise ValueError("The model does not contain the original CSS window")
        solve(HZ.dagger(), kernel(HX))
        solve(HX.dagger(), kernel(HZ))
        for n in range(1, max(C.degrees) + 1):
            solve(C.d(n).dagger(), kernel(C.d(n - 1).dagger()))
        checks.append("source model is a checked finite free model of the original CSS input")
        actual = profile(C)
        layers = {}
        if not isinstance(data["layers"], list):
            raise ValueError("Layers must be a list")
        for raw in data["layers"]:
            _keys(raw, ("degree", "representatives", "actions"), "layer")
            j = _int(raw["degree"], "layer degree", 1)
            if j not in actual or j in layers:
                raise ValueError("Duplicate or spurious cohomology layer")
            layer, reps = actual[j], read(raw["representatives"])
            if reps.shape != (C.rank(j), layer.dimension) or C.d(j) * reps:
                raise ValueError("Invalid layer representatives")
            B = layer.quotient.coordinates(solve(layer.cycles, reps))
            solve(B, eye(ring, layer.dimension))
            if not isinstance(raw["actions"], list) or len(raw["actions"]) != ring.D:
                raise ValueError("Missing translation actions")
            actions = tuple(read(a) for a in raw["actions"])
            for given, computed in zip(actions, layer.actions):
                if B * given != computed * B:
                    raise ValueError("Translation matrix is not the action in the recorded basis")
            layers[j] = SimpleNamespace(degree=j, dimension=layer.dimension, representatives=reps, actions=actions)
        if set(layers) != set(actual):
            raise ValueError("Profile analysis omitted a nonzero layer")
        verify_translation_kernel(layers, ring.D, data["translation_kernel"])
        checks.append("actual cohomology actions and complete common translation kernel Lambda0")
        current = Lattice(data["initial_basis"], max_index=None)
        for layer in layers.values():
            for vector in zip(*current.basis):
                action = eye(ring, layer.dimension)
                for A, exponent in zip(layer.actions, vector):
                    action = action * matrix_power(A, exponent)
                if action != eye(ring, layer.dimension):
                    raise ValueError("Initial retained lattice has nontrivial layer action")
        degrees = sorted(layers)[1:]
        stage = 0
        pending_block = False
        if not isinstance(data["attachments"], list):
            raise ValueError("Attachments must be a list")
        for record in data["attachments"]:
            _keys(record, ("basis", "kill_axes", "gluing") + (("selection",) if version == 2 else ()), "attachment record")
            Lattice(record["basis"], max_index=None)
            if record["basis"] != [list(row) for row in current.basis] or stage >= len(degrees):
                raise ValueError("Attachment lattice or stage does not match the preceding restriction")
            witness = record["gluing"]
            if witness["upper"] != degrees[stage]:
                raise ValueError("Attachments must follow increasing nonzero degrees")
            restricted = CochainComplex(ring,
                {n: rank * current.index for n, rank in C.ranks.items() if n >= 0},
                {n: current.restrict(d) for n, d in C.differentials.items() if n >= 0})
            if witness["complex"] != complex_dict(restricted):
                raise ValueError("Gluing witness is not the actual restricted source model")
            expected_reps = [[j, _matrix_dict(current.pack(layer.representatives))]
                             for j, layer in sorted(layers.items()) if j <= degrees[stage]]
            if witness["representatives"] != expected_reps:
                raise ValueError("Gluing witness changed the recorded layer bases")
            report = verify_gluing(witness)
            if not report.valid:
                raise ValueError("Invalid gluing witness: " + "; ".join(report.errors))
            axes = record["kill_axes"]
            if not isinstance(axes, list) or axes != sorted(set(axes)):
                raise ValueError("Killing axes must be a sorted set")
            for a in axes:
                _int(a, "killing axis", 0)
                if a >= ring.D:
                    raise ValueError("Killing axis outside the ring dimension")
            nonzero = False
            for cls in witness["classes"]:
                M = read(cls["coefficients"])
                tj = layers[degrees[stage]].dimension
                for (_, column), coefficient in M.entries.items():
                    if coefficient:
                        nonzero = True
                        if version == 1 and not set(cls["exterior_basis"][column // tj]) & set(axes):
                            raise ValueError("The restriction does not kill every cocycle support")
            if nonzero:
                if pending_block:
                    raise ValueError("A claimed killing restriction did not produce zero gluing")
                if version == 1:
                    current = Lattice([[v * (ring.p if a in axes else 1) for a, v in enumerate(row)]
                                       for row in current.basis], max_index=None)
                else:
                    from css_decoupling.decoupling.gluing import GluingClass
                    from .lattice_selection import verify_lattice_selection
                    classes = tuple(GluingClass(degrees[stage], c["lower"],
                                               layers[degrees[stage]].dimension,
                                               layers[c["lower"]].dimension, read(c["coefficients"]))
                                    for c in witness["classes"])
                    selection = record["selection"]
                    report = verify_lattice_selection(classes, selection)
                    if not report.valid:
                        raise ValueError("Invalid lattice selection: " + "; ".join(report.errors))
                    transition = selection["relative_basis"]
                    diagonal = all(transition[i][j] == 0 for i in range(ring.D)
                                   for j in range(ring.D) if i != j)
                    expected_axes = [i for i in range(ring.D) if transition[i][i] == ring.p] if diagonal else []
                    if axes != expected_axes:
                        raise ValueError("Coordinate diagnostics disagree with the actual lattice transition")
                    current = Lattice([[sum(current.basis[i][k] * transition[k][j]
                                            for k in range(ring.D)) for j in range(ring.D)]
                                       for i in range(ring.D)], max_index=None)
                pending_block = True
            else:
                if axes:
                    raise ValueError("A zero attachment must not trigger coarse-graining")
                if version == 2 and record["selection"] is not None:
                    raise ValueError("A zero attachment must not carry a killing restriction")
                stage += 1
                pending_block = False
        Lattice(data["final_basis"], max_index=None)
        if stage != len(degrees) or pending_block or data["final_basis"] != [list(row) for row in current.basis]:
            raise ValueError("Incomplete attachment sequence or incorrect final lattice")
        restricted = CochainComplex(ring,
            {n: r * current.index for n, r in C.ranks.items() if n >= 0},
            {n: current.restrict(d) for n, d in C.differentials.items() if n >= 0})
        T = toric_window(ring, {j: l.dimension for j, l in layers.items()}, lowest_degree=0)
        phi = CochainMap(T, restricted, _pairs(data["final_splitting"], read, "final splitting"))
        from css_decoupling.topology.complexes import koszul_window
        for j, layer in sorted(layers.items()):
            offset = sum(koszul_window(ring, i, l.dimension, lowest_degree=0).rank(j)
                         for i, l in layers.items() if i < j)
            if phi.at(j)[:, offset:offset + layer.dimension] != current.pack(layer.representatives):
                raise ValueError("Final splitting does not induce the recorded layer bases")
        checks.append("successive actual kappa classes, verified lattice restrictions, and final splitting")
    except (ValueError, TypeError, KeyError, AttributeError, IndexError) as exc:
        errors.append(str(exc))
    return VerificationReport(not errors, tuple(checks), tuple(errors), "input-to-exact-Lambda0-and-Postnikov-gluing")
