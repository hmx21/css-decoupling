"""Independent certificates for Algorithm 1's sufficient coarse-graining.

Verification binds a complete bounded free model to the actual CSS checks,
recomputes cohomology and layer actions, proves the common kernel complete,
and checks the uniform-cover formula and every restricted differential.
It does not invoke the coarse-graining constructor or solve a splitting.
"""
from copy import deepcopy
from types import SimpleNamespace

from css_decoupling.algebra.laurent import Ring, eye
from css_decoupling.algebra.modules import kernel, solve, using_backend
from css_decoupling.errors import ResourceLimit
from css_decoupling.lattice.retained import Lattice
from css_decoupling.topology.cohomology import cohomology_layer
from css_decoupling.topology.complexes import CochainComplex
from .analysis import complex_dict
from .comparison import _keys, _int, _matrix_dict, _matrix_from_dict, VerificationReport
from .criterion import _check_infinite
from .translation_kernel import verify_translation_kernel

SCHEMA = "css-decoupling.coarse-graining"
SCOPE = "original-CSS-to-uniform-sufficient-lattice-and-bounded-restricted-complex"


def coarse_graining_dict(result, *, include_handoff=False):
    """Serialize the current result fields, including mutable public claims."""
    obstruction = None
    if result.obstruction is not None:
        obstruction = dict(result.obstruction)
        for name in ("cycles", "boundary_coordinates", "cycle_relations", "presentation"):
            obstruction[name] = _matrix_dict(obstruction[name])
        obstruction["ray"] = dict(obstruction["ray"])
    kernel_certificate = None
    if result.translation_kernel is not None:
        kernel_certificate = deepcopy(result.translation_kernel.kernel_certificate)
        kernel_certificate["basis_columns"] = [list(row) for row in result.translation_kernel.basis]
        kernel_certificate["index"] = result.translation_kernel.index
    data = {
        "schema": SCHEMA, "version": 1, "scope": SCOPE,
        "ring": {"prime": result.HX.ring.p, "dimension": result.HX.ring.D},
        "input": {"HX": _matrix_dict(result.HX), "HZ": _matrix_dict(result.HZ)},
        "sector": result.sector, "decouplable": result.decouplable,
        "reason": result.reason, "model": complex_dict(result.complex),
        "layers": [{"degree": j, "representatives": _matrix_dict(layer.representatives),
                    "actions": [_matrix_dict(a) for a in layer.actions]}
                   for j, layer in sorted(result.layers.items())],
        "translation_kernel": kernel_certificate,
        "basis": None if result.lattice is None else [list(row) for row in result.lattice.basis],
        "index": None if result.lattice is None else result.lattice.index,
        "restricted": None if result.restricted is None else complex_dict(result.restricted),
        "obstruction": obstruction,
    }
    if include_handoff and result.cohomology_handoff is not None:
        data["cohomology_handoff"] = deepcopy(result.cohomology_handoff)
    return data


def _bounded_model(data, ring):
    """Decode a bounded free model with arbitrary integer cochain degrees."""
    _keys(data, ("ranks", "differentials"), "bounded model")

    def pairs(raw, read):
        if not isinstance(raw, list):
            raise ValueError("Bounded model data must be pair lists")
        result = {}
        for pair in raw:
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError("Malformed bounded model pair")
            degree = _int(pair[0], "cochain degree")
            if degree in result:
                raise ValueError("Duplicate bounded model degree")
            result[degree] = read(pair[1])
        return result

    return CochainComplex(
        ring, pairs(data["ranks"], lambda r: _int(r, "free rank", 0)),
        pairs(data["differentials"], lambda a: _matrix_from_dict(a, ring, "model map")))


def _verify_source_model(C, HX, HZ, *, kernel=kernel, solve=solve):
    """Check Lemma model, including both terminal kernels and exact negative end."""
    if HX.ring != HZ.ring or HX.ring != C.ring or HX.ncols != HZ.ncols or HX * HZ.dagger():
        raise ValueError("Original CSS checks have incompatible rings/columns or do not commute")
    if C.d(0) != HX or C.d(-1) != HZ.dagger():
        raise ValueError("Bounded model changed the original CSS physical window")
    # Physical exactness of both CSS sectors is an input hypothesis.
    solve(HX.dagger(), kernel(HZ))
    for n in range(min(C.degrees, default=0), 1):
        solve(C.d(n - 1), kernel(C.d(n)))
    # The positive dual tail resolves ker(HX^dagger), including its terminal
    # kernel; arbitrary square-zero extensions do not establish the profile.
    for n in range(1, max(C.degrees, default=0) + 1):
        solve(C.d(n).dagger(), kernel(C.d(n - 1).dagger()))
    return True


def _verified_layers(C, raw_layers, *, stop=None, max_layer_dimension=10000):
    """Recompute and bind every finite layer below an optional infinite one."""
    actual = {}
    for degree in C.degrees:
        if degree <= 0 or (stop is not None and degree >= stop):
            continue
        layer = cohomology_layer(C, degree, max_dimension=max_layer_dimension)
        if layer.dimension:
            if degree >= C.ring.D:
                raise ValueError("Nonzero layer outside the topological CSS degree bound")
            actual[degree] = layer
    if not isinstance(raw_layers, list):
        raise ValueError("Layers must be an ordered list")
    layers = {}
    read = lambda a: _matrix_from_dict(a, C.ring, "layer matrix")
    for raw in raw_layers:
        _keys(raw, ("degree", "representatives", "actions"), "layer")
        degree = _int(raw["degree"], "layer degree", 1)
        if degree in layers or degree not in actual:
            raise ValueError("Duplicate or spurious cohomology layer")
        layer, reps = actual[degree], read(raw["representatives"])
        if reps.shape != (C.rank(degree), layer.dimension) or C.d(degree) * reps:
            raise ValueError("Layer representatives are not cycles of the actual dimension")
        change = layer.quotient.coordinates(solve(layer.cycles, reps))
        solve(change, eye(C.ring, layer.dimension))
        if not isinstance(raw["actions"], list) or len(raw["actions"]) != C.ring.D:
            raise ValueError("Missing original translation matrices")
        actions = tuple(read(a) for a in raw["actions"])
        for given, computed in zip(actions, layer.actions):
            if given.shape != (layer.dimension, layer.dimension) or any(
                    not f.is_constant for f in given.entries.values()):
                raise ValueError("Translation matrices must be constant and correctly sized")
            if change * given != computed * change:
                raise ValueError("Recorded translations differ from the actual action")
        layers[degree] = SimpleNamespace(dimension=layer.dimension, representatives=reps, actions=actions)
    if set(layers) != set(actual):
        raise ValueError("Certificate omitted a nonzero cohomology layer")
    return layers


def verify_coarse_graining(data, *, max_cell_index=4096,
                          max_layer_dimension=10000, max_linear_entries=2000000,
                          backend=None, resolution_timeout=120):
    """Verify a sufficient lattice or an infinite-dimensional obstruction.

    Independent module normal forms check cohomology and model exactness;
    exact cyclic/unipotent witnesses establish kernel completeness without
    discrete-logarithm searches. Resource failures propagate, rather than
    becoming either a negative answer or a false verification report.
    """
    if backend is not None:
        with using_backend(backend, timeout=resolution_timeout):
            return verify_coarse_graining(
                data, max_cell_index=max_cell_index,
                max_layer_dimension=max_layer_dimension,
                max_linear_entries=max_linear_entries)
    checks, errors = [], []
    try:
        for cap, name in ((max_cell_index, "max_cell_index"),
                          (max_layer_dimension, "max_layer_dimension")):
            if cap is not None:
                _int(cap, name, 1)
        _int(max_linear_entries, "max_linear_entries", 1)
        if not isinstance(data, dict) and hasattr(data, "certificate"):
            data = data.certificate()
        _keys(data, ("schema", "version", "scope", "ring", "input", "sector",
                     "decouplable", "reason", "model", "layers", "translation_kernel",
                     "basis", "index", "restricted", "obstruction") +
              (("cohomology_handoff",) if "cohomology_handoff" in data else ()), "coarse-graining")
        if data["schema"] != SCHEMA or _int(data["version"], "version") != 1 or data["scope"] != SCOPE:
            raise ValueError("Unsupported coarse-graining schema, version, or scope")
        _keys(data["ring"], ("prime", "dimension"), "ring")
        ring = Ring(_int(data["ring"]["prime"], "prime", 2),
                    _int(data["ring"]["dimension"], "dimension", 2))
        _keys(data["input"], ("HX", "HZ"), "input")
        read = lambda a: _matrix_from_dict(a, ring, "matrix")
        HX, HZ = read(data["input"]["HX"]), read(data["input"]["HZ"])
        if data["sector"] not in ("X", "Z"):
            raise ValueError("Coarse-graining sector must be X or Z")
        if data["sector"] == "Z":
            HX, HZ = HZ, HX
        C = _bounded_model(data["model"], ring)
        _verify_source_model(C, HX, HZ)
        checks.append("original CSS input, complete bounded free model, and both terminal kernels")
        reason = data["reason"]
        if reason not in ("uniform-sufficient-lattice", "infinite-layer") or type(data["decouplable"]) is not bool:
            raise ValueError("Unknown decision or invalid decouplable flag")
        if data["decouplable"] != (reason == "uniform-sufficient-lattice"):
            raise ValueError("Decision disagrees with the certified reason")
        stop = None
        if reason == "infinite-layer":
            stop = _check_infinite(C, data["obstruction"], read)
        elif data["obstruction"] is not None:
            raise ValueError("A sufficient-cell result cannot contain an obstruction")
        layers = _verified_layers(C, data["layers"], stop=stop,
                                  max_layer_dimension=max_layer_dimension)
        checks.append("actual cohomology bases and original translation actions")
        if stop is not None:
            if data.get("cohomology_handoff") is not None:
                raise ValueError("An obstruction cannot include a successful cohomology handoff")
            if any(data[key] is not None for key in ("translation_kernel", "basis", "index", "restricted")):
                raise ValueError("An infinite-layer result cannot claim a lattice or restricted output")
            checks.append("complete infinite cohomology presentation and standard-monomial ray")
            return VerificationReport(True, tuple(checks), (), SCOPE)
        verify_translation_kernel(layers, ring.D, data["translation_kernel"],
                                  max_linear_entries=max_linear_entries)
        kernel_lattice = Lattice(data["translation_kernel"]["basis_columns"], max_index=None)
        lattice = Lattice(data["basis"], max_index=None)
        if lattice.D != ring.D or _int(data["index"], "index", 1) != lattice.index:
            raise ValueError("Coarse lattice dimension or index is inconsistent")
        factor = ring.p ** max(len(layers) - 1, 0)
        expected = Lattice([[factor * x for x in row] for row in kernel_lattice.basis], max_index=None)
        # Equality of lattices allows a valid signed or sheared column basis;
        # the actual restriction below is checked in the supplied coordinates.
        if lattice.H != expected.H:
            raise ValueError("Coarse lattice is not p**max(m-1,0) times the exact translation kernel")
        checks.append("complete common translation kernel and sufficient uniform-cover formula")
        if max_cell_index is not None and lattice.index > max_cell_index:
            raise ResourceLimit(f"Coarse unit cell has index {lattice.index}; cap is {max_cell_index}")
        restricted = _bounded_model(data["restricted"], ring)
        if restricted.ranks != {n: rank * lattice.index for n, rank in C.ranks.items()}:
            raise ValueError("Restricted free-module ranks do not equal actual cell multiplicities")
        if set(restricted.differentials) != set(C.differentials) or any(
                restricted.differentials[n] != lattice.restrict(d)
                for n, d in C.differentials.items()):
            raise ValueError("Restricted differentials are not the actual source restrictions")
        checks.append("every actual restricted differential, including the resolved negative end")
        if data.get("cohomology_handoff") is not None:
            from css_decoupling.algebra.modules import get_backend
            from .cohomology_handoff import checked_handoff
            from .cone_decoupling import _complex_dict
            handoff = data["cohomology_handoff"]
            checked_handoff(handoff, complex=restricted, backend=get_backend(ring),
                            max_dimension=max_layer_dimension,
                            max_cell_index=max_cell_index, max_total_rank=None)
            if handoff["source"] != _complex_dict(C) or handoff["basis"] != data["basis"]:
                raise ValueError("Handoff changed the Algorithm 1 source or lattice")
            expected = {raw["degree"]: raw["representatives"] for raw in data["layers"]}
            actual = {raw["degree"]: raw["source_representatives"] for raw in handoff["layers"]
                      if raw["layer"]["dimension"]}
            if actual != expected:
                raise ValueError("Handoff changed Algorithm 1's actual representatives")
            checks.append("complete cohomology witnesses transported from the original model")
    except (ValueError, TypeError, KeyError, AttributeError, IndexError) as exc:
        errors.append(str(exc))
    return VerificationReport(not errors, tuple(checks), tuple(errors), SCOPE)
