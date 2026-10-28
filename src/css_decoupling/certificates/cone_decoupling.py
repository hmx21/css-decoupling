"""Independent exact certificates for the direct cone circuit of Sec. V.

The certified target is the direct sum of the supplied restricted blocks.
It carries no claim of smallest toric presentation, minimal ancilla count
for unrelated constructions, or gate/depth optimality.
"""
from __future__ import annotations

from css_decoupling.algebra.laurent import Ring, block, direct_sum, eye, zeros
from css_decoupling.circuits.gates import Circuit, SUM
from css_decoupling.lattice.retained import Lattice
from css_decoupling.topology.complexes import CochainComplex
from css_decoupling.certificates.comparison import (
    VerificationReport, _keys, _int, _list, _matrix_dict, _matrix_from_dict,
    circuit_dict, circuit_from_dict,
)

SCHEMA = "css-decoupling.cone-direct"
SCOPE = "explicit-cone-to-block-diagonal-CSS-circuit"
TARGET_KIND = "restricted-direct-sum-of-input-complexes"
CONVENTIONS = {
    "vectors": "columns",
    "cone": "d_C[n]=[[d_A[n],c[n]],[0,d_B[n]]]",
    "null_homotopy": "c[n]=d_A[n]*h[n]-h[n+1]*d_B[n]",
    "map": "Psi[n]=[[I,h[n]],[0,I]]:C[n]->(A+B)[n]",
    "pauli": "Z=U;X=inverse-dagger(U)",
    "basis": "summand-then-original-generator-then-lattice-representative",
    "schedule": "each-index-is-a-translation-invariant-SUM-family",
}


def _degree_matrices(matrices):
    return [[n, _matrix_dict(value)] for n, value in sorted(matrices.items())]


def _read_degree_matrices(data, ring, context):
    out = {}
    for pair in _list(data, context):
        n, matrix = _list(pair, context + " entry", 2)
        n = _int(n, context + " degree")
        if n in out:
            raise ValueError(f"Duplicate degree in {context}")
        out[n] = _matrix_from_dict(matrix, ring, context + f"[{n}]")
    return out


def _complex_dict(C):
    return {"ranks": [[n, C.rank(n)] for n in C.degrees],
            "differentials": _degree_matrices({n: C.d(n) for n in C.degrees})}


def _read_complex(data, ring, context):
    _keys(data, ("ranks", "differentials"), context)
    ranks = {}
    for pair in _list(data["ranks"], context + ".ranks"):
        n, rank = _list(pair, "rank pair", 2)
        n, rank = _int(n, "degree"), _int(rank, "rank", 0)
        if n in ranks:
            raise ValueError("Duplicate complex degree")
        ranks[n] = rank
    maps = _read_degree_matrices(data["differentials"], ring, context + ".differentials")
    if set(maps) != set(ranks):
        raise ValueError("Every serialized complex degree needs its differential")
    return CochainComplex(ring, ranks, maps)


def cone_certificate_dict(result):
    r = result.presentation.ring
    return {
        "schema": SCHEMA, "version": 1, "scope": SCOPE,
        "ring": {"prime": r.p, "dimension": r.D},
        "conventions": dict(CONVENTIONS),
        "input": {"lower": _complex_dict(result.presentation.lower),
                  "upper": _complex_dict(result.presentation.upper),
                  "coupling": _degree_matrices({n: result.presentation.c(n) for n in result.presentation.degrees})},
        "lattice": result.lattice.metadata(),
        "target": {"kind": TARGET_KIND, "normalization": "none", "claims_minimal_toric_stack": False},
        "homotopy": _degree_matrices(result.homotopy),
        "maps": _degree_matrices(result.maps),
        "inverse_maps": _degree_matrices(result.inverse_maps),
        "checks": {"source_HX": _matrix_dict(result.source.d(0)),
                   "source_HZ": _matrix_dict(result.source.d(-1).dagger()),
                   "target_HX": _matrix_dict(result.target.d(0)),
                   "target_HZ": _matrix_dict(result.target.d(-1).dagger())},
        "circuit": circuit_dict(result.circuit),
        "schedule": {"layers": [list(layer) for layer in result.disjoint_layers], "depth": result.depth},
        "ancillas": dict(result.ancillas),
        "construction": {"homotopy_origin": result.homotopy_origin, "gate_optimization": "none"},
    }


def verify_cone_decoupling(value):
    """Recompute restriction, cone identities, maps, both check sectors and gates.

    This verifier performs no homotopy solve, free-basis computation, or
    profile analysis.  Invertible maps at degrees -1,0,1 provide exact
    changes of stabilizer generators; no topological hypothesis is needed.
    """
    checks, errors = [], []
    try:
        from css_decoupling.decoupling.cone_decoupling import ConeDecouplingResult
        data = cone_certificate_dict(value) if isinstance(value, ConeDecouplingResult) else value
        _keys(data, ("schema", "version", "scope", "ring", "conventions", "input", "lattice",
                     "target", "homotopy", "maps", "inverse_maps", "checks", "circuit", "schedule",
                     "ancillas", "construction"), "direct cone certificate")
        if data["schema"] != SCHEMA or _int(data["version"], "version") != 1 or data["scope"] != SCOPE:
            raise ValueError("Unsupported direct cone schema")
        if data["conventions"] != CONVENTIONS:
            raise ValueError("Unsupported cone/gate conventions")
        _keys(data["ring"], ("prime", "dimension"), "ring")
        ring = Ring(_int(data["ring"]["prime"], "prime", 2), _int(data["ring"]["dimension"], "dimension", 2))
        _keys(data["input"], ("lower", "upper", "coupling"), "input")
        original_A = _read_complex(data["input"]["lower"], ring, "lower")
        original_B = _read_complex(data["input"]["upper"], ring, "upper")
        degrees = tuple(sorted(set(original_A.degrees) | set(original_B.degrees)))
        original_c = _read_degree_matrices(data["input"]["coupling"], ring, "coupling")
        if set(original_c) != set(degrees):
            raise ValueError("Coupling must specify every cone degree")
        for n in degrees:
            if original_c[n].shape != (original_A.rank(n + 1), original_B.rank(n)):
                raise ValueError("Coupling shape does not match the input summands")
            following = original_c.get(n + 1, zeros(ring, original_A.rank(n + 2), original_B.rank(n + 1)))
            if original_A.d(n + 1) * original_c[n] + following * original_B.d(n):
                raise ValueError("Original cone differential does not square to zero")
        metadata = _keys(data["lattice"], ("basis_columns", "index", "representatives"), "lattice")
        lattice = Lattice(metadata["basis_columns"], max_index=None)
        if lattice.D != ring.D:
            raise ValueError("Lattice dimension differs from coefficient ring")
        _int(metadata["index"], "lattice index", 1)
        for rep in _list(metadata["representatives"], "representatives"):
            for coordinate in _list(rep, "representative", ring.D):
                _int(coordinate, "representative coordinate", 0)
        if metadata != lattice.metadata():
            raise ValueError("Inconsistent restriction lattice metadata")
        def restriction(C):
            return CochainComplex(ring, {n: C.rank(n) * lattice.index for n in C.degrees},
                                  {n: lattice.restrict(C.d(n)) for n in C.degrees})
        A, B = restriction(original_A), restriction(original_B)
        c = {n: lattice.restrict(original_c[n]) for n in degrees}
        ranks = {n: A.rank(n) + B.rank(n) for n in degrees}
        source = CochainComplex(ring, ranks, {n: block([
            [A.d(n), c[n]], [zeros(ring, B.rank(n + 1), A.rank(n)), B.d(n)]]) for n in degrees})
        target = CochainComplex(ring, ranks, {n: direct_sum(A.d(n), B.d(n)) for n in degrees})
        checks.append("original cone and exact restriction of both summands and coupling")
        descriptor = _keys(data["target"], ("kind", "normalization", "claims_minimal_toric_stack"), "target")
        if descriptor["kind"] != TARGET_KIND or descriptor["normalization"] != "none" or descriptor["claims_minimal_toric_stack"] is not False:
            raise ValueError("Target must be the unnormalized restricted input summands")
        h = _read_degree_matrices(data["homotopy"], ring, "homotopy")
        maps = _read_degree_matrices(data["maps"], ring, "maps")
        inverses = _read_degree_matrices(data["inverse_maps"], ring, "inverse_maps")
        if any(set(values) != set(degrees) for values in (h, maps, inverses)):
            raise ValueError("Homotopy and both map families must specify every cone degree")
        for n in degrees:
            if h[n].shape != (A.rank(n), B.rank(n)):
                raise ValueError(f"Homotopy shape fails at degree {n}")
            hn = h.get(n + 1, zeros(ring, A.rank(n + 1), B.rank(n + 1)))
            if A.d(n) * h[n] - hn * B.d(n) != c[n]:
                raise ValueError(f"Null-homotopy equation fails at degree {n}")
            expected = block([[eye(ring, A.rank(n)), h[n]],
                              [zeros(ring, B.rank(n), A.rank(n)), eye(ring, B.rank(n))]])
            expected_inverse = block([[eye(ring, A.rank(n)), -h[n]],
                                      [zeros(ring, B.rank(n), A.rank(n)), eye(ring, B.rank(n))]])
            if maps[n] != expected or inverses[n] != expected_inverse:
                raise ValueError(f"Triangular map formula fails at degree {n}")
            if maps[n] * inverses[n] != eye(ring, ranks[n]) or inverses[n] * maps[n] != eye(ring, ranks[n]):
                raise ValueError(f"Two-sided inverse fails at degree {n}")
            next_map = maps.get(n + 1, eye(ring, source.rank(n + 1)))
            next_inverse = inverses.get(n + 1, eye(ring, source.rank(n + 1)))
            if target.d(n) * maps[n] != next_map * source.d(n):
                raise ValueError(f"Forward chain relation fails at degree {n}")
            if source.d(n) * inverses[n] != next_inverse * target.d(n):
                raise ValueError(f"Inverse chain relation fails at degree {n}")
        checks.append("all null-homotopy, triangular inverse and chain equations")
        _keys(data["ancillas"], ("source", "target"), "ancillas")
        if any(_int(data["ancillas"][side], "ancillas", 0) for side in ("source", "target")):
            raise ValueError("Direct cone construction adds no ancillas")
        _keys(data["checks"], ("source_HX", "source_HZ", "target_HX", "target_HZ"), "checks")
        expected_checks = {"source_HX": source.d(0), "source_HZ": source.d(-1).dagger(),
                           "target_HX": target.d(0), "target_HZ": target.d(-1).dagger()}
        for name, expected in expected_checks.items():
            if _matrix_from_dict(data["checks"][name], ring, name) != expected:
                raise ValueError(f"Stored {name} is not the appropriate restricted check matrix")
        U, Uinv = maps.get(0, eye(ring, source.rank(0))), inverses.get(0, eye(ring, source.rank(0)))
        V, Vinv = maps.get(1, eye(ring, source.rank(1))), inverses.get(1, eye(ring, source.rank(1)))
        W, Winv = maps.get(-1, eye(ring, source.rank(-1))), inverses.get(-1, eye(ring, source.rank(-1)))
        if target.d(0) * U != V * source.d(0) or source.d(0) * Uinv != Vinv * target.d(0):
            raise ValueError("X-check intertwining fails")
        if U * source.d(-1) != target.d(-1) * W or Uinv * target.d(-1) != source.d(-1) * Winv:
            raise ValueError("Z-generator intertwining fails")
        if Uinv.dagger() * source.d(0).dagger() != target.d(0).dagger() * Vinv.dagger():
            raise ValueError("Physical X action fails on the X stabilizer generators")
        checks.append("both CSS check sectors mapped in both directions without ancillas")
        circuit = circuit_from_dict(data["circuit"], ring)
        if circuit.qudits != source.rank(0) or any(not isinstance(g, SUM) for g in circuit.gates):
            raise ValueError("Direct cone circuit must use SUMs on the original qudits")
        if circuit.matrix("Z") != U or circuit.matrix("X") != Uinv.dagger() or circuit.inverse().matrix("Z") != Uinv:
            raise ValueError("Gate products disagree with the physical maps")
        schedule = _keys(data["schedule"], ("layers", "depth"), "schedule")
        layers = _list(schedule["layers"], "layers")
        if _int(schedule["depth"], "depth", 0) != len(layers):
            raise ValueError("Stored circuit depth disagrees with schedule")
        seen, ordered = set(), []
        for layer in layers:
            used = set()
            indices = _list(layer, "layer")
            if not indices:
                raise ValueError("Empty circuit layers must be omitted")
            for index in indices:
                index = _int(index, "gate index", 0)
                if index >= len(circuit.gates) or index in seen:
                    raise ValueError("Schedule contains a duplicate or out-of-range gate")
                gate = circuit.gates[index]
                species = {gate.control, gate.target}
                if used & species:
                    raise ValueError("Simultaneous SUM families share a physical qudit species")
                used.update(species)
                seen.add(index)
                ordered.append(gate)
        if len(seen) != len(circuit.gates):
            raise ValueError("Schedule omits a gate")
        scheduled = Circuit(ring, circuit.qudits, tuple(ordered))
        if scheduled.matrix("Z") != U or scheduled.matrix("X") != Uinv.dagger():
            raise ValueError("Layer scheduling changed the physical action")
        checks.append("SUM products and disjoint physical layer schedule")
        metadata = _keys(data["construction"], ("homotopy_origin", "gate_optimization"), "construction")
        if metadata["homotopy_origin"] not in ("supplied", "exact-Laurent-module-solve") or metadata["gate_optimization"] != "none":
            raise ValueError("Unsupported construction metadata or gate-optimality claim")
    except (ValueError, TypeError, KeyError, AttributeError, IndexError) as exc:
        errors.append(str(exc))
    return VerificationReport(not errors, tuple(checks), tuple(errors), SCOPE)
