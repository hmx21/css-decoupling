"""Bind Algorithm 2's input, normalized cohomology, Schanuel maps and circuit."""
from dataclasses import asdict
from copy import deepcopy

from css_decoupling.algebra.laurent import Ring, zeros
from css_decoupling.topology.complexes import toric_stack
from .comparison import VerificationReport, _keys, _int, _matrix_from_dict, certificate_from_dict
from .cone_decoupling import _complex_dict, _read_complex
from .coarse_graining import _verify_source_model
from .decoupling import verify_decoupling
from .schanuel import schanuel_certificate_dict, verify_schanuel

SCHEMA = "css-decoupling.algorithm-two"


def circuit_resources(stable):
    from css_decoupling.circuits.gates import SUM, SWAP
    gates = stable.circuit.gates
    return {"qudits_per_cell": stable.circuit.qudits,
            "sum_translation_families": sum(isinstance(g, SUM) for g in gates),
            "swap_gates_per_cell": sum(isinstance(g, SWAP) for g in gates),
            "sequential_depth_upper_bound": len(gates),
            "interaction_radius": stable.circuit.interaction_radius,
            "ancillas": asdict(stable.ancillas)}


def construction_dict(result):
    physical = None
    if result.coarse_graining is not None:
        from dataclasses import replace
        from css_decoupling.lattice.retained import Lattice
        coarse = result.coarse_graining
        ring = result.complex.ring
        HX, HZ = (_matrix_from_dict(coarse["input"][key], ring, key) for key in ("HX", "HZ"))
        active = replace(result.decoupling, HX=HX if coarse["sector"] == "X" else HZ,
                         HZ=HZ if coarse["sector"] == "X" else HX,
                         lattice=Lattice(coarse["basis"], max_index=None))
        if coarse["sector"] == "X":
            physical = active.certificate()
        else:
            from css_decoupling.decoupling.dual import DualDecouplingResult
            from css_decoupling.circuits.duality import dualize_circuit
            physical = DualDecouplingResult(HX, HZ, active, dualize_circuit(active.circuit), {}).certificate()
    data = {"schema": SCHEMA, "version": 1,
            "ring": {"prime": result.complex.ring.p, "dimension": result.complex.ring.D},
            "model": _complex_dict(result.complex),
            "cohomology": [layer.certificate() for _, layer in sorted(result.layers.items())],
            "comparison": schanuel_certificate_dict(result.comparison),
            "decoupling": result.decoupling.certificate(),
            "configuration": {"linear_backend": result.linear_backend},
            "input_size": dict(result.input_size),
            "resources": circuit_resources(result.decoupling.stabilization),
            "coarse_graining": deepcopy(result.coarse_graining), "physical_decoupling": physical}
    if result.cohomology_handoff is not None:
        data["cohomology_handoff"] = deepcopy(result.cohomology_handoff)
    return data


def verify_construction(data, *, linear_backend=None, bounded_options=None, max_total_rank=100000):
    """Verify mathematics and resources, not a claimed runtime complexity.

    The configuration records a requested backend, not an independently
    attested execution history or a guarantee about a practical GB solver.
    With a bounded Algorithm 2 verifier, attached Algorithm 1 and original-
    code physical provenance are checked separately by the native module
    verifier: those original-lattice computations are outside C5. This is
    an explicit stage boundary, never a fallback after a failed bounded solve.
    """
    from css_decoupling.certificates.bounded_linear import verify_trivial_cohomology
    from css_decoupling.algebra.modules import get_backend, using_backend
    if linear_backend is not None:
        from css_decoupling.algebra.bounded_linear import BoundedLinearBackend
        if linear_backend == "bounded":
            raw = data.certificate() if hasattr(data, "certificate") else data
            spec = raw["ring"]
            selected = BoundedLinearBackend(Ring(spec["prime"], spec["dimension"]), **(bounded_options or {}))
        else:
            selected = linear_backend
        with using_backend(selected):
            return verify_construction(data, bounded_options=bounded_options, max_total_rank=max_total_rank)
    checks, errors = [], []
    try:
        if hasattr(data, "certificate"):
            data = data.certificate()
        _keys(data, ("schema", "version", "ring", "model", "cohomology", "comparison", "decoupling",
                     "configuration", "input_size", "resources", "coarse_graining", "physical_decoupling") +
              (("cohomology_handoff",) if "cohomology_handoff" in data else ()), "Algorithm 2 certificate")
        if data["schema"] != SCHEMA or _int(data["version"], "version") != 1:
            raise ValueError("Unsupported Algorithm 2 certificate")
        _keys(data["ring"], ("prime", "dimension"), "ring")
        ring = Ring(_int(data["ring"]["prime"], "prime", 2), _int(data["ring"]["dimension"], "dimension", 2))
        C = _read_complex(data["model"], ring, "input model")
        if max_total_rank is not None:
            from css_decoupling.errors import ResourceLimit
            _int(max_total_rank, "max_total_rank", 1)
            if sum(C.ranks.values()) > max_total_rank:
                raise ResourceLimit("Input verification exceeds max_total_rank")
        if C.degrees and max(-C.degrees[0], C.degrees[-1]) > ring.D + 2:
            raise ValueError("Model is nonzero outside degrees |n| <= D+2 (Lemma 4)")
        transported = None
        if data.get("cohomology_handoff") is None:
            _verify_source_model(C, C.d(0), C.d(-1).dagger())
        else:
            if data["configuration"].get("linear_backend") != "sympy":
                raise ValueError("Native handoff is incompatible with the declared linear backend")
            from .cohomology_handoff import checked_handoff
            transported = checked_handoff(data["cohomology_handoff"], complex=C, backend=get_backend(ring),
                                           max_cell_index=None, max_total_rank=max_total_rank)
        if not isinstance(data["cohomology"], list):
            raise ValueError("Cohomology witnesses must be an ordered list")
        if transported is not None and data["cohomology"] != [raw for _, raw in sorted(transported.items())]:
            raise ValueError("Construction cohomology differs from its verified transported witnesses")
        multiplicities = {}
        for raw in data["cohomology"]:
            j = _int(raw["degree"], "cohomology degree", 1)
            if j >= ring.D or j in multiplicities or (multiplicities and j <= max(multiplicities)):
                raise ValueError("Repeated or incorrectly ordered cohomology degree")
            if transported is None:
                report = verify_trivial_cohomology(raw, complex=C, backend=get_backend(ring), **(bounded_options or {}))
                if not report.valid:
                    raise ValueError("Invalid normalized cohomology witness: " + "; ".join(report.errors))
            multiplicities[j] = _int(raw["dimension"], "layer dimension", 1)
        T = toric_stack(ring, multiplicities, max_total_rank=max_total_rank)
        comp = data["comparison"]
        report = verify_schanuel(comp)
        if not report.valid:
            raise ValueError("Invalid Schanuel comparison: " + "; ".join(report.errors))
        if comp["ring"] != data["ring"] or comp["input"]["C"] != _complex_dict(C) or comp["input"]["T"] != _complex_dict(T):
            raise ValueError("Comparison is not bound to the input model and declared toric stack")
        # The full homotopy equivalence independently proves that no layer
        # was omitted; no additional Groebner quotient is needed here.
        from .cone_decoupling import _read_degree_matrices
        from css_decoupling.topology.complexes import koszul
        gamma = _read_degree_matrices(comp["input"]["phi"], ring, "Gamma")
        for raw in data["cohomology"]:
            j = raw["degree"]
            offset = sum(koszul(ring, i, t, max_total_rank=max_total_rank).rank(j) for i, t in multiplicities.items() if i < j)
            if gamma[j][:, offset:offset+raw["dimension"]] != _matrix_from_dict(raw["representatives"], ring, "cocycles"):
                raise ValueError("Gamma does not send the chosen generators to the recorded cocycles")
        inner = data["decoupling"]
        if _matrix_from_dict(inner["input"]["HX"], ring, "HX") != C.d(0):
            raise ValueError("Circuit source differs from the supplied model")
        if _matrix_from_dict(inner["input"]["HZ"], ring, "HZ") != C.d(-1).dagger():
            raise ValueError("Circuit Z source differs from the supplied model")
        if inner["lattice"]["basis_columns"] != [[int(i == j) for j in range(ring.D)] for i in range(ring.D)]:
            raise ValueError("Algorithm 2 must retain its input lattice")
        if inner["multiplicities"] != [[j, t] for j, t in sorted(multiplicities.items())]:
            raise ValueError("Circuit multiplicities differ from actual cohomology")
        if inner["identity_checks"] != {"source": comp["ranks"]["free_C"], "target": comp["ranks"]["free_T"]}:
            raise ValueError("Circuit free summands disagree with the Schanuel frame")
        stable = certificate_from_dict(inner["stabilization"])
        for name, raw in comp["lifts"].items():
            if getattr(stable.lifts, name) != _matrix_from_dict(raw, ring, name):
                raise ValueError("Circuit lifts differ from the verified stable comparison")
        report = verify_decoupling(inner)
        if not report.valid:
            raise ValueError("Invalid physical circuit: " + "; ".join(report.errors))
        if data["coarse_graining"] is not None:
            from .coarse_graining import verify_coarse_graining, _bounded_model
            from css_decoupling.algebra.bounded_linear import BoundedLinearBackend
            bounded_origin = isinstance(get_backend(ring), BoundedLinearBackend)

            def verify_attached(verifier, value):
                if bounded_origin:
                    with using_backend("sympy"):
                        return verifier(value)
                return verifier(value)

            coarse = data["coarse_graining"]
            proof = verify_attached(verify_coarse_graining, coarse)
            if not proof.valid or not coarse["decouplable"]:
                raise ValueError("Invalid successful Algorithm 1 source")
            source_model = _bounded_model(coarse["restricted"], ring)
            if _complex_dict(source_model) != _complex_dict(C):
                raise ValueError("Algorithm 2 did not use the exact Algorithm 1 output")
            if data.get("cohomology_handoff") is not None:
                handoff = data["cohomology_handoff"]
                original = _bounded_model(coarse["model"], ring)
                if handoff["source"] != _complex_dict(original) or handoff["basis"] != coarse["basis"]:
                    raise ValueError("Construction handoff changed its Algorithm 1 source or lattice")
                expected_reps = {raw["degree"]: raw["representatives"] for raw in coarse["layers"]}
                given_reps = {raw["degree"]: raw["source_representatives"] for raw in handoff["layers"]
                              if raw["layer"]["dimension"]}
                if given_reps != expected_reps:
                    raise ValueError("Construction handoff changed its Algorithm 1 representatives")
            physical = data["physical_decoupling"]
            proof = verify_attached(verify_decoupling, physical)
            if not proof.valid:
                raise ValueError("Invalid composed original-code circuit: " + "; ".join(proof.errors))
            if coarse["sector"] == "X":
                active, expected_input = physical, coarse["input"]
            else:
                if physical["input"] != coarse["input"]:
                    raise ValueError("Dual circuit changed the original input")
                active = physical["inner"]
                expected_input = {"HX": coarse["input"]["HZ"], "HZ": coarse["input"]["HX"]}
            if active["input"] != expected_input or active["lattice"]["basis_columns"] != coarse["basis"]:
                raise ValueError("Composed circuit input or lattice mismatch")
            if any(active[name] != inner[name] for name in ("stabilization", "multiplicities", "identity_checks")):
                raise ValueError("Composed circuit differs from Algorithm 2 output")
            checks.append("Algorithm 1 output and original-code physical circuit, including sector conversion")
            if bounded_origin:
                checks.append("bounded Algorithm 2; attached Algorithm 1 and original-code provenance checked natively")
        elif data["physical_decoupling"] is not None:
            raise ValueError("Original-code circuit needs its Algorithm 1 source certificate")
        _keys(data["configuration"], ("linear_backend",), "configuration")
        # A legacy provenance label is data, never an executable backend
        # selection. Preserve v1 certificates while checking their equations
        # with the explicitly selected native verifier above.
        if data["configuration"]["linear_backend"] not in ("sympy", "singular", "bounded"):
            raise ValueError("Unknown linear backend label")
        h = max((abs(v) for A in C.differentials.values() for f in A.entries.values()
                 for exponent in f.terms for v in exponent), default=0)
        slots = sum(C.rank(n) * C.rank(n + 1) for n in C.degrees)
        expected = dict(dimension=ring.D, half_width=h, matrix_entries=slots,
                        dense_coefficients=slots * (2*h+1)**ring.D,
                        stored_monomials=sum(len(f.terms) for A in C.differentials.values() for f in A.entries.values()),
                        complex_length=len(C.degrees), total_free_rank=sum(C.ranks.values()))
        _keys(data["input_size"], tuple(expected), "dense input size")
        for name, value in expected.items():
            if _int(data["input_size"][name], name, 0) != value:
                raise ValueError("Incorrect common-box input size")
        expected_resources = circuit_resources(stable)
        _keys(data["resources"], tuple(expected_resources), "resources")
        for name, value in data["resources"].items():
            if name == "ancillas":
                _keys(value, tuple(expected_resources[name]), "ancillas")
                for key, rank in value.items():
                    _int(rank, key, 0)
            else:
                _int(value, name, 0)
        if data["resources"] != expected_resources:
            raise ValueError("Reported circuit resources differ from the actual circuit")
        checks.extend(("complete source model and all trivial finite cohomology layers",
                       "full comparison and source-bound cone Schanuel stable frame",
                       "input-lattice physical circuit, actual free summands, and exact resource counts"))
    except (ValueError, TypeError, KeyError, AttributeError, IndexError) as exc:
        errors.append(str(exc))
    return VerificationReport(not errors, tuple(checks), tuple(errors), "input-complex-to-circuit-Algorithm-2")
