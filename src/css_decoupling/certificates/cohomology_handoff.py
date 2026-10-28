"""Independent small-operator proofs for exact restriction of cohomology.

Restriction is exact and finite free. Completeness is established on the
original operators, never from the equation A*K=0 alone. These native proofs
are explicitly unsupported by a strictly bounded verifier.
"""
from css_decoupling.algebra.laurent import Ring, block, eye, zeros
from css_decoupling.algebra.modules import GeneralBackend
from css_decoupling.lattice.retained import Lattice
from css_decoupling.errors import ResourceLimit
from .comparison import VerificationReport, _keys, _int, _matrix_from_dict
from .cone_decoupling import _read_complex, _complex_dict

SCHEMA = "css-decoupling.cohomology-handoff"
KERNEL_SCHEMA = "css-decoupling.restricted-kernel"
DEFAULT_CELL_INDEX = 4096
DEFAULT_TOTAL_RANK = 100000


def restriction_budget(lattice, total_rank, *, max_cell_index=DEFAULT_CELL_INDEX,
                       max_total_rank=DEFAULT_TOTAL_RANK):
    """Check metadata before coset expansion, identity allocation or kernel work."""
    for value, name in ((max_cell_index, "max_cell_index"), (max_total_rank, "max_total_rank")):
        if value is not None:
            _int(value, name, 1)
    if max_cell_index is not None and lattice.index > max_cell_index:
        raise ResourceLimit(f"Handoff lattice index {lattice.index} exceeds max_cell_index={max_cell_index}")
    if max_total_rank is not None and total_rank > max_total_rank:
        raise ResourceLimit(f"Handoff total free rank {total_rank} exceeds max_total_rank={max_total_rank}")


def _complex_rank(data):
    _keys(data, ("ranks", "differentials"), "handoff model")
    if not isinstance(data["ranks"], list):
        raise ValueError("Model ranks must be a list")
    total = 0
    for pair in data["ranks"]:
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError("Malformed model rank")
        _int(pair[0], "degree")
        total += _int(pair[1], "rank", 0)
    return total


class _IndependentNativeBackend(GeneralBackend):
    """A fresh verification session, never the construction's kernel cache."""


def _verification_backend(ring, backend):
    if isinstance(backend, _IndependentNativeBackend):
        return backend
    native_backend(ring, backend)  # Reject strict bounded/external contexts before any work.
    return _IndependentNativeBackend(ring)


def native_backend(ring, backend=None):
    if backend == "sympy":
        return GeneralBackend(ring)
    if backend == "bounded" or (backend is not None and getattr(backend, "name", "").startswith("bounded-")):
        raise ValueError("Native cohomology handoff is not a bounded kernel proof")
    if type(backend) is GeneralBackend:
        return backend
    from css_decoupling.lattice.restriction_backend import RestrictionBackend
    if isinstance(backend, RestrictionBackend):
        return backend.base_backend
    if backend is not None:
        raise ValueError("Cohomology handoff requires an explicit native SymPy backend")
    return GeneralBackend(ring)


def _complete(operator, generators, backend):
    if generators.nrows != operator.ncols or operator * generators:
        raise ValueError("Original kernel generators are not cycles")
    complete = backend.kernel(operator)
    forward = backend.solve(generators, complete)
    reverse = backend.solve(complete, generators)
    if generators * forward != complete or complete * reverse != generators:
        raise ValueError("Original complete kernel and supplied generators have different images")


def check_restricted_kernel(proof, *, backend=None, max_cell_index=DEFAULT_CELL_INDEX,
                            max_total_rank=DEFAULT_TOTAL_RANK):
    _keys(proof, ("schema", "version", "ring", "kind", "basis", "source"), "restricted kernel")
    if proof["schema"] != KERNEL_SCHEMA or _int(proof["version"], "version") != 1:
        raise ValueError("Unsupported restricted-kernel proof")
    _keys(proof["ring"], ("prime", "dimension"), "ring")
    ring = Ring(_int(proof["ring"]["prime"], "prime", 2),
                _int(proof["ring"]["dimension"], "dimension", 2))
    # Nested layer verification inherits the caller's already validated
    # budget, including explicit None; defaults must not silently reappear.
    if isinstance(backend, _IndependentNativeBackend) and hasattr(backend, "handoff_limits"):
        max_cell_index, max_total_rank = backend.handoff_limits
    selected = _verification_backend(ring, backend)
    lattice = Lattice(proof["basis"], max_index=max_cell_index)
    restriction_budget(lattice, 0, max_cell_index=max_cell_index, max_total_rank=max_total_rank)
    if lattice.D != ring.D:
        raise ValueError("Kernel restriction has another lattice dimension")
    read = lambda value: _matrix_from_dict(value, ring, "original kernel matrix")
    raw = proof["source"]
    if proof["kind"] == "kernel":
        _keys(raw, ("operator", "generators"), "original kernel")
        operator, generators = read(raw["operator"]), read(raw["generators"])
        restriction_budget(lattice, lattice.index * (operator.nrows + operator.ncols + generators.ncols),
                           max_cell_index=max_cell_index, max_total_rank=max_total_rank)
        _complete(operator, generators, selected)
    elif proof["kind"] == "cycle-boundary":
        _keys(raw, ("cycles", "boundary", "coordinates", "cycle_kernel"), "cycle-boundary kernel")
        W, dp, B, K = (read(raw[key]) for key in ("cycles", "boundary", "coordinates", "cycle_kernel"))
        restriction_budget(lattice, lattice.index * (W.nrows + W.ncols + K.ncols + 2 * dp.ncols),
                           max_cell_index=max_cell_index, max_total_rank=max_total_rank)
        if W * B != dp:
            raise ValueError("Original boundary coordinates are false")
        _complete(W, K, selected)
        # W*u=dp*v iff u=K*z+B*v. This generates the ENTIRE relation kernel.
        operator = W.hstack(-dp)
        generators = block([[K, B], [zeros(ring, dp.ncols, K.ncols), eye(ring, dp.ncols)]])
    else:
        raise ValueError("Unknown restricted-kernel proof kind")
    return lattice.restrict(operator), lattice.restrict(generators)


def checked_handoff(data, *, complex=None, backend=None, max_dimension=None,
                    max_cell_index=DEFAULT_CELL_INDEX, max_total_rank=DEFAULT_TOTAL_RANK):
    """Return checked layer dictionaries; all positive degrees include zero layers."""
    from .coarse_graining import _verify_source_model
    from .bounded_linear import verify_trivial_cohomology, _augment
    _keys(data, ("schema", "version", "ring", "source", "target", "basis", "layers"), "cohomology handoff")
    if data["schema"] != SCHEMA or _int(data["version"], "version") != 1:
        raise ValueError("Unsupported cohomology handoff")
    _keys(data["ring"], ("prime", "dimension"), "ring")
    ring = Ring(_int(data["ring"]["prime"], "prime", 2),
                _int(data["ring"]["dimension"], "dimension", 2))
    selected = _verification_backend(ring, backend)
    if max_dimension is not None:
        _int(max_dimension, "max_dimension", 0)
    lattice = Lattice(data["basis"], max_index=max_cell_index)
    restriction_budget(lattice, max(lattice.index * _complex_rank(data["source"]),
                                   _complex_rank(data["target"])),
                       max_cell_index=max_cell_index, max_total_rank=max_total_rank)
    selected.handoff_limits = (max_cell_index, max_total_rank)
    source = _read_complex(data["source"], ring, "original full model")
    target = _read_complex(data["target"], ring, "restricted full model")
    if lattice.D != ring.D:
        raise ValueError("Handoff lattice has another dimension")
    if complex is not None and (complex.ring != ring or _complex_dict(complex) != data["target"]):
        raise ValueError("Handoff changed the Algorithm 2 input complex")
    if target.ranks != {n: lattice.index * r for n, r in source.ranks.items()}:
        raise ValueError("Handoff changed complete restricted ranks")
    for n in source.degrees:
        if target.d(n) != lattice.restrict(source.d(n)):
            raise ValueError("Handoff target is not the actual restriction")
        if target.d(n).dagger() != lattice.restrict(source.d(n).dagger()):
            raise ValueError("Restriction did not preserve the actual dagger convention")
    # Both physical CSS sectors, the full negative resolution and positive
    # dual tail are checked on the ORIGINAL model; exact restriction carries
    # all of these equalities and complete kernels to the supplied target.
    _verify_source_model(source, source.d(0), source.d(-1).dagger(),
                         kernel=selected.kernel, solve=selected.solve)
    if not isinstance(data["layers"], list) or any(not isinstance(item, dict) for item in data["layers"]):
        raise ValueError("Handoff layers must be a list")
    expected = [n for n in source.degrees if n > 0]
    if [item.get("degree") for item in data["layers"]] != expected:
        raise ValueError("Handoff must include every positive degree, including zero layers")
    result = {}
    read = lambda value: _matrix_from_dict(value, ring, "handoff matrix")
    for item in data["layers"]:
        _keys(item, ("degree", "layer", "source_representatives", "source_coordinates",
                     "basis_change", "basis_change_inverse", "boundary_lift"), "handoff layer")
        n = _int(item["degree"], "degree", 1)
        raw = item["layer"]
        if raw["degree"] != n:
            raise ValueError("Handoff layer degree changed")
        first, second = raw["kernel_certificate"], raw["relation_kernel_certificate"]
        if (first["kind"] != "kernel" or second["kind"] != "cycle-boundary"
                or first["basis"] != data["basis"] or second["basis"] != data["basis"]
                or first["ring"] != data["ring"] or second["ring"] != data["ring"]):
            raise ValueError("Layer proof changed its restriction or kind")
        W0 = read(first["source"]["generators"])
        if (read(first["source"]["operator"]) != source.d(n)
                or read(second["source"]["cycles"]) != W0
                or read(second["source"]["boundary"]) != source.d(n-1)):
            raise ValueError("Layer proof is not bound to the original complex")
        report = verify_trivial_cohomology(raw, complex=target, backend=selected)
        if not report.valid:
            raise ValueError("Invalid transported layer: " + "; ".join(report.errors))
        dimension = raw["dimension"]
        if max_dimension is not None and dimension > max_dimension:
            raise ResourceLimit("Transported layer exceeds max_dimension")
        if dimension and not 1 <= n < ring.D:
            raise ValueError("Handoff has cohomology outside the CSS profile degrees")
        original, coordinates = read(item["source_representatives"]), read(item["source_coordinates"])
        change, inverse, lift = (read(item[key]) for key in
                                 ("basis_change", "basis_change_inverse", "boundary_lift"))
        if original.shape != (source.rank(n), dimension) or W0 * coordinates != original:
            raise ValueError("Original representatives are not bound to the cycle coordinates")
        if change.shape != (dimension, dimension) or inverse.shape != change.shape:
            raise ValueError("Transported representative basis change has a wrong shape")
        if any(not f.is_constant for M in (change, inverse) for f in M.entries.values()):
            raise ValueError("Cohomology basis change must be over the prime field")
        if change * inverse != eye(ring, dimension) or inverse * change != eye(ring, dimension):
            raise ValueError("Transported representatives are not a complete basis")
        if change != read(raw["projection"]) * _augment(lattice.pack(coordinates)):
            raise ValueError("Representative change differs from the actual quotient coordinates")
        if lattice.pack(original) - read(raw["representatives"]) * change != target.d(n-1) * lift:
            raise ValueError("Transported representatives lack their actual boundary binding")
        if dimension:
            result[n] = raw
    return result


def verify_cohomology_handoff(data, *, complex=None, backend=None, max_dimension=None,
                              max_cell_index=DEFAULT_CELL_INDEX, max_total_rank=DEFAULT_TOTAL_RANK):
    try:
        checked_handoff(data, complex=complex, backend=backend, max_dimension=max_dimension,
                        max_cell_index=max_cell_index, max_total_rank=max_total_rank)
        return VerificationReport(True, ("complete original model and exact restriction",
            "complete original kernels and cycle-boundary relation kernels",
            "all degrees, trivial translations and representative basis bindings"), (), SCHEMA)
    except (ValueError, TypeError, KeyError, IndexError, ArithmeticError) as exc:
        return VerificationReport(False, (), (str(exc),), SCHEMA)
