"""Transport native Algorithm 1 cohomology without rediscovering it after restriction."""
from copy import deepcopy

from css_decoupling.algebra.laurent import block, eye, zeros
from css_decoupling.algebra.bounded_linear import augmentation, constant_solve
from css_decoupling.algebra.modules import get_backend
from css_decoupling.lattice.restriction_backend import unpack_columns
from css_decoupling.certificates.comparison import _matrix_dict, _matrix_from_dict
from css_decoupling.certificates.cone_decoupling import _complex_dict, _read_complex
from css_decoupling.certificates.cohomology_handoff import (
    SCHEMA, KERNEL_SCHEMA, native_backend, checked_handoff, restriction_budget,
    DEFAULT_CELL_INDEX, DEFAULT_TOTAL_RANK)
from .trivial_cohomology import TrivialCohomologyLayer, TrivialActionQuotient, _quotient


def build_cohomology_handoff(source, lattice, layers=None, *, restricted=None,
                             representatives=None, backend=None,
                             max_cell_index=DEFAULT_CELL_INDEX, max_total_rank=DEFAULT_TOTAL_RANK):
    """Reuse W/B/K when present; missing zero degrees get original-size kernels.

    ``representatives`` is used when reconstructing evidence from an old
    coarse certificate, which never claimed to contain complete kernels.
    Every transported basis is bound by an invertible constant change and
    an actual boundary lift. No verification flag is serialized.
    """
    from css_decoupling.decoupling.splitting import restrict_complex
    restriction_budget(lattice, lattice.index * sum(source.ranks.values()),
                       max_cell_index=max_cell_index, max_total_rank=max_total_rank)
    if restricted is not None:
        restriction_budget(lattice, sum(restricted.ranks.values()),
                           max_cell_index=max_cell_index, max_total_rank=max_total_rank)
    selected = native_backend(source.ring, backend or get_backend(source.ring))
    if layers is None and representatives is None:
        from css_decoupling.algebra.modules import using_backend
        from .cohomology import profile
        with using_backend(selected):
            layers = profile(source)
    target = restricted if restricted is not None else restrict_complex(source, lattice)
    ring = source.ring
    spec = {"prime": ring.p, "dimension": ring.D}
    basis = [list(row) for row in lattice.basis]
    records = []
    for n in source.degrees:
        if n <= 0:
            continue
        layer = (layers or {}).get(n)
        dp, dn = source.d(n-1), source.d(n)
        if layer is None:
            W0 = selected.kernel(dn)
            B0 = selected.solve(W0, dp)
            K0 = selected.kernel(W0)
            original = (representatives or {}).get(n, zeros(ring, source.rank(n), 0))
            coordinates = selected.solve(W0, original)
        else:
            W0, B0, K0 = layer.cycles, layer.boundary_coordinates, layer.cycle_relations
            original = (representatives or {}).get(n, layer.representatives)
            stored_basis = getattr(getattr(layer, "quotient", None), "basis", None)
            coordinates = (stored_basis if stored_basis is not None and W0 * stored_basis == original
                           else selected.solve(W0, original))
        if dn * W0 or W0 * B0 != dp or W0 * K0 or W0 * coordinates != original:
            raise ValueError("Original layer is not bound to the source differentials")
        restriction_budget(lattice, lattice.index * (W0.nrows + W0.ncols + K0.ncols + 2 * dp.ncols),
                           max_cell_index=max_cell_index, max_total_rank=max_total_rank)
        relation0 = block([[K0, B0], [zeros(ring, dp.ncols, K0.ncols), eye(ring, dp.ncols)]])
        W, relation = lattice.restrict(W0), lattice.restrict(relation0)
        U = relation[:W.ncols, :]
        witnesses = []
        for vector in zip(*lattice.basis):
            rhs = (ring.monomial(tuple(vector)) - 1) * W0
            z = selected.solve(dp, rhs)
            if dp * z != rhs:
                raise ArithmeticError("Original lattice translation boundary is false")
            witnesses.append(lattice.restrict(z))
        quotient = _quotient(U)
        if quotient.dimension != original.ncols:
            raise ValueError("Original representative count disagrees with complete restricted cohomology")
        reps = W * quotient.basis
        change = quotient.projection * augmentation(lattice.pack(coordinates))
        inverse = constant_solve(change, eye(ring, quotient.dimension))
        difference = lattice.pack(original) - reps * change
        lift = lattice.pack(selected.solve(dp, unpack_columns(difference, lattice)))
        if target.d(n-1) * lift != difference:
            raise ArithmeticError("Transported representative binding failed")
        kernel_proof = {"schema": KERNEL_SCHEMA, "version": 1, "ring": spec,
                        "kind": "kernel", "basis": basis,
                        "source": {"operator": _matrix_dict(dn), "generators": _matrix_dict(W0)}}
        relation_proof = {"schema": KERNEL_SCHEMA, "version": 1, "ring": spec,
                          "kind": "cycle-boundary", "basis": basis,
                          "source": {"cycles": _matrix_dict(W0), "boundary": _matrix_dict(dp),
                                     "coordinates": _matrix_dict(B0), "cycle_kernel": _matrix_dict(K0)}}
        transported = TrivialCohomologyLayer(n, W, U, quotient, reps, relation, tuple(witnesses),
                        target.d(n-1), target.d(n), kernel_proof, relation_proof)
        records.append({"degree": n, "layer": transported.certificate(),
                        "source_representatives": _matrix_dict(original),
                        "source_coordinates": _matrix_dict(coordinates),
                        "basis_change": _matrix_dict(change), "basis_change_inverse": _matrix_dict(inverse),
                        "boundary_lift": _matrix_dict(lift)})
    return {"schema": SCHEMA, "version": 1, "ring": spec, "source": _complex_dict(source),
            "target": _complex_dict(target), "basis": basis, "layers": records}


def consume_cohomology_handoff(data, complex, *, backend=None, max_dimension=None,
                               max_cell_index=DEFAULT_CELL_INDEX, max_total_rank=DEFAULT_TOTAL_RANK):
    raw_layers = checked_handoff(data, complex=complex, backend=backend, max_dimension=max_dimension,
                                max_cell_index=max_cell_index, max_total_rank=max_total_rank)
    read = lambda value: _matrix_from_dict(value, complex.ring, "transported layer")
    result = {}
    for n, raw in raw_layers.items():
        quotient = TrivialActionQuotient(read(raw["presentation"]), read(raw["basis"]), read(raw["projection"]))
        result[n] = TrivialCohomologyLayer(n, read(raw["cycles"]), read(raw["presentation"]), quotient,
            read(raw["representatives"]), read(raw["relation_kernel"]),
            tuple(read(value) for value in raw["triviality_witnesses"]), complex.d(n-1), complex.d(n),
            deepcopy(raw["kernel_certificate"]), deepcopy(raw["relation_kernel_certificate"]))
    return result


def handoff_from_coarse_certificate(data, *, backend=None, max_cell_index=DEFAULT_CELL_INDEX,
                                    max_total_rank=DEFAULT_TOTAL_RANK):
    """Reconstruct missing ORIGINAL kernels in old v1, never invent saved proofs."""
    if data.get("cohomology_handoff") is not None:
        # A saved dictionary is not validation; at least reject oversized
        # metadata before copying it. Full proof checking occurs on consume.
        from css_decoupling.certificates.cohomology_handoff import _complex_rank
        from css_decoupling.lattice.retained import Lattice
        saved = data["cohomology_handoff"]
        lattice = Lattice(saved["basis"], max_index=max_cell_index)
        restriction_budget(lattice, max(lattice.index * _complex_rank(saved["source"]),
                                       _complex_rank(saved["target"])),
                           max_cell_index=max_cell_index, max_total_rank=max_total_rank)
        return deepcopy(data["cohomology_handoff"])
    from css_decoupling.algebra.laurent import Ring
    from css_decoupling.lattice.retained import Lattice
    if not data.get("decouplable"):
        raise ValueError("Handoff requires a successful coarse-graining certificate")
    ring = Ring(data["ring"]["prime"], data["ring"]["dimension"])
    from css_decoupling.certificates.coarse_graining import _bounded_model
    source = _bounded_model(data["model"], ring)
    target = _bounded_model(data["restricted"], ring)
    representatives = {raw["degree"]: _matrix_from_dict(raw["representatives"], ring, "original representatives")
                       for raw in data["layers"]}
    return build_cohomology_handoff(source, Lattice(data["basis"], max_index=max_cell_index),
                                    restricted=target, representatives=representatives, backend=backend,
                                    max_cell_index=max_cell_index, max_total_rank=max_total_rank)
