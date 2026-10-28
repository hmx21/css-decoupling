"""Raw HX/HZ -> complete input analysis -> paper QCA -> optional circuit."""
from __future__ import annotations
from time import perf_counter

from .algebra import Matrix, eye, zeros, solve, kernel, independent_columns, direct_sum
from ..topology.cohomology import FiniteQuotient
from ..lattice.kernel import translation_kernel
from ..lattice.retained import Lattice, matrix_power
from ..errors import ResourceLimit, NoLaurentSolution, InfiniteLayer
from .qca import construct_qca
from .serialization import digest
from .compact_cell import choose_compact_cell, verify_cell_geometry

SCHEMA = "css-decoupling-2d/qca-circuit/v1"
PRIME_SCHEMA = "css-decoupling-2d/qca-circuit/v2"


class InvalidCSS(ValueError):
    """An exact check of a required input assumption failed."""


class LatticeObstruction(InvalidCSS):
    """A requested translation acts nontrivially on an actual anyon module."""

    def __init__(self, sector, translation, action):
        self.sector, self.translation, self.action = sector, tuple(translation), action
        super().__init__(f"Requested lattice translation {self.translation} acts "
                         f"nontrivially on the {sector} anyon module")


def lattice_policy(mode, basis):
    """Resolve old basis-only calls while rejecting contradictory new requests."""
    if mode is None:
        return "auto" if basis is None else "fixed"
    if mode not in ("auto", "fixed"):
        raise ValueError("lattice_mode must be 'auto' or 'fixed'")
    if (mode == "fixed") != (basis is not None):
        raise ValueError("fixed mode requires lattice_basis; auto mode forbids it")
    return mode


def check_shape(HX, HZ):
    if not isinstance(HX, Matrix) or not isinstance(HZ, Matrix):
        raise TypeError("HX and HZ must be native Laurent matrices")
    if HX.ring != HZ.ring or HX.ncols != HZ.ncols:
        raise InvalidCSS("HX/HZ require identical rings and physical column counts")
    if HX.ring.D != 2:
        raise InvalidCSS("The two-dimensional algorithm requires D=2")
    if HX * HZ.dagger():
        raise InvalidCSS("HX HZ^dagger is nonzero")


def matrix_stats(A):
    exponents = [e for f in A.entries.values() for e in f.terms]
    return {"rows": A.nrows, "columns": A.ncols, "nonzero_entries": len(A.entries),
            "monomials": len(exponents),
            "max_coordinate_range": max((max(map(abs, e)) for e in exponents), default=0),
            "max_l1_range": max((sum(map(abs, e)) for e in exponents), default=0)}


def exactness(HX, HZ, progress=None):
    """Complete physical kernels, with equality witnesses in both sectors."""
    answer = {}
    for label, check, stabilizers in (("Z", HX, HZ.dagger()), ("X", HZ, HX.dagger())):
        if progress:
            progress("input_exactness",{"sector":label,"equation":"ker(check) = im(opposite check dagger)","check_shape":list(check.shape)})
        K = kernel(check)
        try:
            in_image = solve(stabilizers, K)
        except NoLaurentSolution as exc:
            raise InvalidCSS(f"{label} physical kernel contains a finite-support logical operator") from exc
        image_in_kernel = solve(K, stabilizers)
        answer[label] = {"kernel": K, "kernel_in_stabilizers": in_image,
                         "stabilizers_in_kernel": image_in_kernel}
    return answer


def quotient_record(quotient):
    return {"dimension": quotient.dimension, "basis": quotient.basis,
            "actions": list(quotient.actions), "standard_monomials": list(quotient.standard),
            "groebner": quotient.groebner_data}


def generation_witness(source, target):
    return {"source_in_target": solve(target, source), "target_in_source": solve(source, target)}


def stabilized_checks(HX, HZ, ancillas):
    return HX.hstack(zeros(HX.ring, HX.nrows, ancillas)), direct_sum(HZ, eye(HX.ring, ancillas))


def decouple(HX, HZ, *, lattice_basis=None, lattice_mode=None, compile_circuit=False,
             max_index=4096, max_dimension=10000, max_matrix_size=2048,
             max_terms=2_000_000, max_families=200000, progress=None):
    check_shape(HX, HZ)
    mode = lattice_policy(lattice_mode, lattice_basis)
    modern = HX.ring.p != 2 or lattice_mode is not None
    # Explicit fixed mode proves only what was requested, without first
    # constructing the maximal translation kernel. Legacy v1 keeps its proof.
    fixed_only = modern and mode == "fixed"
    selected = Lattice(lattice_basis, max_index=max_index) if mode == "fixed" else None
    if selected is not None and selected.D != 2:
        raise ValueError("The specified lattice must have two-dimensional columns")
    if selected is not None:
        # Lattice accepts Integral coordinates, including SymPy Integer.
        # Canonical JSON uses Python integers; numerical columns are unchanged.
        selected.basis = tuple(tuple(int(v) for v in row) for row in selected.basis)
    for name, cap in (("max_index",max_index),("max_dimension",max_dimension),
                      ("max_matrix_size",max_matrix_size),("max_terms",max_terms),("max_families",max_families)):
        if type(cap) is not int or cap < (0 if name == "max_families" else 1):
            raise ValueError(f"Invalid {name}")
    timings = {}
    start = perf_counter()
    original = {"HX": HX, "HZ": HZ, "p": HX.ring.p, "D": 2}
    input_proof = exactness(HX, HZ, progress)
    timings["input_exactness"] = perf_counter() - start
    mark = perf_counter()
    # Columns are stabilizer supports. Basis changes must be transported back
    # using dagger, not plain transpose.
    reduced = {}
    for sector, H in (("X", HX), ("Z", HZ)):
        if progress:
            progress("redundant_check_basis",{"sector":sector,"equation":"B=A*T; A=B*S; ker(B)=0"})
        data = independent_columns(H.dagger())
        reduced[sector] = {key: data[key] for key in ("basis", "basis_in_input", "input_in_basis")}
    RX, RZ = reduced["X"]["basis"].dagger(), reduced["Z"]["basis"].dagger()
    if RX.ncols != RX.nrows + RZ.nrows:
        raise InvalidCSS("Independent check density q != r+s")
    timings["redundancy"] = perf_counter() - mark
    mark = perf_counter()
    try:
        if progress:
            progress("anyon_quotient_X",{"equation":"coker(reduced HX), complete staircase and translations"})
        QX = FiniteQuotient(RX, max_dimension=max_dimension)
        if progress:
            progress("anyon_quotient_Z",{"equation":"coker(reduced HZ), complete staircase and translations"})
        QZ = FiniteQuotient(RZ, max_dimension=max_dimension)
    except InfiniteLayer as exc:
        raise InvalidCSS("Independent-check anyon cokernel is infinite dimensional") from exc
    if QX.dimension != QZ.dimension:
        raise InvalidCSS("Electric and magnetic anyon dimensions differ")
    layers = {0: QX, 1: QZ}
    full, cell_geometry = None, None
    if not fixed_only:
        if progress:
            progress("full_translation_kernel",{"dimensions":[QX.dimension,QZ.dimension],"equation":"ker(Z^2 -> GL(anyon_X) x GL(anyon_Z))"})
        full = translation_kernel(layers, 2, max_index=max_index)
    if mode == "auto":
        selected, cell_geometry = choose_compact_cell(full, HX.ring, max_index=max_index)
        if progress:
            progress('compact_cell_coordinates', {'basis_columns': selected.metadata()['basis_columns'],
                     'index': selected.index, 'representative_strategy': 'explicit-v1',
                     'equation': 'Same full translation kernel via unimodular basis and unitary coset gauge'})
    for label, Q in (("X", QX), ("Z", QZ)):
        for column in range(2):
            action = eye(HX.ring, Q.dimension)
            for axis in range(2):
                action = action * matrix_power(Q.actions[axis], selected.basis[axis][column])
            if action != eye(HX.ring, Q.dimension):
                raise LatticeObstruction(label, [selected.basis[a][column] for a in range(2)], action)
    if full is not None and not full.contains(selected):
        raise ArithmeticError("Translation-kernel containment disagrees with checked actions")
    timings["anyons_and_full_kernel"] = perf_counter() - mark
    if progress:
        progress("restriction_preflight",{"index":selected.index,"physical_size":HX.ncols*selected.index})
    if HX.ncols * selected.index > max_matrix_size:
        raise ResourceLimit(f"Coarse physical size {HX.ncols * selected.index} exceeds {max_matrix_size}")
    estimated_terms = selected.index * sum(len(f.terms) for A in (HX,HZ,RX,RZ) for f in A.entries.values())
    if estimated_terms > max_terms:
        raise ResourceLimit(f"Restriction term upper bound {estimated_terms} exceeds {max_terms}")
    mark = perf_counter()
    CX, CZ = selected.restrict(RX), selected.restrict(RZ)
    OX, OZ = selected.restrict(HX), selected.restrict(HZ)
    if cell_geometry is not None:
        verify_cell_geometry(full, selected, cell_geometry, HX.ring, checks=((HX, OX), (HZ, OZ)))
    timings["restriction"] = perf_counter() - mark
    mark = perf_counter()
    qca = construct_qca(CX, CZ, QX.dimension, progress=progress,
        restriction_context={"HX":RX,"HZ":RZ,"lattice":selected,"quotient":QX,"max_terms":max_terms})
    U, V = qca["psi"][1], qca["psi_inverse"][1]
    TX, TZ = qca["standard_HX"], qca["standard_HZ"]
    # Compose the already known endpoint and check-reduction maps. These
    # are actual generation witnesses, not new large module-membership solves.
    XT=selected.restrict(reduced["X"]["basis_in_input"])
    XS=selected.restrict(reduced["X"]["input_in_basis"])
    ZT=selected.restrict(reduced["Z"]["basis_in_input"])
    ZS=selected.restrict(reduced["Z"]["input_in_basis"])
    witnesses = {
        "Z":{"source_in_target":qca["psi"][2]*ZS,
             "target_in_source":ZT*qca["psi_inverse"][2]},
        "X":{"source_in_target":qca["psi_inverse"][0].dagger()*XS,
             "target_in_source":XT*qca["psi"][0].dagger()}}
    # Redundant endpoint presentations cannot be isomorphic to the independent
    # target. Record the actual rectangular chain comparison and its retraction.
    to_standard = {0:qca["psi"][0]*selected.restrict(reduced["X"]["basis_in_input"].dagger()),
                   1:U,2:qca["psi"][2]*selected.restrict(reduced["Z"]["input_in_basis"])}
    to_original = {0:selected.restrict(reduced["X"]["input_in_basis"].dagger())*qca["psi_inverse"][0],
                   1:V,2:selected.restrict(reduced["Z"]["basis_in_input"])*qca["psi_inverse"][2]}
    timings["qca"] = perf_counter() - mark
    result = {"schema": PRIME_SCHEMA if modern else SCHEMA, "original": original, "input_sha256": digest(original),
              "conventions": {"coordinates": ["x", "y"], "matrix_action": "columns",
                 "dagger": "transpose and negate Laurent exponents",
                 "psi": "input-to-standard", "psi_inverse": "standard-to-input",
                 "physical_Z": "psi_1", "physical_X": "psi_inverse_1 dagger"},
              "input_exactness": input_proof, "check_reduction": reduced,
              "reduced_HX": RX, "reduced_HZ": RZ,
              "anyons": {"X": quotient_record(QX), "Z": quotient_record(QZ)},
              "translation_kernel": full.kernel_certificate if full is not None else None,
              "lattice": selected.metadata(), "coarse_original_HX": OX, "coarse_original_HZ": OZ,
              "qca": qca, "qca_stabilizers": witnesses, "circuit": None,
              "original_chain_comparison":{"to_standard":to_standard,"to_original":to_original,
                  "original_endpoint_projectors":{i:to_original[i]*to_standard[i] for i in (0,2)},
                  "kind":"retraction_of_redundant_generator_presentation"},
              "circuit_status": "not_requested", "timings_seconds": timings,
              "resources": {"original_HX": matrix_stats(HX), "original_HZ": matrix_stats(HZ),
                 "coarse_HX": matrix_stats(CX), "coarse_HZ": matrix_stats(CZ),
                 "psi_1": matrix_stats(U), "psi_1_inverse": matrix_stats(V),
                 "lattice_index": selected.index, "toric_copies": QX.dimension,
                 "product_X": qca["product_x"], "product_Z": qca["product_z"]},
              "algorithm_source": {"paper": "arXiv:2608.09915v1", "upstream_commit":
                  "7879852262ee7bd0bb69c1a2edae8f1543fcca3e", "construction": "Algorithm 1"}}
    if modern:
        result["lattice_request"] = {"mode": mode,
            "basis_columns": [list(row) for row in selected.basis] if mode == "fixed" else None,
            "kernel_scope": "specified-translations" if fixed_only else "complete"}
        result["conventions"]["stabilizer_eigenvalue"] = "+1"
        result["conventions"]["pauli_phase"] = "pure-X/Z phase zero; mixed X-then-Z multiplicative extension"
        result["algorithm_source"]["extension"] = "prime-field-20261024"
    if cell_geometry is not None:
        result['cell_geometry'] = cell_geometry
    if compile_circuit:
        if progress:
            progress("circuit_compilation",{"equation":"gate product = diag(psi_1,psi_inverse_1)","size":U.nrows})
        compile_result_circuit(result, max_families=max_families,max_matrix_terms=max_terms)
    return result


def compile_result_circuit(result, *, max_families=200000,max_matrix_terms=1_000_000):
    """Compile an already verified QCA; retain it on explicit resource failure."""
    from .circuits import compile_qca
    start = perf_counter()
    Q = result["qca"]
    try:
        C = compile_qca(Q["psi"][1], Q["psi_inverse"][1], max_families=max_families,
                        max_matrix_terms=max_matrix_terms)
    except ResourceLimit as exc:
        result["circuit_status"] = "resource_incomplete"
        result["circuit_error"] = str(exc)
        return result
    n = Q["psi"][1].nrows
    OX, OZ = stabilized_checks(result["coarse_original_HX"], result["coarse_original_HZ"], n)
    TX, TZ = stabilized_checks(Q["standard_HX"], Q["standard_HZ"], n)
    C["source_HX"], C["source_HZ"] = OX, OZ
    C["target_HX"], C["target_HZ"] = TX, TZ
    old=result["qca_stabilizers"]
    C["stabilizers"] = {
        "Z":{"source_in_target":direct_sum(old["Z"]["source_in_target"],Q["psi_inverse"][1]),
             "target_in_source":direct_sum(old["Z"]["target_in_source"],Q["psi"][1])},
        "X":dict(old["X"])}
    C["retained_lattice"] = result["lattice"]["basis_columns"]
    result["circuit"] = C
    result["circuit_status"] = "complete_stabilized"
    result["timings_seconds"]["circuit"] = perf_counter() - start
    return result
