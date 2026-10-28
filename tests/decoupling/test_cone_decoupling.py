"""Section V.C circuits: geometry, general module solving and hostile witnesses."""
from copy import deepcopy
import json

import pytest

from css_decoupling.algebra.laurent import Matrix, Ring, eye, zeros
from css_decoupling.circuits.gates import SUM
from css_decoupling.certificates.cone_decoupling import verify_cone_decoupling
from css_decoupling.decoupling.cone_decoupling import (
    ConePresentation, decouple_cone, solve_cone_homotopy,
)
from css_decoupling.errors import NoLaurentSolution, ResourceLimit
from css_decoupling.models.examples import periodic_toric_cone
from css_decoupling.topology.complexes import CochainComplex


@pytest.mark.parametrize("p", [2, 3, 5])
def test_manuscript_geometric_homotopy_has_one_layer_and_no_ancillas(p):
    cone, basis, h = periodic_toric_cone(Ring(p, 3))
    assert basis == [[1, 0, 0], [0, p, 0], [0, 0, 1]]
    result = decouple_cone(cone, lattice_basis=basis, homotopy=h)
    assert result.ancillas == {"source": 0, "target": 0}
    assert result.depth == 1
    assert result.circuit.qudits == 6 * p
    assert len(result.circuit.gates) == 2 * (p - 1)
    assert all(isinstance(gate, SUM) for gate in result.circuit.gates)
    assert all(gate.displacement == (0, 0, 0) for gate in result.circuit.gates)
    if p == 2:
        assert all(gate.weight == 1 for gate in result.circuit.gates)
        assert result.disjoint_layers == ((0, 1),)
    else:
        # Positive h coefficients s require physical SUM weights -s.
        assert {gate.weight for gate in result.circuit.gates} == set(range(1, p))
        for gate in result.circuit.gates:
            coefficient = result.U[gate.control, gate.target].constant()
            assert gate.weight == -coefficient % p
    assert result.U * result.Uinv == eye(cone.ring, 6*p)
    # The retained target has p-fold generator counts. It is not the smaller
    # coarse-cell canonical toric model, whose physical rank would be six.
    assert result.target.rank(0) == 6*p
    assert result.target_kind == "restricted-direct-sum-of-input-complexes"
    encoded = json.loads(json.dumps(result.certificate()))
    report = verify_cone_decoupling(encoded)
    assert report.valid, report.errors
    assert encoded["target"]["claims_minimal_toric_stack"] is False
    assert encoded["construction"]["gate_optimization"] == "none"


@pytest.mark.parametrize("p,D,degree,axes", [(3, 4, 1, (1, 3)), (5, 5, 2, (0, 4))])
def test_geometric_family_extends_to_other_dimensions_degrees_and_axes(p, D, degree, axes):
    cone, basis, h = periodic_toric_cone(Ring(p, D), degree, axes=axes, coefficient=2)
    result = decouple_cone(cone, lattice_basis=basis, homotopy=h)
    assert result.depth == 1
    assert result.U.ncols == cone.complex().rank(0) * p
    assert verify_cone_decoupling(result).valid


@pytest.mark.parametrize("p", [2, 3])
def test_exact_module_solver_and_original_lattice_obstruction(p):
    cone, basis, h = periodic_toric_cone(Ring(p, 3))
    with pytest.raises(NoLaurentSolution):
        decouple_cone(cone)
    automatic = decouple_cone(cone, lattice_basis=basis)
    geometric = decouple_cone(cone, lattice_basis=basis, homotopy=h)
    assert automatic.homotopy_origin == "exact-Laurent-module-solve"
    assert geometric.homotopy_origin == "supplied"
    assert verify_cone_decoupling(automatic).valid
    assert automatic.source.d(0) == geometric.source.d(0)
    assert automatic.target.d(0) == geometric.target.d(0)
    # The two exact homotopies need not be equal or have ordered gate counts.
    # A different valid module-solve witness may be as short as the geometric
    # one; correctness is the physical equivalence certified on both routes.
    assert verify_cone_decoupling(geometric).valid


def nontoric_cone():
    r = Ring(5, 4)
    x, y, z, w = r.variables
    A = CochainComplex(r, {-1: 2, 0: 2, 1: 2}, {
        -1: Matrix.from_rows(r, [[1, 0], [0, 0]]),
        0: Matrix.from_rows(r, [[0, 0], [0, y]]),
    })
    B = CochainComplex(r, {-1: 1, 0: 3, 1: 2}, {
        -1: Matrix.from_rows(r, [[x], [0], [0]]),
        0: Matrix.from_rows(r, [[0, 1, 0], [0, 0, x-1]]),
    })
    h = {-1: Matrix.from_rows(r, [[x**-1], [w+z]]),
         0: Matrix.from_rows(r, [[2*y, 0, z**-1], [x-1, 3, 0]]),
         1: Matrix.from_rows(r, [[w, 2*y**-1], [x, z]])}
    coupling = {n: A.d(n)*h[n] - h.get(n+1, zeros(r, A.rank(n+1), B.rank(n+1)))*B.d(n)
                for n in A.degrees}
    return ConePresentation(A, B, coupling), h


def test_general_nontoric_rectangular_blocks_and_nonorthogonal_lattice():
    cone, _ = nontoric_cone()
    parsed = ConePresentation.from_complex(cone.complex(), cone.lower.ranks)
    assert parsed.complex().d(0) == cone.complex().d(0)
    basis = [[2, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    result = decouple_cone(parsed, lattice_basis=basis)
    assert result.circuit.qudits == 10
    assert result.target.rank(0) == 10
    assert result.target.d(0)*result.U == result.V*result.source.d(0)
    assert verify_cone_decoupling(result).valid


def test_wrong_homotopy_and_resource_failures_are_distinct():
    cone, basis, h = periodic_toric_cone(Ring(3, 3))
    bad = dict(h)
    bad[0] = zeros(cone.ring, h[0].nrows, h[0].ncols)
    with pytest.raises(ValueError, match="null-homotopy"):
        decouple_cone(cone, lattice_basis=basis, homotopy=bad)
    with pytest.raises(ResourceLimit, match="unknowns"):
        decouple_cone(cone, lattice_basis=basis, max_unknowns=1)
    with pytest.raises(ValueError, match="cap"):
        solve_cone_homotopy(cone, max_unknowns=True)
    with pytest.raises(ValueError, match="dimension"):
        decouple_cone(cone, lattice_basis=[[1, 0], [0, 1]], homotopy=h)


@pytest.fixture(scope="module")
def exact_certificate():
    cone, basis, h = periodic_toric_cone(Ring(3, 3))
    return decouple_cone(cone, lattice_basis=basis, homotopy=h).certificate()


@pytest.mark.parametrize("mutation", ["coupling", "h", "map", "inverse", "X-check", "Z-check",
                                       "gate", "layer-overlap", "schedule-missing", "depth", "ancillas",
                                       "target-claim", "ring", "lattice-bool", "extra-success"])
def test_direct_certificate_tampering_rejected(exact_certificate, mutation):
    data = deepcopy(exact_certificate)
    if mutation == "coupling":
        pair = next(pair for pair in data["input"]["coupling"] if pair[1]["entries"])
        pair[1]["entries"].pop()
    elif mutation in ("h", "map", "inverse"):
        key = {"h": "homotopy", "map": "maps", "inverse": "inverse_maps"}[mutation]
        pair = next(pair for pair in data[key] if pair[0] == 0)
        pair[1]["entries"].pop()
    elif mutation in ("X-check", "Z-check"):
        key = "source_HX" if mutation == "X-check" else "target_HZ"
        data["checks"][key]["entries"].pop()
    elif mutation == "gate":
        data["circuit"]["gates"][0]["weight"] = 0
    elif mutation == "layer-overlap":
        data["schedule"]["layers"][0].append(data["schedule"]["layers"][0][0])
    elif mutation == "schedule-missing":
        data["schedule"]["layers"][0].pop()
    elif mutation == "depth":
        data["schedule"]["depth"] += 1
    elif mutation == "ancillas":
        data["ancillas"]["source"] = 1
    elif mutation == "target-claim":
        data["target"]["claims_minimal_toric_stack"] = True
    elif mutation == "ring":
        data["ring"]["prime"] = 9
    elif mutation == "lattice-bool":
        data["lattice"]["index"] = True
    elif mutation == "extra-success":
        data["success"] = True
    assert not verify_cone_decoupling(data).valid


def test_zero_coupling_and_empty_physical_space():
    r = Ring(3, 2)
    A = CochainComplex(r, {1: 1}, {})
    B = CochainComplex(r, {2: 2}, {})
    cone = ConePresentation(A, B, {})
    result = decouple_cone(cone)
    assert result.depth == 0
    assert result.circuit.qudits == 0
    assert verify_cone_decoupling(result).valid
