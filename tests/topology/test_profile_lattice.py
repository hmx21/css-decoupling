"""Independent module, lattice, profile and free-resolution regression checks."""
from itertools import combinations
from copy import deepcopy
import json

import pytest

from css_decoupling.algebra.laurent import Matrix, Ring, block, eye, zeros
from css_decoupling.topology.complexes import (
    CochainComplex, direct_sum_complexes, disk, koszul, toric_stack,
)
from css_decoupling.lattice.retained import (
    Lattice, ResourceLimit, action_order_multiple, matrix_power, trivializing_lattice,
)
from css_decoupling.algebra.modules import NoLaurentSolution, kernel, solve
from css_decoupling.topology.cohomology import FiniteQuotient, InfiniteLayer, cohomology_layer, profile
from css_decoupling.topology.resolutions import BackendUnavailable, css_complex, image_resolution
from css_decoupling.decoupling.splitting import restrict_complex, splitting_map


@pytest.mark.parametrize("basis", [
    [[2, 1], [0, 2]], [[1, 2], [2, 0]], [[-2, 1], [1, 1]], [[0, 1], [1, 0]],
])
def test_nonorthogonal_lattice_negative_exponents_and_restriction_functor(basis):
    r = Ring(5, 2)
    x, y = r.variables
    lattice = Lattice(basis)
    assert len(lattice.representatives) == lattice.index
    for exponent in [(-7, 9), (3, -5), (-2, -3), (0, 0), (13, 11)]:
        index, shift = lattice.reduce(exponent)
        rep = lattice.representatives[index]
        assert tuple(rep[i] + sum(basis[i][j] * shift[j] for j in range(2))
                     for i in range(2)) == exponent
    A = Matrix.from_rows(r, [[2*x**-3+y, x*y**-2], [x-1, 0]])
    B = Matrix.from_rows(r, [[y**-1, x+2], [3*y, x**-1]])
    assert lattice.restrict(A * B) == lattice.restrict(A) * lattice.restrict(B)
    assert lattice.restrict(A.dagger()) == lattice.restrict(A).dagger()
    assert lattice.pack(A * B) == lattice.restrict(A) * lattice.pack(B)
    assert lattice.restrict(eye(r, 2)) == eye(r, 2*lattice.index)
    C = koszul(r, degree=1)
    assert restrict_complex(C, lattice).validate()


def test_lattice_output_index_cap_and_invalid_basis():
    with pytest.raises(ResourceLimit, match="index"):
        Lattice([[101, 0], [0, 103]], max_index=10000)
    with pytest.raises(ValueError, match="full rank"):
        Lattice([[1, 2], [2, 4]])
    with pytest.raises(ValueError, match="integer"):
        Lattice([[1.5, 0], [0, 1]])
    assert Lattice([[1, 100], [0, 1]], max_index=1).index == 1
    lattice = Lattice([[2, 1], [0, 1]])
    for exponent in [(1.5, 0), (True, 0), ("1", 0)]:
        with pytest.raises(ValueError, match="integers"):
            lattice.reduce(exponent)
    for cap in [0, -1, True, 1.5]:
        with pytest.raises(ValueError, match="cap"):
            Lattice([[1, 0], [0, 1]], max_index=cap)


@pytest.mark.parametrize("p,D", [(3, 4), (5, 5)])
def test_finite_quotient_inverse_actions_and_coordinates(p, D):
    r = Ring(p, D)
    xs = r.variables
    generators = [(xs[0] - 1)**2] + [xs[i] - (2 if i == 1 else 1) for i in range(1, D)]
    A = Matrix.from_rows(r, [generators])
    quotient = FiniteQuotient(A)
    assert quotient.dimension == 2
    assert quotient.coordinates(quotient.basis) == eye(r, 2)
    assert action_order_multiple(quotient.actions[0]) % p == 0
    assert matrix_power(quotient.actions[0], p) == eye(r, 2)
    assert quotient.actions[1] == 2 * eye(r, 2)
    B = Matrix.from_rows(r, [[xs[0]**-2 * xs[1] + xs[2]**3]])
    coords = quotient.coordinates(B)
    residual = B - quotient.basis * coords
    assert A * solve(A, residual) == residual
    lattice = trivializing_lattice({1: quotient}, D, max_index=None)
    for i, action in enumerate(quotient.actions):
        assert matrix_power(action, lattice.basis[i][i]) == eye(r, 2)


def test_characteristic_polynomial_bound_can_exceed_minimal_order():
    r = Ring(3, 3)
    A = Matrix.from_rows(r, [[0, -1], [1, 0]])  # irreducible s^2+1 over F_3
    assert action_order_multiple(A) == 8
    assert matrix_power(A, 4) == eye(r, 2)
    assert matrix_power(A, -1) * A == eye(r, 2)
    with pytest.raises(ValueError, match="singular"):
        action_order_multiple(Matrix.from_rows(r, [[1, 0], [0, 0]]))
    with pytest.raises(ValueError, match="constant"):
        action_order_multiple(Matrix.from_rows(r, [[r.variable(0)]]))
    assert action_order_multiple(eye(r, 0)) == 1
    with pytest.raises(ValueError, match="integer"):
        matrix_power(A, 1.5)
    with pytest.raises(ValueError, match="square"):
        matrix_power(zeros(r, 2, 3), -1)


def test_zero_infinite_and_resource_limited_quotients_are_distinct():
    r = Ring(3, 2)
    x, y = r.variables
    zero = FiniteQuotient(eye(r, 3), max_dimension=0)
    assert zero.dimension == 0
    assert zero.coordinates(Matrix.from_rows(r, [[x], [y**-1], [2]])) == zeros(r, 0, 1)
    assert FiniteQuotient(zeros(r, 0, 2)).dimension == 0
    with pytest.raises(InfiniteLayer):
        FiniteQuotient(Matrix.from_rows(r, [[x-1]]))
    with pytest.raises(InfiniteLayer):
        FiniteQuotient(Matrix.from_rows(r, [[x-1, y-1], [0, 0]]))
    with pytest.raises(ResourceLimit):
        FiniteQuotient(Matrix.from_rows(r, [[(x-1)**3, y-1]]), max_dimension=2)


def test_profile_uses_relations_among_nonfree_kernel_generators():
    r = Ring(3, 3)
    C = koszul(r, degree=2)
    layer = cohomology_layer(C, 1)
    assert layer.cycles.ncols == 3
    assert layer.cycle_relations.ncols > 0
    assert layer.dimension == 0
    layers = profile(C)
    assert {degree: layer.dimension for degree, layer in layers.items()} == {2: 1}


def test_mixed_profile_splitting_and_contractible_disks():
    r = Ring(5, 3)
    C = direct_sum_complexes(toric_stack(r, {1: 1, 2: 1}), disk(r, 0), disk(r, 2))
    layers = profile(C)
    assert {degree: layer.dimension for degree, layer in layers.items()} == {1: 1, 2: 1}
    phi = splitting_map(C, {degree: layer.representatives for degree, layer in layers.items()})
    assert phi.validate()
    with pytest.raises(ResourceLimit, match="unknowns"):
        splitting_map(C, {degree: layer.representatives for degree, layer in layers.items()}, max_unknowns=1)


def glued_complex(p=3):
    """The nonzero Ext^2 class given by contraction along exterior axes 0,1."""
    r = Ring(p, 3)
    A, B = koszul(r, 1), koszul(r, 2)
    degrees = sorted(set(A.degrees) | set(B.degrees))
    maps = {}
    for degree in degrees:
        k = 2 - degree
        gluing = zeros(r, A.rank(degree + 1), B.rank(degree))
        if 2 <= k <= r.D:
            source = list(combinations(range(r.D), k))
            target = list(combinations(range(r.D), k - 2))
            for col, indices in enumerate(source):
                if 0 in indices and 1 in indices:
                    remaining = tuple(a for a in indices if a not in (0, 1))
                    gluing[target.index(remaining), col] = -1 if degree % 2 else 1
        maps[degree] = block([[A.d(degree), gluing],
                              [zeros(r, B.rank(degree + 1), A.rank(degree)), B.d(degree)]])
    return CochainComplex(r, {n: A.rank(n) + B.rank(n) for n in degrees}, maps)


def test_nonzero_gluing_obstructs_original_cell_splitting():
    C = glued_complex()
    layers = profile(C)
    assert {degree: layer.dimension for degree, layer in layers.items()} == {1: 1, 2: 1}
    assert all(action == eye(C.ring, layer.dimension)
               for layer in layers.values() for action in layer.actions)
    with pytest.raises(NoLaurentSolution):
        splitting_map(C, {j: layer.representatives for j, layer in layers.items()})


def test_resolution_clears_laurent_denominators_and_retains_ambient_zero_rows():
    r = Ring(3, 4)
    x, y, z, w = r.variables
    A = Matrix.from_rows(r, [[x**-2*(x-1), y**-1*(y-1), z-1, 0], [0, 0, 0, 0]])
    maps = image_resolution(A, method="native", timeout=20)
    assert maps[0].nrows == 2
    assert maps[0] * solve(maps[0], A) == A
    assert A * solve(A, maps[0]) == maps[0]
    for previous, following in zip(maps, maps[1:]):
        expected = kernel(previous)
        assert previous * following == 0
        assert following * solve(following, expected) == expected
    assert not kernel(maps[-1]).entries


def test_css_completion_retains_physical_window_in_odd_characteristic():
    r = Ring(5, 3)
    original = koszul(r, degree=1)
    HX, HZ = original.d(0), original.d(-1).dagger()
    C = css_complex(HX, HZ)
    assert C.d(0) == HX
    assert C.d(-1) == HZ.dagger()
    assert C.validate()
    assert {degree: layer.dimension for degree, layer in profile(C).items()} == {1: 1}


def test_resolution_truncates_after_terminal_zero_in_d4_multilayer_completion():
    r = Ring(3, 4)
    original = toric_stack(r, {1: 1, 2: 1, 3: 1})
    HX, HZ = original.d(0), original.d(-1).dagger()
    C = css_complex(HX, HZ)
    assert C.validate()
    assert C.d(0) == HX
    assert C.d(-1) == HZ.dagger()


def test_resolution_zero_module_does_not_require_backend():
    r = Ring(3, 4)
    result = image_resolution(zeros(r, 3, 4), method="native")
    assert result == [zeros(r, 3, 0)]


@pytest.fixture(scope="module")
def complete_certificate():
    from css_decoupling.decoupling.pipeline import decouple
    C = koszul(Ring(3, 2), degree=1)
    return decouple(C.d(0), C.d(-1).dagger()).certificate()


def test_full_certificate_json_roundtrip(complete_certificate):
    from css_decoupling.certificates.decoupling import verify_decoupling
    report = verify_decoupling(json.loads(json.dumps(complete_certificate)))
    assert report.valid, report.errors
    assert report.scope == "original-CSS-to-toric-stack-circuit"
    assert "Z stabilizer generators mapped onto one another in both directions" in report.checks


@pytest.mark.parametrize("mutation", [
    "ring", "multiplicity", "duplicate-layer", "boolean-lattice-index",
    "float-lattice-representative", "lattice-basis", "identity-checks", "input-Z",
])
def test_full_certificate_tampering_rejected(complete_certificate, mutation):
    from css_decoupling.certificates.decoupling import verify_decoupling
    data = deepcopy(complete_certificate)
    if mutation == "ring":
        data["stabilization"]["ring"]["prime"] = 5
    elif mutation == "multiplicity":
        data["multiplicities"][0][1] += 1
    elif mutation == "duplicate-layer":
        data["multiplicities"].append(data["multiplicities"][0][:])
    elif mutation == "boolean-lattice-index":
        data["lattice"]["index"] = True
    elif mutation == "float-lattice-representative":
        data["lattice"]["representatives"][0][0] = 0.0
    elif mutation == "lattice-basis":
        data["lattice"]["basis_columns"][0][0] = 2
    elif mutation == "identity-checks":
        data["identity_checks"]["source"] += 1
    elif mutation == "input-Z":
        data["input"]["HZ"]["entries"].pop()
    report = verify_decoupling(data)
    assert not report.valid, mutation
