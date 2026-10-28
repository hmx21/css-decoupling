from copy import deepcopy

import pytest

from css_decoupling.algebra.bounded_linear import BoundedLinearBackend
from css_decoupling.algebra.modules import GeneralBackend
from css_decoupling.algebra.laurent import Matrix, Ring, eye, zeros
from css_decoupling.certificates.bounded_linear import verify_trivial_cohomology
from css_decoupling.topology.cohomology import cohomology_layer
from css_decoupling.topology.complexes import CochainComplex, koszul, toric_stack
from css_decoupling.topology.trivial_cohomology import (
    trivial_cohomology, trivial_cohomology_layer,
)
from css_decoupling.errors import ResourceLimit


@pytest.mark.parametrize("D,p", [(2, 2), (3, 3), (4, 5), (5, 3)])
def test_toric_top_layer_without_groebner(D, p, monkeypatch):
    import css_decoupling.algebra.modules as modules
    ring = Ring(p, D)
    C = koszul(ring, copies=2)
    backend = BoundedLinearBackend(ring)
    monkeypatch.setattr(modules, "sdm_groebner", lambda *a, **kw: pytest.fail("Groebner called"))
    layer = trivial_cohomology_layer(C, 1, backend=backend)
    assert layer.dimension == 2
    assert layer.actions == tuple(eye(ring, 2) for _ in range(D))
    assert layer.quotient.coordinates(layer.quotient.basis) == eye(ring, 2)
    assert verify_trivial_cohomology(layer, complex=C).valid


@pytest.mark.parametrize("p", [2, 3, 5])
def test_multilayer_profile_agrees_with_existing_quotient(p):
    ring = Ring(p, 3)
    C = toric_stack(ring, {1: 1, 2: 1})
    layers = trivial_cohomology(C)
    assert set(layers) == {1, 2}
    for degree, layer in layers.items():
        reference = cohomology_layer(C, degree)
        assert layer.dimension == reference.dimension
        assert verify_trivial_cohomology(layer, complex=C).valid


def test_nontrivial_translation_is_not_erased_by_augmentation():
    ring = Ring(3, 2)
    x = ring.variable(0)
    C = CochainComplex(ring, {0: 1, 1: 1},
                       {0: Matrix.from_rows(ring, [[(x - 1)**2]])})
    with pytest.raises(ValueError, match="not annihilated"):
        trivial_cohomology_layer(C, 1)


def test_exact_acyclic_layer_and_empty_profile():
    ring = Ring(5, 4)
    C = CochainComplex(ring, {0: 2, 1: 2}, {0: eye(ring, 2)})
    backend = BoundedLinearBackend(ring)
    layer = trivial_cohomology_layer(C, 1, backend=backend)
    assert layer.dimension == 0
    assert verify_trivial_cohomology(layer, complex=C).valid
    assert trivial_cohomology(C, backend=backend) == {}


@pytest.mark.parametrize('prime', [2, 3, 5])
def test_relation_coordinates_preserve_actual_action_with_redundant_cycles(prime):
    C = toric_stack(Ring(prime, 3), {1: 1, 2: 1})
    backend = GeneralBackend(C.ring)

    def redundant_kernel(A):
        K = backend.kernel(A)
        return K.hstack(K[:, :1]) if K.ncols else K

    layer = trivial_cohomology_layer(C, 1, backend=backend, kernel=redundant_kernel)
    assert layer.dimension == 1
    for axis, witness in enumerate(layer.triviality_witnesses):
        assert C.d(0) * witness == (C.ring.variable(axis) - 1) * layer.cycles
    assert verify_trivial_cohomology(layer, complex=C).valid


def test_relation_budget_can_use_direct_boundary_shortcut_on_the_same_bounded_backend():
    R = Ring(3, 2)
    C = CochainComplex(R, {0: 2, 1: 2}, {0: eye(R, 2)})
    backend, calls = BoundedLinearBackend(R), []

    def solve_with_relation_limit(A, B):
        calls.append((A, B))
        if len(calls) % 2:
            raise ResourceLimit('relation presentation exceeds this budget')
        return backend.solve(A, B)

    layer = trivial_cohomology_layer(C, 1, backend=backend, solve=solve_with_relation_limit)
    assert len(calls) == 2 * R.D
    assert layer.dimension == 0
    assert verify_trivial_cohomology(layer, complex=C).valid


def test_both_translation_witness_routes_exhausted_remain_unfinished():
    C = koszul(Ring(3, 2))

    def exhausted(A, B):
        raise ResourceLimit('both exact routes are capped')

    with pytest.raises(ResourceLimit, match='both exact routes'):
        trivial_cohomology_layer(C, 1, solve=exhausted)


@pytest.mark.parametrize("field", ["triviality_witnesses", "relation_kernel",
                                   "representatives", "kernel_certificate",
                                   "dimension", "actions"])
def test_layer_tamper(field):
    ring = Ring(3, 2)
    C = koszul(ring)
    layer = trivial_cohomology_layer(C, 1, backend=BoundedLinearBackend(ring))
    data = deepcopy(layer.certificate())
    if field == "dimension":
        data[field] += 1
    elif field == "kernel_certificate":
        data[field]["A"]["rows"] = 1
    elif field in ("triviality_witnesses", "actions"):
        data[field][0]["entries"] = []
    else:
        data[field]["entries"] = []
    assert not verify_trivial_cohomology(data, complex=C).valid


def test_source_binding_and_omitted_kernel_cycles():
    ring = Ring(3, 2)
    C = koszul(ring, copies=2)
    layer = trivial_cohomology_layer(C, 1)
    wrong = koszul(ring, copies=1)
    assert not verify_trivial_cohomology(layer, complex=wrong).valid
    data = layer.certificate()
    data["cycles"]["entries"] = []
    assert not verify_trivial_cohomology(data, complex=C).valid


def test_live_result_mutation_is_checked():
    ring = Ring(5, 2)
    C = koszul(ring)
    layer = trivial_cohomology_layer(C, 1)
    layer.representatives = zeros(ring, 1, 1)
    assert not verify_trivial_cohomology(layer, complex=C).valid


@pytest.mark.parametrize("degree", [True, 1.0, "1"])
def test_degree_requires_an_integer(degree):
    with pytest.raises(ValueError, match="degree must be an integer"):
        trivial_cohomology_layer(koszul(Ring(3, 2)), degree)


@pytest.mark.parametrize("limit", [True, -1, 1.0, "1"])
def test_invalid_dimension_limit_is_bad_input_not_resource_exhaustion(limit):
    C = koszul(Ring(3, 2))
    with pytest.raises(ValueError, match="max_dimension"):
        trivial_cohomology_layer(C, 1, max_dimension=limit)
    with pytest.raises(ValueError, match="max_dimension"):
        trivial_cohomology(CochainComplex(C.ring, {}, {}), max_dimension=limit)


@pytest.mark.parametrize("version", [True, 1.0])
def test_certificate_version_must_be_an_integer(version):
    C = koszul(Ring(3, 2))
    data = trivial_cohomology_layer(C, 1, backend=BoundedLinearBackend(C.ring)).certificate()
    data["version"] = version
    assert not verify_trivial_cohomology(data).valid


def test_certificate_ring_has_exact_fields():
    C = koszul(Ring(3, 2))
    data = trivial_cohomology_layer(C, 1, backend=BoundedLinearBackend(C.ring)).certificate()
    data["ring"]["extra"] = "ignored?"
    assert not verify_trivial_cohomology(data).valid
