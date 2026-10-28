"""Closed points use residue extensions, not enumeration of rational points."""
import pytest

from css_decoupling.algebra.closed_points import closed_point, monic_shear, apply_shear
from css_decoupling.algebra.laurent import Matrix, Ring
from css_decoupling.algebra.residue_fields import ExtensionField, is_irreducible
from css_decoupling.errors import ResourceLimit


def certify(point, generators):
    assert all(point.evaluate(f) == point.field.zero for f in generators)
    field = point.field
    while isinstance(field, ExtensionField):
        assert is_irreducible(field.base, field.modulus)
        field = field.base


def test_no_rational_point_uses_irreducible_residue_field():
    ring = Ring(2, 1)
    x, = ring.variables
    generators = (x*x + x + 1,)
    point = closed_point(ring, generators)
    certify(point, generators)
    assert point.field.order == 4
    assert point.coordinates[0] not in (point.field.zero, point.field.one)


def test_positive_dimensional_ideal_may_require_nonprime_residue():
    ring = Ring(2, 3)
    x, y, z = ring.variables
    generators = (x*x + x + 1,)
    point = closed_point(ring, generators)
    certify(point, generators)
    assert point.field.absolute_degree % 2 == 0


@pytest.mark.parametrize("p", [2, 3, 5])
def test_noncoordinate_shear_point_and_inverse_maps(p):
    ring = Ring(p, 2)
    x, y = ring.variables
    generators = (x*y - 1,)
    weights, scalar = monic_shear(generators[0], (0, 1))
    assert weights and scalar != 0
    transformed = apply_shear(generators[0], 1, weights)
    assert apply_shear(transformed, 1, weights, sign=-1) == generators[0]
    point = closed_point(ring, generators)
    certify(point, generators)
    assert point.coordinates[0] != point.field.zero
    assert point.coordinates[1] != point.field.zero


def test_two_level_tower_and_field_embeddings():
    ring = Ring(2, 2)
    x, y = ring.variables
    generators = (x*x + x + 1, y*y + y + x)
    point = closed_point(ring, generators)
    certify(point, generators)
    assert isinstance(point.field, ExtensionField)
    assert isinstance(point.field.base, ExtensionField)
    assert point.field.absolute_degree == 4
    assert point.field.base.absolute_degree == 2
    assert point.field.pow(point.coordinates[0], 4) == point.coordinates[0]
    assert point.field.pow(point.coordinates[1], 4) != point.coordinates[1]


def test_odd_characteristic_tower_with_a_nonmonic_input_generator():
    ring = Ring(3, 2)
    x, y = ring.variables
    generators = (2*(x*x + 1), y*y - x - 1)
    point = closed_point(ring, generators)
    certify(point, generators)
    assert point.field.absolute_degree == 4


def test_unit_ideal_rejected_even_without_a_constant_input():
    ring = Ring(5, 2)
    x, y = ring.variables
    for generators in ((ring.one(),), (x, 1-x), (x*y-1, x)):
        with pytest.raises(ValueError, match="unit ideal"):
            closed_point(ring, generators)


def test_selected_axes_inactive_coordinates_and_empty_ideal():
    ring = Ring(2, 4)
    x, y, z, w = ring.variables
    generators = (z*z + z + 1,)
    point = closed_point(ring, generators, axes=(2,))
    certify(point, generators)
    assert all(point.coordinates[i] == point.field.zero for i in (0, 1, 3))
    assert point.field.order == 4
    for axes in (None, (), (2, 0)):
        zero = closed_point(ring, (), axes=axes)
        assert zero.coordinates == (0, 0, 0, 0)
        assert zero.field.order == 2
    with pytest.raises(ValueError, match="inactive"):
        closed_point(ring, (z+x,), axes=(2,))


def test_reordered_nonconsecutive_axes_keep_the_original_coordinates():
    ring = Ring(3, 4)
    x, y, z, w = ring.variables
    generators = (w*y - 1,)
    point = closed_point(ring, generators, axes=(3, 1))
    certify(point, generators)
    assert point.coordinates[0] == point.field.zero
    assert point.coordinates[2] == point.field.zero
    assert point.field.mul(point.coordinates[3], point.coordinates[1]) == point.field.one


def test_monic_shear_scalar_leading_and_matrix_roundtrip():
    ring = Ring(5, 3)
    x, y, z = ring.variables
    f = x*y + x*x + z + 2
    weights, scalar = monic_shear(f, (2, 0, 1))
    g = apply_shear(f, 1, weights)
    degree = max(e[1] for e in g.terms)
    leading = [(e, c) for e, c in g.terms.items() if e[1] == degree]
    assert len(leading) == 1
    assert leading[0] == ((0, degree, 0), scalar)
    matrix = Matrix.from_rows(ring, [[f, x+y], [z, 1]])
    image = apply_shear(matrix, 1, weights)
    assert apply_shear(image, 1, weights, sign=-1) == matrix
    assert monic_shear(2*y*y+x, (0, 1)) == ({}, 2)


def test_caps_remain_resource_failures_not_empty_varieties():
    ring = Ring(3, 1)
    x, = ring.variables
    # x^4+1 splits into two irreducible quadratics over F_3.
    with pytest.raises(ResourceLimit, match="max_trials=0"):
        closed_point(ring, (x**4+1,), max_trials=0)
    with pytest.raises(ResourceLimit):
        closed_point(ring, (x**4+1,), max_terms=2)
    other = Ring(5, 2)
    y, z = other.variables
    with pytest.raises(ResourceLimit):
        apply_shear(y**4, 1, {0: 2}, max_terms=2)


def test_invalid_domains_axes_and_shear_parameters():
    ring = Ring(3, 2)
    x, y = ring.variables
    with pytest.raises(ValueError, match="ordinary"):
        closed_point(ring, (x**-1,))
    for axes in ((0, 0), (3,)):
        with pytest.raises(ValueError, match="axes"):
            closed_point(ring, (), axes=axes)
    with pytest.raises(TypeError, match="integer"):
        closed_point(ring, (), axes=(True,))
    with pytest.raises(ValueError, match="zero polynomial"):
        monic_shear(ring.zero())
    with pytest.raises(ValueError, match="remain fixed"):
        apply_shear(x, 0, {0: 1})
