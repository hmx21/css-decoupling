"""Exact kernel tests, including bounded independent finite-action oracles."""
from copy import deepcopy
from itertools import product
from types import SimpleNamespace

import pytest
from sympy import Matrix as ZMatrix

from css_decoupling.algebra.laurent import Ring, Matrix, eye
from css_decoupling.lattice.retained import matrix_power, trivializing_lattice
from css_decoupling.lattice.kernel import translation_kernel
from css_decoupling.certificates.translation_kernel import verify_translation_kernel
from css_decoupling.errors import ResourceLimit


def constant(ring, rows):
    return Matrix(ring, len(rows), len(rows), {(i, j): value for i, row in enumerate(rows)
                                            for j, value in enumerate(row) if value})


def layer(*actions):
    return SimpleNamespace(actions=actions)


def agrees_with_small_period_box(layers, lattice, periods):
    """Test-only exhaustive oracle, bounded to <= 400 exponent tuples."""
    assert __import__("math").prod(periods) <= 400
    image = set()
    for exponents in product(*(range(period) for period in periods)):
        products, is_identity = [], True
        for current in layers.values():
            value = eye(current.actions[0].ring, current.actions[0].nrows)
            for a, exponent in zip(current.actions, exponents):
                value = value * matrix_power(a, exponent)
            is_identity &= value == eye(value.ring, value.nrows)
            products.append(tuple((position, tuple(sorted(poly.terms.items())))
                                  for position, poly in sorted(value.entries.items())))
        coordinates = lattice.B.inv() * ZMatrix(exponents)
        assert is_identity == all(x.q == 1 for x in coordinates)
        image.add(tuple(products))
    assert len(image) == lattice.index


def test_mixed_semisimple_relation_reduces_rectangular_index():
    r = Ring(5, 2)
    a = constant(r, [[2]])
    layers = {1: layer(a, a)}
    result = translation_kernel(layers, 2, max_index=4)
    assert result.index == 4
    assert trivializing_lattice(layers, 2).index == 16
    assert result.basis == ((4, 3), (0, 1))
    agrees_with_small_period_box(layers, result, [4, 4])


def test_extension_field_with_distinct_generator_powers():
    r = Ring(2, 3)
    a = constant(r, [[0, 1], [1, 1]])  # primitive F_4 element
    layers = {1: layer(a, a * a, eye(r, 2))}
    result = translation_kernel(layers, 3)
    assert result.index == 3
    assert result.kernel_statistics["field_degrees"] == [2]
    agrees_with_small_period_box(layers, result, [3, 3, 1])


def test_noncyclic_semisimple_group_and_primitive_idempotents():
    r = Ring(5, 2)
    a, b = constant(r, [[2, 0], [0, 1]]), constant(r, [[1, 0], [0, 2]])
    result = translation_kernel({1: layer(a, b)}, 2)
    assert result.index == 16
    assert result.kernel_statistics["field_degrees"] == [1, 1]
    agrees_with_small_period_box({1: layer(a, b)}, result, [4, 4])


def test_two_extension_components_with_same_individual_minimal_polynomial():
    r = Ring(2, 2)
    a = constant(r, [[0, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 1]])
    b = constant(r, [[0, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 0]])
    layers = {1: layer(a, b)}
    result = translation_kernel(layers, 2)
    assert result.index == 9
    assert sorted(result.kernel_statistics["field_degrees"]) == [2, 2]
    agrees_with_small_period_box(layers, result, [3, 3])


@pytest.mark.parametrize("p", [2, 3, 5])
def test_unipotent_mixed_relation_odd_and_even_characteristic(p):
    r = Ring(p, 2)
    a = constant(r, [[1, 1], [0, 1]])
    layers = {1: layer(a, a)}
    result = translation_kernel(layers, 2)
    assert result.index == p
    agrees_with_small_period_box(layers, result, [p, p])


def test_noncyclic_unipotent_group_and_deeper_filtration():
    r = Ring(3, 2)
    a, b = constant(r, [[1, 1, 0], [0, 1, 1], [0, 0, 1]]), constant(r, [[1, 0, 1], [0, 1, 0], [0, 0, 1]])
    layers = {1: layer(a, b)}
    result = translation_kernel(layers, 2)
    assert result.index == 9
    assert result.kernel_statistics["unipotent_filtration_steps"] == [2]
    agrees_with_small_period_box(layers, result, [3, 3])


def test_unipotent_order_larger_than_characteristic():
    r = Ring(2, 2)
    a = constant(r, [[1, 1, 0], [0, 1, 1], [0, 0, 1]])
    layers = {1: layer(a, a * a)}
    result = translation_kernel(layers, 2)
    assert result.index == 4
    agrees_with_small_period_box(layers, result, [4, 2])


def test_mixed_jordan_parts_and_joint_kernel_across_layers():
    r = Ring(3, 2)
    a = constant(r, [[2, 2], [0, 2]])
    b = constant(r, [[2]])
    layers = {1: layer(a, a), 2: layer(b, eye(r, 1))}
    result = translation_kernel(layers, 2)
    assert result.index == 12
    agrees_with_small_period_box(layers, result, [6, 6])


def test_nontriangular_conjugation_does_not_change_kernel():
    r = Ring(3, 3)
    # P is its own inverse in characteristic 3 for this chosen involution.
    P = constant(r, [[1, 1, 0], [0, 2, 0], [1, 2, 2]])
    assert P * P == eye(r, 3)
    a = constant(r, [[1, 1, 0], [0, 1, 1], [0, 0, 1]])
    b = constant(r, [[1, 0, 1], [0, 1, 0], [0, 0, 1]])
    layers = {1: layer(P * a * P, P * b * P, P * a * b * P)}
    result = translation_kernel(layers, 3)
    assert result.index == 9
    agrees_with_small_period_box(layers, result, [3, 3, 3])


def test_prime_power_pohlig_hellman_digits():
    r = Ring(97, 2)
    a = constant(r, [[5]])  # order 96, so several 2-adic digits
    result = translation_kernel({1: layer(a, matrix_power(a, 17))}, 2)
    assert result.index == 96
    assert result.kernel_statistics["max_baby_steps"] <= 2
    assert result.basis == ((96, 79), (0, 1))


def test_extension_semisimple_with_nonzero_nilpotent_part():
    r = Ring(2, 2)
    # Repeated irreducible F_4 block, coupled by a nilpotent off-diagonal.
    a = constant(r, [[0, 1, 1, 0], [1, 1, 0, 1], [0, 0, 0, 1], [0, 0, 1, 1]])
    layers = {1: layer(a, matrix_power(a, 5))}
    result = translation_kernel(layers, 2)
    assert result.index == 6
    assert result.kernel_statistics["field_degrees"] == [2]
    agrees_with_small_period_box(layers, result, [6, 6])


def test_prime_order_dlog_and_nonprime_field():
    r = Ring(2, 2)
    a = constant(r, [[0, 0, 1], [1, 0, 1], [0, 1, 0]])  # t^3+t+1
    layers = {1: layer(a, matrix_power(a, 3))}
    result = translation_kernel(layers, 2)
    assert result.index == 7
    assert result.kernel_statistics["field_degrees"] == [3]
    agrees_with_small_period_box(layers, result, [7, 7])


def test_joint_large_rectangular_periods_do_not_hit_index_cap():
    r = Ring(3, 12)
    a = constant(r, [[2]])
    result = translation_kernel({1: layer(*([a] * 12))}, 12, max_index=2)
    assert result.index == 2


def test_empty_profile():
    result = translation_kernel({}, 4)
    assert result.index == 1
    assert verify_translation_kernel({}, 4, result.kernel_certificate)


@pytest.mark.parametrize("damage", ["contained", "index", "omit_layer", "order", "logs", "projector", "boolean"])
def test_certificate_rejects_incomplete_kernel_and_tampering(damage):
    r = Ring(5, 2)
    a = constant(r, [[2]])
    layers = {1: layer(a, a)}
    certificate = deepcopy(translation_kernel(layers, 2).kernel_certificate)
    if damage == "contained":
        certificate["basis_columns"] = [[4, 0], [0, 4]]  # all columns act trivially, but not full kernel
        certificate["index"] = 16
    elif damage == "index":
        certificate["index"] = 3
    elif damage == "omit_layer":
        certificate["layers"] = []
    elif damage == "order":
        certificate["layers"][0]["components"][0]["order"] = 2
    elif damage == "logs":
        certificate["layers"][0]["components"][0]["logs"][0] = 0
    elif damage == "projector":
        certificate["layers"][0]["components"][0]["projector"] = []
    else:
        certificate["basis_columns"][1][1] = True
    with pytest.raises(ValueError):
        verify_translation_kernel(layers, 2, certificate)


def test_dlog_resource_limit_is_not_a_smaller_lattice():
    r = Ring(47, 2)
    a = constant(r, [[2]])
    with pytest.raises(ResourceLimit, match="Discrete logarithm"):
        translation_kernel({1: layer(a, a)}, 2, max_dlog_steps=2)


def test_final_index_and_matrix_resource_limits():
    r = Ring(5, 2)
    a = constant(r, [[2]])
    with pytest.raises(ResourceLimit, match="index"):
        translation_kernel({1: layer(a, a)}, 2, max_index=3)
    with pytest.raises(ResourceLimit, match="matrices"):
        translation_kernel({1: layer(eye(r, 3), eye(r, 3))}, 2, max_linear_entries=2)


def test_invalid_actions_are_rejected():
    r = Ring(3, 2)
    a = constant(r, [[1, 1], [0, 1]])
    b = constant(r, [[1, 0], [1, 1]])
    with pytest.raises(ValueError, match="commute"):
        translation_kernel({1: layer(a, b)}, 2)
    with pytest.raises(ValueError, match="singular"):
        translation_kernel({1: layer(constant(r, [[0]]), constant(r, [[1]]))}, 2)


def test_large_order_prime_uses_exact_primality_witness():
    from css_decoupling.lattice.kernel import _prime_certificate
    from css_decoupling.certificates.translation_kernel import _verify_prime
    q = 18446744073709551629
    proof = _prime_certificate(q, factor_limit=10000, cache={})
    _verify_prime(q, proof)
    with pytest.raises(ValueError, match="primality"):
        _verify_prime(q, None)
    proof["base"] = 1
    with pytest.raises(ValueError, match="witness"):
        _verify_prime(q, proof)
