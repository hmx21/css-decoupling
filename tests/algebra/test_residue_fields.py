"""Exact finite-field towers and independently checked irreducible factors."""
from random import Random
import subprocess

import pytest

from css_decoupling.algebra.residue_fields import (
    PrimeField, ExtensionField, poly_add, poly_sub, poly_mul, poly_divmod,
    poly_gcd, poly_powmod, poly_derivative, poly_normalize, poly_monic,
    is_irreducible, irreducible_factor,
)
from css_decoupling.errors import ResourceLimit


def finite_fields():
    return [PrimeField(2), PrimeField(5), ExtensionField(PrimeField(2), (1, 1, 1)),
            ExtensionField(PrimeField(2), (1, 1, 0, 1)),
            ExtensionField(PrimeField(3), (1, 0, 1))]


@pytest.mark.parametrize("field", finite_fields(), ids=["F2", "F5", "F4", "F8", "F9"])
def test_exact_arithmetic_inverse_frobenius_and_distributivity(field):
    rng = Random(8743 + field.order)
    for _ in range(15):
        a, b, c = (field.random(rng) for _ in range(3))
        assert field.pow(a, field.order) == a
        assert field.mul(a, field.add(b, c)) == field.add(field.mul(a, b), field.mul(a, c))
        assert field.add(a, field.neg(a)) == field.zero
        if a != field.zero:
            assert field.mul(a, field.inv(a)) == field.one
            assert field.pow(a, -3) == field.inv(field.pow(a, 3))
    with pytest.raises(ZeroDivisionError):
        field.inv(field.zero)


def test_named_small_field_generator_relations():
    f4, f8, f9 = finite_fields()[2:]
    assert f4.pow(f4.generator, 2) == f4.add(f4.generator, 1)
    assert f8.pow(f8.generator, 3) == f8.add(f8.generator, 1)
    assert f9.pow(f9.generator, 2) == f9.coerce(-1)


def test_two_extension_towers_and_explicit_base_embedding():
    f4 = ExtensionField(PrimeField(2), (1, 1, 1))
    f16 = ExtensionField(f4, (f4.generator, f4.one, f4.one))
    b = f16.generator
    assert f16.order == 16 and f16.absolute_degree == 4
    assert f16.add(f16.pow(b, 2), b) == f16.from_base(f4.generator)
    assert f16.from_base(f4.generator) != f16.generator
    f9 = ExtensionField(PrimeField(3), (1, 0, 1))
    c = f9.add(1, f9.generator)
    f81 = ExtensionField(f9, (f9.neg(c), f9.zero, f9.one))
    assert f81.order == 81 and f81.absolute_degree == 4
    assert f81.pow(f81.generator, 2) == f81.from_base(c)
    for field in (f16, f81):
        rng = Random(field.order)
        for _ in range(8):
            a = field.random(rng)
            assert field.pow(a, field.order) == a
            if a != field.zero:
                assert field.mul(a, field.inv(a)) == field.one


@pytest.mark.parametrize("field", finite_fields()[2:], ids=["F4", "F8", "F9"])
def test_polynomial_division_gcd_and_modular_power_over_extensions(field):
    a = field.generator
    g = (a, field.one, field.one)
    q = (field.one, a, field.one, a)
    remainder = (field.add(a, 1),)
    f = poly_add(field, poly_mul(field, g, q), remainder)
    quotient, recovered = poly_divmod(field, f, g)
    assert quotient == q and recovered == remainder
    assert poly_gcd(field, poly_mul(field, g, q), g) == poly_monic(field, g)
    x = (field.zero, field.one)
    h = (field.one, a, field.one)
    successive = (field.one,)
    for _ in range(13):
        successive = poly_divmod(field, poly_mul(field, successive, x), h)[1]
    assert poly_powmod(field, x, 13, h) == successive


def checked_factor(field, polynomial, **kwargs):
    factor = irreducible_factor(field, polynomial, **kwargs)
    assert factor[-1] == field.one
    assert is_irreducible(field, factor)
    assert not poly_divmod(field, polynomial, factor)[1]
    return factor


def test_no_prime_field_rational_points_and_odd_equal_degree_splitting():
    f3 = PrimeField(3)
    first, second = (2, 1, 1), (2, 2, 1)
    polynomial = poly_mul(f3, first, second)
    assert polynomial == (1, 0, 0, 0, 1)
    assert checked_factor(f3, polynomial, seed=17) in (first, second)
    assert irreducible_factor(f3, polynomial, seed=17) == irreducible_factor(f3, polynomial, seed=17)


def test_characteristic_two_absolute_trace_equal_degree_split():
    f2 = PrimeField(2)
    first, second = (1, 1, 0, 1), (1, 0, 1, 1)
    assert checked_factor(f2, poly_mul(f2, first, second), seed=19) in (first, second)


def test_nonprime_characteristic_two_base_needs_full_absolute_trace():
    f4 = ExtensionField(PrimeField(2), (1, 1, 1))
    a = f4.generator
    first, second = (a, f4.one, f4.one), (f4.add(a, 1), f4.one, f4.one)
    assert is_irreducible(f4, first) and is_irreducible(f4, second)
    assert checked_factor(f4, poly_mul(f4, first, second), seed=41) in (first, second)


def test_odd_extension_field_equal_degree_split():
    f9 = ExtensionField(PrimeField(3), (1, 0, 1))
    a = f9.add(f9.generator, 1)
    first, second = (f9.neg(a), f9.zero, f9.one), (a, f9.zero, f9.one)
    assert is_irreducible(f9, first) and is_irreducible(f9, second)
    assert checked_factor(f9, poly_mul(f9, first, second), seed=5) in (first, second)


def test_factor_over_a_tower_not_just_a_flat_extension():
    f4 = ExtensionField(PrimeField(2), (1, 1, 1))
    f16 = ExtensionField(f4, (f4.generator, f4.one, f4.one))
    roots = [f16.generator, f16.from_base(f4.generator), f16.one]
    factors = [(f16.neg(root), f16.one) for root in roots]
    product = (f16.one,)
    for f in factors:
        product = poly_mul(f16, product, f)
    assert checked_factor(f16, product, seed=8) in factors


def test_inseparable_frobenius_root_uses_inverse_coefficient_frobenius():
    f4 = ExtensionField(PrimeField(2), (1, 1, 1))
    irreducible = (f4.generator, f4.one, f4.one)
    polynomial = poly_mul(f4, irreducible, irreducible)
    assert not poly_derivative(f4, polynomial)
    assert checked_factor(f4, polynomial, max_trials=0) == irreducible
    # Repeated Frobenius roots in a tower, not just over the prime field.
    f16 = ExtensionField(f4, irreducible)
    linear = (f16.generator, f16.one)
    fourth = poly_mul(f16, poly_mul(f16, linear, linear), poly_mul(f16, linear, linear))
    assert checked_factor(f16, fourth, max_trials=0) == linear


def test_squarefree_divisor_with_mixed_multiplicities():
    f3 = PrimeField(3)
    a, b = (1, 0, 1), (1, 1)
    polynomial = poly_mul(f3, poly_mul(f3, poly_mul(f3, a, a), a), b)
    # The derivative removes the entire multiplicity-three factor; selecting
    # the remaining simple factor is still a valid factor of the input.
    assert checked_factor(f3, polynomial) == b


def test_distinct_degree_selection_and_prime_above_machine_word_backend_bound():
    field = PrimeField(2147483659)
    linear = (2, 1)
    quadratic = (1, 0, 1)
    assert checked_factor(field, poly_mul(field, linear, quadratic), max_trials=0) == linear


def test_trial_cap_does_not_become_a_false_irreducibility_answer(monkeypatch):
    field = PrimeField(2)
    polynomial = (0, 1, 1)
    with pytest.raises(ResourceLimit, match="max_trials=0"):
        irreducible_factor(field, polynomial, max_trials=0)
    calls = []
    def always_zero(self, rng):
        calls.append(1)
        return self.zero
    monkeypatch.setattr(PrimeField, "random", always_zero)
    with pytest.raises(ResourceLimit, match="max_trials=3"):
        irreducible_factor(field, polynomial, max_trials=3)
    assert len(calls) == 6
    assert checked_factor(field, (1, 1, 1), max_trials=0) == (1, 1, 1)


def test_rabin_rejects_reducible_and_inseparable_moduli():
    field = PrimeField(2)
    assert not is_irreducible(field, ())
    assert not is_irreducible(field, (1,))
    assert not is_irreducible(field, (1, 0, 1))
    assert not is_irreducible(field, poly_mul(field, (1, 1, 1), (1, 1, 0, 1)))
    with pytest.raises(ValueError, match="reducible"):
        ExtensionField(field, (1, 0, 1))
    with pytest.raises(ValueError, match="monic"):
        ExtensionField(PrimeField(3), (1, 0, 2))


@pytest.mark.parametrize("fake", [(1, 0, 0, 0, 1), (0, 1)])
def test_returned_splitter_candidate_is_rechecked_not_trusted(monkeypatch, fake):
    import css_decoupling.algebra.residue_fields as implementation
    # First fake is reducible; second is irreducible but does not divide f.
    monkeypatch.setattr(implementation, "_equal_degree_factor", lambda *args: fake)
    with pytest.raises(ArithmeticError, match="Rabin/divisibility"):
        irreducible_factor(PrimeField(3), (1, 0, 0, 0, 1))


@pytest.mark.parametrize("polynomial", [(), (0,), (1,)])
def test_factor_requires_nonconstant_input(polynomial):
    with pytest.raises(ValueError, match="nonconstant"):
        irreducible_factor(PrimeField(2), polynomial)


def test_no_external_process_or_field_enumeration_required(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Finite residue-field primitive invoked a process")
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    field = ExtensionField(PrimeField(3), (1, 0, 1))
    polynomial = poly_mul(field, (field.generator, field.one), (field.one, field.one))
    checked_factor(field, polynomial)
