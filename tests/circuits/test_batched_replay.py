"""Independent elementary products for generic commuting-batch replay."""
from random import Random
from unittest.mock import patch

import pytest

from css_decoupling.algebra.laurent import Matrix, Ring, eye
from css_decoupling.circuits.gates import Circuit, SUM, SWAP


def _elementary_product(circuit, action):
    result = eye(circuit.ring, circuit.qudits)
    for gate in circuit.gates:
        result = gate.matrix(circuit.ring, circuit.qudits, action) * result
    return result


@pytest.mark.parametrize("prime", [2, 3, 5])
@pytest.mark.parametrize("action", ["Z", "X"])
def test_large_commuting_batches_match_individual_gates_and_use_convolution(prime, action):
    R = Ring(prime, 3)
    gates = [SWAP(0, 3), SWAP(1, 2)]
    for target, source, length in ((0, 1, 72), (2, 3, 72), (1, 0, 80), (3, 2, 80)):
        gates.extend(SUM(target, source, 1 + k % (prime - 1), (k - 31, -k, k % 3))
                     for k in range(length))
    # End one batch by a SWAP, then force both kinds of dependency conflict.
    gates.extend((SWAP(0, 2), SUM(2, 1, 2, (-1, 2, 0)),
                  SUM(1, 3, 1, (2, 0, -1)), SUM(0, 1, 1, (0, -1, 1))))
    circuit = Circuit(R, 4, tuple(gates))
    original = Matrix.__mul__
    products = []

    def observe(left, right):
        if isinstance(right, Matrix):
            products.append((left.shape, right.shape))
        return original(left, right)

    with patch.object(Matrix, "__mul__", observe):
        actual = circuit.matrix(action)
    assert products  # This input must exercise the polynomial batch path.
    assert actual == _elementary_product(circuit, action)
    assert circuit.inverse().matrix(action) == _elementary_product(circuit.inverse(), action)
    assert circuit.inverse().matrix(action) * actual == eye(R, circuit.qudits)


@pytest.mark.parametrize("prime", [2, 3, 5])
@pytest.mark.parametrize("seed", range(4))
def test_interacting_random_gates_preserve_order_and_both_pauli_actions(prime, seed):
    R, rng = Ring(prime, 2), Random(51013 + seed)
    gates = []
    for index in range(24):
        first, second = rng.sample(range(4), 2)
        if index % 7 == 0:
            gates.append(SWAP(first, second))
        else:
            gates.append(SUM(first, second, rng.randrange(-prime, prime + 1),
                             (rng.randrange(-1, 2), rng.randrange(-1, 2))))
    circuit = Circuit(R, 4, tuple(gates))
    for action in ("Z", "X"):
        assert circuit.matrix(action) == _elementary_product(circuit, action)


def test_cancelling_terms_and_swap_do_not_alias_source_rows():
    R = Ring(5, 2)
    first = tuple(SUM(0, 1, k % 4 + 1, (k - 35, -k)) for k in range(80))
    middle = (SWAP(0, 2), SUM(1, 0, 3, (-1, 2)), SWAP(1, 2))
    circuit = Circuit(R, 3, first + middle)
    reverse = circuit.inverse()
    doubled = Circuit(R, 3, circuit.gates + reverse.gates)
    for action in ("Z", "X"):
        assert circuit.matrix(action) == _elementary_product(circuit, action)
        assert doubled.matrix(action) == eye(R, 3)


def test_coefficient_cancellation_does_not_skip_a_dependency_barrier():
    R = Ring(3, 2)
    # A dependency separates two equal-and-opposite updates. Merging them
    # across that barrier would incorrectly return the intervening gate.
    gates = (SUM(0, 1, 1, (1, 0)), SUM(1, 2, 1, (0, 1)),
             SUM(0, 1, -1, (1, 0)))
    circuit = Circuit(R, 3, gates)
    for action in ("Z", "X"):
        assert circuit.matrix(action) == _elementary_product(circuit, action)
        assert circuit.matrix(action) != Circuit(R, 3, (gates[1],)).matrix(action)
