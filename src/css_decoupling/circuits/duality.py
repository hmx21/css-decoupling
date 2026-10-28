"""Fourier conjugation of translation-invariant gate families."""
from .gates import Circuit, SUM, SWAP


def dualize_circuit(circuit: Circuit) -> Circuit:
    """Apply U_Z -> U_Z^(-dagger), including SUM sign and displacement reversal."""
    gates = []
    for gate in circuit.gates:
        if isinstance(gate, SUM):
            gates.append(SUM(gate.target, gate.control, (-gate.weight) % circuit.ring.p,
                             tuple(-a for a in gate.displacement)))
        else:
            gates.append(SWAP(gate.left, gate.right))
    return Circuit(circuit.ring, circuit.qudits, tuple(gates))
