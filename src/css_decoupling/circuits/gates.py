"""Translation invariant SUM/SWAP circuits over any prime Laurent ring.

Column vectors encode Pauli representatives.  ``SUM(i,j,w,alpha)`` has
control species i in cell r+alpha and target species j in cell r, and acts
on computational states as |a,b> -> |a,b+w*a>.  Consequently its Z matrix
is I-w*x**alpha*e_i*e_j^T; its X matrix is the inverse dagger of that matrix.
A gate here denotes the whole translation invariant layer, not one cell.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from css_decoupling.algebra.laurent import Matrix, Poly, Ring, eye


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


@dataclass(frozen=True)
class SUM:
    control: int
    target: int
    weight: int
    displacement: tuple[int, ...]

    def __post_init__(self) -> None:
        _integer(self.control, "control")
        _integer(self.target, "target")
        _integer(self.weight, "weight")
        object.__setattr__(self, "displacement", tuple(self.displacement))
        for value in self.displacement:
            _integer(value, "displacement coordinate")
        if self.control < 0 or self.target < 0 or self.control == self.target:
            raise ValueError("SUM requires distinct nonnegative species")

    def inverse(self) -> SUM:
        return SUM(self.control, self.target, -self.weight, self.displacement)

    def matrix(self, ring: Ring, qudits: int, action: str = "Z") -> Matrix:
        if len(self.displacement) != ring.D:
            raise ValueError("SUM displacement dimension does not match ring")
        if max(self.control, self.target) >= qudits:
            raise ValueError("SUM species is outside circuit")
        if action == "Z":
            entry = (self.control, self.target)
            value = ring.monomial(self.displacement, -self.weight)
        elif action == "X":
            entry = (self.target, self.control)
            value = ring.monomial(tuple(-a for a in self.displacement), self.weight)
        else:
            raise ValueError("Pauli action must be 'Z' or 'X'")
        return eye(ring, qudits) + Matrix(ring, qudits, qudits, {entry: value})


@dataclass(frozen=True)
class SWAP:
    left: int
    right: int

    def __post_init__(self) -> None:
        _integer(self.left, "left")
        _integer(self.right, "right")
        if self.left < 0 or self.right < 0 or self.left == self.right:
            raise ValueError("SWAP requires distinct nonnegative species")

    def inverse(self) -> SWAP:
        return self

    def matrix(self, ring: Ring, qudits: int, action: str = "Z") -> Matrix:
        if action not in ("Z", "X"):
            raise ValueError("Pauli action must be 'Z' or 'X'")
        if max(self.left, self.right) >= qudits:
            raise ValueError("SWAP species is outside circuit")
        entries = {(i, i): ring.one() for i in range(qudits)
                   if i not in (self.left, self.right)}
        entries[self.left, self.right] = ring.one()
        entries[self.right, self.left] = ring.one()
        return Matrix(ring, qudits, qudits, entries)


@dataclass(frozen=True)
class Circuit:
    ring: Ring
    qudits: int
    gates: tuple[SUM | SWAP, ...]

    def __post_init__(self) -> None:
        _integer(self.qudits, "qudits")
        if self.qudits < 0:
            raise ValueError("qudits must be nonnegative")
        object.__setattr__(self, "gates", tuple(self.gates))
        for gate in self.gates:
            if isinstance(gate, SUM):
                if len(gate.displacement) != self.ring.D:
                    raise ValueError("SUM displacement dimension does not match ring")
                if max(gate.control, gate.target) >= self.qudits:
                    raise ValueError("SUM species is outside circuit")
            elif isinstance(gate, SWAP):
                if max(gate.left, gate.right) >= self.qudits:
                    raise ValueError("SWAP species is outside circuit")
            else:
                raise TypeError("Circuit accepts only SUM and SWAP gates")

    def matrix(self, action: Literal["Z", "X"] = "Z") -> Matrix:
        """Replay the actual gate sequence, grouping independent row updates.

        In a consecutive SUM batch whose target rows and source rows are
        disjoint, all cross-products of elementary updates vanish. Its exact
        product is therefore I+N, with N the sum of every recorded weighted
        monomial. A conflicting SUM or any SWAP ends the batch. Large
        products use sparse polynomial matrix convolution; small batches
        retain direct mutable-support row updates. No circuit template or
        saved product is assumed, including for inverse and X-action replay.
        """
        if action not in ("Z", "X"):
            raise ValueError("Pauli action must be 'Z' or 'X'")
        p = self.ring.p
        zero_exponent = (0,) * self.ring.D
        rows = [{i: {zero_exponent: 1}} for i in range(self.qudits)]
        batch, targets, sources = {}, set(), set()

        def add_terms(target, column, terms):
            accumulated = rows[target].setdefault(column, {})
            for exponent, coefficient in terms.items():
                value = (accumulated.get(exponent, 0) + coefficient) % p
                if value:
                    accumulated[exponent] = value
                else:
                    accumulated.pop(exponent, None)
            if not accumulated:
                rows[target].pop(column, None)

        def flush():
            if batch:
                source_work = {source: sum(len(f) for f in rows[source].values())
                               for source in sources}
                products = sum(len(f) * source_work[source]
                               for (_, source), f in batch.items())
                input_terms = sum(map(len, batch.values())) + sum(source_work.values())
                if products >= max(4096, 4 * input_terms):
                    # Compact only the participating species. In particular
                    # neither a full identity matrix nor untouched rows are
                    # multiplied for a batch of local transvections.
                    target_ids, source_ids = sorted(targets), sorted(sources)
                    target_index = {value: i for i, value in enumerate(target_ids)}
                    source_index = {value: i for i, value in enumerate(source_ids)}
                    operator = Matrix(self.ring, len(target_ids), len(source_ids), {
                        (target_index[target], source_index[source]): Poly(self.ring, terms)
                        for (target, source), terms in batch.items()})
                    source_matrix = Matrix(self.ring, len(source_ids), self.qudits, {
                        (i, column): Poly(self.ring, terms)
                        for i, source in enumerate(source_ids)
                        for column, terms in rows[source].items()})
                    delta = operator * source_matrix
                    for (target, column), polynomial in delta.entries.items():
                        add_terms(target_ids[target], column, polynomial.terms)
                else:
                    for (target, source), coefficients in batch.items():
                        for column, terms in rows[source].items():
                            accumulated = rows[target].setdefault(column, {})
                            for shift, coefficient in coefficients.items():
                                for exponent, value in terms.items():
                                    translated = tuple(a + b for a, b in zip(exponent, shift))
                                    value = (accumulated.get(translated, 0) + coefficient * value) % p
                                    if value:
                                        accumulated[translated] = value
                                    else:
                                        accumulated.pop(translated, None)
                            if not accumulated:
                                rows[target].pop(column, None)
            batch.clear()
            targets.clear()
            sources.clear()

        for gate in self.gates:
            if isinstance(gate, SWAP):
                flush()
                rows[gate.left], rows[gate.right] = rows[gate.right], rows[gate.left]
                continue
            if action == "Z":
                target, source = gate.control, gate.target
                shift, coefficient = gate.displacement, (-gate.weight) % p
            else:
                target, source = gate.target, gate.control
                shift, coefficient = tuple(-a for a in gate.displacement), gate.weight % p
            if not coefficient:
                continue
            if target in sources or source in targets:
                flush()
            targets.add(target)
            sources.add(source)
            coefficients = batch.setdefault((target, source), {})
            value = (coefficients.get(shift, 0) + coefficient) % p
            if value:
                coefficients[shift] = value
            else:
                coefficients.pop(shift, None)
            if not coefficients:
                batch.pop((target, source), None)
        flush()
        return Matrix(self.ring, self.qudits, self.qudits,
                      {(i, j): value for i, row in enumerate(rows) for j, value in row.items()})

    def inverse(self) -> Circuit:
        return Circuit(self.ring, self.qudits,
                       tuple(g.inverse() for g in reversed(self.gates)))

    @property
    def layer_count(self) -> int:
        return len(self.gates)

    @property
    def interaction_radius(self) -> int:
        """Maximum L1 displacement in the current coarse lattice coordinates."""
        return max((sum(abs(a) for a in g.displacement)
                    for g in self.gates if isinstance(g, SUM)), default=0)


def _block_swap(a: int, b: int) -> list[SWAP]:
    """Realize [A,B] -> [B,A] in O(a+b) bookkeeping and at most a+b swaps."""
    current = list(range(a + b))
    positions = list(range(a + b))
    desired = list(range(a, a + b)) + list(range(a))
    gates = []
    for row, value in enumerate(desired):
        other = positions[value]
        if row != other:
            old = current[row]
            current[row], current[other] = value, old
            positions[value], positions[old] = row, other
            gates.append(SWAP(row, other))
    return gates


def compile_theta(f: Matrix, g: Matrix) -> Circuit:
    """Compile Theta(f,g) directly, without matrix elimination or gate search.

    f:A->B and g:B->A may be rectangular and need not be invertible.
    Execution order is the block permutation, lower block -g, upper block f.
    There is exactly one SUM layer per nonzero monomial of f and g.
    """
    if f.ring != g.ring or (f.nrows, f.ncols) != (g.ncols, g.nrows):
        raise ValueError("Theta factors must have opposite shapes over one ring")
    a, b = f.ncols, f.nrows
    gates: list[SUM | SWAP] = list(_block_swap(a, b))
    for (i, j), polynomial in sorted(g.entries.items()):
        for exponent, coefficient in sorted(polynomial.terms.items()):
            gates.append(SUM(b + i, j, coefficient % f.ring.p, exponent))
    for (i, j), polynomial in sorted(f.entries.items()):
        for exponent, coefficient in sorted(polynomial.terms.items()):
            gates.append(SUM(i, b + j, (-coefficient) % f.ring.p, exponent))
    return Circuit(f.ring, a + b, tuple(gates))
