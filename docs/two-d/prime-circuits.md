# Prime-field stable circuits and exact transport

This page describes the retained advanced stable/transport APIs. The unified
default circuit request uses the [same-register code-circuit contract](no-input-ancilla.md),
which separately records target-code equivalence and does not add input ancillas.

The circuit modules accept `F_p[x^+-1,y^+-1]` for every prime `p` supported by
the exact Laurent algebra. These are exact physical CSS QCA implementations
on the declared extended register. They do not replace a QCA by an operation
that agrees only on the code space, and they do not promise minimum ancillas.

## Physical gates

For odd primes a gate record uses `kind: SUM`. Its integer `coefficient` is
the canonical physical weight `w`, with `0 < w < p`; `offset: e` means that
the control is at cell `n+e` and the target at `n`. The gate is

```text
|a,b> -> |a,b+w*a>
Z action: I - w*x^e E_control,target
X action: I + w*x^-e E_target,control
```

An elementary factor `E_ij(a*x^e)` therefore becomes a SUM of weight `-a`.
Chronological products multiply on the left. Inverse replay reverses the
order and negates every gate weight or elementary coefficient. A mere reversal
without this negation is not an inverse at odd primes. Every layer is checked
for disjoint species, so each infinite translation family is a matching.
For `p=2`, records remain `CNOT` with coefficient 1 and the old meanings.

## Two exact constructions

The six-shear compiler retains its algebraic identity over every prime field:
it implements `diag(U,U_inverse)` with `q` additional `|0>` qudits. The
ancilla action is the actual `U_inverse`; its two-way stabilizer witnesses
are verified. The original determinant `c*x^a*y^b` is cancelled completely.

The stable elementary route implements `W = diag(U,B) = S*F`. `F` is the
supplied finite SUM circuit, and `B` is the explicit added-register action.
The factorizer's `balance` policy balances the complete determinant unit,
including `c`, on an added `|0>` qudit when needed. Identity stabilization may
add further qudits. This route needs no unspecified scalar gate.

Under `separate`, `S` remains explicit:

```text
S_ii = coefficients[i] * x^exponents[i]
X(S) = (S_inverse).dagger()
W = S * F
```

Both the exponents and the nonzero field coefficients are part of the
certificate. `constant_unit` is the product of the species coefficients,
and `determinant_exponent` is the sum of their exponent vectors. These two
aggregate invariants do not determine whether `S` is identity. Countershifts
or reciprocal scalings on distinct species remain nontrivial transport.
No part of `S` is counted as a gate layer. A scalar-only transport is local,
but it is still not implemented by the separately recorded SUM stream.

The circuit implements the original system QCA exactly iff `S` is identity
on every original species. It implements the full extended `W` exactly iff
`S=I` on all species. The ancilla stabilizer change is `S_ancilla^-1 B`.
The wrapper binds actual circuit targets as `HX_target = HX_standard*S`
and `HZ_target = HZ_standard*(S_inverse).dagger()`.

## Schema compatibility

| Artifact | Binary format retained | Odd-prime format |
| --- | --- | --- |
| Six-shear core | `css-decoupling-2d-circuit-v1` | `css-decoupling-2d-circuit-v2` |
| Elementary core | `css-decoupling-2d-direct-circuit/v1` | `css-decoupling-2d-direct-circuit/v2` |
| Direct code wrapper | `css-decoupling-2d/direct-decoupling/v1` | `css-decoupling-2d/direct-decoupling/v2` |

Each old circuit format is restricted to `p=2`. Odd-prime formats declare
their characteristic and use SUM records. Direct v2 transport has
`kind: diagonal_species_laurent_unit` and required `coefficients` in addition
to `exponents`. Its nontrivial mode is `transport_separated`; binary v1 keeps
`shift_separated`. The field `remaining_transport_coefficients` accompanies
`remaining_transport` in the v2 reader report. Resource fields whose historical
names contain `qubits` count prime-dimensional sites in v2. Shift resource
fields count displacements only; v2 adds `separate_unit_species_count` and
`separate_scaling_species_count`.

A binary direct wrapper can contain a modern binary QCA source certificate:
the source schema version also records lattice-request policy and is independent
of the circuit format. All readers validate the actual coefficient field.

## Verification scope

Readers check Laurent inverse identities, original/extended QCA binding,
the complete `S*F=W` equation, elementary products, actual gate products,
ancilla generation and declared resources. Odd-prime readers also explicitly
replay inverse Z and both X streams with the appropriate signs. They do not
call the constructor or factorizer. Tests include independent computational-
basis permutations for weighted SUMs at `p=3,5`, noncommuting factors, negative
offsets, nontrivial scalar units, balanced and separate transport, skew fixed
lattices, binary compatibility, serialization, and deliberately altered data.
These finite tests validate implementations; they are not a general proof.
