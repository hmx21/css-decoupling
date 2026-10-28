# Decoupling on the original physical register

The no-input-ancilla route keeps exactly the coarse input's q physical qudits.
Its target has t toric sectors and pX/pZ product sectors, with
`q = 2*t + pX + pZ`. These product factors are formed from the existing
physical register; they are not additional qudits prepared at the input.
This route is separate from the retained stable Higman and inverse-register
compilers. Their schemas and their added-ancilla meanings are unchanged.

```python
from css_decoupling.two_d.code_circuits import (
    compile_qca_no_ancilla, verify_no_ancilla_result,
)

artifact = compile_qca_no_ancilla(qca_result, factorization="auto")
report = verify_no_ancilla_result(artifact, expected_HX=HX, expected_HZ=HZ)
assert report["input_ancillas"] == 0
```

The physical count here is after the already declared coarse-graining.
Coarse-graining groups existing sites; it does not prepare ancillary sites.

## Initial QCA and actual circuit

When fewer than three physical species are present, the compiler first checks
whether the original coarse checks and the standard target already generate
exactly the same X and Z stabilizer modules. Only if all four exact membership
witnesses exist does it return the identity circuit, retaining the original U
and `K=U_inverse` as a verified target-code automorphism. This branch does not
factor U and does not infer code equality from q=2 or from a toric multiplicity.
Its certificate has null `initial_factorization` and `correction`, with an
explicit code-equality reason. Supplied factor data is always checked through
the factor route. Failed code equality proceeds to actual factorization;
resource failures are not treated as equality or as mathematical impossibility.

Otherwise the original-rank factorizer supplies an actual elementary word F and diagonal
Laurent units S such that `U = S*F`, where U is the saved source-to-standard
QCA. The new correction G uses only the same q species and produces

```text
actual_z = G*F = K*U,    K = G*S_inverse.
```

The certificate independently proves that K preserves the standard toric/
product stabilizer modules in both directions. It also verifies the actual
circuit directly against the original coarse checks and the standard target.
It does not claim `actual_z = U` unless their matrices are equal. The X action
is the inverse dagger of the actual Z action, with zero phases on pure X/Z
generators. Equality of code images is not asserted to be equality of the
original QCA or of every encoded logical transformation.

This is the distinction made in the paper's Supplemental VI.1: eliminating
transport can change the QCA while retaining the required code decoupling.
[Wang et al., arXiv:2608.09915v1](https://arxiv.org/html/2608.09915v1)

## Existing target product species

If a product species j already occurs in the standard target, the correction
uses it as an anchor. For each toric species i with transport unit u, the
signed six-shear identity realizes `diag(u,u_inverse)` on species (i,j).
The resulting K is identity on every toric species and a diagonal unit on
product species only. Such units preserve the translation-invariant |0> or
|+> stabilizer module. The full unit, including its nonzero field coefficient,
is handled this way without adding a species or requiring a scalar gate.
Product-only targets require no correction gates.

The anchor was already a physical qudit at the input; it becomes a product
factor after the initial part of the circuit. Using that existing qudit is
not the stable compiler's addition of a new initialized input register.

## Pure toric targets

For a toric pair write

```text
S_pair = diag(c1*x^a*y^b, c2*x^c*y^d),
C = diag(c1,c2),       M = diag(x^a*y^b,x^c*y^d),
f = (y-1)*(x^(c-a)-1)/(x-1),
g = x^(a-c)*(x-1)*(y^(b-d)-1)/(y-1),
R = E12(f)*E21(g),     G_pair = R*C.
```

The fractions here are finite geometric Laurent sums, including the signed
negative-power formula. For the canonical X check `h=(x-1,y-1)`, direct
multiplication gives `h*M=x^a*y^d*h*R`, over every prime field. This is the
signed form of the paper's Eq. VI.3. The resulting K has X- and Z-generator
changes `x^-a*y^-d` and `x^-c*y^-b`, respectively. Both reverse changes and
the physical gate products are checked, rather than relying on a formula label.

When `p>2`, C can require an explicit on-site multiplication Clifford:
`MULTIPLY(multiplier=m)` sends `|a>` to `|m*a>`. Its Z exponent is multiplied
by `m_inverse` and its X exponent by m; to realize Z scalar c, the compiler
uses `m=c_inverse`. These are declared physical gates and counted separately
from weighted SUMs. Qubit output uses CNOTs and has no nontrivial scalar unit.

This extra gate type is sometimes necessary with a fixed pure-toric register.
For example, over F5 take one toric pair and initial transport `diag(2,1)`.
A CSS automorphism K of the canonical pair satisfies `h*K=v*h` for a Laurent
unit v. Comparing linear terms at x=y=1 gives `K(1,1)=v(1,1)*I`, so its
augmentation determinant is a square. Every SUM/SWAP-only matrix has
determinant +1 or -1, both squares in F5. Thus a no-input-ancilla correction
for this nonsquare determinant cannot always be implemented with only those
gates. An on-site multiplication suffices. Independent computational-basis
permutation tests check this gate's Pauli action and a complete mixed stream.

## Certificate and scope

The new schema is `css-decoupling-2d/no-input-ancilla/v1`. It stores the
original QCA result, a verified original-rank `U=S*F` factorization, the
correction's actual gates and Pauli matrices, the verified target-code
automorphism, and the resulting same-q circuit and original-check generation
witnesses. The independent reader calls neither the factorizer nor the
correction constructor. It replays the correction's four X/Z forward/inverse
streams, verifies the initial core's actual gates, and checks the exact
chronological concatenation and matrix composition.

`input_ancillas=0` and both physical register sizes are independently checked.
Target product counts are reported separately. The old stable formats cannot
be relabeled as no-input-ancilla results: any positive added register is rejected.

The transport correction handles empty, one-site product and two-site toric
targets, including negative and oblique supercell coordinates. It does not
infer that every rank-two special linear Laurent matrix is elementary.
In particular, checks formed by applying a rank-two Cohn matrix to the standard
toric pair need not generate the same physical stabilizer modules. They cannot
be sent through the identity branch just because their anyon content is toric.
Original-rank factorization must actually supply a checked word, or report
its unresolved case/resource limit. A rank-at-least-three factorization
theorem must not be used as a rank-two oracle. The target correction itself
does not add a larger cell or any physical species to hide that obligation.

The current default q=2 path also includes bounded code-level row reduction
and an exact toric-automorphism bridge; see
[original-register factorization](unstabilized-factorization.md). This extends
the successful cases beyond already-identical codes and simple unit pivots,
but does not turn an exhausted search into a mathematical rejection. The
unified interface retains the verified QCA and reports `circuit_incomplete`
with the reason when this construction does not finish.

Limits bound actual factors, gate families and matrix/geometric-sum terms;
resource exhaustion remains an unfinished computation. Finite tests are
implementation evidence, not a new formal proof or a minimum-depth claim.
