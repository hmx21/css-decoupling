# Elementary factors on the original register

The original-register route factors only the q-by-q Z-support matrix U. Its inverse is
used as an exact witness and to obtain the X action; no 2q-by-2q symplectic
matrix is constructed for factorization. Every elementary index is checked
against the original q. The result has `added_ancillas=0`.

```python
from css_decoupling.two_d import factor_unstabilized

data = factor_unstabilized(U, U_inverse, method="auto")
```

The result equation is `U = S F`, with a declared diagonal Laurent unit S and
an actual elementary word F. Records are chronological, so a list E1,...,Em
means `F=Em ... E1`. If `S=I`, this is a factorization of U itself. Otherwise
the unit factor remains explicit; a determinant different from one is never
misrepresented as a pure transvection product. The X factors are derived by
inverse dagger in the same chronological order.

## Implemented construction

The rank-at-least-three construction follows
[Park and Woodburn, alg-geom/9405003v1](https://arxiv.org/html/alg-geom/9405003v1).
The implementation uses the original physical coordinates as the spare slots
required by its identities. It does not append identity qudits or call the
stable Higman matrix compiler.

`elementary_columns.py` reduces a unimodular column with an explicit Bezout
witness. It records unit and existing-zero-slot reductions, Laurent
normalization, polynomial coordinate changes, resultants and closed-point
patches. Finite-field points may lie in extension fields; the code does not
restrict its coverage to rational points. The final elementary word and both
inverse identities are checked in the original Laurent ring.

`unstable_local.py` implements local SL3 degree reduction. `unstable.py`
constructs the normality words, clears their denominators with explicit patch
radii, combines the cover, and pulls back the polynomial and monomial
coordinate changes. Intermediate symbolic calculations do not define new
physical qudits. The delivered word acts on the original register.

For q>=3, `method="auto"` first tries an explicitly capped fixed-rank fast
reduction. A verified complete word is returned immediately; otherwise the
spent probe quota is charged before entering the general route. `"suslin"`
enters the complete fixed-rank organization directly, including its exact
unit, support-reduction and zero-slot base cases. A deterministic comparison
of the normalized matrix and its known inverse selects one general direction;
an inverse-direction word is inverted back to factors of the original U.
It does not factor both Pauli blocks. `"fast"` requests only the incomplete
unit/cancellation reductions. No mode silently invokes the stabilized compiler.

The code uses CPython and SymPy only. Time, factor, term and operation limits
can stop construction. Product counters cover the instrumented sparse
convolutions, not every internal SymPy operation. A single symbolic operation
can outlive a cooperative time check. Murthy recursion also has a 256-level
structural protection. Resource stops are incomplete computations, not
non-elementarity certificates.

## From the word to the requested code circuit

`compile_qca_no_ancilla` constructs a target-code correction G and returns
the actual circuit action `A=G F=K U`, where `K=G S^-1` is independently
verified to preserve the standard target code. It preserves both U and A.
Their equality is asserted only when it actually holds. Target product-state
factors occupy existing physical sites and are counted separately from added
input qudits, whose count is always zero. See
[target correction and gate conventions](no-input-ancilla.md).

## Low rank and precise scope

The general fixed-rank factorization theorem used here requires q>=3. For
q<3 the software returns an explicitly verified factorization, or uses the
verified same-code identity branch when both source stabilizer modules already
equal their standard target modules. It does not label an arbitrary two-site
code canonical merely because its anyon content is toric.

For q=2, the default code-level `factorization="auto"` additionally invokes
an explicitly bounded row-reduction search and a checked toric target-
automorphism bridge. Successful paths return an actual word for the specified
QCA, without receiving a planted construction matrix or model-specific answer.
Search width/depth, generated nodes and quotient work have explicit limits.
This is a positive construction procedure, not a general E2 membership decision
algorithm. Its failure retains a structured reason and does not trigger blocking,
stabilization or the rank-at-least-three routine on a two-by-two matrix.

Nonstandard rank-two examples remain explicitly incomplete when the available
factorization does not finish. Their fast-path failure is not a proof of
non-elementarity. A recent primary discussion states that general
`SL2(F_p[x^+-1,y^+-1])=E2` remains open:
[Ionin and Semidetnov, introduction](https://arxiv.org/html/2609.04275v1).
This does not rule out exploiting code-equivalent representatives; it prevents
using the rank-three theorem as an unimplemented rank-two oracle.

Grouping existing neighboring cells for gate synthesis is being investigated
separately. It introduces no physical ancillas but can change the individual
gate layers' periods. It is not an implicit fallback of the original-cell
interface, and no such grouping is silently performed in fixed lattice mode.
