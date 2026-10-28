# The two-dimensional QCA construction over prime fields

This note records the mathematical adaptation implemented by `qca.py` and
`models.py`. It is separate from the article and does not change its stated
qubit scope. The source construction is [arXiv:2608.09915v1, Supplemental
IV.3–IV.4](https://arxiv.org/html/2608.09915v1): it compares extension
classes, then repairs a comparison map by a chain homotopy. In IV.3 the
one-dimensional extension group is over F₂ and consequently has just one
nonzero element. Over Fₚ there are p−1 nonzero elements; the replacement
needed here is normalization by a nonzero scalar, not identifying all of
these elements with 1. The argument below supplies that replacement.

## Domain and hypotheses

Let k=Fₚ, with p prime, R=k[x^±1,y^±1], and J=(x−1,y−1). Matrix columns are
physical Pauli support vectors and dagger transposes and reverses Laurent
exponents. A valid input satisfies both physical exactness conditions:

```
ker(HX) = im(HZ†),       ker(HZ) = im(HX†).
```

The pipeline retains complete kernel and generator witnesses, reduces
redundant checks to independent generators, computes both finite anyon
cokernels, and restricts to a lattice acting trivially on both of them.
The QCA stage assumes these preparations have completed. In particular,
the reduced checks have q=r+s and both cokernels are k^t with trivial x,y
action. The zero-anyon case is expressed as annihilation by J; it does not
require the annihilator of the zero module to equal J.

The signed toric block is

```
HX = (x−1, y−1),        HZ† = (1−y, x−1)^T.
```

Its consecutive maps multiply to zero in every characteristic. The
regular sequence x−1,y−1 gives its exactness. Products add identity blocks.
In characteristic two these matrices are exactly the earlier matrices.

## Why nonzero extension classes still give toric blocks

The presentation `R -> R² -> J` with maps `(1−y,x−1)^T` and `(x−1,y−1)`
gives `Ext¹_R(J,R)=R/J=k`: applying Hom(−,R) makes the final cokernel the
quotient by J. Consequently extensions of `R^px ⊕ J^t` by `R^r` are
represented by r-by-t matrices over k.

For one J summand, the zero class gives R⊕J. Every nonzero scalar class
is obtained from the toric extension by an automorphism of its left R
endpoint, since every nonzero element of k is a unit of R. Its middle
module therefore remains R². This step is where nontrivial inverses in
Fₚ are essential.

For a general class θ of rank ℓ, invertible constant row and column
operations put θ into rank normal form. They lift to R-module
automorphisms. With Δ=t−ℓ, the middle module is isomorphic to
`R^(q−Δ) ⊕ J^Δ`. If it were free of rank q and Δ>0, localization at J
would contradict the minimal generator count: J/J² has k-dimension 2,
so the displayed module needs q+Δ generators over the local ring,
whereas a free rank-q module needs q. Thus a free middle module forces
rank θ=t. Constant GL(r,k) acts transitively on full-column-rank
matrices, and its action alone aligns θ with the standard class while
keeping the right endpoint fixed. This establishes the required chain
isomorphism under the stated hypotheses.

## Signed construction and exact inverse

Reduction modulo J is evaluation at x=y=1. The odd-prime clearing routine
normalizes each pivot by its multiplicative inverse and maintains

```
X = Lx augmentation(HX) P,
Z = Lz augmentation(HZ) P^(−T).
```

A column addition by c on one check induces the transposed inverse
addition by −c on the other. This preserves CSS commutation; binary XOR
is retained only as a characteristic-two optimization.

Write D=Z† and TD=standard_HZ† after clearing. The comparison equations
are `X phi1=standard_HX` and `D phi2=phi1 TD`, with the product columns
fixed. The previous existence argument and uniqueness up to chain
homotopy show that the toric block of phi2 has an invertible constant
augmentation B. The implementation computes B and its actual field
inverse; it never replaces a nonzero scalar by 1.

Replacing that block by B gives phi2′. Since the difference has entries
in J and the signed toric differential generates J, solve
`eta TD=phi2′−phi2` and set `phi1′=phi1+D eta`. Then
`D phi2′=phi1′ TD`. The corrected endpoint has block form `[I C;0 B]`
and inverse `[I −C B^−1;0 B^−1]`.

The physical inverse is also constructed, not inferred from a flag.
If V is the standard-to-input physical map, choose Q0 with
`standard_HX Q0=psi0 HX`, lift `V Q0−I=HZ† H`, and set
`Q=Q0−standard_HZ† psi2 H`. The implementation checks both `VQ=I` and
`QV=I`, both chain directions, and the complete comparison/homotopy
identities. The physical Z action is Q and the X action is V†, hence
`X_action† Z_action=I` over Fₚ as required.

## Scope of the implementation evidence

`tests/test_prime_qca.py` checks p=3,5,7 signed toric blocks, nonunit
constant pivots, weighted Laurent changes, nontrivial extension scalars,
product/empty cases, and tampered maps. At p=3,5 it additionally checks
redundant weighted generators and nontrivial anyon translations on an
oblique, negatively oriented coarse lattice. Existing restriction tests
cover finite-quotient lifting over p=2,3,5. These are software checks,
not a proof by finite enumeration or a Lean formalization.

The complete module solver remains the fallback for odd primes. Binary
bounded solvers may return no candidate; that does not assert
nonmembership. Resource exhaustion remains incomplete computation.

This note concerns QCA construction. It makes no ancilla-free circuit
claim, no complexity improvement claim, and no claim that the published
qubit BB/color-code catalog has been validated at odd primes. Its saved
reference periods and counts retain their original qubit meaning.
