# Prime-field elementary factorization

The native factorization and Park column-normalization primitives accept
`Ring(p, 2)` for every prime `p`. They use CPython and SymPy, preserve negative
Laurent exponents, and return actual elementary factors. The implementation
does not use an external CAS, a fraction-field kernel as a module basis, or
an unbounded search for factors. The original binary schema and algorithm
identifiers remain supported. Saved binary acceptance results retain their
original scope; they are not evidence of odd-prime acceptance.

This page describes the prime-field algebra behind `suslin.py`, `elementary.py`,
and `laurent_to_polynomial.py`. A physical circuit and a complete decoupling
result have additional independent verification requirements.

## Factor and transport conventions

A factor record `(row=i, column=j, coefficient=a)` denotes
`E_ij(a)=I+a e_i e_j^T`, with `i != j`. Records are chronological: if they
are `E1,...,Em`, the actual product is `F=Em ... E1`. The inverse record has
coefficient `-a`; reversing the list alone is insufficient for odd primes.

The returned equation is

```
W = U direct-sum B = S F,
W^-1 = V direct-sum B^-1 = F^-1 S^-1,
S = diag(c_i x^a_i y^b_i).
```

`shift_exponents[i]` stores `(a_i,b_i)`. Odd-prime results additionally contain
`transport_coefficients[i]=c_i`, a canonical nonzero residue modulo `p`.
Binary results omit that field and all its implied coefficients are one.
No scalar coefficient is discarded when a determinant is written as a unit.
`S` is residual unit transport; its nonzero displacement is not a SWAP layer.
Its constant scaling is also explicitly retained in the algebraic action.

In `separate` mode, Higman ancillas have identity action and the final transport
entry is the complete determinant `c x^a y^b`. In `balance` mode, if that unit
is not one, one additional species has action `(c x^a y^b)^-1`. This balances
both its scalar and its displacement, so `S=I`. A determinant which is constant
but differs from one still requires balancing under this policy. The policy
does not claim the minimum number of ancillas or gates.

Both factorization entry points replay their actual elementary word and its
inverse and check the stated extended equations. The physical circuit verifier
must independently replay the gates as well. For the package's physical SUM
convention, `E_ij(a)` corresponds to SUM weight `-a` on each supported monomial.

## Signs and units

Let `w(u)=E_ij(u) E_ji(-u^-1) E_ij(u)`, a written product. Its two-coordinate
matrix is `[[0,u],[-u^-1,0]]`. Thus

```
diag(u,u^-1) = w(u) w(1)^-1.
```

The six signed elementary coefficients, in written order, are
`u,-u^-1,u,-1,1,-1`; chronological order is the reverse. This identity applies
to every Laurent unit, including units with nontrivial constant coefficient.

The fast reducer applies `w(1)` as a signed swap, preserving determinant one.
The PID reducer tracks actual permutation matrices: a true swap is
`w(1) diag(-1,1)`. Its diagonal sign is included in the internal word and
therefore in the determinant calculation. Left-operation records and written
right-operation words have different order conventions; the final matrix
identities check their conversion.

When factoring an inverse is cheaper, `V_ext=S_V F_V` is pulled back as

```
U_ext = S_V^-1 (S_V F_V^-1 S_V^-1).
```

The returned word reverses and negates the original elementary coefficients,
then multiplies each by the corresponding full scalar-monomial ratio. The
returned transport coefficients and ancilla action are inverted as well.

## Euclidean division and the projector

For PID coefficients in `A=F_p[x,x^-1]`, valuations are first removed. At each
polynomial division step, the quotient coefficient is the remainder's leading
coefficient times the inverse of the divisor's leading coefficient modulo `p`.
Subtracting that multiple strictly lowers the polynomial degree. The returned
Laurent quotient and remainder satisfy `a=q b+r`, with the remainder's support
span smaller than the divisor's. XOR cancellation is not used for odd primes.

Higman linearization gives an exact elementary relation between a stabilized
polynomial matrix and a pencil `C(y)=C0+y C1`. Specializing the recorded
relation at `y=1` gives a checked inverse of `C(1)` over `A`. Define

```
C(1)^-1 C(y) = I + (y-1) T.
```

If `C(y)` is Laurent invertible, its determinant is a unit times `y^k`.
After specialization this implies
`det(I+(y-1)T)=y^k`, or
`charpoly(T)=t^(N-k)(t-1)^k`. Cayley–Hamilton bounds both nilpotent parts by
the pencil size `N`. Choose a power `M=p^s >= N`. In characteristic `p`,
`(I+N1)^M=I+N1^M=I` on the generalized one-eigenspace, and `N0^M=0` on the
generalized zero-eigenspace. Consequently `E=T^M` is an idempotent.

The code uses bounded binary matrix powering to compute each `p`th power;
it does not perform `p` multiplications or retain a power-of-two rule at odd
primes. It then explicitly checks `E^2=E`, constructs complete PID image/kernel
bases and both inverses, and checks the reconstruction of the pencil.
Nilpotent kernel recursion strictly decreases rank. The resulting triangular
pencil has diagonal units one or `y`, and its elementary reduction is recorded.

The scalar-weighted fast cancellation heuristic also uses exact arithmetic.
For a shifted source row/column, support decrease equals
`overlap + cancellations - source_terms`. Only overlaps with a coefficient
ratio that actually cancels count as cancellations. These two counts coincide
over `F2`, but need not coincide for odd primes. This heuristic is optional
and is not the completeness argument for the Higman/PID fallback.

## Park column preprocessing

For a supplied column `v` and Bezout row `h` with `h v=1`, an elementary update
`v <- E_ij(a)v` is accompanied by `h <- h E_ij(-a)`. The leading monomial of
the first component is normalized using both its exponent and the inverse of
its coefficient. Lower-degree cancellation uses subtraction.

After the coordinate changes, `v0=1 mod xy`, and the other components lie in
the ideal `(xy)`. If multiplying `h` by `(xy)^c` clears its negative exponents,
write `t=1-v0=(xy)g`. The polynomial row

```
g^c ((xy)^c h) + (1+t+...+t^(c-1)) e0^T
```

has product `t^c+(1-t)(1+...+t^(c-1))=1` with `v`. The signs of `t` and the
inverse elementary updates matter at odd primes. The independent reader
replays the supplied word and coordinate changes and checks the resulting
polynomial Bezout equation. This preprocessing remains distinct from a
completed matrix factorization.

## Validation and resource scope

`tests/test_prime_factorization.py` includes forced Higman controls at `p=3,5`
with a nonunit Cohn matrix, negative exponents and determinant
`2 x^-1 y`; mixed zero/one projectors; nonconstant PID kernels and longer
nilpotent flags; true permutations; scalar units; signed residual and inverse
pullbacks; seeded mixed-coefficient words; Park witness replay and tampering;
and explicit resource stops. The replay uses generic matrix multiplication,
independently of the factorizer's sparse update routines. Existing binary
tests continue to cover their original inputs.

Time, operation, term, factor, convolution-pair and linearized-size caps remain
finite. Native time checks are cooperative and are not a process-level hard
deadline. Exhaustion is `ResourceLimit`, not a claim that the requested
factorization does not exist. These finite tests are implementation evidence;
they are not a general formal proof, a polynomial running-time guarantee,
or a claim of minimum stabilization.
