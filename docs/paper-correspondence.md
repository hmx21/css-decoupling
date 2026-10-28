# Correspondence with the manuscript

This page maps the higher-dimensional manuscript, *Toric codes all the way
down: decoupling translation-invariant topological CSS codes in higher
dimensions* (arXiv version 1), to the general implementation. Section,
equation, lemma and line numbers refer to that version. Passing tests and the
worked example are finite checks of the implementation; they are neither a
proof of the manuscript's statements nor a formal verification.

## Conventions

- Checks are matrices over `F_p[x_1^+-1, ..., x_D^+-1]`, with checks as rows
  and qudits as columns (Sec. II A). The dagger transposes and negates
  exponents.
- The physical qudits sit in degree 0 of the complex `C_code`, with
  `d^0 = HX` and `d^-1 = HZ^dagger` (Eq. (1)). `C.d(n)` maps degree n to n+1.
- Lattice bases are integer matrices whose columns are the basis vectors.
- The algorithms work in the X sector. A Z-sector request exchanges HX and
  HZ and maps the circuit back by on-site Fourier conjugation (Sec. IV B).

## Algorithm 1: coarse-graining

Entry point: `coarse_grain(HX, HZ)`; independent check:
`verify_coarse_graining`.

| Line | Implementation |
| --- | --- |
| 1. Bounded free complex `C_code` (Lemma 4) | `topology.resolutions.css_complex(..., complete_negative=True)`. Kernels are resolved by the native Schreyer algorithm (`algebra.schreyer`), as in Appendix A2; no Quillen–Suslin step is used. |
| 2. Layers `E_X^j = H^j(C_code)` | `topology.cohomology.cohomology_layer`, degree by degree. Nonzero layers outside `1 <= j <= D-1` are rejected as invalid input (Lemma 3). |
| 3–4. Stop on an infinite layer | `InfiniteLayer` gives a result with `decouplable=False`, `reason="infinite-layer"` and a witness. |
| 5. Translation action and `Lambda_triv` | `lattice.kernel.translation_kernel`: the complete common kernel of the actions, not a rectangular sublattice. |
| 6. `Lambda_suff = p^max(m-1,0) Lambda_triv` | Computed from the kernel basis and the number m of nonzero layers. `max_cell_index` is checked before any coset is expanded. |
| 7. Return `C_code` restricted to `F_p[Lambda_suff]` | `decoupling.splitting.restrict_complex`, stored as `result.restricted` (Lemma 8). |

Invalid input and exhausted resources raise exceptions. They are never
reported as "cannot be decoupled".

## Algorithm 2: decoupling circuit

Entry point: `decouple_complex(C)`, where `C` is a complex as in Lemma 4 or a
successful `coarse_grain` result; independent check: `verify_construction`.
The input must vanish outside degrees `|n| <= D+2` (Lemma 4). Algorithm 2
does not recompute free resolutions or the translation kernel.

| Line | Implementation |
| --- | --- |
| 1. Cohomology bases, `t_j`, representing cocycles | `topology.trivial_cohomology.trivial_cohomology`. Boundary witnesses for `(x_i - 1)` acting on each cocycle prove that translations act trivially; this is not inferred by setting `x_i = 1`. |
| 2. Cochain map `Gamma: C_stack -> C_code`, cone contraction, homotopy inverse `Delta` (Appendix C4) | `decoupling.splitting.splitting_map` in nonnegative degrees and exact lifts in negative degrees give `Gamma` (`result.phi`). `topology.complexes.homotopy_inverse` gives `Delta` (`result.psi`) and the homotopies. `contraction_strategy` selects how these equations are solved; see below. |
| 3. Stable isomorphism of check cokernels (Lemma 10) | `decoupling.comparison.paper_stable_cokernel_isomorphism`: the cone of the lifted rho, its splitting, the stably free frame `Q + R^k = R^a` and `b = rank(C_code^2) + k`. Each extra free direction whose frame row contains a Laurent unit is then cancelled by an elementary completion (`algebra.cancellation.cancel_unit_directions`), so `k` and `a` shrink without a Quillen–Suslin step; the remaining frame still satisfies `Q + R^k = R^a`, "for some `k` and `a`" as in Appendix C4. The adjoined identity X-type checks are reported as `identity_checks`. |
| 4. Invertible `U` and `V` (Lemma 11) | `decoupling.stabilization.stabilize`, using the block matrix `Theta(f, g)` and its inverse `Theta(g, f)`. Added ancillas are reported per cell. |
| 5. SUM and on-site SWAP layers (Lemma 12) | `circuits.gates.compile_theta`: one SUM layer per monomial of each block and an on-site block permutation. For unequal block sizes the permutation is generally not a single layer of SWAP gates. The reported depth, `sequential_depth_upper_bound`, is the length of the gate list. |

The verifiers recompute the commuting square of Eq. (3), the inverses of `U`
and `V`, and the action of the returned gate sequence, bound to the original
checks. They never trust a stored success flag.

### Contraction strategies

Appendix C4 contracts `G = Cone(Gamma)` and later splits `Cone(rho)`.

- `sections` builds the contraction from sections
  `sigma^n` chosen degree by degree from the top, so it squares to zero, and
  splits `Cone(rho)` the same way. This is the construction written in
  Appendix C4.
- `normalized` (the native default) first solves
  `d_G^{n-1} h^n = I - h^{n+1} d_G^n` degree by degree, as in the last
  sentence of that paragraph, after cancelling unit blocks. It then replaces
  `h` by `h d h`, which squares to zero, and transfers this contraction to
  `Cone(rho)` instead of solving there again. For a single top layer it
  first builds `Delta` from the explicit structure of the toric target. These
  shortcuts are implementation choices without a counterpart in the
  manuscript.
- `auto` selects `normalized` for the native backend and `sections` for the
  bounded backend.

All strategies are checked against the same identities.

### Proposition 1

Proposition 1 concerns `linear_backend="bounded"`
(`BoundedLinearBackend`), which linearizes coefficients within the degree
bounds of Appendix C5. `construction_size(C)` reports the dense common-box
size counted there. The default `linear_backend="sympy"` solves the same
equations with Gröbner bases. Its results are verified, but its running time
inherits no bound from Proposition 1. Algorithm 1 is outside the scope of
Proposition 1.

The bounded backend is a reference implementation. To find the complete
kernel of a matrix that unit operations do not reduce, it must expand to the
full degree bound `(2md)^(2^(2D))`, which is at least `2^16` already for
D = 2. With its default ceilings it therefore stops with `ResourceLimit` on
every such input, including the two-dimensional toric code; it completes
inputs whose kernels unit operations settle, such as contractible
complexes. A stop is a resource limit, never a mathematical answer.

## Other results

| Manuscript | Implementation |
| --- | --- |
| Theorem 1 on a specified superlattice | `check_lattice(HX, HZ, basis)`, with reasons `split-profile`, `infinite-layer`, `nontrivial-layer-translation` or `nonzero-gluing`; negative results carry witnesses (`verify_lattice_criterion`). |
| Gluing classes and their values `f_I` (Lemma 1); restricted values `kappa(mu_1, ..., mu_n)` (Lemma 9) | `gluing_classes`, `GluingClass.coefficients` and `GluingClass.evaluate`. |
| Coarse-graining less than line 6 of Algorithm 1 (Appendix B3, Proposition 2, Corollary 2) | `analyze_profile` and the default `coarse_graining="adaptive"` of `decouple`, using `select_lattice` and `subspace_preimage`. `ProfileAnalysis.lattice_guarantee` states whether the cell is proved minimal. `coarse_graining="uniform"` uses line 6 instead. |
| Logical qudits on a torus (Corollary 1, Lemma 13) | `periodic_code` (exact count on a finite torus), `periodic_layer` and `periodic_profile`. |
| Sec. V, the glued code | [`examples/paper_section_v.py`](../examples/paper_section_v.py) recomputes each stated claim from Eq. (16). `css-decouple example glued` writes the same input for the CLI. |

## Choices outside the manuscript

- Unit cancellation of the Schanuel frame (above) is an implementation step.
  It keeps every identity of Appendix C4 and removes directions the
  construction does not need: the toric code `TC_1` with D = 3, p = 2 now
  uses 5 ancillas and 24 gate layers instead of 21 and 96, and the glued code
  of Sec. V with p = 2 uses 21 ancillas instead of 67. The remaining ancillas
  are those of the block construction of Lemma 11.

- `comparison_method="free-basis"` replaces the stably free frame of
  Appendix C4 by an actual free basis computed with Quillen–Suslin. It is
  not the construction of the manuscript; with unit cancellation, the default
  route reached the same ancilla counts on the examples above.
- In dimension two, `decouple` uses the specialized algorithm of Wang et al.
  by default; `algorithm="general"` selects the construction described here.
- The Lean formalization of the article is a separate repository,
  [css-decoupling-lean](https://github.com/hmx21/css-decoupling-lean). It does
  not verify this package.
