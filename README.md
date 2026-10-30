# css-decoupling

Exact decoupling of translation-invariant topological CSS codes into stacks of
toric codes, over qudits of any prime dimension p and in any spatial
dimension D ≥ 2.

Given the check matrices of a code over the Laurent polynomial ring
`F_p[x_1^±1, ..., x_D^±1]`, the package decides on which superlattices the
code can be decoupled, and constructs the decoupling: a finite-depth local
circuit of SUM and SWAP gates, possibly with ancillas, that maps the code onto
a stack of higher-form toric codes. Every result is computed with exact
arithmetic over `F_p` and checked by an independent verifier.

This package accompanies the article

> M. He, Y. Wang, H. Song, and Y. Gu, *Toric codes all the way down:
> decoupling translation-invariant topological CSS codes in higher
> dimensions*, [arXiv:2610.40592](https://arxiv.org/abs/2610.40592) (2026).

Version 0.1.0 is archived on Zenodo; the DOI
[10.5281/zenodo.23752877](https://doi.org/10.5281/zenodo.23752877)
always resolves to the latest version.

It implements both algorithms of that article. In dimension two it also
provides a prime-field version of the algorithm of Wang, Ni, He, Liu, and Gu,
*Decoupling 2D translation-invariant topological CSS codes*,
[arXiv:2608.09915](https://arxiv.org/abs/2608.09915).
The Lean 4 formalization of the article's results is a separate repository,
[css-decoupling-lean](https://github.com/hmx21/css-decoupling-lean).

## Install

```sh
python -m pip install "git+https://github.com/hmx21/css-decoupling.git@v0.1.0"
```

or, from a clone of this repository, `python -m pip install .`.
Python 3.11 or newer and SymPy 1.13–1.14 are required; no other computer
algebra system is used. The distribution is `css-decoupling`, the import is
`css_decoupling`, and the command-line tool is `css-decouple`
(`python -m css_decoupling` is equivalent).

## Quick start

Decouple the two-dimensional toric code for p = 3:

```python
from css_decoupling import Ring, Matrix, decouple, verify_result, save_result

R = Ring(p=3, D=2)
x, y = R.variables
HX = Matrix.from_rows(R, [[x - 1, y - 1]])
HZ = Matrix.from_rows(R, [[1 - y**-1, x**-1 - 1]])

result = decouple(HX, HZ, dimension=2, output="circuit")
assert result.status == "circuit_ready"
assert verify_result(result).valid
save_result(result, "out/toric-circuit.json")  # choose a new path
```

`HX` and `HZ` have checks as rows and qudits as columns. The dagger of a
matrix is its transpose with every exponent negated. The input must satisfy
the topological condition `ker HX = im HZ^†` and `ker HZ = im HX^†`.
With `output="qca"` (the default) the result is the Clifford quantum cellular
automaton of the decoupling; with `output="circuit"` it is also factored into
gates. Check `result.status` before using a circuit.

The same workflow from the command line:

```sh
css-decouple example toric --dimension 2 --prime 3 --output out/input.json
css-decouple run out/input.json --format circuit --output out/circuit.json
css-decouple verify out/circuit.json --input out/input.json
```

Output paths are never overwritten.

## The glued code of the article

`css-decouple example glued --prime 3 --output out/glued.json` writes the
three-dimensional glued code of Sec. III B of the article. The script
[`examples/paper_section_v.py`](examples/paper_section_v.py) recomputes every
claim that Sec. V makes about it:

```sh
python examples/paper_section_v.py --prime 2 --prime 3 --prime 5
```

It checks the layers and the translation kernel, the gluing class and its
plaquette values, the superlattices of Algorithm 1 and the smallest
superlattices, a verified circuit onto TC_1 ⊕ TC_2, and the number of logical
qudits on small tori. Finite checks of this kind support the implementation;
they do not prove the article's statements.

## The algorithms of the article

| Article | Function |
| --- | --- |
| Algorithm 1 (coarse-graining) | `coarse_grain(HX, HZ)`, checked by `verify_coarse_graining` |
| Algorithm 2 (decoupling circuit) | `decouple_complex(C)`, checked by `verify_construction` |
| Theorem 1 on a given superlattice | `check_lattice(HX, HZ, basis)` |
| Gluing classes and their values (Lemmas 1 and 9) | `gluing_classes`, `GluingClass.evaluate` |
| Smallest superlattices (Proposition 2, Corollary 2) | `analyze_profile`, `select_lattice` |
| Logical qudits on a torus (Corollary 1) | `periodic_code`, `periodic_profile` |

`decouple` runs the whole pipeline. For D ≥ 3, or with
`algorithm="general"` for D = 2, it uses the construction of the article.
By default (`coarse_graining="adaptive"`) it coarse-grains only as much as the
gluing classes require; `coarse_graining="uniform"` uses the superlattice
`p^(m-1) Λ_triv` of Algorithm 1 instead. A fixed superlattice can be requested
with `lattice_mode="fixed"` and `lattice_basis`, whose columns are the basis
vectors.

Three implementation choices are worth knowing:

- **Linear algebra.** The default `linear_backend="sympy"` solves the linear
  systems over Laurent polynomial rings with Gröbner bases. Its results are
  verified, but its running time is not covered by the polynomial bound of
  Proposition 1. The reference backend `linear_backend="bounded"` follows the
  degree bounds of Proposition 1; these bounds are so large that it stops with
  `ResourceLimit` on most inputs, including the two-dimensional toric code.
- **Contraction.** `contraction_strategy="sections"` follows Appendix C4 of
  the article step by step. The default for the SymPy backend,
  `"normalized"`, solves the same degree-by-degree equations and then takes
  shortcuts that the article does not describe. Both are checked against the
  same identities.
- **Ancillas.** Extra free directions of the stably free module in
  Appendix C4 that contain a Laurent unit are cancelled before the circuit
  is built. This reduces the number of ancillas and gates and keeps every
  identity of the construction.

[docs/paper-correspondence.md](docs/paper-correspondence.md) maps every line
of both algorithms to the code.

## Dimension two

For D = 2, `decouple` uses the specialized algorithm of arXiv:2608.09915 by
default. That article treats qubits; this package uses a prime-field version,
described in
[docs/two-d/prime-qca-mathematics.md](docs/two-d/prime-qca-mathematics.md).
The specialized route keeps the original register and adds no input ancillas.
It always returns a verified QCA. For some inputs of rank two, the circuit
request returns `circuit_incomplete` together with the verified QCA and a
reason: a bounded search or resource stop is not a proof that no circuit
exists. Circuits for odd p may contain an on-site multiplication gate in
addition to weighted SUM gates; it is recorded explicitly.

## Results and verification

A computation either succeeds, reports a mathematical obstruction with a
witness (for example an infinite-dimensional layer or a nonzero gluing class),
rejects invalid input, or stops at a resource limit. These outcomes are kept
apart; a resource stop never means that a code cannot be decoupled.

Verifiers recompute the defining identities from the stored data and the
original input, for example the commuting square of the decoupling, the
inverses of the block matrices and the action of the returned gates. They do
not trust a stored success flag.

## Documentation

- [Unified API, result states and JSON input](docs/unified-api.md)
- [Lower-level entry points of the general algorithm](docs/general-reference.md)
- [Correspondence with the article](docs/paper-correspondence.md)
- [Two-dimensional prime-field construction](docs/two-d/prime-qca-mathematics.md)
  and [same-register circuit contract](docs/two-d/no-input-ancilla.md)
- [Contributing](CONTRIBUTING.md) and [changelog](CHANGELOG.md)

## Tests

```sh
python -m pip install -e ".[dev]"
python -m pytest
```

The suite runs in a few minutes; continuous integration tests the installed
wheel on Linux, macOS and Windows.

## Citation

If you use this package, please cite the article above and the software;
see [CITATION.cff](CITATION.cff). For the two-dimensional algorithm, please
also cite arXiv:2608.09915.

## License and attribution

MIT; see [LICENSE](LICENSE). Parts of `src/css_decoupling/two_d/` are adapted
from [topological-css-code-decoupling](https://github.com/nzy1997/topological-css-code-decoupling)
(MIT, by Zhongyi Ni and Mingxin He); see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

The code was developed by Mingxin He with the AI coding assistant Codex
(OpenAI).
