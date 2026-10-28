# Source notices and license scope

The unified distribution is **css-decoupling**, imported as `css_decoupling`.
Project-owned implementation and documentation use the MIT license in
[LICENSE](LICENSE), with copyright held by Mingxin He. This does not replace upstream notices
or claim that this project is the upstream repository below.

## Two-dimensional algorithm and example catalog

Some material in `src/css_decoupling/two_d/` is adapted from
[topological-css-code-decoupling](https://github.com/nzy1997/topological-css-code-decoupling),
commit `7879852262ee7bd0bb69c1a2edae8f1543fcca3e`, copyright 2026 The Authors.
The original MIT text is preserved verbatim in
[LICENSES/topological-css-code-decoupling-MIT.txt](LICENSES/topological-css-code-decoupling-MIT.txt).
Keep that notice with redistributed copies or substantial portions of the
adapted code and data.

The upstream Python/SageMath implementation is attributed to **Mingxin He**;
the upstream Julia/Oscar implementation is attributed to **Zhongyi Ni**.
Adapted material includes the attributed Algorithm 1 implementation and the
published example/BB-code catalog. Original per-file attribution comments
remain in the source. The original upstream citation metadata is retained
in [CITATION.upstream.cff](CITATION.upstream.cff).

The associated article is *Decoupling 2D translation-invariant topological
CSS codes*, by Yifei Wang, Zhongyi Ni, Mingxin He, Jinguo Liu and Yingfei Gu,
[arXiv:2608.09915v1](https://arxiv.org/abs/2608.09915v1).
The prime-field adaptation, independent native implementations, later
factorizers, unified API and certificate integration are distinguished from
that pinned upstream snapshot; neither the article nor its historical
results are rewritten by this merge.

## Shared native algebra

The two-dimensional code uses the package's own algebra modules
(`src/css_decoupling/algebra/`, `lattice/`, `topology/`), which are
project-owned and covered by the project's MIT license.

Historical migration manifests and audit hashes retain their original
dates and license observations. In particular, an old record that no local
license existed at its snapshot date is historical provenance, not a claim
that the current MIT-licensed source lacks a license. Private workspace
paths and historical audit dumps are not required runtime package data.

## Dependencies and references

SymPy and its mpmath dependency are installed as separate distributions with
their own licenses and notices. Their source distributions are not bundled
into this wheel. SageMath, Singular, Julia and Oscar are not runtime backends.
References to those systems in upstream attribution or historical comparisons
do not make them package dependencies.

Mathematical references explain algorithms; they do not by themselves confer
licenses to copy another implementation. Contributors must preserve the
source and license of any new adapted code. The manuscript and external
reference documents are not included in this software license by implication.
