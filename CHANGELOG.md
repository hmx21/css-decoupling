# Changelog

## 0.1.0 — 2026-10-28

First public release, accompanying the article *Toric codes all the way
down: decoupling translation-invariant topological CSS codes in higher
dimensions*.

- Algorithm 1 (`coarse_grain`) and Algorithm 2 (`decouple_complex`) of the
  article, with independent verifiers.
- The unified entry point `decouple` and the command-line tool
  `css-decouple`, with QCA and circuit outputs, adaptive or uniform
  coarse-graining, and fixed superlattices.
- Gluing classes, smallest superlattices, the decoupling criterion on a given
  superlattice, and logical-qudit counts on finite tori.
- A prime-field version of the two-dimensional algorithm of arXiv:2608.09915.
- `examples/paper_section_v.py`, which recomputes the claims of Sec. V of the
  article.
