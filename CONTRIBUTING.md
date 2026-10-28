# Contributing

Use Python 3.11 or newer and SymPy. Core execution must remain native Python;
do not introduce an external CAS executable or an automatic backend fallback.
Install development tools from the project root:

```sh
python -m pip install -e ".[dev]"
python -m pytest --import-mode=importlib
```

Editable installation is convenient for development. Before a release, also
build a wheel, install it without editable links, and run the tests from
outside the source tree, as the continuous-integration workflow does.

## Mathematical changes

State the input assumptions and the exact algebraic identities affected by a
change. Check both directions of maps, inverse/dagger conventions, prime-field
signs and original-input binding. Finite tests support implementation claims;
they are not a proof of a classification theorem or a formalization in Lean.

Construction and verification must remain separate. A reader may inspect
the saved witnesses and recompute complete algebraic conditions, but must
not trust a success flag or call the construction routine to accept an artifact.
Resource limits, invalid input, certified obstructions and an unfinished
factorization are different outcomes.

For two-dimensional circuit work, distinguish a target product factor made
from existing sites from an extra initialized input ancilla. Do not replace
an unresolved original-register problem with a stabilized circuit or a larger
gate-period cell without an explicit option and certificate contract. Suslin's
rank-at-least-three result is not a general rank-two factorization oracle.

For general-algorithm changes, preserve the meaning of existing stable
directions, source complexes, lattice certificates and QCA/circuit views.

## Tests, evidence and attribution

Add focused tests for substantive behavior, including odd-prime signs,
negative Laurent exponents, redundant checks, nonrectangular lattices and
altered certificates where relevant. Avoid duplicating an implementation as
its only test oracle. Keep large campaigns separate from the fast test suite.
Never overwrite a saved exact certificate to make a rerun look clean.

Retain per-file authorship and applicable notices when adapting code. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Contributions must be
compatible with the project's MIT license.

Report problems and propose changes through the issues and pull requests of
[the GitHub repository](https://github.com/hmx21/css-decoupling).
