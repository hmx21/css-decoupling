# Advanced general-algorithm reference

The unified `decouple` entry point is documented in [unified-api.md](unified-api.md).
The former general Python interface remains available as `decouple_general`.
Its direct keyword arguments and algorithm-specific result type are distinct
from the unified result envelope.

```python
from css_decoupling import decouple_general, save_decoupling, verify_decoupling

general = decouple_general(HX, HZ, backend="sympy", resolution_method="native")
assert verify_decoupling(general).valid
save_decoupling(general, "out/general-certificate.json")
```

Use new output paths. `save_decoupling`, `load_decoupling` and `save_input`
retain their historical lower-level contracts; the unified counterparts are
`save_result`, `load_result` and `save_css_input`.

## Stages and choices

- `coarse_grain` implements the paper's Algorithm 1: a complete free model,
  actual layers and translation actions, the full common kernel, and a
  sufficient lattice removing gluing.
- `decouple_complex` implements Algorithm 2 on an already supplied complete
  coarse-grained complex. It does not redo the original input analysis.
- `analyze_profile` and the default adaptive construction use gluing
  information to avoid unnecessary coarse-graining when the implemented
  lattice methods allow it.
- `check_lattice` checks a specified lattice with input-bound positive or
  negative evidence; a resource stop is not a negative theorem certificate.

`coarse_graining="adaptive"` is distinct from `"uniform"`. The latter composes
the paper's sufficient-lattice formula with Algorithm 2. Construction
strategies `normalized` and `sections` differ in how exact
homotopy and cone equations are solved; see
[paper-correspondence.md](paper-correspondence.md). They do not change an input's mathematical
assumptions or waive the final verification conditions.

The default C4 comparison uses a Schanuel stable-free framework and records
all resulting extra directions. `comparison_method="free-basis"` selects
the separate implemented actual-free-basis/QS route. QS row completion,
module basis construction and elementary circuit factorization are different
operations; an available completion matrix is not by itself a gate list.

## Native backends and cost scope

Ordinary Laurent module work uses `backend="sympy"`; free resolutions use
`resolution_method="native"`. External CAS execution is not an option.
Algorithm 2 separately supports `linear_backend="bounded"`, with finite-field
coefficient equations and an explicit degree-bound contract. Its general
kernel bounds can exceed practical resource limits. The practical SymPy
route does not automatically inherit a proved running-time bound for every
internal operation.

The manuscript's fixed-dimension Algorithm 2 cost statement concerns dense
coefficient size in a common exponent box. It is not a uniform polynomial
claim in dimension, an Algorithm 1 guarantee, or a bound for every sparse
input and every practical backend. Coarse-cell index, Koszul ranks, Gröbner
support and actual output can be large. Time checks are cooperative; hard
process limits require an external supervisor of the Python process.

## CLI and historical evidence

`css-decouple general --help` exposes the preserved advanced CLI. To use its
historical raw-input format, create input with `save_input`; the unified CLI
uses `save_css_input` and the portable input schema instead. Existing source
certificates and their validators keep their prior meanings.

Successful general QCA/circuit views include stable physical registers,
their order, ancilla states, the selected lattice, the target toric
multiplicities and both X/Z actions. A standalone QCA inverse or symplectic
check does not prove decoupling from the original input; retain and verify
the source certificate.

Large research examples have finite, source-specific acceptance histories;
some remain resource-incomplete. No general performance, minimum-ancilla,
minimum-gate or higher-dimensional Lean formalization claim is added by the
unified packaging release. Workspace-only historical audits remain separate
from this portable reference.
