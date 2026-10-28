# Unified API and file contracts

The public distribution is `css-decoupling`. Both algorithms are installed
under `css_decoupling`; the dedicated implementation is an internal subpackage
`css_decoupling.two_d`. Start with the shared public `Ring` and `Matrix` types;
both algorithms use them and the same exact algebra modules.

## Entry point

```python
decouple(
    HX, HZ, dimension=None, p=None,
    *, algorithm="auto", output="qca",
    lattice_mode="auto", lattice_basis=None, options=None,
)
```

Dimension and prime are inferred from the matrices when omitted. Explicit
values must agree with their ring; neither argument changes the coefficient
field. `HX` and `HZ` share that ring and their physical column count.

`auto` chooses `two_d` for D=2 and `general` for D>2. Explicit `two_d` rejects
other dimensions; explicit `general` also works in D=2. The selected algorithm
is recorded in every result. A failed specialized circuit request does not
silently switch to the general stable algorithm.

`output` is `qca` or `circuit`, defaulting to `qca`. In the general algorithm,
the underlying construction currently produces its stable circuit even when
only a QCA view is requested. This selection controls the public output, not
a claimed optimization of every internal stage.

## Lattices

`lattice_mode="auto"` accepts no basis. `"fixed"` requires a full-rank
D-by-D integer `lattice_basis`, with basis vectors as columns. For example,
`[[2,1],[0,1]]` has basis vectors `(2,0)` and `(1,1)`.

The two-dimensional fixed mode checks the actual actions of those translations
on both anyon modules without first computing the full translation kernel.
It preserves the requested basis. An obstruction or a resource limit is not
permission to refine the lattice. A larger internal gate-period construction
is not an automatic fallback.

For the general algorithm, automatic-lattice evidence certifies the actual
profile/gluing refinement starting from the complete translation kernel. It
does not claim a globally smallest lattice in every dimension.

## Options and resources

`options` is an algorithm-specific dictionary; unknown or incompatible options
are rejected rather than silently discarded. Common limits include
`max_cell_index` and `max_layer_dimension`. The specialized path additionally
accepts `max_matrix_size` and `max_terms`. Circuit settings belong inside
`circuit_options`, for example:

```python
result = decouple(
    HX, HZ, algorithm="two_d", output="circuit",
    lattice_mode="fixed", lattice_basis=[[1,1],[0,1]],
    options={
        "max_cell_index": 4096,
        "circuit_options": {
            "factorization": "auto",
            "timeout": 120,
            "max_families": 200000,
            "rank_two_options": {"search_depth": 20, "max_nodes": 100000},
        },
    },
)
```

The rank-two search options are `search_width`, `search_depth`, `max_nodes`,
`max_quotient_steps` and `max_quotient_terms`. They bound an explicit procedure;
they do not turn unsuccessful search into an algebraic nonexistence result.
Other circuit limits include `max_steps`, `max_matrix_terms`, `max_factors`,
`max_products`, `max_patches`, `max_field_trials` and `seed`.
`max_factors` and `max_matrix_terms` must be positive; `max_families` may be
zero to impose a zero-gate budget.

The general route accepts its documented construction options, including
native `backend="sympy"`, `resolution_method="native"`, and the separate
Algorithm 2 choice `linear_backend="sympy"` or `"bounded"`. Bounded linear
solving is not an external backend and does not silently fall back to Gröbner
solving on failure. See [general reference](general-reference.md).

## Results and independent verification

`UnifiedResult` exposes `algorithm`, `status`, `qca`, `circuit`, `diagnostics`
and `backend_evidence`, plus `.certificate()` for its unified envelope.

| Status | Meaning |
| --- | --- |
| `qca_ready` | The requested QCA is available with its source evidence |
| `circuit_ready` | The requested circuit and its relationship to the QCA are available |
| `circuit_incomplete` | A verified QCA is retained, but the requested circuit did not complete |

For an incomplete circuit, `circuit` is `None`. Diagnostics retain the reason
and, when available, the phase and remaining-problem details. Invalid input,
certified obstructions and resource failures before a QCA exists are separate
failures; the table is not a claim that every input returns a ready result.

The shared `qca` describes the initial physical Pauli action. For a specialized
code-equivalent circuit, `circuit.action` describes the actual gate action and
`circuit.relation_to_qca` distinguishes `exact` from `code_equivalent`.
The latter is certified through a target-code automorphism, not assumed to
equal the initial matrix. The general QCA includes its declared stable
registers and ancillas.

```python
from css_decoupling import save_result, load_result, verify_result

report = verify_result(result)
assert report.valid, report.errors
save_result(result, "out/result.json")
loaded = load_result("out/result.json")  # Complete independent verification by default.
assert verify_result(loaded, expected_HX=HX, expected_HZ=HZ).valid
```

The new envelope uses schema `css-decoupling.result`. It binds the public
view to algorithm-specific evidence; it does not reinterpret old certificates.
Existing lower-level constructors and readers remain explicitly available.
New unified writes reject existing destinations and final symlinks and create
required parent directories. Use fresh paths for every run.

## Portable raw input and CLI

```python
from css_decoupling import save_css_input, load_css_input, css_input_dict

save_css_input(HX, HZ, "out/input.json", lattice_mode="auto")
raw = load_css_input("out/input.json")
assert raw["dimension"] == HX.ring.D
```

New raw exports use schema `css-decoupling.input`, version 1, with explicit
`ring.prime`, `ring.dimension`, sparse `HX`/`HZ`, `lattice_mode` and
`lattice_basis`. Matrix entries carry exact finite-field coefficients and
integer exponent vectors. Generate this format with the exporter instead
of evaluating polynomial strings. The loader also imports the former raw
general and two-dimensional formats with domain checks.

```sh
css-decouple run out/input.json --algorithm two_d --format qca --output out/qca.json
css-decouple run out/input.json --algorithm general --format circuit --output out/general.json
css-decouple verify out/qca.json --input out/input.json
css-decouple general --help
```

CLI options may come from `--options` or `--options-file`; duplicate keys and
nonfinite numbers are rejected. `--circuit-options` supplies the nested
two-dimensional settings. New circuit incompletion uses exit status 3 while
preserving the unified QCA result. The explicit `general` subcommand retains
the older general CLI and its advanced options and formats.

## Rank-two scope

Two physical qudits per retained cell do not mean that the matrices are
already the canonical square-toric checks. Some noncanonical presentations
compile successfully; others remain incomplete under current limits.
The implementation may use a checked same-code representative, but does not
apply a rank-at-least-three Suslin result to rank two, infer non-elementarity
from a stalled heuristic, add input ancillas, or enlarge the gate period
without an explicit different contract. An odd-prime resource stop is not
an unsupported coefficient field.
