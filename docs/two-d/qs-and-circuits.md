# QS completion and QCA circuit construction

Quillen–Suslin completion is implemented in this package. It is a different
operation from elementary factorization of a physical QCA matrix. Neither
operation is an unimplemented external-CAS oracle.

The unified default circuit route is the original-register construction in
[unstabilized-factorization.md](unstabilized-factorization.md), followed by
the [target-code correction](no-input-ancilla.md). The stable routes described
below remain explicit lower-level alternatives; they are not an automatic
fallback for an incomplete no-input-ancilla request.

| Operation | Implementation | Checked output |
| --- | --- | --- |
| Complete a unimodular Laurent row | `algebra/quillen_suslin.py:complete_laurent_row` | `row*U=e1`, `UV=VU=I` |
| Remove redundant check generators | `algebra.py:independent_columns` and `_split_injection` | An independent image basis and both directions of generation from original checks |
| Construct the paper's QCA | `qca.py:construct_qca` | Corrected chain maps, their Laurent inverses and physical X/Z action |
| Factor on the original register | `unstable.py`, with the bounded code-specific `rank_two.py` path | Actual fixed-size factors and explicit unit transport; incomplete cases retain a reason |
| Construct a stable elementary word | `suslin.py:factor_suslin`, reached by `elementary.py` | Actual factors of the declared extended QCA with explicit transport |
| Compile factors to physical gates | `direct_circuits.py` | Actual weighted SUM streams, signed inverses, ancilla and Pauli identities |

The QS routine uses Laurent normalization, ordinary polynomial completion,
monic changes, local Horrocks reduction, closed points and denominator
patching. It has unit, two-entry and one-variable shortcuts, but its general
path is also implemented. The redundancy stage invokes this routine when it
must cancel terminal relations in a finite free resolution. This is a module
basis problem, not a request to decompose an arbitrary QCA into gates.

For the explicit stable `compile_qca_direct` interface, `factorization="auto"` uses exact fast reductions
and passes a remaining problem to the implemented stable Higman/PID route.
`factorization="higman"` forces that general stable route. Only the explicit
`factorization="fast"` option stops when its incomplete heuristic reductions
cannot continue.

```python
from css_decoupling.two_d import compile_qca_direct

result = compile_qca_direct(qca_result, factorization="auto", transport="balance")
general = compile_qca_direct(qca_result, factorization="higman", transport="balance")
```

The separate retained compiler uses the explicit six-shear identity for
`diag(U,U_inverse)` and adds q zero-state qudits. The stable elementary route
instead introduces the identity stabilization it needs and, under `balance`,
balances the full determinant unit on an added zero-state qudit if necessary.
Under `separate`, any scalar-monomial transport remains explicit and is not
counted as a gate layer. Both routes retain exact physical action, rather
than silently replacing the QCA by a map agreeing only on the code space.

The retained stable interfaces do not promise an unstabilized factorization,
minimum ancillas, minimum depth or completion within every finite resource
budget. A resource stop is an unfinished computation, not a proof that QS or
the circuit does not exist. The default stable route's completeness claim is
limited to its stated two-variable prime-field construction, not every
possible Suslin or no-ancilla compilation problem.

All runtime algorithms use native CPython and SymPy. Historical Sage/Julia
authorship and comparison records remain attributed to their sources; they
are not optional executable backends. See [original-register factorization](unstabilized-factorization.md),
[prime-field factorization](prime-factorization.md) and
[actual gate and transport contracts](prime-circuits.md).
