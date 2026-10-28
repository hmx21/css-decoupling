"""Certificates for elementary factorizations and separate Laurent-unit transport.

This module is not a Quillen--Suslin or Suslin factorization algorithm.  It
compiles a supplied chronological elementary list and checks it independently
against actual weighted SUM families. Over F_p[x^+-1,y^+-1], it certifies W=S*F,
where W=diag(U,B), F is the actual finite-depth circuit, and S is a separately
declared diagonal Laurent-unit QCA, including every nonzero scalar. S is
never encoded as a SWAP gate/layer. Binary certificates retain their v1 form.

The added register consists of a Z-product state. B need not be identity.
Its action under F is S_ancilla^-1*B, checked in both directions. Only S=I
permits the claim that the finite-depth circuit implements W exactly.
If only ancillas carry separate shifts, F still implements the original U
on the system with its Z-product ancillas returned to zero. The certificate
distinguishes these system and extended-matrix implementation claims.
Full original/target stabilizer binding belongs to the caller's code wrapper.
"""
from __future__ import annotations

from .algebra import Matrix, Ring, block_diag, eye, zeros
from .circuits import CircuitResourceLimit, _validate_layer, _apply_gate_rows


SCHEMA = "css-decoupling-2d-direct-circuit/v1"
PRIME_SCHEMA = "css-decoupling-2d-direct-circuit/v2"
_FACTOR_KEYS = {"row", "column", "coefficient"}
_KEYS = {"schema", "field_characteristic", "dimension", "mode", "original_qca",
         "original_inverse", "extended_qca", "extended_inverse", "ancilla_action",
         "ancilla_inverse", "transport", "factors", "layers", "z_action",
         "z_inverse", "x_action", "x_inverse", "ancillas", "equivalence",
         "resources"}


def _integer(value, label, minimum=None):
    if type(value) is not int or (minimum is not None and value < minimum):
        raise ValueError(f"{label} must be an exact integer" +
                         (f" >= {minimum}" if minimum is not None else ""))
    return value


def _equal_metadata(actual, expected):
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            _equal_metadata(actual[k], v) for k, v in expected.items())
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _equal_metadata(a, b) for a, b in zip(actual, expected))
    return actual == expected


def _terms(matrix):
    return sum(len(value.terms) for value in matrix.entries.values())


def _bounded(matrix, cap):
    if _terms(matrix) > cap:
        raise CircuitResourceLimit("Direct circuit matrix monomial budget exceeded")
    return matrix


def _ring(ring):
    if not isinstance(ring, Ring) or ring.D != 2:
        raise ValueError("Direct circuits require a two-dimensional prime ring")


def _inverse_pair(A, B, cap):
    if (not isinstance(A, Matrix) or not isinstance(B, Matrix) or
            A.ring != B.ring or A.shape != B.shape or A.nrows != A.ncols):
        raise ValueError("Expected square Laurent matrices with a common ring and shape")
    _ring(A.ring)
    _bounded(A, cap); _bounded(B, cap)
    if A.nrows > cap:
        raise CircuitResourceLimit("Direct circuit identity monomial budget exceeded")
    identity = eye(A.ring, A.nrows)
    if _bounded(A*B, cap) != identity or _bounded(B*A, cap) != identity:
        raise ValueError("Direct circuit requires actual two-sided inverse identities")


def elementary_factor(row, column, coefficient):
    """Return E_(row,column)(coefficient), with a serializable 1x1 Matrix.

    Its row is physical SUM control, its column physical target. A monomial
    coefficient a is realized with physical weight -a. Matrix
    action is on support columns; a later factor multiplies on the left.
    A zero coefficient is allowed but still requires distinct species.
    """
    _integer(row, "factor row", 0); _integer(column, "factor column", 0)
    if row == column:
        raise ValueError("An elementary factor requires distinct species")
    if not isinstance(coefficient, Matrix) or coefficient.shape != (1, 1):
        raise ValueError("An elementary coefficient must be a 1x1 Laurent Matrix")
    _ring(coefficient.ring)
    return {"row": row, "column": column, "coefficient": coefficient.copy()}


def _factor(ring, n, factor):
    if not isinstance(factor, dict) or set(factor) != _FACTOR_KEYS:
        raise ValueError("Malformed elementary factor")
    row = _integer(factor["row"], "factor row", 0)
    col = _integer(factor["column"], "factor column", 0)
    coefficient = factor["coefficient"]
    if row == col or max(row, col) >= n:
        raise ValueError("Elementary factor species are equal or out of range")
    if (not isinstance(coefficient, Matrix) or coefficient.shape != (1, 1) or
            coefficient.ring != ring):
        raise ValueError("Elementary coefficient matrix has the wrong ring or shape")
    return row, col, coefficient[0, 0]


def elementary_product(ring, qubits, factors, *, reverse=False,
                       max_factors=200_000, max_matrix_terms=1_000_000):
    """Replay supplied polynomial elementary factors by exact row updates.

    This does not use a factorization trace, the gate expansion or a declared
    target matrix. Reverse order and negated coefficients give the inverse.
    Budgets bound the supplied list and retained supports, not every temporary
    allocation or elapsed time; external process limits remain appropriate.
    """
    _ring(ring)
    n = _integer(qubits, "qubits", 0)
    max_factors = _integer(max_factors, "max_factors", 0)
    cap = _integer(max_matrix_terms, "max_matrix_terms", 0)
    if type(reverse) is not bool:
        raise ValueError("reverse must be boolean")
    if not isinstance(factors, list):
        raise ValueError("Elementary factors must be a chronological list")
    if len(factors) > max_factors or n > cap:
        raise CircuitResourceLimit("Elementary factor or identity budget exceeded")
    rows = [{i: ring.one()} for i in range(n)]
    count = n
    for item in reversed(factors) if reverse else factors:
        i, j, a = _factor(ring, n, item)
        if reverse:
            a = -a
        if len(a.terms) > cap:
            raise CircuitResourceLimit("Elementary coefficient monomial budget exceeded")
        # i != j guarantees that iteration does not mutate its source row.
        for col, value in rows[j].items():
            previous = rows[i].get(col, ring.zero())
            updated = previous + a*value
            count += len(updated.terms)-len(previous.terms)
            if updated:
                rows[i][col] = updated
            else:
                rows[i].pop(col, None)
            if count > cap:
                raise CircuitResourceLimit("Elementary product monomial budget exceeded")
    return Matrix(ring, n, n, {(i,j):f for i,row in enumerate(rows) for j,f in row.items()})


def direct_gate_product(ring, qubits, layers, *, pauli="Z", reverse=False,
                        max_families=200_000, max_matrix_terms=1_000_000):
    """Replay the actual supplied SUM stream with sparse row operations.

    Gate and infinite-lattice layer validation are the same as the original
    compiler. This routine neither uses elementary factors nor regenerates a
    compiler recipe. It maintains the exact retained monomial count by row
    deltas, avoiding a full sparse-matrix product/reindex for each dependent
    gate group. All four chronological/reversed X/Z modes are supported.

    The term cap applies after each actual gate, including intermediate gates
    which later cancel. It is therefore a resource cap on this replay, not the
    original grouped algorithm's different retained-state cap. No search
    failure or cap exhaustion means that an elementary factorization is absent.
    """
    _ring(ring)
    n = _integer(qubits, "qubits", 0)
    family_cap = _integer(max_families, "max_families", 0)
    term_cap = _integer(max_matrix_terms, "max_matrix_terms", 0)
    if pauli not in ("X", "Z") or type(reverse) is not bool:
        raise ValueError("Pauli mode must be X/Z and reverse must be boolean")
    if not isinstance(layers, list):
        raise ValueError("Actual gate layers must be a list")
    count = 0
    for layer in layers:
        _validate_layer(ring, n, layer)
        count += len(layer)
        if count > family_cap:
            raise CircuitResourceLimit("Direct CNOT-family verification budget exceeded")
    if n > term_cap:
        raise CircuitResourceLimit("Direct gate identity monomial budget exceeded")
    rows = [{i:ring.one()} for i in range(n)]
    retained_terms = n
    for layer in reversed(layers) if reverse else layers:
        for gate in reversed(layer) if reverse else layer:
            retained_terms += _apply_gate_rows(ring, rows, gate, pauli, reverse=reverse)
            if retained_terms > term_cap:
                raise CircuitResourceLimit("Direct gate intermediate monomial budget exceeded")
    return Matrix(ring, n, n, {(i,j):f for i,row in enumerate(rows) for j,f in row.items()})


def _shifts(ring, n, exponents, coefficients=None):
    if not isinstance(exponents, list) or len(exponents) != n:
        raise ValueError("One shift exponent pair is required per circuit species")
    values = []
    for exponent in exponents:
        if not isinstance(exponent, list) or len(exponent) != 2:
            raise ValueError("Shift exponents must be two-integer lists")
        values.append([_integer(e, "shift exponent") for e in exponent])
    coefficients = [1] * n if coefficients is None else coefficients
    if (not isinstance(coefficients, list) or len(coefficients) != n or
            any(type(c) is not int or not 0 < c < ring.p for c in coefficients)):
        raise ValueError("Transport requires one canonical nonzero field coefficient per species")
    S = Matrix(ring, n, n, {(i,i):ring.monomial(tuple(e), coefficients[i]) for i,e in enumerate(values)})
    Sinv = Matrix(ring, n, n, {(i,i):ring.monomial(tuple(-k for k in e), pow(coefficients[i], -1, ring.p))
                              for i,e in enumerate(values)})
    return values, S, Sinv


def _transport(exponents, S, Sinv):
    result = {"kind": "diagonal_species_translation", "application": "after_circuit",
            "exponents": exponents, "z_action": S, "z_inverse": Sinv,
            "x_action": Sinv.dagger(), "x_inverse": S.dagger(),
            "constant_unit": 1,
            "determinant_exponent": [sum(e[k] for e in exponents) for k in range(2)],
            "included_in_gate_layers": False}
    if S.ring.p != 2:
        coefficients = [next(iter(S[i,i].terms.values())) for i in range(S.nrows)]
        constant = 1
        for c in coefficients:
            constant = constant*c % S.ring.p
        result.update(kind="diagonal_species_laurent_unit", coefficients=coefficients,
                      constant_unit=constant)
    return result


def _equivalence_labels(ring, exact):
    if exact:
        return "exact", "exact_extended_qca"
    if ring.p == 2:
        return "shift_separated", "shift_separated_extended_qca"
    return "transport_separated", "unit_transport_separated_extended_qca"


def _transport_identity(S, count):
    return all(S[i,i] == 1 for i in range(count))


def _geometry(ring, q, a):
    n = q+a
    J, K = zeros(ring, n, q), zeros(ring, n, a)
    for i in range(q):
        J[i,i] = 1
    for i in range(a):
        K[q+i,i] = 1
    return J, K


def _resources(q, a, factors, layers, exponents, S):
    count = 0
    range_l1 = range_linf = 0
    for layer in layers:
        for gate in layer:
            count += 1
            e = gate["offset"]
            range_l1 = max(range_l1, sum(abs(k) for k in e))
            range_linf = max(range_linf, max(map(abs, e)))
    result = {"original_qubits_per_cell":q, "added_Z_ancillas_per_cell":a,
            "added_X_ancillas_per_cell":0, "circuit_qubits_per_cell":q+a,
            "elementary_factor_count":len(factors), "depth":len(layers),
            "gate_family_count":count, "gates_per_cell":count,
            "range_l1_in_supercell_coordinates":range_l1,
            "range_linf_in_supercell_coordinates":range_linf,
            "translation_subgroup_in_supercell_coordinates":[[1,0],[0,1]],
            "scheduling":"species_disjoint_translation_families",
            "separate_shift_species_count":sum(any(e) for e in exponents),
            "separate_shift_range_l1":max((sum(map(abs,e)) for e in exponents), default=0)}
    if S.ring.p != 2:
        result["separate_unit_species_count"] = sum(S[i,i] != 1 for i in range(S.nrows))
        result["separate_scaling_species_count"] = sum(
            next(iter(S[i,i].terms.values())) != 1 for i in range(S.nrows))
    return result


def compile_direct_circuit(U, V, factors, *, added_ancillas=0,
                           ancilla_action=None, ancilla_inverse=None,
                           shift_exponents=None, transport_coefficients=None, max_factors=200_000,
                           max_families=200_000, max_matrix_terms=1_000_000):
    """Compile supplied factors; success is conditional on their actual product.

    W=diag(U,B)=S*F is required. S has entries c_i*x^e_i, supplied by
    transport_coefficients and shift_exponents. No search or general
    elementary-factorization theorem is invoked. A nontrivial S remains
    separate QCA transport even when its total determinant is one.
    """
    cap = _integer(max_matrix_terms, "max_matrix_terms", 0)
    family_cap = _integer(max_families, "max_families", 0)
    _inverse_pair(U, V, cap)
    a = _integer(added_ancillas, "added_ancillas", 0)
    q, ring = U.nrows, U.ring
    if q+a > cap:
        raise CircuitResourceLimit("Direct circuit identity monomial budget exceeded")
    if (ancilla_action is None) != (ancilla_inverse is None):
        raise ValueError("Supply both ancilla action and its inverse")
    B = eye(ring,a) if ancilla_action is None else ancilla_action
    Binv = eye(ring,a) if ancilla_inverse is None else ancilla_inverse
    _inverse_pair(B, Binv, cap)
    if B.ring != ring or B.shape != (a,a):
        raise ValueError("Ancilla count/ring does not match its actual action")
    n = q+a
    exponents, S, Sinv = _shifts(ring, n, [[0,0] for _ in range(n)] if shift_exponents is None else shift_exponents,
                               transport_coefficients)
    F = elementary_product(ring,n,factors,max_factors=max_factors,max_matrix_terms=cap)
    W, Winv = block_diag(U,B), block_diag(V,Binv)
    if _bounded(S*F,cap) != W:
        raise ValueError("Supplied elementary factors do not satisfy extended_qca = shift * circuit")
    normalized = []
    layers = []
    for item in factors:
        row,col,coefficient = _factor(ring,n,item)
        normalized.append(elementary_factor(row,col,item["coefficient"]))
        for exponent, value in sorted(coefficient.terms.items()):
            if len(layers) >= family_cap:
                raise CircuitResourceLimit("Direct CNOT family budget exceeded")
            layers.append([{"kind":"CNOT" if ring.p == 2 else "SUM", "control":row, "target":col,
                            "offset":list(exponent), "coefficient":-value % ring.p}])
    Finv = _bounded(Winv*S,cap)
    J,K = _geometry(ring,q,a)
    # S is diagonal: its ancilla restriction preserves the coordinate split.
    Sa, Sainv = S[q:,q:], Sinv[q:,q:]
    C, Cinv = _bounded(Sainv*B,cap), _bounded(Binv*Sa,cap)
    exact = _transport_identity(S, n)
    system_exact = _transport_identity(S, q)
    mode, kind = _equivalence_labels(ring, exact)
    result = {"schema":SCHEMA if ring.p == 2 else PRIME_SCHEMA, "field_characteristic":ring.p, "dimension":2,
        "mode":mode,
        "original_qca":U.copy(), "original_inverse":V.copy(),
        "extended_qca":W, "extended_inverse":Winv,
        "ancilla_action":B.copy(), "ancilla_inverse":Binv.copy(),
        "transport":_transport(exponents,S,Sinv), "factors":normalized, "layers":layers,
        "z_action":F, "z_inverse":Finv, "x_action":Finv.dagger(), "x_inverse":F.dagger(),
        "ancillas":{"type":"Z", "state":"zero", "count":a, "z_generators":K,
                    "forward_generator_change":C, "backward_generator_change":Cinv},
        "equivalence":{"kind":kind,
                       "equation":"extended_qca = shift * z_action" if ring.p == 2 else "extended_qca = transport * z_action",
                       "original_qca_implemented_by_circuit":system_exact,
                       "extended_qca_implemented_by_circuit":exact,
                       "input_injection":J, "output_projection":J.transpose()},
        "resources":_resources(q,a,normalized,layers,exponents,S)}
    verify_direct_circuit(result,max_factors=max_factors,max_families=family_cap,max_matrix_terms=cap)
    return result


def verify_direct_circuit(bundle, *, max_factors=200_000, max_families=200_000,
                          max_matrix_terms=1_000_000):
    """Verify the supplied factors and actual gate stream without a factorizer.

    This core verifier does not certify original HX/HZ or their stabilizers.
    Its exact equation lets a separate code verifier check the actual F on
    shift-twisted targets: S^-1 on Z supports and S^dagger on X supports.
    """
    cap = _integer(max_matrix_terms, "max_matrix_terms", 0)
    _integer(max_factors,"max_factors",0); _integer(max_families,"max_families",0)
    if not isinstance(bundle,dict) or set(bundle) != _KEYS or bundle["schema"] not in (SCHEMA, PRIME_SCHEMA):
        raise ValueError("Unsupported or malformed direct circuit certificate")
    U,V = bundle["original_qca"],bundle["original_inverse"]
    _inverse_pair(U,V,cap)
    q,ring = U.nrows,U.ring
    if (bundle["schema"] != (SCHEMA if ring.p == 2 else PRIME_SCHEMA) or
            not _equal_metadata([bundle["field_characteristic"],bundle["dimension"]],[ring.p,2])):
        raise ValueError("Direct circuit domain metadata or schema differs")
    ancillas = bundle["ancillas"]
    if not isinstance(ancillas,dict) or set(ancillas) != {
            "type","state","count","z_generators","forward_generator_change","backward_generator_change"}:
        raise ValueError("Malformed direct ancilla certificate")
    a = _integer(ancillas["count"],"ancilla count",0)
    n = q+a
    if n > cap:
        raise CircuitResourceLimit("Direct circuit identity monomial budget exceeded")
    B,Binv = bundle["ancilla_action"],bundle["ancilla_inverse"]
    _inverse_pair(B,Binv,cap)
    if B.ring != ring or B.shape != (a,a):
        raise ValueError("Ancilla action shape/ring differs")
    transport = bundle["transport"]
    if not isinstance(transport,dict) or "exponents" not in transport:
        raise ValueError("Missing explicit shift transport")
    exponents,S,Sinv = _shifts(ring,n,transport["exponents"],
                             transport.get("coefficients") if ring.p != 2 else None)
    if not _equal_metadata(transport,_transport(exponents,S,Sinv)):
        raise ValueError("Shift transport metadata or actual action differs")
    exact = _transport_identity(S, n)
    system_exact = _transport_identity(S, q)
    mode, kind = _equivalence_labels(ring, exact)
    if bundle["mode"] != mode:
        raise ValueError("Nontrivial species transport cannot be called an exact circuit")
    W,Winv = _bounded(block_diag(U,B),cap),_bounded(block_diag(V,Binv),cap)
    if bundle["extended_qca"] != W or bundle["extended_inverse"] != Winv:
        raise ValueError("Extended QCA differs from its actual ancilla action")
    F,Finv = bundle["z_action"],bundle["z_inverse"]
    for matrix in (F,Finv):
        if (not isinstance(matrix,Matrix) or matrix.ring != ring or
                matrix.shape != (n,n)):
            raise ValueError("Actual circuit action/inverse has wrong type, dimension or domain")
        _bounded(matrix,cap)
    if _bounded(S*F,cap) != W or _bounded(Finv*Sinv,cap) != Winv:
        raise ValueError("Exact shift/QCA/circuit equations failed")
    if bundle["x_action"] != Finv.dagger() or bundle["x_inverse"] != F.dagger():
        raise ValueError("Physical X action is not the inverse dagger of the actual Z action")
    # U/V and B/Binv were checked in both directions, and W/Winv are their
    # actual block diagonals. S/Sinv are constructed opposite monomial shifts.
    # The two checked equations imply F=Sinv*W and Finv=Winv*S; consequently
    # F*Finv=Sinv*W*Winv*S=I and Finv*F=Winv*S*Sinv*W=I. Expanding those two
    # products again adds no condition. The declared daggers therefore obey
    # both Pauli pairing identities. Pure SUM X/Z strings acquire no phases.
    factors,layers = bundle["factors"],bundle["layers"]
    if elementary_product(ring,n,factors,max_factors=max_factors,max_matrix_terms=cap) != F:
        raise ValueError("Actual elementary factor product differs from the circuit matrix")
    if direct_gate_product(ring,n,layers,max_families=max_families,max_matrix_terms=cap) != F:
        raise ValueError("Actual CNOT gate product differs from the circuit matrix")
    # For odd primes explicitly replay the signed inverse and both physical
    # Pauli streams. Legacy qubit verification retains its established cost.
    if ring.p != 2:
        if elementary_product(ring,n,factors,reverse=True,max_factors=max_factors,max_matrix_terms=cap) != Finv:
            raise ValueError("Inverse elementary product differs from the circuit inverse")
        for pauli, reverse, expected in (("Z",True,Finv), ("X",False,bundle["x_action"]),
                                         ("X",True,bundle["x_inverse"])):
            if direct_gate_product(ring,n,layers,pauli=pauli,reverse=reverse,
                                   max_families=max_families,max_matrix_terms=cap) != expected:
                raise ValueError("Actual signed SUM stream differs from the physical Pauli action")
    # No compiler factor trace or regenerated expected gate list is trusted.
    J,K = _geometry(ring,q,a)
    C,Cinv = Sinv[q:,q:]*B, Binv*S[q:,q:]
    expected_ancillas = {"type":"Z","state":"zero","count":a,"z_generators":K,
                        "forward_generator_change":C,"backward_generator_change":Cinv}
    if not _equal_metadata(ancillas,expected_ancillas):
        raise ValueError("Actual circuit ancilla stabilizer witnesses differ")
    if (_bounded(F*K,cap) != K*C or _bounded(Finv*K,cap) != K*Cinv or
            _bounded(C*Cinv,cap) != eye(ring,a) or _bounded(Cinv*C,cap) != eye(ring,a)):
        raise ValueError("Added Z ancillas fail bidirectional stabilizer generation")
    expected_equivalence = {"kind":kind,
        "equation":"extended_qca = shift * z_action" if ring.p == 2 else "extended_qca = transport * z_action",
        "original_qca_implemented_by_circuit":system_exact,
        "extended_qca_implemented_by_circuit":exact,
        "input_injection":J,"output_projection":J.transpose()}
    if not _equal_metadata(bundle["equivalence"],expected_equivalence):
        raise ValueError("QCA/circuit equivalence scope or subsystem identification differs")
    if W*J != J*U or Winv*J != J*V or J.transpose()*J != eye(ring,q):
        raise ValueError("Extended action does not restrict to the original QCA")
    if system_exact and (F*J != J*U or Finv*J != J*V or
                         bundle["x_action"]*J != J*V.dagger() or
                         bundle["x_inverse"]*J != J*U.dagger()):
        raise ValueError("The actual circuit does not implement its claimed original system action")
    if not _equal_metadata(bundle["resources"],_resources(q,a,factors,layers,exponents,S)):
        raise ValueError("Direct circuit resource metadata differs from actual gates")
    return True
