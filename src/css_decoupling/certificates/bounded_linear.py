"""Independent checks of bounded linear and translation-trivial layer evidence."""
from __future__ import annotations

from itertools import combinations_with_replacement
from math import comb

from css_decoupling.algebra.laurent import Matrix, Ring, eye, zeros
from css_decoupling.certificates.comparison import (
    VerificationReport, _int, _keys, _matrix_from_dict,
)
from css_decoupling.errors import ResourceLimit


def _dense_eliminate(rows, width, p, rhs=None):
    """Verifier's dense forward/backward Gaussian elimination."""
    extra = 0 if rhs is None else max((max(row, default=-1) for row in rhs), default=-1) + 1
    data = [[row.get(j, 0) % p for j in range(width)] +
            ([] if rhs is None else [rhs[i].get(j, 0) % p for j in range(extra)])
            for i, row in enumerate(rows)]
    rank, pivots = 0, []
    for column in range(width):
        pivot = next((i for i in range(rank, len(data)) if data[i][column]), None)
        if pivot is None:
            continue
        data[rank], data[pivot] = data[pivot], data[rank]
        inverse = pow(data[rank][column], -1, p)
        data[rank] = [v * inverse % p for v in data[rank]]
        for i, row in enumerate(data):
            if i == rank or not row[column]:
                continue
            factor = row[column]
            data[i] = [(a - factor * b) % p for a, b in zip(row, data[rank])]
        pivots.append(column)
        rank += 1
        if rank == len(data):
            break
    consistent = all(not any(row[width:]) for row in data[rank:])
    return data, pivots, consistent


def _constant_rank(A):
    rows = [dict() for _ in range(A.nrows)]
    for (i, j), f in A.entries.items():
        if not f.is_constant:
            raise ValueError("Finite-field witness contains a nonconstant entry")
        rows[i][j] = f.constant()
    return len(_dense_eliminate(rows, A.ncols, A.ring.p)[1])


def _augment(A):
    return Matrix(A.ring, A.nrows, A.ncols,
                  {ij: sum(poly.terms.values()) for ij, poly in A.entries.items()})


def _lifted_coefficients(A, B, degree, caps):
    """Independent column-first assembly of the localization coefficient map."""
    variables, D = 2 * A.ring.D, A.ring.D
    d = max([2] + [sum(map(abs, exponent)) for M in (A, B) if M is not None
                   for f in M.entries.values() for exponent in f.terms])
    width = (A.ncols + A.nrows * D) * comb(degree + variables, variables)
    height = A.nrows * comb(degree + d + variables, variables)
    for count, limit, label in (
            (width, caps["max_unknowns"], "unknowns"),
            (height, caps["max_equations"], "equations"),
            (height * (width + (0 if B is None else B.ncols)),
             caps["max_matrix_entries"], "elimination entries")):
        if limit is not None and count > limit:
            raise ResourceLimit(f"Independent bounded verification needs {count} {label}; cap is {limit}")

    def monomials(max_degree):
        result = []
        for size in range(max_degree + 1):
            for multiset in combinations_with_replacement(range(variables), size):
                powers = [0] * variables
                for variable in multiset:
                    powers[variable] += 1
                result.append(tuple(powers))
        return sorted(result)

    unknown_monomials = monomials(degree)
    equation_monomials = monomials(degree + d)
    lookup = {powers: i for i, powers in enumerate(equation_monomials)}
    height_per_row = len(equation_monomials)
    rows = [dict() for _ in range(height)]
    rhs = [dict() for _ in range(height)]
    for component in range(A.ncols + A.nrows * D):
        if component < A.ncols:
            terms = [(i, tuple(max(a, 0) for a in exponent) +
                      tuple(max(-a, 0) for a in exponent), coefficient)
                     for (i, j), f in A.entries.items() if j == component
                     for exponent, coefficient in f.terms.items()]
        else:
            i, axis = divmod(component - A.ncols, D)
            powers = [0] * variables
            powers[axis] = powers[D + axis] = 1
            terms = [(i, tuple(powers), 1), (i, (0,) * variables, -1)]
        for offset, monomial in enumerate(unknown_monomials):
            column = component * len(unknown_monomials) + offset
            for i, powers, coefficient in terms:
                exponent = tuple(a + b for a, b in zip(powers, monomial))
                row = i * height_per_row + lookup[exponent]
                rows[row][column] = (rows[row].get(column, 0) + coefficient) % A.ring.p
    if B is not None:
        for (i, j), f in B.entries.items():
            for exponent, coefficient in f.terms.items():
                powers = tuple(max(a, 0) for a in exponent) + tuple(max(-a, 0) for a in exponent)
                rhs[i * height_per_row + lookup[powers]][j] = coefficient
    return rows, rhs, unknown_monomials, width


def _project_nullspace(A, monomials, vectors):
    entries = {}
    for column, vector in enumerate(vectors):
        for position, coefficient in vector.items():
            component, offset = divmod(position, len(monomials))
            if component >= A.ncols:
                continue
            powers = monomials[offset]
            exponent = tuple(powers[i] - powers[i + A.ring.D] for i in range(A.ring.D))
            terms = entries.setdefault((component, column), {})
            terms[exponent] = (terms.get(exponent, 0) + coefficient) % A.ring.p
    result = Matrix(A.ring, A.ncols, len(vectors), entries)
    nonzero_columns = sorted({j for _, j in result.entries})
    return result[:, nonzero_columns]


def _kernel_check(data, *, caps):
    from css_decoupling.algebra.bounded_linear import _bound_within, degree_bound
    _keys(data, ("schema", "version", "operation", "ring", "A", "B", "result",
                 "bound", "method", "evidence"), "bounded linear")
    if data["schema"] != "css-decoupling.bounded-linear" or _int(data["version"], "version") != 1:
        raise ValueError("Unsupported bounded linear certificate")
    _keys(data["ring"], ("prime", "dimension"), "ring")
    ring = Ring(_int(data["ring"]["prime"], "prime", 2),
                _int(data["ring"]["dimension"], "dimension", 1))
    A = _matrix_from_dict(data["A"], ring, "A")
    B = None if data["B"] is None else _matrix_from_dict(data["B"], ring, "B")
    result = None if data["result"] is None else _matrix_from_dict(data["result"], ring, "result")
    if B is not None and B.nrows != A.nrows:
        raise ValueError("Linear certificate row counts disagree")
    if data["bound"] != degree_bound(A, B):
        raise ValueError("The recorded theorem bound does not match the polynomial lift")
    method, operation, evidence = data["method"], data["operation"], data["evidence"]
    if not isinstance(evidence, dict):
        raise ValueError("Linear evidence must be a mapping")
    if operation == "solve":
        if B is None or result is None or A * result != B:
            raise ValueError("Laurent solution does not satisfy A X=B")
        # A concrete polynomial solution is complete positive evidence; no
        # completeness or search-history claim is needed for this operation.
        if method not in ("zero-rhs", "verified-solution", "bounded-coefficients"):
            raise ValueError("Unknown solution evidence method")
        if method == "bounded-coefficients":
            degree = _int(evidence["degree"], "searched degree", 0)
            if caps["max_degree"] is not None and degree > caps["max_degree"]:
                raise ResourceLimit("Solution coefficient degree exceeds verification budget")
            rows, _, _, width = _lifted_coefficients(A, B, degree, caps)
            if evidence != {"degree": degree, "coefficient_unknowns": width,
                            "coefficient_equations": len(rows)}:
                raise ValueError("Coefficient search size metadata is false")
        return A, result, B
    if operation not in ("kernel", "no-solution"):
        raise ValueError("Unknown linear certificate operation")
    if operation == "kernel":
        if B is not None or result is None or result.nrows != A.ncols or A * result:
            raise ValueError("Invalid homogeneous kernel witness")
    elif B is None or result is not None:
        raise ValueError("Nonmembership must bind a right-hand side and have no solution")
    if method == "zero-operator":
        if A or (operation == "kernel" and result != eye(ring, A.ncols)) or (
                operation == "no-solution" and not B):
            raise ValueError("Invalid zero-operator proof")
    elif method == "domain-column":
        if operation != "kernel" or not A or A.ncols != 1 or result != zeros(ring, 1, 0):
            raise ValueError("Invalid integral-domain injectivity proof")
    elif method == "augmentation-obstruction":
        for count, limit in (
                (A.ncols, caps["max_unknowns"]), (A.nrows, caps["max_equations"]),
                (A.nrows * (A.ncols + (0 if B is None else B.ncols)), caps["max_matrix_entries"])):
            if limit is not None and count > limit:
                raise ResourceLimit("Specialization nonmembership verification exceeds the finite-field budget")
        if operation != "no-solution" or _constant_rank(_augment(A).hstack(_augment(B))) <= _constant_rank(_augment(A)):
            raise ValueError("No finite-field specialization obstruction exists")
    elif method == "unit-reduction":
        r = _int(evidence["rank"], "unit rank", 1)
        L = _matrix_from_dict(evidence["L"], ring, "L")
        V = _matrix_from_dict(evidence["V"], ring, "V")
        Li = _matrix_from_dict(evidence["Linv"], ring, "Linv")
        Vi = _matrix_from_dict(evidence["Vinv"], ring, "Vinv")
        if L.shape != (A.nrows, A.nrows) or V.shape != (A.ncols, A.ncols):
            raise ValueError("Unit reduction dimensions are false")
        if L * Li != eye(ring, A.nrows) or Li * L != eye(ring, A.nrows) or (
                V * Vi != eye(ring, A.ncols) or Vi * V != eye(ring, A.ncols)):
            raise ValueError("Elementary transformations lack two-sided inverses")
        reduced = L * A * V
        if r > min(A.shape) or reduced[:r, :r] != eye(ring, r) or reduced[:r, r:] or reduced[r:, :r]:
            raise ValueError("Unit reduction is not the claimed block diagonal matrix")
        child_A, child_result, child_B = _kernel_check(evidence["child"], caps=caps)
        if child_A != reduced[r:, r:] or evidence["child"]["operation"] != operation:
            raise ValueError("Unit reduction child is not bound to the residual")
        if operation == "kernel":
            if result != V * zeros(ring, r, child_result.ncols).vstack(child_result):
                raise ValueError("Kernel was not transported by the invertible basis change")
        elif child_B != (L * B)[r:, :]:
            raise ValueError("Nonmembership child has the wrong residual RHS")
    elif method == "exhausted-bound":
        degree = _int(evidence["degree"], "complete degree", 0)
        if _bound_within(degree_bound(A, B), degree) != degree:
            raise ValueError("A partial degree search cannot prove completeness")
        if caps["max_degree"] is not None and degree > caps["max_degree"]:
            raise ResourceLimit("Complete coefficient verification exceeds the degree budget")
        rows, rhs, monomials, width = _lifted_coefficients(A, B, degree, caps)
        reduced, pivots, consistent = _dense_eliminate(
            rows, width, ring.p, rhs if B is not None else None)
        if operation == "no-solution":
            if consistent:
                raise ValueError("The complete bounded system is actually consistent")
        else:
            free = [j for j in range(width) if j not in set(pivots)]
            if caps["max_matrix_entries"] is not None and width * len(free) > caps["max_matrix_entries"]:
                raise ResourceLimit("Complete kernel verification output exceeds the entry budget")
            vectors = [{j: 1, **{pivot: -reduced[i][j] % ring.p
                                 for i, pivot in enumerate(pivots) if reduced[i][j]}}
                       for j in free]
            expected = _project_nullspace(A, monomials, vectors)
            if result != expected:
                raise ValueError("The output omits or changes complete coefficient nullspace generators")
    else:
        raise ValueError("Unknown kernel/nonmembership proof method")
    return A, result, B


def verify_bounded_linear(certificate, *, max_degree=64, max_unknowns=10000,
                          max_equations=20000, max_matrix_entries=2_000_000):
    """Check actual identities, and a complete proof for kernel/nonmembership."""
    caps = dict(max_degree=max_degree, max_unknowns=max_unknowns,
                max_equations=max_equations, max_matrix_entries=max_matrix_entries)
    try:
        _kernel_check(certificate, caps=caps)
        return VerificationReport(True, ("input-bound linear proof",), (), "bounded-linear")
    except (ValueError, TypeError, KeyError, IndexError, ArithmeticError) as exc:
        return VerificationReport(False, (), (str(exc),), "bounded-linear")


def verify_trivial_cohomology(certificate, *, complex=None, backend=None,
                             max_degree=64, max_unknowns=10000,
                             max_equations=20000, max_matrix_entries=2_000_000):
    """Verify a full layer, optionally bound to an externally supplied complex.

    Kernel completeness is checked from bounded certificates when present.
    Otherwise an independent exact module backend recomputes the kernel and
    checks that its entire image is contained in the supplied generators.
    """
    from css_decoupling.algebra.modules import GeneralBackend
    data = certificate.certificate() if hasattr(certificate, "certificate") else certificate
    try:
        _keys(data, ("schema", "version", "ring", "degree", "dprev", "dnext", "cycles",
                     "relation_kernel", "presentation", "basis", "projection",
                     "representatives", "dimension", "actions", "triviality_witnesses",
                     "kernel_certificate", "relation_kernel_certificate"), "trivial cohomology")
        if data["schema"] != "css-decoupling.trivial-cohomology" or _int(data["version"], "version") != 1:
            raise ValueError("Unsupported trivial cohomology certificate")
        _keys(data["ring"], ("prime", "dimension"), "ring")
        ring = Ring(_int(data["ring"]["prime"], "prime", 2),
                    _int(data["ring"]["dimension"], "dimension", 1))
        degree = _int(data["degree"], "degree")
        names = ("dprev", "dnext", "cycles", "relation_kernel", "presentation",
                 "basis", "projection", "representatives")
        matrices = {name: _matrix_from_dict(data[name], ring, name) for name in names}
        dp, dn, W, K, U, E, Q, reps = (matrices[name] for name in names)
        if dn.ncols != dp.nrows or dn * dp:
            raise ValueError("Input differentials do not form a complex")
        if complex is not None and (complex.ring != ring or complex.d(degree) != dn or complex.d(degree - 1) != dp):
            raise ValueError("Layer evidence is not bound to the supplied source complex")
        local_backend = backend or GeneralBackend(ring)
        caps = dict(max_degree=max_degree, max_unknowns=max_unknowns,
                    max_equations=max_equations, max_matrix_entries=max_matrix_entries)

        def complete_kernel(operator, generators, proof):
            if generators.nrows != operator.ncols or operator * generators:
                raise ValueError("Claimed kernel generators are not cycles")
            if proof is not None:
                if not isinstance(proof, dict):
                    raise ValueError("Kernel completeness proof must be a mapping")
                if proof.get("schema") == "css-decoupling.restricted-kernel":
                    from .cohomology_handoff import check_restricted_kernel
                    proof_operator, proof_generators = check_restricted_kernel(proof, backend=local_backend)
                else:
                    proof_operator, proof_generators, _ = _kernel_check(proof, caps=caps)
                    if proof["operation"] != "kernel":
                        raise ValueError("A non-kernel certificate cannot prove kernel completeness")
                if proof_operator != operator or proof_generators != generators:
                    raise ValueError("Kernel completeness certificate is not bound to this layer")
            else:
                complete = local_backend.kernel(operator)
                lift = local_backend.solve(generators, complete)
                if generators * lift != complete:
                    raise ValueError("The supplied cycles omit part of the complete kernel")

        complete_kernel(dn, W, data["kernel_certificate"])
        operator = W.hstack(-dp)
        complete_kernel(operator, K, data["relation_kernel_certificate"])
        if U != K[:W.ncols, :]:
            raise ValueError("Relations are not the projection of the complete relation kernel")
        if len(data["triviality_witnesses"]) != ring.D:
            raise ValueError("A translation generator lacks a triviality witness")
        for axis, witness_data in enumerate(data["triviality_witnesses"]):
            witness = _matrix_from_dict(witness_data, ring, "triviality witness")
            if dp * witness != (ring.variable(axis) - 1) * W:
                raise ValueError("Translation does not act trivially on the given cycles")
        dimension = _int(data["dimension"], "dimension", 0)
        if E.shape != (W.ncols, dimension) or Q.shape != (dimension, W.ncols):
            raise ValueError("Quotient basis and projection have wrong dimensions")
        if any(not f.is_constant for M in (E, Q) for f in M.entries.values()):
            raise ValueError("Augmented quotient basis must be over the prime field")
        U0 = _augment(U)
        if Q * E != eye(ring, dimension) or Q * U0 or _constant_rank(U0) + dimension != W.ncols:
            raise ValueError("Finite quotient witnesses do not describe the full quotient")
        if reps != W * E or dn * reps:
            raise ValueError("Cohomology representatives are not the declared basis cocycles")
        if len(data["actions"]) != ring.D or any(
                _matrix_from_dict(action, ring, "action") != eye(ring, dimension)
                for action in data["actions"]):
            raise ValueError("Declared translation actions are not the verified identity")
        return VerificationReport(True, ("source complex", "complete kernels", "trivial actions",
                                         "full augmented quotient"), (), "trivial-cohomology")
    except (ValueError, TypeError, KeyError, IndexError, ArithmeticError) as exc:
        return VerificationReport(False, (), (str(exc),), "trivial-cohomology")
