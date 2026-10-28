"""Paper Algorithm 1: comparison maps, homotopy correction and exact QCA.

The construction follows arXiv:2608.09915v1, Supplement IV.4, and the
MIT-licensed implementation by Mingxin He (upstream commit
7879852262ee7bd0bb69c1a2edae8f1543fcca3e; Copyright 2026 The Authors).
This implementation uses native Laurent module solves and always constructs
both directions.  It performs no circuit decomposition.

Matrices act on columns.  The complex is C2 --HZ.dagger()--> C1 --HX--> C0.
``psi[i]`` maps input to standard; ``psi_inverse[i]`` maps standard to input.
The physical Z map is psi[1]; the physical X map is psi_inverse[1].dagger().
"""
from __future__ import annotations

from .algebra import Matrix, inverse, solve, ResourceLimit, FiniteQuotient, zeros
from .models import standard_code
from ..algebra.sparse_linear import sparse_binary_solution


def _eye(ring, n):
    return Matrix(ring, n, n, {(i, i): 1 for i in range(n)})


def augmentation(matrix):
    """Evaluate at x=y=1, retaining the Laurent matrix type."""
    return Matrix(matrix.ring, matrix.nrows, matrix.ncols, {
        ij: sum(poly.terms.values()) % matrix.ring.p
        for ij, poly in matrix.entries.items()
    })


def _binary_rows(matrix):
    rows = [0] * matrix.nrows
    for (i, j), polynomial in matrix.entries.items():
        if sum(polynomial.terms.values()) % 2:
            rows[i] |= 1 << j
    return rows


def _binary_matrix(ring, rows, ncols):
    entries = {}
    for i, value in enumerate(rows):
        while value:
            low = value & -value
            entries[i, low.bit_length() - 1] = 1
            value ^= low
    return Matrix(ring, len(rows), ncols, entries)


def _column_add(rows, target, source):
    bit = 1 << target
    for i, value in enumerate(rows):
        if (value >> source) & 1:
            rows[i] = value ^ bit


def _column_swap(rows, first, second):
    if first == second:
        return
    mask = (1 << first) | (1 << second)
    for i, value in enumerate(rows):
        if ((value >> first) ^ (value >> second)) & 1:
            rows[i] = value ^ mask


def _constant_inverse(matrix):
    """Invert over the exact prime field, retaining the binary fast path."""
    if matrix.nrows != matrix.ncols:
        raise ValueError("Expected a square constant matrix")
    if any(not f.is_constant for f in matrix.entries.values()):
        raise ValueError("Expected constant clearing map")
    n = matrix.nrows
    if matrix.ring.p != 2:
        prime = matrix.ring.p
        rows = _field_rows(matrix)
        witness = [[int(i == j) for j in range(n)] for i in range(n)]
        for j in range(n):
            pivot = next((i for i in range(j, n) if rows[i][j]), None)
            if pivot is None:
                raise ValueError("Singular clearing map")
            rows[j], rows[pivot] = rows[pivot], rows[j]
            witness[j], witness[pivot] = witness[pivot], witness[j]
            scale = pow(rows[j][j], -1, prime)
            rows[j] = [scale*c % prime for c in rows[j]]
            witness[j] = [scale*c % prime for c in witness[j]]
            for i in range(n):
                if i != j and rows[i][j]:
                    scale = -rows[i][j]
                    _field_row_add(rows, i, j, scale, prime)
                    _field_row_add(witness, i, j, scale, prime)
        return _field_matrix(matrix.ring, witness, n)
    rows = _binary_rows(matrix)
    witness = [1 << i for i in range(n)]
    for j in range(n):
        pivot = next((i for i in range(j, n) if (rows[i] >> j) & 1), None)
        if pivot is None:
            raise ValueError("Singular clearing map")
        rows[j], rows[pivot] = rows[pivot], rows[j]
        witness[j], witness[pivot] = witness[pivot], witness[j]
        for i in range(n):
            if i != j and (rows[i] >> j) & 1:
                rows[i] ^= rows[j]
                witness[i] ^= witness[j]
    return _binary_matrix(matrix.ring, witness, n)


def _field_rows(matrix):
    rows = [[0] * matrix.ncols for _ in range(matrix.nrows)]
    for (i, j), polynomial in matrix.entries.items():
        rows[i][j] = sum(polynomial.terms.values()) % matrix.ring.p
    return rows


def _field_matrix(ring, rows, ncols):
    return Matrix(ring, len(rows), ncols,
                  {(i, j): c for i, row in enumerate(rows)
                   for j, c in enumerate(row) if c})


def _field_row_add(rows, target, source, coefficient, prime):
    rows[target] = [(a + coefficient*b) % prime
                    for a, b in zip(rows[target], rows[source])]


def _field_column_add(rows, target, source, coefficient, prime):
    for row in rows:
        row[target] = (row[target] + coefficient*row[source]) % prime


def _clear_css_mod_j_prime(HX, HZ):
    """Weighted elimination with the contragredient physical basis change.

    Maintain X=Lx*augmentation(HX)*P and
    Z=Lz*augmentation(HZ)*P^(-T). A column addition on either check
    therefore induces the opposite coefficient on the other check.
    """
    ring, q, prime = HX.ring, HX.ncols, HX.ring.p
    x, z = _field_rows(HX), _field_rows(HZ)
    lx = [[int(i == j) for j in range(HX.nrows)] for i in range(HX.nrows)]
    lz = [[int(i == j) for j in range(HZ.nrows)] for i in range(HZ.nrows)]
    physical = [[int(i == j) for j in range(q)] for i in range(q)]

    def clear(active, other, row_map, start_col, x_side):
        rank = 0
        while rank < len(active) and start_col + rank < q:
            first_col = start_col + rank
            pivot = next(((i, j) for i in range(rank, len(active))
                          for j in range(first_col, q) if active[i][j]), None)
            if pivot is None:
                break
            row, col = pivot
            active[rank], active[row] = active[row], active[rank]
            row_map[rank], row_map[row] = row_map[row], row_map[rank]
            for rows in (active, other, physical):
                for values in rows:
                    values[first_col], values[col] = values[col], values[first_col]
            scale = pow(active[rank][first_col], -1, prime)
            active[rank] = [scale*c % prime for c in active[rank]]
            row_map[rank] = [scale*c % prime for c in row_map[rank]]
            for i in range(len(active)):
                if i != rank and active[i][first_col]:
                    coefficient = -active[i][first_col]
                    _field_row_add(active, i, rank, coefficient, prime)
                    _field_row_add(row_map, i, rank, coefficient, prime)
            for col in range(first_col + 1, q):
                coefficient = -active[rank][col]
                if not coefficient:
                    continue
                _field_column_add(active, col, first_col, coefficient, prime)
                _field_column_add(other, first_col, col, -coefficient, prime)
                if x_side:
                    _field_column_add(physical, col, first_col, coefficient, prime)
                else:
                    _field_column_add(physical, first_col, col, -coefficient, prime)
            rank += 1
        return rank

    px = clear(x, z, lx, 0, True)
    pz = clear(z, x, lz, px, False)
    Lx, Lz = _field_matrix(ring, lx, HX.nrows), _field_matrix(ring, lz, HZ.nrows)
    P = _field_matrix(ring, physical, q)
    cleared_HX = Lx * HX * P
    cleared_HZ = Lz * HZ * _constant_inverse(P).T
    if (augmentation(cleared_HX) != _field_matrix(ring, x, q) or
            augmentation(cleared_HZ) != _field_matrix(ring, z, q)):
        raise ArithmeticError("Prime-field clearing witness failed")
    return {"cleared_HX": cleared_HX, "cleared_HZ": cleared_HZ,
            "xi0": _constant_inverse(Lx), "xi1": P, "xi2": Lz.T,
            "product_x": px, "product_z": pz}


def clear_css_mod_j(HX, HZ):
    """Expose product-state blocks modulo J=(x-1,y-1).

    Returned xi maps run from the cleared complex to the input complex.
    The operation records full Laurent identities; augmentation is used only
    to decide finite-field pivots, never to establish Laurent exactness.
    """
    if HX.ring != HZ.ring or HX.ncols != HZ.ncols:
        raise ValueError("CSS checks must have a common ring and column count")
    if HX.ring.D != 2:
        raise ValueError("The two-dimensional QCA construction requires D=2")
    if HX * HZ.dagger():
        raise ValueError("CSS checks do not commute")
    if HX.ring.p != 2:
        return _clear_css_mod_j_prime(HX, HZ)
    ring, q = HX.ring, HX.ncols
    x, z = _binary_rows(HX), _binary_rows(HZ)
    lx = [1 << i for i in range(HX.nrows)]
    lz = [1 << i for i in range(HZ.nrows)]
    physical = [1 << i for i in range(q)]

    def clear(active, other, row_map, start_col, x_side):
        rank = 0
        available = ((1 << q) - 1) ^ ((1 << start_col) - 1)
        while rank < len(active) and start_col + rank < q:
            first_col = start_col + rank
            candidates = available & ~((1 << first_col) - 1)
            pivot = next(((i, (active[i] & candidates & -(active[i] & candidates)).bit_length() - 1)
                          for i in range(rank, len(active)) if active[i] & candidates), None)
            if pivot is None:
                break
            row, col = pivot
            active[rank], active[row] = active[row], active[rank]
            row_map[rank], row_map[row] = row_map[row], row_map[rank]
            for rows in (active, other, physical):
                _column_swap(rows, first_col, col)
            for i in range(len(active)):
                if i != rank and (active[i] >> first_col) & 1:
                    active[i] ^= active[rank]
                    row_map[i] ^= row_map[rank]
            tail = active[rank] & candidates & ~(1 << first_col)
            while tail:
                low = tail & -tail
                col = low.bit_length() - 1
                _column_add(active, col, first_col)
                _column_add(other, first_col, col)
                if x_side:
                    _column_add(physical, col, first_col)
                else:
                    _column_add(physical, first_col, col)
                tail ^= low
            rank += 1
        return rank

    px = clear(x, z, lx, 0, True)
    pz = clear(z, x, lz, px, False)
    Lx = _binary_matrix(ring, lx, HX.nrows)
    Lz = _binary_matrix(ring, lz, HZ.nrows)
    P = _binary_matrix(ring, physical, q)
    Pinv = _constant_inverse(P)
    cleared_HX = Lx * HX * P
    cleared_HZ = Lz * HZ * Pinv.T
    if augmentation(cleared_HX) != _binary_matrix(ring, x, q):
        raise ArithmeticError("X clearing witness failed")
    if augmentation(cleared_HZ) != _binary_matrix(ring, z, q):
        raise ArithmeticError("Z clearing witness failed")
    return {
        "cleared_HX": cleared_HX, "cleared_HZ": cleared_HZ,
        "xi0": _constant_inverse(Lx), "xi1": P, "xi2": Lz.T,
        "product_x": px, "product_z": pz,
    }


def invert_physical_chain(V,HX,HZ,TX,TZ,p0,p2,px,pz,t,*,progress=None,max_expansion_terms=2_000_000,
                          restriction_context=None):
    """Invert a chain isomorphism by standard lifting and its exact defect.

    Choose Q0 with TX*Q0=p0*HX. Then E=V*Q0-I is in im(HZ†).
    If HZ†*H=E, Q=Q0-TZ†*p2*H satisfies V*Q=I by the checked
    endpoint chain identities. Both actual inverse products are checked below.
    This avoids generic inversion of a large unimodular physical matrix.
    """
    def report(stage,**details):
        if progress is not None:progress(stage,details)
    R=V.ring
    expected_x,expected_z=standard_code(t,product_x=px,product_z=pz,ring=R)
    if TX!=expected_x or TZ!=expected_z:
        raise ValueError("Chain inversion requires the declared standard complex")
    rhs=p0*HX
    report("inverse_preimage",coefficient_shape=list(TX.shape),
           equation="standard_HX Q0 = psi0 HX by explicit Laurent telescoping")
    required=sum(abs(e[0])+abs(e[1]) for (i,_),f in rhs.entries.items() if i>=px for e in f.terms)
    if type(max_expansion_terms) is not int or max_expansion_terms<0:
        raise ValueError("Invalid inverse expansion budget")
    if required>max_expansion_terms:
        raise ResourceLimit(f"Standard inverse preimage needs at most {required} terms; cap {max_expansion_terms}")
    Q0=Matrix(R,V.ncols,V.nrows)
    Q0[:px,:]=rhs[:px,:]
    for (i,j),f in rhs.entries.items():
        if i<px:continue
        if sum(f.terms.values())%R.p:
            raise ValueError("Toric syndrome preimage has nonzero augmentation")
        a={};b={}
        for (u,v),c in f.terms.items():
            sign=1 if u>0 else -1
            for k in range(min(0,u),max(0,u)):
                e=(k,v);a[e]=(a.get(e,0)+sign*c)%R.p
            sign=1 if v>0 else -1
            for k in range(min(0,v),max(0,v)):
                e=(0,k);b[e]=(b.get(e,0)+sign*c)%R.p
        Q0[px+pz+2*(i-px),j]=R(a)
        Q0[px+pz+2*(i-px)+1,j]=R(b)
    if TX*Q0!=rhs:raise ArithmeticError("Standard inverse preimage failed")
    defect=V*Q0-_eye(R,V.nrows)
    report("inverse_homotopy",coefficient_shape=list(HZ.dagger().shape),
           equation="HZ_dagger H = psi_inverse1 Q0 - I")
    if restriction_context is None:
        H=solve(HZ.dagger(),defect)
    else:
        # Keep inexpensive checked coarse solutions when available. A bounded
        # miss requests the complete fine-coordinate route, never nonmembership.
        H=sparse_binary_solution(HZ.dagger(),defect)
        stats={"route":"coarse_bounded_positive"}
        if H is None:
            from .restricted_lifting import solve_restricted
            H=solve_restricted(restriction_context["HZ"].dagger(),HZ.dagger(),defect,
                               restriction_context["lattice"],stats=stats,
                               max_terms=restriction_context.get("max_terms",2_000_000),timeout=120)
        report("inverse_lift_statistics",coefficient_shape=list(HZ.dagger().shape),
               equation="Verified restriction transport of input_D H = defect",statistics=stats)
    report("inverse_assembly",physical_shape=list(V.shape),
           equation="Q = Q0 - standard_HZ_dagger psi2 H; verify VQ=QV=I")
    Q=Q0-TZ.dagger()*(p2*H)
    if V*Q!=_eye(R,V.nrows) or Q*V!=_eye(R,V.nrows):
        raise ArithmeticError("Chain-based physical inverse failed both actual products")
    return Q


def construct_qca(HX, HZ, t=None, progress=None, restriction_context=None):
    """Construct both QCA directions from independent, translation-trivial checks.

    The caller must validate complete kernels/images, independent generators,
    and trivial action of both translations on the two cokernels.  This routine
    checks dimensions, commutation, all construction equations and all actual
    two-sided inverses.  A failed solve is not reinterpreted as non-topological.
    An optional ``progress(stage, details)`` callback runs before each expensive
    mathematical step; it is observational and its return value is ignored.
    """
    if progress is not None and not callable(progress):
        raise TypeError("progress must be callable or None")
    if restriction_context is not None:
        if set(restriction_context)-{"HX","HZ","lattice","quotient","max_terms"}:
            raise ValueError("Unknown restriction-context field")
        L=restriction_context["lattice"]
        if L.restrict(restriction_context["HX"])!=HX or L.restrict(restriction_context["HZ"])!=HZ:
            raise ValueError("Restriction context does not match QCA input")
        cap=restriction_context.get("max_terms",2_000_000)
        if type(cap) is not int or cap<1:
            raise ValueError("Invalid restriction lifting term budget")
        quotient=restriction_context.get("quotient")
        if quotient is not None:
            fine=restriction_context["HX"]
            if (not isinstance(quotient,FiniteQuotient) or quotient.presentation!=fine or
                    quotient.ring!=fine.ring or type(quotient.dimension) is not int or quotient.dimension<0):
                raise ValueError("Restriction quotient does not match fine presentation")
            E=quotient.basis
            if (not isinstance(E,Matrix) or E.ring!=fine.ring or E.shape!=(fine.nrows,quotient.dimension) or
                    quotient.coordinates(E)!=_eye(fine.ring,quotient.dimension) or
                    quotient.coordinates(fine)!=zeros(fine.ring,quotient.dimension,fine.ncols) or
                    len(quotient.actions)!=fine.ring.D or
                    any(T!=quotient.coordinates(fine.ring.variable(i)*E) for i,T in enumerate(quotient.actions))):
                raise ValueError("Restriction quotient basis/actions do not verify")

    def report(stage, **details):
        if progress is not None:
            progress(stage, details)

    report("clearing_mod_j", HX_shape=list(HX.shape), HZ_shape=list(HZ.shape),
           equation="Find constant xi with HX xi1 = xi0 standard_HX mod J and HZ_dagger xi2 = xi1 standard_HZ_dagger mod J")
    clearing = clear_css_mod_j(HX, HZ)
    X, Z = clearing["cleared_HX"], clearing["cleared_HZ"]
    px, pz = clearing["product_x"], clearing["product_z"]
    inferred_t = X.nrows - px
    if t is None:
        t = inferred_t
    if isinstance(t, bool) or not isinstance(t, int) or t < 0:
        raise ValueError("Toric multiplicity must be a nonnegative integer")
    if t != inferred_t or HZ.nrows - pz != t or HX.ncols != px + pz + 2*t:
        raise ValueError("Ranks do not match an independent translation-trivial complex")
    TX, TZ = standard_code(t, product_x=px, product_z=pz, ring=HX.ring)
    if augmentation(X) != augmentation(TX) or augmentation(Z) != augmentation(TZ):
        raise ArithmeticError("Cleared matrices do not have the standard augmentation")
    D, TD = Z.dagger(), TZ.dagger()
    q = X.ncols
    # Solve in the original sparse physical coordinates. Multiplication by
    # the constant clearing basis can otherwise densify the coefficient
    # matrix before the expensive Laurent lift. F = xi1*phi1 is the same
    # comparison map expressed in the input physical basis.
    F = Matrix(HX.ring, q, q)
    xi0, xi1, xi2 = (clearing[f"xi{i}"] for i in range(3))
    original_D = HZ.dagger()
    solved = list(range(px)) + list(range(px+pz, q))
    if solved:
        report("solve_phi1", coefficient_shape=list(HX.shape),
               target_shape=[TX.nrows, len(solved)],
               equation="HX F = xi0 standard_HX; F=xi1 phi1; Z-product columns fixed")
        if restriction_context is None:
            F[:, solved] = solve(HX, xi0*TX[:, solved])
        else:
            right=xi0*TX[:,solved]
            candidate=sparse_binary_solution(HX,right)
            stats={"route":"coarse_bounded_positive"}
            if candidate is None:
                from .restricted_lifting import solve_restricted
                candidate=solve_restricted(restriction_context["HX"],HX,right,
                    restriction_context["lattice"],quotient=restriction_context.get("quotient"),
                    max_terms=restriction_context.get("max_terms",2_000_000),timeout=120,stats=stats)
            F[:,solved]=candidate
            report("comparison_lift_statistics",coefficient_shape=list(HX.shape),
                   equation="Verified restricted source-ring lift HX F = xi0 standard_HX",statistics=stats)
    F[:, px:px+pz] = original_D*xi2[:, :pz]
    phi1 = _constant_inverse(xi1)*F
    if X * phi1 != TX:
        raise ArithmeticError("Comparison equation X phi1 = standard_X failed")
    phi2 = Matrix(HX.ring, HZ.nrows, HZ.nrows)
    for i in range(pz):
        phi2[i, i] = 1
    if t:
        report("solve_phi2", coefficient_shape=list(original_D.shape),
               target_shape=[phi1.nrows, t],
               equation="HZ_dagger G = F standard_HZ_dagger; G=xi2 phi2")
        phi2[:, pz:] = _constant_inverse(xi2)*solve(original_D, F*TD[:, pz:])
    if D * phi2 != phi1 * TD:
        raise ArithmeticError("Comparison equation D phi2 = phi1 standard_D failed")
    phi2_corrected = phi2.copy()
    phi2_corrected[pz:, pz:] = augmentation(phi2[pz:, pz:])
    eta = Matrix(HX.ring, HZ.nrows, q)
    if t:
        report("solve_eta", coefficient_shape=list(TD.T.shape),
               target_shape=[HZ.nrows, t],
               equation="eta standard_HZ_dagger = phi2_corrected - phi2")
        eta[pz:, :] = solve(TD.T, (phi2_corrected - phi2)[pz:, :].T).T
    if eta * TD != phi2_corrected - phi2:
        raise ArithmeticError("Homotopy equation eta standard_D = delta_phi2 failed")
    phi1_corrected = phi1 + D * eta
    backward = {
        0: clearing["xi0"],
        1: F + original_D*(xi2*eta),
        2: clearing["xi2"] * phi2_corrected,
    }
    forward = {0: _constant_inverse(backward[0])}
    report("inverse2", coefficient_shape=list(backward[2].shape),
           equation="psi_inverse[2] psi[2] = I and psi[2] psi_inverse[2] = I")
    # The corrected endpoint has the proved block form [I C;0 B], where B
    # is an invertible constant t-by-t matrix. Use its explicit inverse,
    # avoiding a large generic Laurent solve. The verifier still checks
    # both complete endpoint inverse products.
    B_inverse = _constant_inverse(phi2_corrected[pz:, pz:])
    endpoint_inverse = _eye(HX.ring,HZ.nrows)
    endpoint_inverse[pz:, pz:] = B_inverse
    endpoint_inverse[:pz, pz:] = -phi2_corrected[:pz, pz:]*B_inverse
    forward[2] = endpoint_inverse*_constant_inverse(xi2)
    report("inverse1", coefficient_shape=list(backward[1].shape),
           equation="psi_inverse[1] psi[1] = I and psi[1] psi_inverse[1] = I")
    forward[1]=invert_physical_chain(backward[1],HX,HZ,TX,TZ,forward[0],forward[2],px,pz,t,
                                     progress=progress,restriction_context=restriction_context)
    result = {
        **clearing, "input_HX": HX, "input_HZ": HZ,
        "toric_copies": t, "standard_HX": TX, "standard_HZ": TZ,
        "psi": forward, "psi_inverse": backward,
        "phi1": phi1, "phi2": phi2, "eta": eta,
        "phi1_corrected": phi1_corrected, "phi2_corrected": phi2_corrected,
        "physical_Z": forward[1], "physical_X": backward[1].dagger(),
        "direction": "psi: input-to-standard; psi_inverse: standard-to-input",
    }
    report("verify_qca", physical_shape=list(forward[1].shape),
           equation="Verify both chain directions, both inverses, Pauli pairing, stabilizer generation and homotopy equations")
    verify_qca(result)
    return result


def verify_qca(result):
    """Check saved maps by exact multiplication, without reconstructing them."""
    X, Z = result["input_HX"], result["input_HZ"]
    TX, TZ = result["standard_HX"], result["standard_HZ"]
    p, b = result["psi"], result["psi_inverse"]
    sizes = {0: X.nrows, 1: X.ncols, 2: Z.nrows}
    if X.ring != Z.ring or X.ring.D != 2:
        raise ValueError("QCA ring mismatch")
    expected_x, expected_z = standard_code(result["toric_copies"],
        product_x=result["product_x"], product_z=result["product_z"], ring=X.ring)
    if TX != expected_x or TZ != expected_z:
        raise ValueError("QCA target is not the declared toric/product standard code")
    for i, size in sizes.items():
        identity = _eye(X.ring, size)
        if p[i]*b[i] != identity or b[i]*p[i] != identity:
            raise ValueError(f"QCA two-sided inverse failed in degree {i}")
    equalities = (
        (lambda: TX*p[1] == p[0]*X, "forward X chain"),
        (lambda: TZ.dagger()*p[2] == p[1]*Z.dagger(), "forward Z chain"),
        (lambda: X*b[1] == b[0]*TX, "backward X chain"),
        (lambda: Z.dagger()*b[2] == b[1]*TZ.dagger(), "backward Z chain"),
        (lambda: b[1].dagger()*X.dagger() == TX.dagger()*b[0].dagger(), "X stabilizer forward"),
        (lambda: p[1].dagger()*TX.dagger() == X.dagger()*p[0].dagger(), "X stabilizer backward"),
    )
    for check, label in equalities:
        if not check():
            raise ValueError(f"QCA {label} equation failed")
    if X*Z.dagger() or TX*TZ.dagger():
        raise ValueError("QCA source or target checks do not commute")
    if result["physical_Z"] != p[1] or result["physical_X"] != b[1].dagger():
        raise ValueError("QCA declared physical maps disagree with chain maps")
    if result["physical_X"].dagger()*result["physical_Z"] != _eye(X.ring, X.ncols):
        raise ValueError("QCA Pauli pairing failed")
    # These witnesses are part of the construction certificate, not trusted
    # annotations.  Checking them catches a changed clearing or homotopy.
    cx, cz = result["cleared_HX"], result["cleared_HZ"]
    xi0, xi1, xi2 = (result[f"xi{i}"] for i in range(3))
    phi1, phi2, eta = (result[k] for k in ("phi1", "phi2", "eta"))
    pc1, pc2 = result["phi1_corrected"], result["phi2_corrected"]
    px, pz = result["product_x"], result["product_z"]
    if augmentation(cx) != augmentation(TX) or augmentation(cz) != augmentation(TZ):
        raise ValueError("QCA clearing augmentation failed")
    if any(not f.is_constant for xi in (xi0, xi1, xi2) for f in xi.entries.values()):
        raise ValueError("QCA clearing maps must be over the constant field")
    if phi1[:, px:px+pz] != cz.dagger()[:, :pz]:
        raise ValueError("QCA comparison product columns failed")
    expected_pc2 = phi2.copy()
    expected_pc2[pz:, pz:] = augmentation(phi2[pz:, pz:])
    if pc2 != expected_pc2 or eta[:pz, :]:
        raise ValueError("QCA prescribed augmentation correction failed")
    construction = (
        lambda: X*xi1 == xi0*cx,
        lambda: Z.dagger()*xi2 == xi1*cz.dagger(),
        lambda: cx*phi1 == TX,
        lambda: cz.dagger()*phi2 == phi1*TZ.dagger(),
        lambda: eta*TZ.dagger() == pc2-phi2,
        lambda: pc1 == phi1+cz.dagger()*eta,
        lambda: cz.dagger()*pc2 == pc1*TZ.dagger(),
        lambda: b[0] == xi0,
        lambda: b[1] == xi1*pc1,
        lambda: b[2] == xi2*pc2,
    )
    if not all(check() for check in construction):
        raise ValueError("QCA comparison/homotopy witness failed")
    return True
