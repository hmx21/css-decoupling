"""Positive-only C4 section solves with a structural degree bound.

The right-hand projection is retained as a matrix factor.  Finite-field
coefficient systems never materialize a Laurent Kronecker matrix. A missed
finite support is ResourceLimit, not Laurent nonmembership. The bound is
conditional until an actual boundary lift (or the returned X) supplies the
missing image inclusion. No minors, projective bases, or h*d*h are computed.
"""
from .laurent import Matrix, Poly, eye, zeros
from .sparse_linear import _ShiftedExponents
from css_decoupling.errors import ResourceLimit


def terms(A):
    return sum(len(f.terms) for f in A.entries.values())


def product_work(A, B):
    counts = {}
    for (i, _), f in B.entries.items():
        counts[i] = counts.get(i, 0)+len(f.terms)
    return sum(len(f.terms)*counts.get(j, 0) for (_, j), f in A.entries.items())


def _shapes(d, R, hnext, s, q, Pprev):
    matrices = (d, R, hnext, s, q, Pprev)
    if not all(isinstance(A, Matrix) for A in matrices):
        raise ValueError("Structural factors must be Laurent matrices")
    ring, m, r = d.ring, d.nrows, d.ncols
    if any(A.ring != ring for A in matrices):
        raise ValueError("Structural factors have different rings")
    if (R.shape != (m, m) or hnext.nrows != m or s.nrows != m
            or q.ncols != r or Pprev.shape != (r, r)):
        raise ValueError("Malformed structural section dimensions")
    return ring, m, r


def c4_degree_bound(d, R, hnext, s, q, Pprev, *, boundary_lift=None):
    """Symbolic NSS bound; the image inclusion is never a boolean assertion.

    With H covering all actual known factors, delta=max((2D-1)H,D+1),
    E=max(3,m*delta)^(D+1), B=E+(m-1)delta+(4D-2)H. For a genuine C4
    boundary projection this bounds an S=F[x,z] lift, hence a Laurent box.
    No enormous power is evaluated. A supplied lift is actually checked.
    """
    ring, m, r = _shapes(d, R, hnext, s, q, Pprev)
    H = max((abs(v) for A in (d, R, hnext, s, q, Pprev)
             for f in A.entries.values() for e in f.terms for v in e), default=0)
    if boundary_lift is not None:
        if (not isinstance(boundary_lift, Matrix) or boundary_lift.ring != ring
                or boundary_lift.shape != (r, m) or d*boundary_lift != R):
            raise ValueError("Unverified structural boundary lift")
        if (R*d != d or d*Pprev or q*Pprev != q or Pprev*Pprev != Pprev
                or R*hnext or R*s):
            raise ValueError("Boundary lift does not certify the other C4 structural conditions")
    delta = max((2*ring.D-1)*H, ring.D+1)
    return dict(dimension=ring.D, target_rank=m, actual_half_width=H,
                delta=delta, base=max(3, m*delta), power=ring.D+1,
                additive=(m-1)*delta+(4*ring.D-2)*H if m else 0,
                zero_degree=not H or not m,
                formula="0 if zero_degree else base**power+additive",
                qualification="verified-boundary-lift" if boundary_lift is not None else "conditional",
                support_guarantee="some solution has canonical lift weight <= B; box [-B,B]^D suffices")


def _weight(e):
    return sum(e)+(len(e)+1)*max(0, -min(e, default=0))


def bound_contains(bound, value):
    if bound['zero_degree']:
        return value == 0
    target = value-bound['additive']
    if target <= 1:
        return True
    # Saturated exponentiation; storage depends on the exponent we check,
    # not on the gigantic theoretical degree ceiling.
    base, exponent, result = min(bound['base'], target), bound['power'], 1
    while exponent:
        if exponent & 1:
            result = min(target, result*base)
        exponent >>= 1
        if exponent:
            base = min(target, base*base)
    return result >= target


def right_annihilator_by_units(B, *, max_terms=2000000, max_product_work=50000000):
    """Eliminate known right constraints; return (N,J-indices), or None.

    When the residual vanishes, all X B=0 are X=Y N and N[:,J]=I.
    Only coordinates of this equation change; original complex ranks do not.
    """
    residual, N, active = B.copy(), eye(B.ring, B.nrows), list(range(B.nrows))
    while residual:
        positions = [at for at, value in residual.entries.items() if value.is_unit]
        if not positions:
            return None
        row_counts, col_counts = {}, {}
        for (i, j), f in residual.entries.items():
            row_counts[i] = row_counts.get(i, 0)+len(f.terms)
            col_counts[j] = col_counts.get(j, 0)+len(f.terms)
        i, j = min(positions, key=lambda at: (
            (row_counts[at[0]]-1)*(col_counts[at[1]]-1),
            row_counts[at[0]]+col_counts[at[1]], at))
        rows = [k for k in range(residual.nrows) if k != i]
        cols = [k for k in range(residual.ncols) if k != j]
        column = residual[rows, j:j+1]*(residual[i, j]**-1)
        row, old = residual[i:i+1, cols], N[i:i+1, :]
        if product_work(column, row)+product_work(column, old) > max_product_work:
            return None
        residual = residual[rows, cols]-column*row
        N = N[rows, :]-column*old
        active.pop(i)
        if terms(residual)+terms(N) > max_terms:
            return None
    if N[:, active] != eye(B.ring, len(active)):
        raise ArithmeticError("Right constraint elimination lost its coordinate section")
    return N, active


def _support(D, radius, limit):
    # Two declared finite families: a Laurent l1 ball and projected S-total
    # degree monomials. Neither is labelled a complete theorem search.
    found = set()
    def visit(prefix, remaining):
        if len(prefix) == D:
            found.add(prefix)
            if len(found) > limit:
                raise ResourceLimit("Structural support exceeds coefficient budget")
            return
        for value in range(-remaining, remaining+1):
            visit(prefix+(value,), remaining-abs(value))
    visit((), radius)
    def canonical(prefix, remaining):
        if len(prefix) == D:
            for z in range(remaining+1):
                found.add(tuple(e-z for e in prefix))
                if len(found) > limit:
                    raise ResourceLimit("Structural support exceeds coefficient budget")
            return
        for value in range(remaining+1):
            canonical(prefix+(value,), remaining-value)
    canonical((), radius)
    return sorted(found, key=lambda e: (sum(map(abs, e)), e))


class _Rows:
    def __init__(self, p, max_equations, max_terms):
        self.p, self.maximum, self.max_terms = p, max_equations, max_terms
        self.data, self.contributions = {}, 0

    def add(self, key, column, coefficient, *, rhs=False):
        self.contributions += 1
        if self.contributions > self.max_terms:
            raise ResourceLimit("Structural coefficient contributions exceed budget")
        if key not in self.data:
            if len(self.data) >= self.maximum:
                raise ResourceLimit("Structural coefficient equations exceed budget")
            self.data[key] = ({}, {})
        target = self.data[key][int(rhs)]
        value = (target.get(column, 0)+coefficient) % self.p
        if value:
            target[column] = value
        else:
            target.pop(column, None)


def _eliminate(rows, width, right_width, p, *, max_steps, max_coefficient_bits,
               max_output_terms=None):
    """Sparse coefficient rows, dense binary pivots, shared RHS elimination."""
    nonzeros = sum(len(a)+len(b) for a, b in rows)
    field_bits = (p-1).bit_length()
    # Keep input rows sparse and account for actual pivots as they appear.
    # Reserving a hypothetical full width-by-width dense basis can reject a
    # sparse/inconsistent system before its first inexpensive pivot.
    sparse_bits = (nonzeros*(max(1, width.bit_length(), right_width.bit_length())+field_bits)
                   +len(rows)*right_width*field_bits)
    basis_bits = 0
    bits = sparse_bits+2*(width+right_width)*field_bits
    if bits > max_coefficient_bits:
        raise ResourceLimit("Structural sparse coefficient rows exceed bit budget")
    def reserve(extra):
        nonlocal basis_bits, bits
        estimate = sparse_bits+basis_bits+extra+2*(width+right_width)*field_bits
        if estimate > max_coefficient_bits:
            raise ResourceLimit("Structural actual pivot storage exceeds bit budget")
        basis_bits += extra
        bits = max(bits, estimate)
    basis, steps = {}, 0
    if p == 2:
        mask = (1 << width)-1
        for a, b in rows:
            row = sum(1 << k for k in a) | sum(1 << (width+k) for k in b)
            while row and (pivot := (row & -row).bit_length()-1) < width:
                if pivot not in basis:
                    reserve(row.bit_length())
                    basis[pivot] = row
                    break
                row ^= basis[pivot]
                steps += 1
                if steps > max_steps:
                    raise ResourceLimit("Structural finite-field elimination step budget")
            else:
                if row:
                    return None, dict(rank=len(basis), steps=steps, reason="finite-support-inconsistent")
        if len(basis)*right_width > max_steps-steps:
            raise ResourceLimit("Structural back substitution step budget")
        steps += len(basis)*right_width
        reserve(width*right_width)
        columns = [0]*right_width
        for pivot in sorted(basis, reverse=True):
            row = basis[pivot]
            tail, right = (row & mask) ^ (1 << pivot), row >> width
            for j in range(right_width):
                if ((right >> j)&1) ^ ((tail & columns[j]).bit_count()&1):
                    columns[j] |= 1 << pivot
        output_terms = sum(column.bit_count() for column in columns)
        if max_output_terms is not None and output_terms > max_output_terms:
            raise ResourceLimit("Structural coefficient solution exceeds output term budget")
        reserve(output_terms*(width.bit_length()+right_width.bit_length()+field_bits))
        answer = {}
        for j, column in enumerate(columns):
            while column:
                bit = column & -column
                answer[bit.bit_length()-1, j] = 1
                column ^= bit
    else:
        for original, original_rhs in rows:
            a, b = original.copy(), original_rhs.copy()
            while a:
                pivot = min(a)
                if pivot not in basis:
                    inv = pow(a[pivot], -1, p)
                    reserve(len(a)*(width.bit_length()+field_bits)
                            +len(b)*(right_width.bit_length()+field_bits))
                    basis[pivot] = ({i: v*inv % p for i, v in a.items()},
                                    {i: v*inv % p for i, v in b.items()})
                    break
                other, right = basis[pivot]
                scale = a[pivot]
                steps += len(other)+len(right)
                if steps > max_steps:
                    raise ResourceLimit("Structural prime-field elimination step budget")
                for target, source in ((a, other), (b, right)):
                    for i, v in source.items():
                        value = (target.get(i, 0)-scale*v) % p
                        if value:
                            target[i] = value
                        else:
                            target.pop(i, None)
            else:
                if b:
                    return None, dict(rank=len(basis), steps=steps, reason="finite-support-inconsistent")
        values, answer = {}, {}
        output_terms = 0
        for pivot in sorted(basis, reverse=True):
            a, b = basis[pivot]
            value = b.copy()
            for k, coefficient in a.items():
                if k == pivot:
                    continue
                for j, v in values.get(k, {}).items():
                    steps += 1
                    if steps > max_steps:
                        raise ResourceLimit("Structural prime-field substitution step budget")
                    new = (value.get(j, 0)-coefficient*v) % p
                    if new:
                        value[j] = new
                    else:
                        value.pop(j, None)
            output_terms += len(value)
            if max_output_terms is not None and output_terms > max_output_terms:
                raise ResourceLimit("Structural coefficient solution exceeds output term budget")
            # Account for both the per-pivot values and decoded answer before
            # retaining either. This is coefficient storage, not a Python RSS cap.
            reserve(len(value)*(width.bit_length()+2*right_width.bit_length()+2*field_bits))
            values[pivot] = value
            answer.update({(pivot, j): v for j, v in value.items()})
    return answer, dict(rank=len(basis), steps=steps, coefficient_bit_estimate=bits)


def _coefficient_preflight(width, contributions, rhs_equations, caps, *, shared):
    """Count actual add calls, not a bound on distinct coefficient equations."""
    if width > caps['max_unknowns']:
        prefix = 'Shared' if shared else 'Joint'
        raise ResourceLimit(prefix+" structural coefficient unknowns exceed budget")
    if contributions > caps['max_terms']:
        raise ResourceLimit("Structural coefficient contributions exceed budget")
    if rhs_equations > caps['max_equations']:
        raise ResourceLimit("Structural coefficient equations exceed budget")


def _joint_term_counts(d, hnext, s, q):
    left, right = {}, {}
    for A in (d, q):
        for (_, column), f in A.entries.items():
            left[column] = left.get(column, 0)+len(f.terms)
    for A in (hnext, s):
        for (row, _), f in A.entries.items():
            right[row] = right.get(row, 0)+len(f.terms)
    return left, right


def _left_coefficients(A, B, support, caps):
    width = A.ncols*len(support)
    if width > caps['max_unknowns']:
        raise ResourceLimit("Shared structural coefficient unknowns exceed budget")
    # Multi-RHS entries can share (row, exponent): terms(B) is an exact
    # contribution count, but must not be used as an equation lower bound.
    _coefficient_preflight(width, len(support)*terms(A)+terms(B), 0, caps, shared=True)
    by_column = {}
    for (i, k), f in A.entries.items():
        by_column.setdefault(k, []).append((i, f))
    rows = _Rows(A.ring.p, caps['max_equations'], caps['max_terms'])
    for (i, j), f in B.entries.items():
        for e, c in f.terms.items():
            rows.add((i, e), j, c, rhs=True)
    variables = [(k, e) for e in support for k in range(A.ncols)]
    shifted = _ShiftedExponents()
    for index, (k, shift) in enumerate(variables):
        shifted.set_shift(shift)
        for i, f in by_column.get(k, ()):
            for e, c in f.terms.items():
                rows.add((i, shifted(e)), index, c)
    answer, info = _eliminate(list(rows.data.values()), width, B.ncols, A.ring.p,
                              max_steps=caps['max_steps'], max_coefficient_bits=caps['max_coefficient_bits'],
                              max_output_terms=caps['max_terms'])
    info.update(unknowns=width, equations=len(rows.data), contributions=rows.contributions)
    if answer is None:
        return None, info
    if len(answer) > caps['max_terms']:
        raise ResourceLimit("Shared coefficient solution exceeds term budget")
    entries = {}
    for (index, j), c in answer.items():
        k, e = variables[index]
        entries.setdefault((k, j), {})[e] = c
    X = Matrix(A.ring, A.ncols, B.ncols, {at: Poly(A.ring, f) for at, f in entries.items()})
    if product_work(A, X) > caps['max_product_work']:
        raise ResourceLimit("Shared coefficient left-equation verification exceeds product budget")
    if A*X != B:
        raise ValueError("Structural coefficient solution failed its left equation")
    return X, info


def _joint_coefficients(d, R, hnext, s, q, variables, caps):
    if len(variables) > caps['max_unknowns']:
        raise ResourceLimit("Joint structural coefficient unknowns exceed budget")
    left_counts, right_counts = _joint_term_counts(d, hnext, s, q)
    rhs_terms = terms(R)
    contributions = rhs_terms+sum(left_counts.get(i, 0)+right_counts.get(j, 0)
                                  for i, j, _ in variables)
    # Each R term has its own ('d', i, j, exponent) equation key. Operator
    # contributions may collide, so their count is not an equation bound.
    _coefficient_preflight(len(variables), contributions, rhs_terms, caps, shared=False)
    rows = _Rows(d.ring.p, caps['max_equations'], caps['max_terms'])
    lefts, rights = [], []
    for name, A in (('d', d), ('q', q)):
        columns = {}
        for (i, k), f in A.entries.items():
            columns.setdefault(k, []).append((i, f))
        lefts.append((name, columns))
    for name, A in (('hnext', hnext), ('s', s)):
        indexed = {}
        for (k, j), f in A.entries.items():
            indexed.setdefault(k, []).append((j, f))
        rights.append((name, indexed))
    for (i, j), f in R.entries.items():
        for e, c in f.terms.items():
            rows.add(('d', i, j, e), 0, c, rhs=True)
    shifted = _ShiftedExponents()
    for index, (i, j, shift) in enumerate(variables):
        shifted.set_shift(shift)
        for name, columns in lefts:
            for row, f in columns.get(i, ()):
                for e, c in f.terms.items():
                    rows.add((name, row, j, shifted(e)), index, c)
        for name, indexed in rights:
            for column, f in indexed.get(j, ()):
                for e, c in f.terms.items():
                    rows.add((name, i, column, shifted(e)), index, c)
    answer, info = _eliminate(list(rows.data.values()), len(variables), 1, d.ring.p,
                              max_steps=caps['max_steps'], max_coefficient_bits=caps['max_coefficient_bits'],
                              max_output_terms=caps['max_terms'])
    info.update(unknowns=len(variables), equations=len(rows.data), contributions=rows.contributions)
    if answer is None:
        return None, info
    entries = {}
    for (index, _), c in answer.items():
        i, j, e = variables[index]
        entries.setdefault((i, j), {})[e] = c
    return Matrix(d.ring, d.ncols, d.nrows, {at: Poly(d.ring, f) for at, f in entries.items()}), info


def solve_structural_section(d, R, hnext, s, q, Pprev, *, seed=None, max_radius=4,
                             max_unknowns=100000, max_equations=200000, max_terms=5000000,
                             max_steps=10000000, max_coefficient_bits=2000000000,
                             max_product_work=50000000, return_certificate=False, observer=None,
                             try_sparse_seed=False):
    """Return a verified X, or incomplete; no unbounded support search/GB.

    seed is an optional actual left lift d*seed=R, not frozen higher h.
    Right constraints are eliminated by exact units when possible, allowing
    a shared left solve for all surviving columns. Otherwise factorized
    coefficient constraints are assembled directly for just this X.
    """
    ring, m, r = _shapes(d, R, hnext, s, q, Pprev)
    caps = dict(max_unknowns=max_unknowns, max_equations=max_equations,
                max_terms=max_terms, max_steps=max_steps, max_coefficient_bits=max_coefficient_bits,
                max_product_work=max_product_work)
    for name, value in dict(caps, max_radius=max_radius, max_product_work=max_product_work).items():
        if type(value) is not int or value < 0:
            raise ValueError(name+" must be a nonnegative integer")
    if type(try_sparse_seed) is not bool:
        raise ValueError("try_sparse_seed must be boolean")
    def multiply(A, B):
        if product_work(A, B) > max_product_work:
            raise ResourceLimit("Structural exact verification product exceeds budget")
        return A*B
    if multiply(R, d) != d or multiply(d, Pprev) or multiply(q, Pprev) != q:
        raise ValueError("Invalid structural projection relations")
    if multiply(Pprev, Pprev) != Pprev:
        raise ValueError("Previous homology projector is not idempotent")
    bound = c4_degree_bound(d, R, hnext, s, q, Pprev)
    attempts = []
    def emit(info):
        attempts.append(info)
        if observer:
            observer(info)
    emit(dict(strategy='preflight', budgets=dict(caps, max_radius=max_radius),
              degree_bound=bound, complete_search=False))
    def finish(X, method):
        if (not isinstance(X, Matrix) or X.ring != ring or X.shape != (r, m)
                or multiply(d, X) != R or multiply(X, hnext)
                or multiply(X, s) or multiply(q, X)):
            raise ValueError("Structural section failed an original linear constraint")
        if terms(X) > max_terms:
            raise ResourceLimit("Structural section output support exceeds budget")
        if not bound_contains(bound, max((_weight(e) for f in X.entries.values()
                                         for e in f.terms), default=0)):
            raise ResourceLimit("Positive section exceeds the certified structural degree ceiling")
        # A positive solution itself certifies im(R) subset im(d). With
        # Rd=d this also proves idempotence and the reverse inclusion.
        from css_decoupling.certificates.structural_c4 import structural_section_dict
        cert = structural_section_dict(d, R, hnext, s, q, Pprev, X, method=method)
        return (X, cert) if return_certificate else X
    if seed is None and try_sparse_seed and ring.p == 2:
        # A bounded positive-only candidate preserves already-good sparse
        # choices. It is accepted only after every original side condition,
        # never frozen merely because its left equation holds.
        from .sparse_linear import sparse_binary_solution
        left = d.vstack(q) if q else d
        rhs = R.vstack(zeros(ring, q.nrows, m)) if q else R
        seed = sparse_binary_solution(left, rhs, max_terms=min(max_terms, 1000000),
            max_unknowns=min(max_unknowns, 20000),
            max_equations=min(max_equations, 40000),
            max_steps=min(max_steps, 4000000),
            max_coefficient_bits=min(max_coefficient_bits, 640000000),
            max_product_work=max_product_work)
    if seed is not None:
        if (not isinstance(seed, Matrix) or seed.ring != ring or seed.shape != (r, m)
                or multiply(d, seed) != R):
            raise ValueError("Structural seed is not an actual boundary lift")
        try:
            if not multiply(seed, hnext) and not multiply(seed, s) and not multiply(q, seed):
                return finish(seed.copy(), 'verified-seed')
        except ResourceLimit:
            pass
    right = hnext.hstack(s)
    quotient = right_annihilator_by_units(right, max_terms=max_terms, max_product_work=max_product_work)
    if quotient is not None:
        N, active = quotient
        left = d.vstack(q) if q else d
        rhs = R[:, active].vstack(zeros(ring, q.nrows, len(active))) if q else R[:, active]
        for radius in range(max_radius+1):
            try:
                support = _support(ring.D, radius, max_unknowns//max(1, r))
                support = [e for e in support if bound_contains(bound, _weight(e))]
                Y, info = _left_coefficients(left, rhs, support, caps)
                emit(dict(strategy='shared-coefficients', radius=radius, **info))
                if Y is not None:
                    X = multiply(Y, N)
                    return finish(X, 'annihilator-shared-coefficients')
            except ResourceLimit as exc:
                emit(dict(strategy='shared-coefficients', radius=radius, resource=str(exc)))
                break
    # The support seed is only an ansatz, not a correction or normalization.
    left_counts, right_counts = _joint_term_counts(d, hnext, s, q)
    rhs_terms = terms(R)
    full_shift_contributions = m*sum(left_counts.values())+r*sum(right_counts.values())
    def plans():
        if seed is not None and terms(seed)+r*m <= max_unknowns:
            zero = (0,)*ring.D
            width, contributions = r*m, rhs_terms+full_shift_contributions
            for (i, j), f in seed.entries.items():
                nonconstant = sum(e != zero for e in f.terms)
                width += nonconstant
                contributions += nonconstant*(left_counts.get(i, 0)+right_counts.get(j, 0))
            try:
                _coefficient_preflight(width, contributions, rhs_terms, caps, shared=False)
            except ResourceLimit as exc:
                emit(dict(strategy='seed-support', resource=str(exc)))
            else:
                variables = {(i, j, e) for (i, j), f in seed.entries.items() for e in f.terms}
                variables.update((i, j, zero) for i in range(r) for j in range(m))
                yield 'seed-support', sorted(variables)
        for radius in range(max_radius+1):
            try:
                support = _support(ring.D, radius, max_unknowns//max(1, r*m))
            except ResourceLimit:
                break
            support = [e for e in support if bound_contains(bound, _weight(e))]
            name = f'finite-radius-{radius}'
            try:
                _coefficient_preflight(r*m*len(support),
                    rhs_terms+len(support)*full_shift_contributions, rhs_terms, caps, shared=False)
            except ResourceLimit as exc:
                emit(dict(strategy=name, resource=str(exc)))
            else:
                yield name, [(i, j, e) for e in support for j in range(m) for i in range(r)]
    for name, variables in plans():
        try:
            X, info = _joint_coefficients(d, R, hnext, s, q, variables, caps)
            emit(dict(strategy=name, **info))
            if X is not None:
                return finish(X, 'factorized-coefficients')
        except ResourceLimit as exc:
            emit(dict(strategy=name, resource=str(exc)))
    exc = ResourceLimit("C4 structural coefficient supports/budgets exhausted; Laurent solvability undecided")
    exc.structural_attempts = attempts
    exc.structural_degree_bound = bound
    if quotient is not None:
        # A call-local candidate, not a success flag or a global cache. The
        # descending caller rechecks its binding and every quotient identity
        # before using it with another exact left solver.
        N, active = quotient
        exc.structural_quotient = dict(constraints=right.copy(), rhs=R.copy(),
                                       N=N.copy(), active=tuple(active))
    raise exc
