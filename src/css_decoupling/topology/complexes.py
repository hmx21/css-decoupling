"""Finite free cochain complexes over an explicit Laurent ring.

All differentials raise cohomological degree.  The physical qudits of a
CSS complex occupy degree zero.  Matrix solves in this module are Laurent
module solves, never linear algebra over the fraction field.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb
from typing import Callable, Mapping

from css_decoupling.algebra.laurent import Matrix, Ring, block, direct_sum, eye, zeros
from css_decoupling.errors import ResourceLimit


DEFAULT_KOSZUL_RANK_LIMIT = 100000


@dataclass
class CochainComplex:
    ring: Ring
    ranks: Mapping[int, int]
    differentials: Mapping[int, Matrix]

    def __post_init__(self):
        self.ranks = dict(self.ranks)
        self.differentials = dict(self.differentials)
        if any(not isinstance(n, int) or not isinstance(r, int) or r < 0
               for n, r in self.ranks.items()):
            raise ValueError("Complex degrees and nonnegative ranks must be integers")
        for n, differential in self.differentials.items():
            if differential.ring != self.ring:
                raise ValueError("Differential belongs to another ring")
            if (differential.nrows, differential.ncols) != (self.rank(n + 1), self.rank(n)):
                raise ValueError(f"Wrong differential shape in degree {n}")
        self.validate()

    @property
    def degrees(self):
        if not self.ranks:
            return ()
        return tuple(range(min(self.ranks), max(self.ranks) + 1))

    def rank(self, degree):
        return self.ranks.get(degree, 0)

    def d(self, degree):
        return self.differentials.get(
            degree, zeros(self.ring, self.rank(degree + 1), self.rank(degree)))

    def validate(self):
        for n in self.degrees:
            if self.d(n + 1) * self.d(n) != zeros(self.ring, self.rank(n + 2), self.rank(n)):
                raise ValueError(f"Differentials do not square to zero in degree {n}")
        return True


@dataclass
class CochainMap:
    source: CochainComplex
    target: CochainComplex
    components: Mapping[int, Matrix]

    def __post_init__(self):
        if self.source.ring != self.target.ring:
            raise ValueError("A cochain map requires the same coefficient ring")
        self.components = dict(self.components)
        self.validate()

    @property
    def degrees(self):
        return tuple(sorted(set(self.source.degrees) | set(self.target.degrees)))

    def at(self, degree):
        return self.components.get(degree, zeros(self.source.ring,
                                                 self.target.rank(degree),
                                                 self.source.rank(degree)))

    def validate(self):
        for n, value in self.components.items():
            if value.ring != self.source.ring or (value.nrows, value.ncols) != (
                    self.target.rank(n), self.source.rank(n)):
                raise ValueError(f"Wrong cochain map shape in degree {n}")
        for n in self.degrees:
            if self.target.d(n) * self.at(n) != self.at(n + 1) * self.source.d(n):
                raise ValueError(f"Map does not commute with differential in degree {n}")
        return True


def identity_map(complex):
    return CochainMap(complex, complex, {n: eye(complex.ring, complex.rank(n))
                                        for n in complex.degrees})


def direct_sum_complexes(*complexes):
    if not complexes:
        raise ValueError("At least one complex is needed to specify a ring")
    ring = complexes[0].ring
    if any(c.ring != ring for c in complexes):
        raise ValueError("Complexes belong to different rings")
    degrees = sorted(set().union(*(c.degrees for c in complexes)))
    ranks = {n: sum(c.rank(n) for c in complexes) for n in degrees}
    maps = {n: direct_sum(*(c.d(n) for c in complexes)) for n in degrees}
    return CochainComplex(ring, ranks, maps)


def disk(ring, degree, rank=1):
    """The contractible disk with identity from ``degree`` to ``degree+1``."""
    return CochainComplex(ring, {degree: rank, degree + 1: rank},
                          {degree: eye(ring, rank)})


def _koszul_size(ring, degree, copies, lowest_degree, max_total_rank):
    if not isinstance(degree, int) or not 1 <= degree < ring.D:
        raise ValueError("Toric degree must satisfy 1 <= degree < D")
    if not isinstance(copies, int) or copies < 0:
        raise ValueError("The number of copies must be nonnegative")
    if not isinstance(lowest_degree, int) or lowest_degree > degree:
        raise ValueError("The lowest Koszul degree must be an integer <= its top degree")
    if max_total_rank is not None and (not isinstance(max_total_rank, int) or max_total_rank < 1):
        raise ValueError("Koszul rank limit must be a positive integer or None")
    largest_k = min(ring.D, degree - lowest_degree)
    rank = 0
    for k in range(largest_k + 1):
        rank += copies * comb(ring.D, k)
        if max_total_rank is not None and rank > max_total_rank:
            raise ResourceLimit(f"Koszul model needs more than {max_total_rank} total free rank "
                                f"(D={ring.D}, degree={degree}, copies={copies})")
    return largest_k, rank


def _koszul_model(ring, degree, copies, lowest_degree, max_total_rank):
    largest_k, _ = _koszul_size(ring, degree, copies, lowest_degree, max_total_rank)
    if not copies:
        return CochainComplex(ring, {}, {})
    bases = [list(combinations(range(ring.D), k)) for k in range(largest_k + 1)]
    ranks = {degree - k: len(bases[k]) * copies for k in range(largest_k + 1)}
    differentials = {}
    for k in range(1, largest_k + 1):
        rows = {indices: i for i, indices in enumerate(bases[k - 1])}
        entries = {}
        for column, indices in enumerate(bases[k]):
            for position, axis in enumerate(indices):
                row = rows[indices[:position] + indices[position + 1:]]
                value = (-1) ** position * (ring.variable(axis) - ring.one())
                for copy in range(copies):
                    entries[row * copies + copy, column * copies + copy] = value
        differentials[degree - k] = Matrix(ring, len(bases[k - 1]) * copies,
                                           len(bases[k]) * copies, entries)
    return CochainComplex(ring, ranks, differentials)


def koszul(ring, degree=1, copies=1, *, max_total_rank=DEFAULT_KOSZUL_RANK_LIMIT):
    """Koszul model of ``copies`` toric codes TC_degree in dimension D.

    Increasing subsets index exterior powers; the copy index varies fastest.
    K_k occupies cochain degree ``degree-k`` and the differential is
    sum_a (-1)^a (x_{i_a}-1) e_{I without i_a}.  This sign convention works
    in odd characteristic as well as characteristic two.
    """
    return _koszul_model(ring, degree, copies, degree - ring.D, max_total_rank)


def koszul_window(ring, degree=1, copies=1, *, lowest_degree=-1,
                  max_total_rank=DEFAULT_KOSZUL_RANK_LIMIT):
    """Only the Koszul degrees needed above a specified physical window.

    The default keeps K_0,...,K_(j+1), hence the physical X and Z maps.
    ``lowest_degree=0`` retains just the comparison-map terms.  For fixed
    degree j the total rank is polynomial in D, avoiding the complete
    exterior algebra's 2**D allocation.  A rank preflight precedes enumeration.
    """
    return _koszul_model(ring, degree, copies, lowest_degree, max_total_rank)


def _toric_model(ring, profile, lowest_degree, max_total_rank):
    if any(not isinstance(j, int) or not 1 <= j < ring.D or
           not isinstance(t, int) or t < 0 for j, t in profile.items()):
        raise ValueError("Profile requires integer 1 <= j < D and multiplicities >= 0")
    total = 0
    for j, t in sorted(profile.items()):
        low = j - ring.D if lowest_degree is None else lowest_degree
        _, rank = _koszul_size(ring, j, t, low, max_total_rank)
        total += rank
        if max_total_rank is not None and total > max_total_rank:
            raise ResourceLimit(f"Toric stack needs total free rank {total}; cap is {max_total_rank}")
    pieces = [_koszul_model(ring, j, t, j - ring.D if lowest_degree is None else lowest_degree,
                            max_total_rank) for j, t in sorted(profile.items()) if t]
    return direct_sum_complexes(*pieces) if pieces else CochainComplex(ring, {}, {})


def toric_stack(ring, profile, *, max_total_rank=DEFAULT_KOSZUL_RANK_LIMIT):
    """A direct sum ordered first by excitation degree, then subset and copy."""
    return _toric_model(ring, profile, None, max_total_rank)


def toric_window(ring, profile, *, lowest_degree=-1,
                 max_total_rank=DEFAULT_KOSZUL_RANK_LIMIT):
    """Direct sum of truncated Koszul models, with aggregate rank preflight."""
    return _toric_model(ring, profile, lowest_degree, max_total_rank)


def mapping_cone(phi):
    """Cone(phi)^n = target^n + source^(n+1), differential [d,phi;0,-d]."""
    C, T = phi.target, phi.source
    degrees = sorted(set(C.degrees) | {n - 1 for n in T.degrees})
    ranks = {n: C.rank(n) + T.rank(n + 1) for n in degrees}
    differentials = {
        n: block([[C.d(n), phi.at(n + 1)],
                  [zeros(C.ring, T.rank(n + 2), C.rank(n)), -T.d(n + 1)]])
        for n in degrees}
    return CochainComplex(C.ring, ranks, differentials)


def validate_homotopy(source, target, left, right, homotopy, *, min_degree=None):
    """Check left-right = d_target h + h d_source in every degree."""
    for n in sorted(set(source.degrees) | set(target.degrees)):
        if min_degree is not None and n < min_degree:
            continue
        hn = homotopy.get(n, zeros(source.ring, target.rank(n - 1), source.rank(n)))
        hnext = homotopy.get(n + 1, zeros(source.ring, target.rank(n), source.rank(n + 1)))
        if target.d(n - 1) * hn + hnext * source.d(n) != left.at(n) - right.at(n):
            raise ValueError(f"Invalid homotopy in degree {n}")
    return True


def solve_contraction(complex, solve: Callable, *, min_degree=None):
    """Cancel exact unit disks, solve the residual, and verify the pullback.

    Unit cancellation is an explicit chain deformation retract. The original
    differential is unchanged, and the final contraction is checked there.
    A residual without unit entries still uses the complete Laurent solver.
    """
    from .contractions import solve_contraction_with_units
    return solve_contraction_with_units(complex, solve,
                                        residual_solver=_solve_contraction_descending,
                                        min_degree=min_degree)


def _solve_contraction_descending(complex, solve: Callable, *, min_degree=None):
    """Contract a bounded acyclic free complex by descending Laurent solves.

    ``solve(A,B)`` must return X with A X = B in the Laurent ring, raising
    a precise no-solution/resource exception otherwise.  This routine never
    asks for a basis of a projective cycle module.  An acyclic bounded free
    complex is split exact; descending induction makes every RHS a boundary.
    """
    h = {}
    ring = complex.ring
    for n in reversed(complex.degrees):
        if min_degree is not None and n < min_degree:
            continue
        hnext = h.get(n + 1, zeros(ring, complex.rank(n), complex.rank(n + 1)))
        rhs = eye(ring, complex.rank(n)) - hnext * complex.d(n)
        hn = solve(complex.d(n - 1), rhs)
        if (hn.nrows, hn.ncols) != (complex.rank(n - 1), complex.rank(n)) or complex.d(n - 1) * hn != rhs:
            raise ValueError(f"Solver returned an invalid contraction in degree {n}")
        h[n] = hn
    zero = CochainMap(complex, complex, {})
    validate_homotopy(complex, complex, identity_map(complex), zero, h, min_degree=min_degree)
    return h


def homotopy_inverse(phi, solve: Callable, *, return_contraction=False, strategy="direct"):
    """Produce a full inverse from a verified comparison-cone contraction.

    ``weak`` uses unit cancellation and complete descending Laurent solves;
    ``direct`` (the public ``sections`` strategy) additionally imposes
    square-zero while solving. Every strategy verifies inverse homotopies in every degree. Schanuel's
    strong conditions are separate, not waived by a weak comparison cone.
    """
    if strategy not in ("direct", "weak"):
        raise ValueError("Comparison contraction strategy must be direct or weak")
    C, T = phi.target, phi.source
    cone = mapping_cone(phi)
    if strategy == "direct":
        from .direct_splitting import solve_strong_contraction
        h = solve_strong_contraction(cone, solve, name="gamma")
    else:
        h = solve_contraction(cone, solve)
    psi, hc, ht = {}, {}, {}
    for n in sorted(set(C.degrees) | set(T.degrees)):
        hn = h.get(n, zeros(C.ring, cone.rank(n - 1), cone.rank(n)))
        psi[n] = hn[C.rank(n - 1):, :C.rank(n)]
        hc[n] = hn[:C.rank(n - 1), :C.rank(n)]
        hprev = h.get(n - 1, zeros(C.ring, cone.rank(n - 2), cone.rank(n - 1)))
        ht[n] = -hprev[C.rank(n - 2):, C.rank(n - 1):]
    inverse = CochainMap(C, T, psi)
    phipsi = CochainMap(C, C, {n: phi.at(n) * inverse.at(n) for n in C.degrees})
    psiphi = CochainMap(T, T, {n: inverse.at(n) * phi.at(n) for n in T.degrees})
    validate_homotopy(C, C, identity_map(C), phipsi, hc)
    validate_homotopy(T, T, identity_map(T), psiphi, ht)
    return (inverse, hc, ht, h) if return_contraction else (inverse, hc, ht)


def homotopy_inverse_window(phi, *, physical_boundary, solve: Callable):
    """Comparison inverse in degrees >=0, with homotopies in degrees >=1.

    ``phi`` compares the nonnegative truncations Tpos->Cpos.  The original
    Z-check boundary is supplied separately, mapping C^-1->C^0.  The cone
    starts at degree -1 with C^-1+T^0; its contraction is needed only from
    degree zero upwards.  No negative free resolution or Phi^-1 is used.

    Exactness at cone degree zero follows from H^1(phi) being injective and
    ker(d_C^0)=im(physical_boundary).  In particular no assertion about the
    degree-zero cohomology of the brutally truncated Tpos is necessary.
    """
    C, T = phi.target, phi.source
    if any(n < 0 for n in C.degrees + T.degrees):
        raise ValueError("Window comparison complexes must have no negative degrees")
    if physical_boundary.ring != C.ring or physical_boundary.nrows != C.rank(0):
        raise ValueError("Physical boundary must map into C^0 over the same ring")
    if C.d(0) * physical_boundary != zeros(C.ring, C.rank(1), physical_boundary.ncols):
        raise ValueError("Physical boundary is not a cycle")
    degrees = sorted({-1} | set(C.degrees) | {n - 1 for n in T.degrees})
    ranks = {n: C.rank(n) + T.rank(n + 1) for n in degrees}
    ranks[-1] += physical_boundary.ncols
    differentials = {n: block([[C.d(n), phi.at(n + 1)],
                               [zeros(C.ring, T.rank(n + 2), C.rank(n)), -T.d(n + 1)]])
                     for n in degrees if n >= 0}
    differentials[-1] = block([[physical_boundary, phi.at(0)],
                               [zeros(C.ring, T.rank(1), physical_boundary.ncols), -T.d(0)]])
    cone = CochainComplex(C.ring, ranks, differentials)
    h = solve_contraction(cone, solve, min_degree=0)
    psi, hc, ht = {}, {}, {}
    for n in sorted(set(C.degrees) | set(T.degrees)):
        hn = h.get(n, zeros(C.ring, cone.rank(n - 1), cone.rank(n)))
        offset = physical_boundary.ncols if n == 0 else C.rank(n - 1)
        psi[n] = hn[offset:, :C.rank(n)]
        if n >= 1:
            hc[n] = hn[:C.rank(n - 1), :C.rank(n)]
            hprev = h.get(n - 1, zeros(C.ring, cone.rank(n - 2), cone.rank(n - 1)))
            prior_offset = physical_boundary.ncols if n == 1 else C.rank(n - 2)
            ht[n] = -hprev[prior_offset:, C.rank(n - 1):]
    inverse = CochainMap(C, T, psi)
    phipsi = CochainMap(C, C, {n: phi.at(n) * inverse.at(n) for n in C.degrees})
    psiphi = CochainMap(T, T, {n: inverse.at(n) * phi.at(n) for n in T.degrees})
    validate_homotopy(C, C, identity_map(C), phipsi, hc, min_degree=1)
    validate_homotopy(T, T, identity_map(T), psiphi, ht, min_degree=1)
    return inverse, hc, ht


def solve_matrix_equations(ring, shapes, equations, solve: Callable):
    """Solve simultaneous equations sum A X_key B = rhs.

    ``shapes`` maps variable keys to (rows,columns); each equation is
    ``(terms,rhs)`` where a term is ``(A,key,B)``.  Signs are represented in
    A or B.  This vectorizes only matrix entries, without a coefficient
    degree bound or polynomial enumeration.  A single Laurent module solve
    supplies the polynomial coefficients of all unknown matrices.
    """
    # A X I = B is already a matrix RHS membership problem. Forming
    # I tensor A repeats the same presentation once per column and loses
    # the backend's shared reduction across all right-hand sides.
    if len(shapes) == 1 and len(equations) == 1:
        terms, rhs = equations[0]
        if len(terms) == 1:
            A, key, B = terms[0]
            if any(matrix.ring != ring for matrix in (A, B, rhs)):
                raise ValueError("Matrix equations require the specified common ring")
            rows, cols = shapes[key]
            if (A.ncols, B.nrows, A.nrows, B.ncols) != (rows, cols, rhs.nrows, rhs.ncols):
                raise ValueError(f"Incompatible matrix equation term for {key!r}")
            if B == eye(ring, cols):
                result = solve(A, rhs)
                if result.ring != ring or result.shape != (rows, cols) or A * result != rhs:
                    raise ValueError("Invalid simultaneous matrix-equation solution")
                return {key: result}
    offsets, width = {}, 0
    for key, (rows, cols) in shapes.items():
        offsets[key] = width
        width += rows * cols
    entries, rhs_entries, row_offset = {}, {}, 0
    for terms, rhs in equations:
        for (i, j), value in rhs.entries.items():
            rhs_entries[row_offset + j * rhs.nrows + i, 0] = value
        for A, key, B in terms:
            rows, cols = shapes[key]
            if (A.ncols, B.nrows, A.nrows, B.ncols) != (rows, cols, rhs.nrows, rhs.ncols):
                raise ValueError(f"Incompatible matrix equation term for {key!r}")
            for (i, k), a in A.entries.items():
                for (ell, j), b in B.entries.items():
                    index = (row_offset + j * rhs.nrows + i,
                             offsets[key] + ell * rows + k)
                    entries[index] = entries.get(index, ring.zero()) + a * b
        row_offset += rhs.nrows * rhs.ncols
    operator = Matrix(ring, row_offset, width, entries)
    rhs = Matrix(ring, row_offset, 1, rhs_entries)
    result = solve(operator, rhs)
    if operator * result != rhs:
        raise ValueError("Invalid simultaneous matrix-equation solution")
    answer = {}
    for key, (rows, cols) in shapes.items():
        answer[key] = Matrix(ring, rows, cols, {
            (i, j): result[offsets[key] + j * rows + i, 0]
            for j in range(cols) for i in range(rows)})
    return answer
