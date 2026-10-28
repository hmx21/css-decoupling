"""Dimension-independent example inputs; none selects a special solver path.

The glued family extends the manuscript's displayed three-dimensional
example by exterior contractions.  These constructors return ordinary CSS
matrices that pass through exactly the same automatic pipeline as user input.
"""
from itertools import combinations
from math import comb

from css_decoupling.algebra.laurent import Matrix, block, eye, zeros
from css_decoupling.topology.complexes import CochainComplex, koszul, koszul_window, toric_window


def css_checks(complex):
    """Return (H_X,H_Z) from a complex's physical window."""
    return complex.d(0), complex.d(-1).dagger()


def toric_css(ring, profile):
    return css_checks(toric_window(ring, profile))


def _exterior_coupling(ring, lower_degree, upper_degree, n, axes, coefficient):
    source_k, target_k = upper_degree - n, lower_degree - n - 1
    source_rank = comb(ring.D, source_k) if 0 <= source_k <= ring.D else 0
    target_rank = comb(ring.D, target_k) if 0 <= target_k <= ring.D else 0
    entries = {}
    if source_rank and target_rank:
        sources = list(combinations(range(ring.D), source_k))
        targets = {indices: i for i, indices in enumerate(combinations(range(ring.D), target_k))}
        for column, indices in enumerate(sources):
            if not set(axes).issubset(indices):
                continue
            remainder = list(indices)
            sign = -1 if ((len(axes) + 1) * (n + 1)) % 2 else 1
            for axis in axes:
                position = remainder.index(axis)
                sign *= (-1) ** position
                remainder.pop(position)
            entries[targets[tuple(remainder)], column] = sign * coefficient
    return Matrix(ring, target_rank, source_rank, entries)


def glued_toric_complex(ring, lower_degree=1, upper_degree=2, *,
                       axes=None, coefficient=1):
    """Two layers coupled by an exterior Ext representative.

    Let r=upper_degree-lower_degree+1.  The coupling from the upper toric
    summand in degree n to the lower one in degree n+1 is

      (-1)^((r+1)(n+1)) * coefficient * i_(a_r)...i_(a_1).

    Each i_a contracts the ordered exterior basis by the a-th dual vector.
    The degree-dependent sign makes this coupling anticommute with the
    Koszul differential.  For lower=1, upper=2, axes=(0,1), coefficient=1,
    D=3, the two physical off-diagonal entries are exactly -1 and +1 as
    displayed in the manuscript (including in odd characteristic).

    A nonzero constant coefficient represents nonzero gluing; zero gives a
    split control.  Contraction axes are enumerated in strictly increasing
    order.  Exterior bases are the actual output bases, not candidate maps.
    """
    if not isinstance(lower_degree, int) or not isinstance(upper_degree, int) or not (
            1 <= lower_degree < upper_degree < ring.D):
        raise ValueError("Require 1 <= lower_degree < upper_degree < D")
    order = upper_degree - lower_degree + 1
    axes = tuple(range(order)) if axes is None else tuple(axes)
    if len(axes) != order or tuple(sorted(set(axes))) != axes or any(
            not isinstance(axis, int) or not 0 <= axis < ring.D for axis in axes):
        raise ValueError("Contraction axes must be distinct increasing valid indices")
    coefficient = ring(coefficient)
    if not coefficient.is_constant:
        raise ValueError("This example family uses a constant Ext coefficient")
    lower = koszul(ring, lower_degree)
    upper = koszul(ring, upper_degree)
    degrees = sorted(set(lower.degrees) | set(upper.degrees))
    ranks = {n: lower.rank(n) + upper.rank(n) for n in degrees}
    differentials = {}
    for n in degrees:
        coupling = _exterior_coupling(ring, lower_degree, upper_degree, n, axes, coefficient)
        differentials[n] = block([[lower.d(n), coupling],
                                  [zeros(ring, upper.rank(n + 1), lower.rank(n)), upper.d(n)]])
    return CochainComplex(ring, ranks, differentials)


def glued_css(ring, lower_degree=1, upper_degree=2, *, axes=None, coefficient=1):
    return css_checks(glued_toric_complex(ring, lower_degree, upper_degree,
                                         axes=axes, coefficient=coefficient))


def glued_toric_chain(ring, degrees, *, coefficient=1):
    """Any increasing list of layers with successive nonzero gluing.

    Consecutive layers are coupled by the first r coordinate contractions,
    where r is their degree gap plus one.  Every contraction contains axis
    zero; hence products of two neighboring couplings vanish.  The resulting
    upper-triangular differential squares to zero without adding hidden
    higher blocks.  Only degrees >=-1 are allocated.  This is an ordinary
    example constructor, never a branch in the decoupling algorithm.
    """
    degrees = tuple(degrees)
    if len(degrees) < 2 or tuple(sorted(set(degrees))) != degrees or any(
            not isinstance(j, int) or not 1 <= j < ring.D for j in degrees):
        raise ValueError("Glued chain needs at least two strictly increasing valid layer degrees")
    coefficient = ring(coefficient)
    if not coefficient.is_constant:
        raise ValueError("This example family uses a constant Ext coefficient")
    pieces = [koszul_window(ring, j) for j in degrees]
    all_degrees = sorted(set().union(*(C.degrees for C in pieces)))
    ranks = {n: sum(C.rank(n) for C in pieces) for n in all_degrees}
    differentials = {}
    for n in all_degrees:
        rows = []
        for i, C in enumerate(pieces):
            row = []
            for k, T in enumerate(pieces):
                if i == k:
                    row.append(C.d(n))
                elif k == i + 1:
                    axes = tuple(range(degrees[k] - degrees[i] + 1))
                    row.append(_exterior_coupling(ring, degrees[i], degrees[k], n, axes, coefficient))
                else:
                    row.append(zeros(ring, C.rank(n + 1), T.rank(n)))
            rows.append(row)
        differentials[n] = block(rows)
    return CochainComplex(ring, ranks, differentials)


def shear_css(HX, HZ, target, control, coefficient):
    """Conjugate by one invertible Laurent transvection on physical Z data."""
    if HX.ring != HZ.ring or HX.ncols != HZ.ncols:
        raise ValueError("CSS checks must share a ring and physical columns")
    if not 0 <= target < HX.ncols or not 0 <= control < HX.ncols or target == control:
        raise ValueError("A shear needs two distinct physical coordinates")
    shear = Matrix(HX.ring, HX.ncols, HX.ncols, {(target, control): coefficient})
    U, Uinv = eye(HX.ring, HX.ncols) + shear, eye(HX.ring, HX.ncols) - shear
    return HX * Uinv, HZ * U.dagger()


def periodic_toric_cone(ring, lower_degree=1, *, axes=(0, 1), coefficient=1):
    """Explicit adjacent-layer cone and its geometric p-periodic null-homotopy.

    Return ``(presentation, lattice_basis, homotopy)`` for use with
    ``decouple_cone``.  For axes a<b, restrict only axis b by p.  In the
    coset fiber, Q e_s=s e_(s-1) for 1<=s<p and Q e_0=0.  Then
    [Q,x_b-1]=I and h^n=(-1)^n coefficient*i_a*Q cancels the exterior
    coupling.  No module solve or search is used in this example witness.

    The target remains the two restricted toric complexes, with p times
    their original generator counts.  It is not the smaller canonical
    coarse-lattice toric presentation.  All physical SUM families in this
    witness have disjoint species and can run in one layer.  In D=3,p=2
    there are two CNOT translation families in that layer (the figure
    shows the x_1,x_2 family, while x_1,x_3 plaquettes give the other).
    """
    from css_decoupling.decoupling.cone_decoupling import ConePresentation
    if isinstance(lower_degree, bool) or not isinstance(lower_degree, int) or not 1 <= lower_degree < ring.D - 1:
        raise ValueError("Adjacent toric layers require 1 <= lower_degree < D-1")
    axes = tuple(axes)
    if len(axes) != 2 or any(isinstance(a, bool) or not isinstance(a, int) for a in axes) or not (
            0 <= axes[0] < axes[1] < ring.D):
        raise ValueError("Choose two increasing coordinate axes")
    coefficient = ring(coefficient)
    if not coefficient.is_constant:
        raise ValueError("The periodic exterior example requires a constant coefficient")
    a, b = axes
    lower = koszul_window(ring, lower_degree)
    upper = koszul_window(ring, lower_degree + 1)
    degrees = sorted(set(lower.degrees) | set(upper.degrees))
    coupling = {n: _exterior_coupling(ring, lower_degree, lower_degree + 1, n, axes, coefficient)
                for n in degrees}
    cone = ConePresentation(lower, upper, coupling)
    basis = [[(ring.p if i == b else 1) if i == j else 0 for j in range(ring.D)]
             for i in range(ring.D)]
    homotopy = {}
    for n in degrees:
        k = lower_degree + 1 - n
        source = list(combinations(range(ring.D), k)) if 0 <= k <= ring.D else []
        target = ({indices: i for i, indices in enumerate(combinations(range(ring.D), k - 1))}
                  if 0 <= k - 1 <= ring.D else {})
        entries = {}
        for col, indices in enumerate(source):
            if a not in indices:
                continue
            row = target[tuple(i for i in indices if i != a)]
            sign = (-1 if n % 2 else 1) * (-1 if indices.index(a) % 2 else 1)
            for fiber in range(1, ring.p):
                entries[row * ring.p + fiber - 1, col * ring.p + fiber] = sign * fiber * coefficient
        homotopy[n] = Matrix(ring, lower.rank(n) * ring.p, upper.rank(n) * ring.p, entries)
    return cone, basis, homotopy
