"""Exact cancellation of Laurent-unit disks before cone contraction.

Only the supplied auxiliary complex changes presentation.  Inclusion,
projection and a homotopy retain an explicit equivalence to the original
complex; no CSS input or physical window is edited.  A residual complex
without unit pivots is passed to the ordinary complete contraction solver.
"""
from dataclasses import dataclass
from numbers import Integral

from css_decoupling.algebra.laurent import Matrix, eye, zeros
from css_decoupling.topology.complexes import (
    CochainComplex, CochainMap, identity_map, validate_homotopy,
)


@dataclass(frozen=True)
class UnitPivot:
    """A cancelled differential entry in the current, shrinking coordinates."""

    degree: int
    row: int
    column: int
    unit: object


@dataclass
class UnitReduction:
    """A verified deformation retract: p*i=1 and 1-i*p=d*h+h*d."""

    source: CochainComplex
    residual: CochainComplex
    inclusion: CochainMap
    projection: CochainMap
    homotopy: dict
    steps: tuple

    def validate(self):
        """Multiply the final identities in the original coordinates."""
        C, E = self.source, self.residual
        if self.inclusion.source != E or self.inclusion.target != C or (
                self.projection.source != C or self.projection.target != E):
            raise ValueError("Unit reduction maps have incompatible complexes")
        C.validate()
        E.validate()
        self.inclusion.validate()
        self.projection.validate()
        for n in sorted(set(C.degrees) | set(E.degrees)):
            if self.projection.at(n)*self.inclusion.at(n) != eye(C.ring, E.rank(n)):
                raise ValueError("Unit reduction projection and inclusion are not inverse on the residual")
        # I and P were independently checked as chain maps above.  Check the
        # homotopy identities directly; constructing another CochainMap for
        # I*P would redundantly multiply the large composite by every d.
        for n in C.degrees:
            hn = self.homotopy.get(n, zeros(C.ring, C.rank(n-1), C.rank(n)))
            hnext = self.homotopy.get(n+1, zeros(C.ring, C.rank(n), C.rank(n+1)))
            if C.d(n-1)*hn + hnext*C.d(n) != eye(C.ring, C.rank(n)) - self.inclusion.at(n)*self.projection.at(n):
                raise ValueError(f"Invalid unit-reduction homotopy in degree {n}")
        return True


def _minimum_degree(value):
    if value is not None and (isinstance(value, bool) or not isinstance(value, Integral)):
        raise ValueError("min_degree must be an integer or None")
    return None if value is None else int(value)


def _unit_pivot(A):
    """Choose a unit with small prospective row/column fill within one degree."""
    positions = [at for at, value in A.entries.items() if value.is_unit]
    if not positions:
        return None
    rows, columns = {}, {}
    for (i, j), value in A.entries.items():
        rows[i] = rows.get(i, 0) + len(value.terms)
        columns[j] = columns.get(j, 0) + len(value.terms)
    return min(positions, key=lambda at: (
        (rows[at[0]]-1)*(columns[at[1]]-1), rows[at[0]]+columns[at[1]], at))


def cancel_unit_disks(C, *, min_degree=None):
    """Cancel unit disks from high differential degree to low degree.

    For the unit a=d^n[i,j], let h=a^-1 e_j e_i^T.  With row i and column j
    removed, the local inclusion/projection are

      i_n=(1-h*d) inclusion_j,       p_n=projection_j,
      i_(n+1)=inclusion_i,           p_(n+1)=projection_i*(1-d*h).

    The new differential is the Schur complement
    ``d[!i,!j]-d[!i,j]*a^-1*d[i,!j]``.  The adjacent maps only delete the
    corresponding row or column.  Local homotopies are accumulated as
    ``H <- H + I*h*P``, before updating the accumulated I and P.

    In a window beginning at m, disks with target degree at least m may be
    cancelled, including d^(m-1).  The resulting retract identities hold
    in every degree, even outside the requested contraction window.
    Each pivot reduces total free rank by two, so this process is finite.
    Failure to find another unit says nothing about residual contractibility.
    """
    if not isinstance(C, CochainComplex):
        raise TypeError("Unit cancellation requires a CochainComplex")
    min_degree = _minimum_degree(min_degree)
    C.validate()
    ring = C.ring
    ranks, differentials = dict(C.ranks), dict(C.differentials)
    inclusions = {n: eye(ring, C.rank(n)) for n in C.degrees}
    projections = {n: eye(ring, C.rank(n)) for n in C.degrees}
    homotopy = {n: zeros(ring, C.rank(n-1), C.rank(n)) for n in C.degrees}
    steps = []

    def differential(n):
        return differentials.get(n, zeros(ring, ranks.get(n+1, 0), ranks.get(n, 0)))

    for n in reversed(C.degrees):
        if min_degree is not None and n+1 < min_degree:
            continue
        while True:
            A = differential(n)
            position = _unit_pivot(A)
            if position is None:
                break
            i, j = position
            unit = A[i, j]
            inverse = unit ** -1
            keep_rows = [k for k in range(A.nrows) if k != i]
            keep_columns = [k for k in range(A.ncols) if k != j]
            row, column = A[i:i+1, keep_columns], A[keep_rows, j:j+1]
            source_column = inclusions[n][:, j:j+1]
            target_row = projections[n+1][i:i+1, :]
            homotopy[n+1] = homotopy[n+1] + (source_column*inverse)*target_row

            # Compose the actual local i,p into original/residual coordinates.
            inclusions[n] = inclusions[n][:, keep_columns] - (source_column*inverse)*row
            inclusions[n+1] = inclusions[n+1][:, keep_rows]
            projections[n] = projections[n][keep_columns, :]
            projections[n+1] = projections[n+1][keep_rows, :] - (column*inverse)*target_row

            previous, following = differential(n-1), differential(n+1)
            differentials[n] = A[keep_rows, keep_columns] - (column*inverse)*row
            differentials[n-1] = previous[keep_columns, :]
            differentials[n+1] = following[:, keep_rows]
            ranks[n] -= 1
            ranks[n+1] -= 1
            steps.append(UnitPivot(n, i, j, unit))

    residual = CochainComplex(ring, ranks, differentials)
    answer = UnitReduction(C, residual, CochainMap(residual, C, inclusions),
                           CochainMap(C, residual, projections), homotopy, tuple(steps))
    answer.validate()
    return answer


def solve_contraction_with_units(C, solve, *, residual_solver, min_degree=None):
    """Reduce, solve the residual by the caller's complete algorithm, pull back.

    ``residual_solver(E, solve, min_degree=...)`` must be the ordinary
    descending solver, not this wrapper.  It is not called for an empty
    requested residual window.  Otherwise all no-solution/resource failures
    propagate unchanged.  The returned dict uses the original complex's
    ranks and is checked by ``d*h+h*d=I`` in every requested degree.
    """
    min_degree = _minimum_degree(min_degree)
    reduction = cancel_unit_disks(C, min_degree=min_degree)
    requested = tuple(n for n in C.degrees if min_degree is None or n >= min_degree)
    E = reduction.residual
    residual_h = (residual_solver(E, solve, min_degree=min_degree)
                  if any(E.rank(n) for n in requested) else {})
    answer = {}
    for n in requested:
        small = residual_h.get(n, zeros(C.ring, E.rank(n-1), E.rank(n)))
        if not isinstance(small, Matrix) or small.ring != C.ring or small.shape != (E.rank(n-1), E.rank(n)):
            raise ValueError("Residual contraction has incompatible dimensions")
        answer[n] = (reduction.homotopy[n]
                     + reduction.inclusion.at(n-1)*small*reduction.projection.at(n))
    validate_homotopy(C, C, identity_map(C), CochainMap(C, C, {}),
                      answer, min_degree=min_degree)
    return answer
