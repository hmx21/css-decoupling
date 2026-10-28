"""Canonical CSS models and all published BB inputs (no reference outputs).

Paper: Wang, Ni, He, Liu, Gu, arXiv:2608.09915v1 (2026).
BB/color structural data are adapted from the MIT-licensed Python companion
by Mingxin He, commit 7879852262ee7bd0bb69c1a2edae8f1543fcca3e.
Copyright (c) 2026 The Authors.  See the distributed upstream MIT notice.
"""
from __future__ import annotations

from dataclasses import dataclass
from .algebra import Matrix, Ring


def standard_code(toric_copies=1, *, product_x=0, product_z=0, ring=None):
    """Return (HX,HZ) with X products, Z products, then toric pairs.

    An X product has stabilizer X and state |+>; a Z product has stabilizer Z
    and state |0>. The signed Koszul pair works over every prime field.
    """
    ring = Ring(2, 2) if ring is None else ring
    if ring.D != 2:
        raise ValueError("The standard model requires a two-variable Laurent ring")
    counts = (toric_copies, product_x, product_z)
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in counts):
        raise ValueError("Toric and product counts must be nonnegative integers")
    t, px, pz = counts
    q = 2*t + px + pz
    HX, HZ = Matrix(ring, px+t, q), Matrix(ring, pz+t, q)
    x, y = ring.variables
    for i in range(px):
        HX[i, i] = 1
    for i in range(pz):
        HZ[i, px+i] = 1
    for i in range(t):
        c = px+pz+2*i
        HX[px+i, c], HX[px+i, c+1] = x-1, y-1
        HZ[pz+i, c], HZ[pz+i, c+1] = 1-y**-1, x**-1-1
    return HX, HZ


toric_code = standard_code


def product_state(*, x_count=0, z_count=0, ring=None):
    return standard_code(0, product_x=x_count, product_z=z_count, ring=ring)


def two_generator_css(f, g):
    """Return HX=[f,g], HZ=[-bar(g),bar(f)] over a prime Laurent ring.

    This guarantees CSS commutation, not the topological input hypotheses.
    The published BB/color catalog retains its original qubit interpretation.
    """
    if f.ring != g.ring or f.ring.D != 2:
        raise ValueError("Two-generator CSS inputs require the same two-variable ring")
    return (Matrix.from_rows(f.ring, [[f, g]]),
            Matrix.from_rows(f.ring, [[-g.involution(), f.involution()]]))


def _polynomial(exponents, ring):
    result = ring.zero()
    for exponent in exponents:
        result += ring.monomial(exponent)
    return result


@dataclass(frozen=True)
class BBCodeInstance:
    collection: str
    row: int
    f_exponents: tuple
    g_exponents: tuple
    reference_square_period: int
    superlattice_basis: tuple
    reference_q: int
    reference_p_x: int
    reference_p_z: int
    reference_t: int

    @property
    def identifier(self):
        return f"{self.collection}-{self.row:02d}"

    def polynomials(self, ring=None):
        ring = Ring(2, 2) if ring is None else ring
        return _polynomial(self.f_exponents, ring), _polynomial(self.g_exponents, ring)

    def checks(self, ring=None):
        # Catalog periods and product/toric counts are published qubit data.
        # A prime-field re-evaluation is a new input: construct it explicitly
        # with polynomials(ring) and two_generator_css, without these claims.
        if ring is not None and ring != Ring(2, 2):
            raise ValueError("Published BB catalog checks retain their qubit ring")
        return two_generator_css(*self.polynomials(ring))

    def as_reference_record(self):
        return {
            "id": self.identifier, "collection": self.collection, "row": self.row,
            "f_exponents": [list(e) for e in self.f_exponents],
            "g_exponents": [list(e) for e in self.g_exponents],
            "reference_square_period": self.reference_square_period,
            "superlattice_basis": [list(v) for v in self.superlattice_basis],
            "reference_q": self.reference_q, "reference_p_x": self.reference_p_x,
            "reference_p_z": self.reference_p_z, "reference_t": self.reference_t,
        }


@dataclass(frozen=True)
class ColorCodeInstance:
    label: str
    reference_square_period: int
    superlattice_basis: tuple
    reference_q: int
    reference_p_x: int
    reference_p_z: int
    reference_t: int

    def checks(self, ring=None):
        ring = Ring(2, 2) if ring is None else ring
        if ring != Ring(2, 2):
            raise ValueError("The paper color codes require the qubit ring")
        x, y = ring.variables
        if self.label == "666":
            return two_generator_css(1+x+x*y, 1+y+x*y)
        if self.label == "488":
            check = Matrix.from_rows(ring, [[1+y, 1+y, 1+x, 1+x],
                                          [x*y, y, x*y, x]])
            return check, check.copy()
        raise ValueError(f"Unknown color code: {self.label}")


COLOR_CODE_666 = ColorCodeInstance("666", 3, ((3, 0), (2, 1)), 6, 1, 1, 2)
COLOR_CODE_488 = ColorCodeInstance("488", 2, ((2, 0), (1, 1)), 8, 2, 2, 2)


def color_code_666(ring=None):
    return COLOR_CODE_666.checks(ring)


def color_code_488(ring=None):
    return COLOR_CODE_488.checks(ring)



def _benchmark_case(row, a, b, period, basis, q, p, t):
    """Build a row with ``f=1+x+a`` and ``g=1+y+b``."""
    return BBCodeInstance(
        "benchmark",
        row,
        ((0, 0), (1, 0), a),
        ((0, 0), (0, 1), b),
        period,
        basis,
        q,
        p,
        p,
        t,
    )


def _table_case(row, f, g, period, basis, q, p_x, p_z, t):
    """Build a row from the paper's arbitrary-polynomial table."""
    return BBCodeInstance(
        "table",
        row,
        tuple(f),
        tuple(g),
        period,
        basis,
        q,
        p_x,
        p_z,
        t,
    )


BENCHMARK_BB_CODE_INSTANCES = (
    _benchmark_case(1, (1, 1), (1, 1), 3, ((2, 1), (1, 2)), 6, 1, 2),
    _benchmark_case(2, (-1, 1), (1, 1), 7, ((3, 1), (2, 3)), 14, 4, 3),
    _benchmark_case(3, (2, 0), (2, 0), 3, ((2, 1), (1, 2)), 6, 1, 2),
    _benchmark_case(4, (-1, 0), (0, -1), 3, ((3, 0), (0, 3)), 18, 5, 4),
    _benchmark_case(5, (1, 1), (1, -1), 7, ((3, 2), (1, 3)), 14, 4, 3),
    _benchmark_case(6, (-1, 0), (3, 2), 3, ((3, 0), (0, 3)), 18, 5, 4),
    _benchmark_case(7, (0, -2), (-2, 0), 21, ((5, 2), (2, 5)), 42, 16, 5),
    _benchmark_case(8, (0, -2), (2, 0), 15, ((3, 3), (0, 5)), 30, 11, 4),
    _benchmark_case(9, (-1, 1), (-1, -1), 31, ((5, 3), (3, 8)), 62, 26, 5),
    _benchmark_case(10, (-2, -1), (2, 1), 3, ((1, 1), (0, 3)), 6, 1, 2),
    _benchmark_case(11, (-1, 3), (3, -1), 12, ((12, 0), (0, 12)), 288, 136, 8),
    _benchmark_case(12, (-2, 0), (-2, 2), 7, ((7, 0), (0, 7)), 98, 43, 6),
    _benchmark_case(13, (-2, 1), (1, -2), 63, ((15, 6), (6, 15)), 378, 181, 8),
    _benchmark_case(14, (-1, 2), (-2, -1), 217, ((19, 10), (3, 13)), 434, 209, 8),
    _benchmark_case(15, (-3, 1), (-5, 0), 21, ((1, 1), (0, 21)), 42, 16, 5),
    _benchmark_case(16, (-2, 1), (1, 2), 105, ((9, 1), (3, 12)), 210, 98, 7),
    _benchmark_case(17, (-1, -2), (1, -1), 63, ((7, 3), (0, 9)), 126, 57, 6),
    _benchmark_case(18, (0, 2), (-4, 1), 73, ((14, 5), (5, 7)), 146, 64, 9),
    _benchmark_case(19, (-1, 2), (0, -4), 105, ((35, 7), (0, 21)), 1470, 725, 10),
    _benchmark_case(20, (0, -4), (4, 0), 255, ((15, 13), (0, 17)), 510, 239, 16),
    _benchmark_case(21, (-4, 0), (-3, 2), 21, ((21, 0), (0, 21)), 882, 431, 10),
    _benchmark_case(22, (-2, -5), (-1, -3), 42, ((6, 6), (2, 16)), 168, 77, 7),
    _benchmark_case(23, (-8, -1), (5, 1), 21, ((4, 1), (3, 6)), 42, 16, 5),
    _benchmark_case(24, (1, -5), (1, 4), 186, ((18, 42), (10, 44)), 744, 363, 9),
    _benchmark_case(25, (-1, -1), (5, 0), 105, ((15, 6), (10, 11)), 210, 98, 7),
    _benchmark_case(26, (1, 3), (2, -2), 63, ((35, 7), (7, 14)), 882, 432, 9),
    _benchmark_case(27, (-1, -2), (2, -1), 217, ((13, 3), (10, 19)), 434, 209, 8),
    _benchmark_case(28, (-1, 3), (1, 3), 186, ((16, 10), (14, 32)), 744, 363, 9),
    _benchmark_case(29, (2, 2), (-4, 1), 889, ((101, 5), (4, 9)), 1778, 879, 10),
    _benchmark_case(30, (-1, 3), (3, 0), 217, ((17, 9), (8, 17)), 434, 209, 8),
)


TABLE_BB_CODE_INSTANCES = (
    _table_case(1, ((1, 0), (0, 0)), ((1, 0), (0, 1)), 1, ((1, 0), (0, 1)), 2, 0, 0, 1),
    _table_case(2, ((1, 0), (0, 2)), ((0, 2), (0, 1)), 1, ((1, 0), (0, 1)), 2, 0, 0, 1),
    _table_case(3, ((2, 1), (1, 1), (1, 0)), ((1, 1), (0, 2)), 3, ((2, 1), (1, 2)), 6, 1, 1, 2),
    _table_case(4, ((2, 0), (1, 0), (0, 0)), ((1, 0), (0, 1)), 3, ((2, 1), (1, 2)), 6, 1, 1, 2),
    _table_case(5, ((2, 1), (1, 1), (0, 0)), ((1, 1), (1, 0)), 3, ((3, 0), (0, 1)), 6, 1, 1, 2),
    _table_case(6, ((2, 2), (1, 0), (0, 0)), ((1, 2), (1, 1)), 3, ((3, 0), (0, 1)), 6, 1, 1, 2),
    _table_case(7, ((3, 1), (3, 0), (2, 1), (1, 1)), ((3, 1), (0, 2)), 4, ((1, 1), (0, 4)), 8, 1, 1, 3),
    _table_case(8, ((2, 3), (1, 0), (0, 0)), ((1, 3), (1, 2), (1, 1)), 3, ((3, 0), (0, 3)), 18, 5, 5, 4),
    _table_case(9, ((1, 0), (0, 1)), ((0, 4), (0, 3), (0, 2), (0, 1)), 4, ((2, 2), (1, 3)), 8, 1, 1, 3),
    _table_case(10, ((0, 1), (-1, 2), (-2, 2), (-2, 1)), ((1, 1), (0, 2), (0, 0), (-2, 0)), 3, ((1, 1), (0, 3)), 6, 0, 0, 3),
    _table_case(11, ((1, 2), (1, 0), (-1, 3), (-2, 4)), ((1, 3), (1, 1), (0, 2), (0, 0)), 40, ((2, 2), (0, 40)), 160, 71, 71, 9),
    _table_case(12, ((3, -2), (2, -1), (0, 2), (0, 0)), ((2, 1), (2, -1), (1, 0), (1, -2)), 40, ((2, 2), (0, 40)), 160, 71, 71, 9),
    _table_case(13, ((3, 1), (0, 0), (-1, 3), (-2, 4)), ((3, 3), (1, 1), (0, 2), (-2, 0)), 24, ((4, 4), (0, 24)), 192, 76, 76, 20),
    _table_case(14, ((3, 0), (1, 0), (-1, 3), (-2, 4)), ((3, 1), (2, 0), (1, 1), (0, 0)), 56, ((2, 2), (0, 56)), 224, 101, 101, 11),
    _table_case(15, ((3, 0), (0, 2), (0, 1)), ((2, 0), (1, 0), (0, 3)), 12, ((12, 0), (0, 12)), 288, 136, 136, 8),
    _table_case(16, ((27, 0), (6, 0), (0, 1)), ((24, 0), (15, 0), (0, 0)), 93, ((93, 0), (0, 1)), 186, 78, 78, 15),
    _table_case(17, ((9, 0), (0, 2), (0, 1)), ((8, 0), (1, 0), (0, 0)), 63, ((36, 27), (27, 36)), 1134, 551, 551, 16),
    _table_case(18, ((3, 0), (0, 2), (0, 1)), ((2, 0), (1, 0), (0, 0)), 3, ((3, 0), (0, 3)), 18, 5, 5, 4),
    _table_case(19, ((3, 1), (2, 2), (0, 0)), ((3, 2), (2, 0), (0, 0)), 105, ((21, 3), (0, 5)), 210, 98, 98, 7),
    _table_case(20, ((3, 2), (2, 3), (0, 0)), ((3, 1), (2, 0), (0, 0)), 217, ((14, 23), (7, 27)), 434, 209, 209, 8),
    _table_case(21, ((3, 2), (2, 0), (0, 0)), ((3, 0), (2, 1), (0, 0)), 7, ((7, 0), (0, 1)), 14, 4, 4, 3),
    _table_case(22, ((3, 1), (2, 0), (0, 0)), ((3, 2), (2, 0), (0, 0)), 7, ((7, 0), (0, 1)), 14, 4, 4, 3),
    _table_case(23, ((3, 0), (2, 0), (0, 1)), ((3, 0), (2, 0), (0, 0)), 7, ((7, 0), (0, 1)), 14, 4, 4, 3),
    _table_case(24, ((6, 2), (3, 0), (1, 3), (0, 0)), ((6, 1), (5, 1), (4, 1), (3, 2)), 120, ((8, 6), (0, 30)), 480, 230, 230, 10),
    _table_case(25, ((6, 0), (3, 0), (1, 1), (0, 0)), ((6, 1), (5, 1), (4, 1), (3, 0)), 8, ((8, 0), (0, 4)), 64, 25, 25, 7),
    _table_case(26, ((5, 1), (5, 0), (3, 0), (0, 0)), ((5, 1), (3, 0), (2, 1), (0, 1)), 186, ((18, 2), (12, 22)), 744, 362, 362, 10),
    _table_case(27, ((5, 3), (4, 4), (3, 5), (0, 5)), ((5, 1), (4, 3), (3, 3), (0, 2)), 56, ((28, 2), (0, 4)), 224, 103, 103, 9),
    _table_case(28, ((5, 1), (4, 0), (3, 1), (0, 1)), ((5, 1), (4, 1), (3, 1), (0, 0)), 56, ((28, 2), (0, 4)), 224, 103, 103, 9),
    _table_case(29, ((3, 0), (0, 7), (0, 2)), ((2, 0), (1, 0), (0, 3)), 126, ((12, 6), (6, 66)), 1512, 744, 744, 12),
)


BB_CODE_INSTANCES = BENCHMARK_BB_CODE_INSTANCES + TABLE_BB_CODE_INSTANCES

# Completed Julia result case_016 is absent from the upstream Python paper
# catalog. Preserve the original 59 Python identifiers; include this additional
# published input under its source-specific identifier for complete accounting.
JULIA_EXTRA_BB_CODE_INSTANCES = (
    BBCodeInstance("julia-case", 16,
        ((0, 0), (1, 0), (-3, 1)), ((0, 0), (0, 1), (3, 2)),
        63, ((21, 0), (0, 63)), 2646, 1312, 1312, 11),
)
PUBLISHED_BB_CODE_INSTANCES = BB_CODE_INSTANCES + JULIA_EXTRA_BB_CODE_INSTANCES


def select_bb_code_instances(selectors=None):
    """Select rows by stable identifier, failing on unknown identifiers."""
    if not selectors:
        return BB_CODE_INSTANCES
    requested = set(selectors)
    by_identifier = {case.identifier: case for case in BB_CODE_INSTANCES}
    unknown = sorted(requested - set(by_identifier))
    if unknown:
        raise ValueError(f"Unknown BB-code row identifiers: {', '.join(unknown)}")
    return tuple(case for case in BB_CODE_INSTANCES if case.identifier in requested)
