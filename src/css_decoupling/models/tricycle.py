"""Raw Laurent checks for the published trivariate tricycle (TT) family.

The convention is Eqs. (13)--(14) of Jacob, McLauchlan and Browne,
https://arxiv.org/html/2508.08191v3#S2, with finite shift transposes replaced
by Laurent adjoints.  The three named examples reproduce Table I literally.
Their published periods and [[n,k,d]] describe finite quotients only: they
are not assumptions about infinite-lattice exactness, charge dimension,
translation periods, or a tractable decoupling cell.  None of those answers
is passed to the solver by these constructors.

This literature family is binary and three-dimensional.  Use the signed
Koszul constructors for general prime-dimensional examples.
"""
from dataclasses import dataclass
from types import MappingProxyType

from css_decoupling.algebra.laurent import Matrix, Ring
from css_decoupling.topology.complexes import CochainComplex


SOURCE = "https://arxiv.org/html/2508.08191v3#S2.T1"


def _require_tt_ring(ring):
    if not isinstance(ring, Ring) or (ring.p, ring.D) != (2, 3):
        raise ValueError("The published TT convention requires Ring(2, 3)")


def tricycle_complex(ring, polynomials):
    """Return the raw length-three TT complex, without assuming exactness.

    For the supplied ``(a,b,c)``, ``H_X=[a,b,c]`` and
    ``H_Z^dagger=[[0,c,b],[c,0,a],[b,a,0]]``.  The degree -2 map is
    ``[a,b,c]^T``.  These products vanish in characteristic two for any
    triple; that alone does not establish topological order.
    """
    _require_tt_ring(ring)
    values = tuple(polynomials)
    if len(values) != 3:
        raise ValueError("A tricycle needs exactly three Laurent polynomials")
    a, b, c = (ring(value) for value in values)
    return CochainComplex(ring, {-2: 1, -1: 3, 0: 3, 1: 1}, {
        -2: Matrix.from_rows(ring, [[a], [b], [c]]),
        -1: Matrix.from_rows(ring, [[0, c, b], [c, 0, a], [b, a, 0]]),
        0: Matrix.from_rows(ring, [[a, b, c]]),
    })


def tricycle_css(ring, polynomials):
    """Return ordinary ``(H_X,H_Z)``; no period or resolution is supplied."""
    complex = tricycle_complex(ring, polynomials)
    return complex.d(0), complex.d(-1).dagger()


@dataclass(frozen=True)
class TricycleCase:
    """Literal polynomial supports and separately labelled finite data.

    ``published_nkd`` is a citation, not a distance calculation by this
    package.  All exponent triples are in the original (x,y,z) basis.
    """

    name: str
    supports: tuple
    periodic_lengths: tuple
    published_nkd: tuple
    source: str = SOURCE

    def polynomials(self, ring=None):
        ring = Ring(2, 3) if ring is None else ring
        _require_tt_ring(ring)
        return tuple(sum((ring.monomial(exponent) for exponent in support),
                         ring.zero()) for support in self.supports)

    def checks(self, ring=None):
        ring = Ring(2, 3) if ring is None else ring
        return tricycle_css(ring, self.polynomials(ring))


TRICYCLE_CASES = MappingProxyType({
    "tt72": TricycleCase(
        "tt72",
        (((0, 0, 0), (0, 1, 0), (1, 2, 0)),
         ((0, 0, 0), (0, 1, 1), (2, 2, 0)),
         ((0, 0, 0), (1, 2, 1), (2, 1, 0))),
        (4, 3, 2), (72, 6, 6)),
    "tt180": TricycleCase(
        "tt180",
        (((0, 0, 0), (2, 3, 1), (4, 1, 0)),
         ((0, 0, 0), (3, 0, 0), (4, 0, 2)),
         ((0, 0, 0), (3, 3, 0), (4, 1, 2))),
        (5, 4, 3), (180, 12, 8)),
    "tt432": TricycleCase(
        "tt432",
        (((0, 0, 0), (1, 1, 3), (3, 4, 2)),
         ((0, 0, 0), (3, 1, 2), (3, 2, 3)),
         ((0, 0, 0), (4, 3, 3), (5, 0, 2))),
        (6, 6, 4), (432, 12, 12)),
})


def tricycle_case(name="tt72", *, ring=None):
    """Build one Table I raw input; finite metadata is in TRICYCLE_CASES."""
    try:
        case = TRICYCLE_CASES[name]
    except KeyError as exc:
        raise ValueError("Unknown TT case; choose " + ", ".join(TRICYCLE_CASES)) from exc
    return case.checks(ring)
