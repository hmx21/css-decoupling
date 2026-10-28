"""Synthetic inter-layer inputs for exercising the general algorithm.

These are mathematical test constructions, not literature code families.
They extend ``models.examples`` with sums of exterior representatives and
optional changes of presentation.  Only raw check matrices reach the solver.
"""
from numbers import Integral
from types import MappingProxyType

from sympy import Matrix as IntegerMatrix

from css_decoupling.algebra.laurent import Matrix, Ring, block, zeros
from css_decoupling.models.examples import (
    _exterior_coupling, css_checks, glued_toric_chain, shear_css,
)
from css_decoupling.topology.complexes import CochainComplex, koszul_window


def _check_pair(HX, HZ):
    if not isinstance(HX, Matrix) or not isinstance(HZ, Matrix) or (
            HX.ring != HZ.ring or HX.ncols != HZ.ncols):
        raise ValueError("CSS checks must share a Laurent ring and physical columns")


def lattice_relabel_css(HX, HZ, basis):
    """Apply the lattice automorphism x^v -> x^(Bv) to both checks.

    The integer columns of ``basis`` give images of the old unit steps.
    Requiring determinant +/-1 distinguishes relabelling from restriction
    to a sublattice.  Coefficients and negative exponents remain exact.
    """
    _check_pair(HX, HZ)
    rows = tuple(tuple(row) for row in basis)
    D = HX.ring.D
    if len(rows) != D or any(len(row) != D for row in rows) or any(
            isinstance(value, bool) or not isinstance(value, Integral)
            for row in rows for value in row):
        raise ValueError("A lattice relabelling needs a D by D integer matrix")
    rows = tuple(tuple(int(value) for value in row) for row in rows)
    if abs(IntegerMatrix(rows).det()) != 1:
        raise ValueError("A lattice relabelling must be unimodular")

    def transform(A):
        return Matrix(A.ring, A.nrows, A.ncols, {
            ij: A.ring({tuple(sum(rows[i][j] * exponent[j] for j in range(D))
                              for i in range(D)): coefficient
                        for exponent, coefficient in polynomial.terms.items()})
            for ij, polynomial in A.entries.items()})

    return transform(HX), transform(HZ)


def redundant_css_checks(HX, HZ):
    """Append one translated dependent row to each nonempty check matrix.

    Every original row is retained.  The appended row is the first row
    multiplied by x_0^-1, plus twice the last row when there are two or more
    rows.  Thus the generated Laurent modules are exactly unchanged.
    """
    _check_pair(HX, HZ)

    def append(A):
        if not A.nrows:
            return A.copy()
        extra = A[:1, :] * A.ring.variable(0) ** -1
        if A.nrows > 1:
            extra = extra + 2 * A[-1:, :]
        return A.vstack(extra)

    return append(HX), append(HZ)


def glued_stress_complex(ring, degrees=(1, 2), *, coefficient=1, couplings=None):
    """Construct signed multi-layer gluing, optionally with a sum of wedges.

    With ``couplings=None`` use the existing successive gluing family.
    For two layers ``couplings`` may instead contain ``(axes,scalar)``
    pairs, e.g. ``(((0,1),1),((2,3),1))`` gives epsilon_12+epsilon_34.
    This is one rank-four two-form coupling two one-dimensional layers,
    not a direct sum of two independent low layers.  Each term has exterior
    degree ``upper-lower+1`` and a constant coefficient in F_p.
    """
    degrees = tuple(degrees)
    if len(degrees) < 2 or any(isinstance(j, bool) or not isinstance(j, Integral)
                             or not 1 <= j < ring.D for j in degrees) or (
            tuple(sorted(set(degrees))) != degrees):
        raise ValueError("Require at least two increasing degrees with 1 <= j < D")
    degrees = tuple(int(j) for j in degrees)
    if couplings is None:
        return glued_toric_chain(ring, degrees, coefficient=coefficient)
    if len(degrees) != 2:
        raise ValueError("Explicit exterior sums currently require exactly two layers")
    if ring(coefficient) != ring.one():
        raise ValueError("Give explicit coupling coefficients instead of coefficient")
    lower, upper = degrees
    order = upper - lower + 1
    terms = []
    for axes, scalar in couplings:
        axes = tuple(axes)
        if len(axes) != order or any(isinstance(a, bool) or not isinstance(a, Integral)
                                    or not 0 <= a < ring.D for a in axes) or (
                tuple(sorted(set(axes))) != axes):
            raise ValueError("Each exterior term needs distinct increasing valid axes")
        scalar = ring(scalar)
        if not scalar.is_constant:
            raise ValueError("Exterior coefficients must be constants")
        terms.append((tuple(int(a) for a in axes), scalar))
    low, high = koszul_window(ring, lower), koszul_window(ring, upper)
    ns = sorted(set(low.degrees) | set(high.degrees))
    maps = {}
    for n in ns:
        coupling = zeros(ring, low.rank(n + 1), high.rank(n))
        for axes, scalar in terms:
            coupling = coupling + _exterior_coupling(ring, lower, upper, n, axes, scalar)
        maps[n] = block([[low.d(n), coupling],
                         [zeros(ring, high.rank(n + 1), low.rank(n)), high.d(n)]])
    return CochainComplex(ring, {n: low.rank(n) + high.rank(n) for n in ns}, maps)


def glued_stress_css(ring, degrees=(1, 2), *, coefficient=1, couplings=None,
                     lattice_basis=None, shear=False, redundant_checks=False):
    """Return raw glued checks with optional equivalent presentations.

    ``shear`` applies one invertible physical transvection with coefficient
    x_last^-1 + 2*x_0.  ``lattice_basis`` is a unimodular coordinate change,
    not a suggested decoupling lattice.  No precomputed splitting, charge
    basis or target multiplicity is attached to the returned matrices.
    """
    HX, HZ = css_checks(glued_stress_complex(
        ring, degrees, coefficient=coefficient, couplings=couplings))
    if shear:
        HX, HZ = shear_css(HX, HZ, 0, HX.ncols - 1,
                          ring.variable(ring.D - 1) ** -1 + 2 * ring.variable(0))
    if lattice_basis is not None:
        HX, HZ = lattice_relabel_css(HX, HZ, lattice_basis)
    if redundant_checks:
        HX, HZ = redundant_css_checks(HX, HZ)
    return HX, HZ


def rank_four_glued_css(ring):
    """Two one-dimensional layers glued by epsilon_12+epsilon_34.

    This form has rank four over every prime field.  In particular killing
    only the first plane does not remove the second plane's obstruction.
    Reported kappa coefficients use the solver's recorded cohomology bases;
    their common nonzero scalar need not be one in those bases.
    """
    if ring.D < 4:
        raise ValueError("The rank-four exterior form requires dimension at least four")
    return glued_stress_css(ring, (1, 2), couplings=(((0, 1), 1), ((2, 3), 1)))


# Registration data contains input choices only, never claimed solver outputs.
STRESS_CASES = MappingProxyType({
    "gluing_d4_p3_ext3": (4, 3, (1, 3)),
    "gluing_d4_p3_rank4": (4, 3, (1, 2)),
    "gluing_d4_p3_three_layers": (4, 3, (1, 2, 3)),
    "gluing_d5_p5_ext3": (5, 5, (1, 3)),
    "gluing_d5_p5_three_layers": (5, 5, (1, 2, 3)),
    "gluing_d4_p3_ext3_scrambled": (4, 3, (1, 3)),
})


def stress_case(name):
    """Return one registered artificial raw input, with no solver shortcuts."""
    try:
        D, p, degrees = STRESS_CASES[name]
    except KeyError as exc:
        raise ValueError("Unknown stress case; choose " + ", ".join(STRESS_CASES)) from exc
    options = {}
    if name == "gluing_d4_p3_rank4":
        options["couplings"] = (((0, 1), 1), ((2, 3), 1))
    if name.endswith("_scrambled"):
        basis = [[int(i == j) for j in range(D)] for i in range(D)]
        basis[0][1] = 1
        basis[2][2] = -1
        options.update(lattice_basis=basis, shear=True, redundant_checks=True)
    return glued_stress_css(Ring(p, D), degrees, **options)
