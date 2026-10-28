"""Exact module membership over GF(p)[x_1,...,x_D], without localization.

Laurent ``Matrix`` objects are reused as a sparse container, but negative
exponents are rejected at the boundary.  No inverse variable, localization
relation, or monomial-as-unit pivot is introduced.  Extended standard-basis
coefficients provide an actual polynomial solution, not a fraction-field one.
"""
from numbers import Integral

from sympy import GF
from sympy.polys.agca.modules import ModuleOrder
from sympy.polys.distributedmodules import sdm_groebner, sdm_nf_buchberger
from sympy.polys.orderings import monomial_key

from css_decoupling.algebra.laurent import Matrix, Poly, zeros
from css_decoupling.errors import ResourceLimit


class NoPolynomialSolution(ValueError):
    """A complete polynomial standard basis proves module nonmembership.

    This is intentionally not a NoLaurentSolution: a polynomial equation
    can fail here and still have a Laurent solution after localization.
    """


def _limit(value, name):
    if value is not None and (isinstance(value, bool) or not isinstance(value, Integral) or value < 1):
        raise ValueError(f"{name} must be a positive integer or None")
    return None if value is None else int(value)


def _check_polynomial(matrix, name):
    if not isinstance(matrix, Matrix):
        raise ValueError(f"{name} must be a Laurent Matrix containing ordinary polynomials")
    if any(e < 0 for polynomial in matrix.entries.values() for powers in polynomial.terms for e in powers):
        raise ValueError(f"{name} has negative exponents; polynomial membership does not localize")


def _columns(matrix, order, field):
    columns = {}
    for (row, col), polynomial in matrix.entries.items():
        columns.setdefault(col, []).extend(((row,) + powers, field(coefficient))
                                           for powers, coefficient in polynomial.terms.items())
    for column in columns.values():
        column.sort(key=lambda term: order(term[0]), reverse=True)
    return columns


class _CooperativeLimits:
    """Check visible SDM vectors around every normal-form call.

    SymPy constructs S-polynomials and critical-pair lists internally, and a
    single normal-form call may allocate before returning.  These are explicit
    cooperative complexity caps, not hard process-memory or wall-clock limits.
    """

    def __init__(self, max_basis, max_terms):
        self.max_basis, self.max_terms = max_basis, max_terms

    def basis(self, basis):
        if self.max_basis is not None and len(basis) > self.max_basis:
            raise ResourceLimit("Polynomial standard basis exceeds max_basis")

    def terms(self, *vectors):
        if self.max_terms is not None and sum(map(len, vectors)) > self.max_terms:
            raise ResourceLimit("Polynomial standard-basis witnesses exceed max_terms")

    def normal_form(self, f, G, O, K, phantom=None, *, building_basis=True):
        self.basis(G)
        extra = [] if phantom is None else [phantom[0], *phantom[1]]
        self.terms(f, *G, *extra)
        result = sdm_nf_buchberger(f, G, O, K, phantom=phantom)
        remainder, coefficient = (result, []) if phantom is None else result
        if building_basis and remainder and self.max_basis is not None and len(G) + 1 > self.max_basis:
            raise ResourceLimit("Polynomial standard basis exceeds max_basis")
        self.terms(remainder, coefficient, *G, *extra)
        return result


def solve_polynomial(A, B, *, max_basis=100000, max_terms=2000000):
    """Solve A*X=B in the ordinary polynomial ring, or prove nonmembership.

    ``A`` and ``B`` must contain nonnegative exponents and share a prime-field
    Laurent ``Ring`` and their ambient row count.  The returned ``X`` has
    nonnegative exponents and is verified by direct multiplication.

    ``max_basis`` and ``max_terms`` are cooperative checks at input, SDM
    normal-form boundaries, and output.  There is no degree/support cutoff.
    Resource exhaustion is never converted to ``NoPolynomialSolution``.
    The routine always uses native SymPy finite-field arithmetic, independently
    of the currently selected Laurent module backend.
    """
    _check_polynomial(A, "A")
    _check_polynomial(B, "B")
    if A.ring != B.ring or A.nrows != B.nrows:
        raise ValueError("Polynomial solve requires a common ring and matching ambient rows")
    limits = _CooperativeLimits(_limit(max_basis, "max_basis"), _limit(max_terms, "max_terms"))
    if limits.max_terms is not None and sum(len(f.terms) for M in (A, B) for f in M.entries.values()) > limits.max_terms:
        raise ResourceLimit("Polynomial solve input exceeds max_terms")
    if not B:
        return zeros(A.ring, A.ncols, B.ncols)
    if not A:
        raise NoPolynomialSolution("A nonzero right-hand side lies outside the zero polynomial image")
    ring, field = A.ring, GF(A.ring.p)
    order = ModuleOrder(monomial_key("lex"), monomial_key("grevlex"), True)
    source = _columns(A, order, field)
    active = sorted(source)
    generators = [source[j] for j in active]
    limits.basis(generators)
    limits.terms(*generators)
    # No extra localization generator is present.  SymPy's extended graph
    # tracks the coefficients of these original polynomial columns only.
    basis, transition = sdm_groebner(generators, limits.normal_form, order, field, extended=True)
    limits.basis(basis)
    limits.terms(*basis, *transition)
    rhs, answer, solution_terms = _columns(B, order, field), {}, 0
    for j in sorted(rhs):
        remainder, phantom = limits.normal_form(rhs[j], basis, order, field,
                                               phantom=([], transition), building_basis=False)
        if remainder:
            raise NoPolynomialSolution(f"Right-hand column {j} is not in the polynomial image")
        # Reduction tracks B + A*phantom = remainder, so the solution is
        # minus the phantom.  Keeping this sign matters in odd characteristic.
        for monomial, coefficient in phantom:
            position, powers = monomial[0], monomial[1:]
            if not 0 <= position < len(active) or len(powers) != ring.D or any(e < 0 for e in powers):
                raise ArithmeticError("Polynomial standard-basis lift has invalid coordinates")
            entry = answer.setdefault((active[position], j), {})
            previous = entry.get(powers, 0)
            value = (previous - int(coefficient)) % ring.p
            solution_terms += bool(value) - bool(previous)
            if value:
                entry[powers] = value
            else:
                entry.pop(powers, None)
            if limits.max_terms is not None and solution_terms > limits.max_terms:
                raise ResourceLimit("Polynomial solution exceeds max_terms")
    result = Matrix(ring, A.ncols, B.ncols,
                    {ij: Poly(ring, coefficients) for ij, coefficients in answer.items()})
    _check_polynomial(result, "Polynomial solution")
    if A * result != B:
        raise ArithmeticError("Polynomial solution failed its exact multiplication certificate")
    return result
