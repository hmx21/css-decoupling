"""Exact comparison inverses using the standard Koszul target resolution.

The target has an explicit F_p-linear contraction onto its augmentation.
Applying it to individual columns constructs Laurent matrices; it is never
claimed to be an R-linear contraction of the noncontractible toric complex.
Every resulting chain map, inverse homotopy and mapping-cone contraction is
checked in the original coordinates.
"""
from dataclasses import dataclass
import logging

from css_decoupling.algebra.laurent import block, eye, zeros
from css_decoupling.algebra.bounded_linear import augmentation, constant_solve
from css_decoupling.errors import NoLaurentSolution, ResourceLimit
from css_decoupling.topology.complexes import (
    CochainMap, identity_map, koszul, mapping_cone, validate_homotopy,
)
from css_decoupling.topology.koszul_lifts import koszul_preimage

log = logging.getLogger(__name__)


def toric_cone_solver(phi, layers, solve, *, max_terms=2000000):
    """A checked block solver for current Gamma-section left equations.

    Below the sole top layer j, a cone cycle (c,t) has t in im(dT): at
    n=j-1 this uses injectivity of H^j(phi), not just the cycle equation.
    Choose -dT*yt=t by the explicit Koszul lift, then solve
    dC*yc=c-phi*yt in the original source coordinates.  No weak inverse
    homotopies or products hC*hC*phi are constructed.  Higher/right section
    constraints are imposed and checked by the direct splitting caller.
    """
    if len(layers) != 1:
        return None
    j, layer = next(iter(layers.items()))
    C, T, R = phi.target, phi.source, phi.source.ring
    if C.d(j) or any(C.rank(n) for n in C.degrees if n > j):
        return None
    standard = koszul(R, j, layer.dimension)
    if any(T.rank(n) != standard.rank(n) or T.d(n) != standard.d(n)
           for n in set(T.degrees) | set(standard.degrees)):
        return None
    G = mapping_cone(phi)
    operators = {G.d(n-1).key(): n for n in G.degrees if n < j}

    def solve_section(A, B):
        n = operators.get(A.key())
        if n is None or B.ring != R or A.nrows != B.nrows:
            return solve(A, B)
        # Two explicit positive constructions, not a support-radius search.
        # A small expanded-coordinate solution can beat a source-ring lift;
        # choose by actual stored support, making no global sparsity claim.
        from css_decoupling.algebra.sparse_linear import sparse_binary_solution
        candidate = sparse_binary_solution(A, B)
        try:
            bottom = B[C.rank(n):, :]
            if T.rank(n+1):
                yt = koszul_preimage(-bottom, j-n-1, copies=layer.dimension,
                                     max_terms=max_terms)
            else:
                if bottom:
                    raise NoLaurentSolution("Nonzero empty target block")
                yt = zeros(R, T.rank(n), B.ncols)
            right = B[:C.rank(n), :] - phi.at(n)*yt
            # Limit support union across unrelated RHS columns. A complete
            # answer requires every chunk to pass its exact positive solve;
            # a single miss discards the shortcut and uses the caller's
            # original complete backend. No truncated kernel is computed.
            pieces = []
            if R.p == 2 and right.ncols > 16:
                for start in range(0, right.ncols, 16):
                    piece = sparse_binary_solution(C.d(n-1), right[:, start:start+16])
                    if piece is None:
                        pieces = []
                        break
                    pieces.append(piece)
            yc = pieces[0].hstack(*pieces[1:]) if pieces else solve(C.d(n-1), right)
            X = yc.vstack(yt)
            if X.shape != (A.ncols, B.ncols) or A*X != B:
                raise ValueError("Gamma block section failed its actual Laurent equation")
        except (NoLaurentSolution, ResourceLimit):
            # Failure of this choice of the target lift is not a negative
            # conclusion about the full cone system. Preserve the backend.
            return candidate if candidate is not None else solve(A, B)
        terms = lambda M: sum(len(f.terms) for f in M.entries.values())
        if candidate is not None and terms(candidate) <= terms(X):
            return candidate
        return X

    return solve_section


@dataclass
class ToricComparisonInverse:
    psi: CochainMap
    hC: dict
    hT: dict
    cone_contraction: dict


def toric_physical_preimage(T, profile, rhs, *, max_terms=2000000):
    """Lift a cycle through the actual direct-sum toric physical differential."""
    if rhs.ring != T.ring or rhs.nrows != T.rank(1):
        raise ValueError("Toric syndrome has incompatible rows or coefficient ring")
    pieces, offset = [], 0
    for j, copies in sorted(profile.items()):
        piece = koszul(T.ring, j, copies)
        rows = piece.rank(1)
        pieces.append(koszul_preimage(rhs[offset:offset + rows, :], j - 1,
                                      copies=copies, max_terms=max_terms))
        offset += rows
    if offset != rhs.nrows:
        raise ValueError("Toric profile disagrees with its physical differential")
    result = zeros(T.ring, 0, rhs.ncols).vstack(*pieces)
    if T.d(0) * result != rhs:
        raise ArithmeticError("Toric preimage failed the actual target differential")
    return result


def cone_contraction_from_inverse(phi, psi, hC, hT):
    """Align inverse homotopies and explicitly contract Cone(phi).

    Write f=phi,g=psi,A=hC*f-f*hT. Set hT'=g*hC*f+hT-hT*g*f and
    k=A*hT-hC*hC*f. Then delta(k)=f*hT'-hC*f. Thus the blocks
    [[hC,k],[g,-hT']] contract [[dC,f],[0,-dT]]. Signs are valid over
    every prime field. The full matrix identity is checked, not assumed.
    """
    C, T, R = phi.target, phi.source, phi.source.ring
    if psi.source != C or psi.target != T:
        raise ValueError("Inverse comparison has incompatible source and target")
    def hc(n):
        return hC.get(n, zeros(R, C.rank(n - 1), C.rank(n)))
    def ht(n):
        return hT.get(n, zeros(R, T.rank(n - 1), T.rank(n)))
    adjusted = {}
    for n in T.degrees:
        adjusted[n] = psi.at(n - 1) * hc(n) * phi.at(n) + ht(n) - ht(n) * psi.at(n) * phi.at(n)
    validate_homotopy(T, T, identity_map(T), CochainMap(T, T, {
        n: psi.at(n) * phi.at(n) for n in T.degrees}), adjusted)
    G, H = mapping_cone(phi), {}
    for n in G.degrees:
        A = hc(n) * phi.at(n) - phi.at(n - 1) * ht(n)
        k = A * ht(n + 1) - hc(n) * hc(n + 1) * phi.at(n + 1)
        H[n] = block([[hc(n), k], [psi.at(n),
            -adjusted.get(n + 1, zeros(R, T.rank(n), T.rank(n + 1)))]])
    validate_homotopy(G, G, identity_map(G), CochainMap(G, G, {}), H)
    return adjusted, H


def toric_comparison_inverse(phi, layers, solve, *, max_terms=2000000):
    """Return an exact inverse for a top single layer, or None if inapplicable.

    The condition concerns an actual cochain presentation, not a model name.
    C has one positive cohomology layer in its highest nonzero degree j;
    W=I there and phi^j induces an invertible augmentation map. The
    source still uses complete Laurent equations for its inverse homotopy.
    Other profiles/presentations retain the general cone algorithm.
    """
    if len(layers) != 1:
        return None
    j, layer = next(iter(layers.items()))
    C, T, R = phi.target, phi.source, phi.source.ring
    if C.d(j) or any(C.rank(n) for n in C.degrees if n > j):
        return None
    if layer.cycles != eye(R, C.rank(j)):
        return None
    copies = layer.dimension
    if copies < 1:
        return None
    standard = koszul(R, j, copies)
    if any(T.rank(n) != standard.rank(n) or T.d(n) != standard.d(n)
           for n in set(T.degrees) | set(standard.degrees)):
        return None
    projection = layer.quotient.projection
    if projection.shape != (copies, C.rank(j)):
        return None
    # A different valid choice of representatives may mix the top copies
    # or differ by boundaries.  Only its induced finite quotient map must
    # be inverted; demanding an exact polynomial identity would reject
    # such equivalent presentations.  All subsequent Laurent identities
    # are still checked on the actual source and target complexes.
    induced = augmentation(projection * phi.at(j))
    try:
        inverse = constant_solve(induced, eye(R, copies))
    except NoLaurentSolution:
        return None
    if inverse * induced != eye(R, copies):
        raise ArithmeticError("Top comparison map has no two-sided quotient inverse")
    projection = inverse * projection

    def preimage(rhs, target_degree):
        if not T.rank(target_degree - 1):
            if rhs:
                raise ArithmeticError("A comparison cycle escaped the bottom Koszul injection")
            return zeros(R, 0, rhs.ncols)
        return koszul_preimage(rhs, j - target_degree, copies=copies, max_terms=max_terms)

    delta = {j: projection}
    for n in range(j - 1, min((*C.degrees, *T.degrees)) - 1, -1):
        rhs = delta.get(n + 1, zeros(R, T.rank(n + 1), C.rank(n + 1))) * C.d(n)
        delta[n] = preimage(rhs, n + 1)
    psi = CochainMap(C, T, delta)

    hT = {}
    for n in reversed(T.degrees):
        following = hT.get(n + 1, zeros(R, T.rank(n), T.rank(n + 1)))
        rhs = eye(R, T.rank(n)) - psi.at(n) * phi.at(n) - following * T.d(n)
        hT[n] = preimage(rhs, n)
    hC = {}
    for n in reversed(C.degrees):
        following = hC.get(n + 1, zeros(R, C.rank(n), C.rank(n + 1)))
        rhs = eye(R, C.rank(n)) - phi.at(n) * psi.at(n) - following * C.d(n)
        log.info("source inverse homotopy degree %s: %s RHS monomials", n,
                 sum(len(f.terms) for f in rhs.entries.values()))
        hC[n] = solve(C.d(n - 1), rhs)
        log.info("source inverse homotopy degree %s: %s solution monomials", n,
                 sum(len(f.terms) for f in hC[n].entries.values()))
        if hC[n].shape != (C.rank(n - 1), C.rank(n)) or C.d(n - 1) * hC[n] != rhs:
            raise ArithmeticError("Source comparison homotopy failed its exact identity")
    validate_homotopy(C, C, identity_map(C), CochainMap(C, C, {
        n: phi.at(n) * psi.at(n) for n in C.degrees}), hC)
    validate_homotopy(T, T, identity_map(T), CochainMap(T, T, {
        n: psi.at(n) * phi.at(n) for n in T.degrees}), hT)
    adjusted, H = cone_contraction_from_inverse(phi, psi, hC, hT)
    return ToricComparisonInverse(psi, hC, adjusted, H)
