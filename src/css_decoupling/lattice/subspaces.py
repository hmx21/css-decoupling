"""Lift finite-field subspaces to their full integer preimage lattices."""
from numbers import Integral

from css_decoupling.algebra.exterior import integer_rows
from css_decoupling.algebra.laurent import Ring
from .finite_algebra import nullspace, rref, modular_kernel
from .retained import Lattice


def subspace_preimage(W, *, p, D=None, max_index=None):
    """Return {z in Z^D: z mod p belongs to span(columns(W))}.

    Dependent spanning columns are allowed; zero columns use D empty rows.
    No coset representatives are expanded by this construction.
    """
    if D is not None and (isinstance(D, bool) or not isinstance(D, Integral) or D < 1):
        raise ValueError("Subspace dimension must be a positive integer")
    W = integer_rows(W, nrows=D, name="subspace generators")
    D = len(W)
    ring = Ring(p, D)
    columns = list(zip(*W))
    dimension = len(rref(columns, D, ring.p)[1])
    annihilator = nullspace(columns, D, ring.p)
    basis = modular_kernel(annihilator, D, ring.p)
    result = Lattice([[int(basis[i, j]) for j in range(D)] for i in range(D)], max_index=max_index)
    if result.index != ring.p ** (D - dimension):
        raise ArithmeticError("Subspace preimage has the wrong index")
    if any(sum(row[i] * result.basis[i][j] for i in range(D)) % ring.p
           for row in annihilator for j in range(D)):
        raise ArithmeticError("Subspace preimage violates its defining congruences")
    return result
