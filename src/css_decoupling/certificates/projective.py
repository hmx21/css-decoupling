"""Independent exact verification of a projective-complement free basis."""
from css_decoupling.algebra.laurent import Matrix, eye


def verify_projective_basis(E, P, Q):
    """Verify that P:R^a->im(E) and Q:im(E)->R^a are inverse maps.

    This checker performs sparse ring arithmetic only.  It does not call the
    constructing solver, a module backend, or an asserted rank oracle.
    """
    if not all(isinstance(A, Matrix) for A in (E, P, Q)):
        raise TypeError("Projective witnesses must be exact Laurent matrices")
    if E.ring != P.ring or E.ring != Q.ring:
        raise ValueError("Projective witness rings differ")
    if E.nrows != E.ncols or P.nrows != E.nrows or Q.shape != (P.ncols, E.nrows):
        raise ValueError("Projective witness dimensions differ")
    if E * E != E:
        raise ValueError("Projective presentation is not idempotent")
    if P * Q != E or Q * P != eye(E.ring, P.ncols):
        raise ValueError("Projective basis is not an inverse factorization of E")
    return True
