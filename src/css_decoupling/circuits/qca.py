"""CSS-preserving Laurent Pauli action, separate from a gate decomposition.

Vectors are columns, with all X coordinates before all Z coordinates. Single
X/Z generator images have phase zero; mixed Paulis use multiplicative extension
in X-then-Z order. This local object alone does not certify a code decoupling.
"""
from dataclasses import dataclass

from css_decoupling.algebra.laurent import Matrix, direct_sum, eye


@dataclass(frozen=True)
class QCA:
    """A candidate physical Z action and its Laurent inverse.

    X and inverse-X are determined by the inverse dagger. Shape/ring checks
    are immediate; validate() additionally checks both inverse products.
    Source-bound exported outputs use the complete decoupling certificate.
    """
    z_action: Matrix
    z_inverse: Matrix

    def __post_init__(self):
        if not isinstance(self.z_action, Matrix) or not isinstance(self.z_inverse, Matrix):
            raise TypeError("QCA actions must be Laurent matrices")
        if (self.z_action.ring != self.z_inverse.ring
                or self.z_action.nrows != self.z_action.ncols
                or self.z_inverse.shape != self.z_action.shape):
            raise ValueError("QCA actions must be square, with the same ring and shape")

    @property
    def ring(self):
        return self.z_action.ring

    @property
    def qudits(self):
        return self.z_action.nrows

    @property
    def x_action(self):
        return self.z_inverse.dagger()

    @property
    def x_inverse(self):
        return self.z_action.dagger()

    def matrix(self, action="Z"):
        if action == "Z":
            return self.z_action.copy()
        if action == "X":
            return self.x_action
        raise ValueError("QCA action must be X or Z")

    def inverse(self):
        return QCA(self.z_inverse.copy(), self.z_action.copy())

    def fourier_dual(self):
        """Exchange the physical X/Z actions without reordering species."""
        return QCA(self.x_action, self.x_inverse)

    def symplectic_matrix(self):
        """Materialize diag(X,Z) only when the full 2q matrix is requested."""
        return direct_sum(self.x_action, self.z_action)

    def validate(self):
        """Check local reversibility; this does not certify source-code mapping."""
        identity = eye(self.ring, self.qudits)
        if self.z_action * self.z_inverse != identity or self.z_inverse * self.z_action != identity:
            raise ValueError("QCA matrices are not two-sided Laurent inverses")
        return True
