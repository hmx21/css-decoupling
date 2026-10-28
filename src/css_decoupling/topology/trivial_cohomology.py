"""C5 cohomology from complete kernels and augmentation, with checked actions.

This construction is valid only when the augmentation ideal annihilates the
actual cohomology.  Explicit boundary witnesses establish that condition;
it is never inferred from evaluating a presentation at x_i=1.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from css_decoupling.algebra.bounded_linear import (
    augmentation, constant_kernel, constant_solve,
)
from css_decoupling.algebra.laurent import Matrix, eye, zeros
from css_decoupling.algebra.modules import get_backend
from css_decoupling.errors import NoLaurentSolution, ResourceLimit


class TrivialActionQuotient:
    """Finite quotient coordinates when (x_i-1) annihilation is witnessed."""

    def __init__(self, presentation, basis, projection):
        self.presentation = presentation
        self.ring = presentation.ring
        self.basis = basis
        self.projection = projection
        self.dimension = basis.ncols
        self.actions = tuple(eye(self.ring, self.dimension) for _ in range(self.ring.D))

    def coordinates(self, vectors):
        if vectors.ring != self.ring or vectors.nrows != self.presentation.nrows:
            raise ValueError("Quotient coordinates need matching ambient module and ring")
        return self.projection * augmentation(vectors)


@dataclass
class TrivialCohomologyLayer:
    degree: int
    cycles: Matrix
    presentation: Matrix
    quotient: TrivialActionQuotient
    representatives: Matrix
    relation_kernel: Matrix
    triviality_witnesses: tuple[Matrix, ...]
    previous_differential: Matrix
    next_differential: Matrix
    kernel_certificate: dict | None = None
    relation_kernel_certificate: dict | None = None

    @property
    def dimension(self):
        return self.quotient.dimension

    @property
    def actions(self):
        return self.quotient.actions

    def certificate(self):
        from css_decoupling.certificates.comparison import _matrix_dict
        ring = self.cycles.ring
        return {
            "schema": "css-decoupling.trivial-cohomology", "version": 1,
            "ring": {"prime": ring.p, "dimension": ring.D}, "degree": self.degree,
            "dprev": _matrix_dict(self.previous_differential),
            "dnext": _matrix_dict(self.next_differential),
            "cycles": _matrix_dict(self.cycles),
            "relation_kernel": _matrix_dict(self.relation_kernel),
            "presentation": _matrix_dict(self.presentation),
            "basis": _matrix_dict(self.quotient.basis),
            "projection": _matrix_dict(self.quotient.projection),
            "representatives": _matrix_dict(self.representatives),
            "dimension": self.dimension,
            "actions": [_matrix_dict(A) for A in self.actions],
            "triviality_witnesses": [_matrix_dict(A) for A in self.triviality_witnesses],
            "kernel_certificate": deepcopy(self.kernel_certificate),
            "relation_kernel_certificate": deepcopy(self.relation_kernel_certificate),
        }


def _quotient(presentation):
    """A complement to im(U(1)) and the dual quotient map, by Gaussian elimination."""
    relation = augmentation(presentation)
    # Rows of projection span the annihilator of the relation image.
    projection = constant_kernel(relation.T).T
    basis = constant_solve(projection, eye(presentation.ring, projection.nrows))
    if projection * relation or projection * basis != eye(presentation.ring, projection.nrows):
        raise ArithmeticError("Augmented quotient failed its finite-field identities")
    return TrivialActionQuotient(presentation, basis, projection)


def _validate_dimension_limit(max_dimension):
    if max_dimension is not None and (
            isinstance(max_dimension, bool) or not isinstance(max_dimension, int) or max_dimension < 0):
        raise ValueError("max_dimension must be a nonnegative integer or None")


def trivial_cohomology_layer(C, degree, *, backend=None, kernel=None, solve=None,
                            max_dimension=10000):
    """Compute a layer using W=ker(d), proj ker[W,-dprev], and augmentation.

    ``kernel`` must supply complete module generators, not merely some cycles.
    The optional backend is used consistently for both kernel and solve calls.
    Resource exceptions are preserved; failure to prove triviality is not
    converted into a zero-dimensional cohomology group.
    """
    if isinstance(degree, bool) or not isinstance(degree, int):
        raise ValueError("Cohomology degree must be an integer")
    _validate_dimension_limit(max_dimension)
    C.validate()
    backend = backend or get_backend(C.ring)
    kernel_function = kernel or backend.kernel
    solve_function = solve or backend.solve
    dn, dp = C.d(degree), C.d(degree - 1)
    W = kernel_function(dn)
    kernel_certificate = deepcopy(getattr(backend, "last_certificate", None)) if kernel is None else None
    if W.ring != C.ring or W.nrows != C.rank(degree) or dn * W:
        raise ArithmeticError("Kernel backend returned invalid cycles")
    relation_operator = W.hstack(-dp)
    relations = kernel_function(relation_operator)
    relation_certificate = deepcopy(getattr(backend, "last_certificate", None)) if kernel is None else None
    if relations.nrows != relation_operator.ncols or relation_operator * relations:
        raise ArithmeticError("Kernel backend returned invalid cycle-boundary relations")
    U = relations[:W.ncols, :]
    witnesses = []
    for axis in range(C.ring.D):
        rhs = (C.ring.variable(axis) - 1) * W
        try:
            try:
                # With relations=[U;V], the checked kernel equation gives
                # W*U=dp*V.  Solving in the relation presentation removes
                # redundant cycle coordinates before forming the physical
                # boundary witness.  Completeness makes this equivalent to
                # the direct boundary equation, not a truncation criterion.
                coordinate_rhs = (C.ring.variable(axis) - 1) * eye(C.ring, W.ncols)
                coordinates = solve_function(U, coordinate_rhs)
                if U * coordinates != coordinate_rhs:
                    raise ArithmeticError("Relation-coordinate translation witness is invalid")
                witness = relations[W.ncols:, :] * coordinates
            except ResourceLimit:
                # The same backend may have a cheaper exact shortcut for dp,
                # especially in the strictly bounded coefficient backend.
                witness = solve_function(dp, rhs)
        except NoLaurentSolution as exc:
            raise ValueError(f"Cohomology in degree {degree} is not annihilated by x_{axis + 1}-1") from exc
        if dp * witness != rhs:
            raise ArithmeticError("Translation triviality boundary witness is invalid")
        witnesses.append(witness)
    quotient = _quotient(U)
    if max_dimension is not None and quotient.dimension > max_dimension:
        raise ResourceLimit(f"Trivial layer has dimension {quotient.dimension}; cap is {max_dimension}")
    return TrivialCohomologyLayer(
        degree, W, U, quotient, W * quotient.basis, relations, tuple(witnesses),
        dp, dn, kernel_certificate, relation_certificate)


def trivial_cohomology(C, *, backend=None, kernel=None, solve=None, max_dimension=10000):
    """All nonzero positive layers of a translation-trivial profile."""
    _validate_dimension_limit(max_dimension)
    result = {}
    for degree in C.degrees:
        if degree <= 0:
            continue
        layer = trivial_cohomology_layer(C, degree, backend=backend, kernel=kernel,
                                        solve=solve, max_dimension=max_dimension)
        if layer.dimension:
            if degree >= C.ring.D:
                raise ValueError("Nonzero layer outside the topological CSS degree bound")
            result[degree] = layer
    return result
