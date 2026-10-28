"""Failure categories shared by all general algorithm backends."""


class ResourceLimit(RuntimeError):
    """An explicit resource cap was reached; the mathematical answer is unknown."""


class BackendUnavailable(RuntimeError):
    """The requested exact backend is unavailable or cannot represent the domain."""


class NoLaurentSolution(ValueError):
    """An exact algebraic witness proves Laurent module nonmembership."""


class InfiniteLayer(ValueError):
    """The initial module contains a certified infinite standard-monomial ray."""
