"""Keep the original nonzero homology projector when retracting a cone.

The projector below is deliberately not aligned with the unit-disk retract:
both correction terms H*P and (i*p)*P*H are nonzero.  Testing P=0, or a
projector commuting with the retract, would miss these two essential terms.
"""
import pytest

from css_decoupling.algebra.laurent import Matrix, Ring, eye, zeros
from css_decoupling.algebra.modules import solve
from css_decoupling.decoupling.schanuel import _contract_around_projector
from css_decoupling.errors import ResourceLimit
from css_decoupling.topology.complexes import CochainComplex, direct_sum_complexes, disk
from css_decoupling.topology.contractions import cancel_unit_disks


def _projected_example(prime, nonunit_residual):
    ring = Ring(prime, 3)
    x, y, _ = ring.variables
    # In degree zero the coordinates begin with: homology, incoming disk,
    # outgoing disk.  The two Laurent parameters change both the section
    # and projection without changing the differential.
    C = direct_sum_complexes(CochainComplex(ring, {0: 1}, {}),
                             disk(ring, -1), disk(ring, 0))
    if nonunit_residual:
        # Neither a nor b is a Laurent unit, but b-x*a=1 over every prime
        # field.  This exact Koszul complex needs the residual solver.
        a, b = x + 1, x*x + x + 1
        E = CochainComplex(ring, {-1: 1, 0: 2, 1: 1}, {
            -1: Matrix.from_rows(ring, [[-b], [a]]),
            0: Matrix.from_rows(ring, [[a, b]]),
        })
        C = direct_sum_complexes(C, E)
    u, v = 2*x**-1 + y, x + 3*y**-2
    section = Matrix(ring, C.rank(0), 1, {(0, 0): 1, (1, 0): v})
    projection = Matrix(ring, 1, C.rank(0), {(0, 0): 1, (0, 2): u})
    projectors = {n: section*projection if n == 0 else zeros(ring, C.rank(n), C.rank(n))
                  for n in C.degrees}
    return C, section, projection, projectors


def _snapshot(C, projectors):
    return (dict(C.ranks), {n: value.key() for n, value in C.differentials.items()},
            {n: value.key() for n, value in projectors.items()})


def _assert_projected_identity(C, projectors, homotopy):
    for n in C.degrees:
        following = homotopy.get(n + 1, zeros(C.ring, C.rank(n), C.rank(n + 1)))
        assert (C.d(n - 1)*homotopy[n] + following*C.d(n)
                == eye(C.ring, C.rank(n)) - projectors[n])


@pytest.mark.parametrize("prime", [2, 3, 5])
@pytest.mark.parametrize("nonunit_residual", [False, True])
def test_nonzero_projector_needs_both_transfer_corrections(prime, nonunit_residual):
    C, section, projection, projectors = _projected_example(prime, nonunit_residual)
    before = _snapshot(C, projectors)
    assert projection*section == eye(C.ring, 1)
    assert projectors[0] and projectors[0]*projectors[0] == projectors[0]

    reduction = cancel_unit_disks(C)
    retract = reduction.inclusion.at(0)*reduction.projection.at(0)
    assert reduction.homotopy[0]*projectors[0]
    assert retract*projectors[0]*reduction.homotopy[1]
    # The homology projector is not interchangeable with the retract's
    # own choice of residual inclusion/projection.
    assert retract*projectors[0] != projectors[0]*retract

    raw = _contract_around_projector(C, projectors, solve)
    _assert_projected_identity(C, projectors, raw)
    normalized = {n: raw[n]*C.d(n - 1)*raw[n] for n in C.degrees}
    _assert_projected_identity(C, projectors, normalized)
    for n in C.degrees:
        previous = normalized.get(n - 1, zeros(C.ring, C.rank(n - 2), C.rank(n - 1)))
        assert not previous*normalized[n]
    assert not normalized[0]*section
    assert not projection*normalized[1]
    assert _snapshot(C, projectors) == before


@pytest.mark.parametrize("mutation", ["not-idempotent", "not-cycle", "hits-boundaries", "wrong-shape"])
def test_invalid_homology_projectors_are_rejected(mutation):
    C, _, _, projectors = _projected_example(5, True)
    projectors = {n: value.copy() for n, value in projectors.items()}
    if mutation == "not-idempotent":
        projectors[0] = 2*projectors[0]
    elif mutation == "not-cycle":
        # Still idempotent, but its image is the outgoing disk coordinate.
        projectors[0] = Matrix(C.ring, C.rank(0), C.rank(0), {(2, 2): 1})
        assert projectors[0]*projectors[0] == projectors[0]
    elif mutation == "hits-boundaries":
        # An idempotent onto boundaries consists of cycles, but cannot be
        # the homology projector required by the contraction formula.
        projectors[0] = Matrix(C.ring, C.rank(0), C.rank(0), {(1, 1): 1})
        assert projectors[0]*projectors[0] == projectors[0]
    else:
        projectors[0] = zeros(C.ring, C.rank(0) - 1, C.rank(0))
    with pytest.raises(ValueError, match="Invalid homology projector"):
        _contract_around_projector(C, projectors, solve)


def test_false_residual_solution_is_rejected_on_the_actual_equation():
    C, _, _, projectors = _projected_example(3, True)
    with pytest.raises(ValueError, match="Residual cone splitting"):
        _contract_around_projector(
            C, projectors, lambda A, B: zeros(C.ring, A.ncols, B.ncols))


def test_residual_budget_failure_is_not_a_projector_or_gluing_obstruction():
    C, _, _, projectors = _projected_example(3, True)

    def limited(A, B):
        raise ResourceLimit("residual test budget")

    with pytest.raises(ResourceLimit, match="residual test budget"):
        _contract_around_projector(C, projectors, limited)
