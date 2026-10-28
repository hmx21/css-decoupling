"""Actual cancellation of every stable free direction, with hostile oracles."""
import pytest

from css_decoupling.algebra.laurent import Ring, Matrix, eye, zeros, direct_sum
from css_decoupling.algebra.cancellation import (
    cancel_stable_frame, complete_unimodular_row_basic, UnimodularCompletionUnavailable,
)
from css_decoupling.errors import NoLaurentSolution


@pytest.mark.parametrize("p", [2, 3, 5])
def test_two_entry_bezout_completion_with_no_unit_entry(p):
    ring = Ring(p, 3)
    x, y, z = ring.variables
    a = x + y * z ** -2
    row = Matrix.from_rows(ring, [[a, 1 - a]])
    assert not any(f.is_unit for f in row.entries.values())
    U, Uinv = complete_unimodular_row_basic(row)
    assert row * U == Matrix.from_rows(ring, [[1, 0]])
    assert U * Uinv == Uinv * U == eye(ring, 2)


def test_unit_completion_at_noninitial_position_and_single_scalar():
    ring = Ring(5, 2)
    x, y = ring.variables
    row = Matrix.from_rows(ring, [[x + y, x - 1, 3 * x ** -2 * y, 1 - y]])
    U, Uinv = complete_unimodular_row_basic(row)
    assert row * U == Matrix.from_rows(ring, [[1, 0, 0, 0]])
    assert U * Uinv == Uinv * U == eye(ring, 4)
    scalar = Matrix.from_rows(ring, [[2 * x ** -3 * y]])
    U, Uinv = complete_unimodular_row_basic(scalar)
    assert scalar * U == eye(ring, 1)
    assert Uinv == scalar


def test_nonunimodular_pair_is_not_completed():
    ring = Ring(3, 2)
    x, y = ring.variables
    with pytest.raises(NoLaurentSolution):
        complete_unimodular_row_basic(Matrix.from_rows(ring, [[x - 1, y - 1]]))


def test_long_row_without_unit_does_not_claim_general_completion():
    ring = Ring(5, 2)
    x, y = ring.variables
    row = Matrix.from_rows(ring, [[x - 1, y - 1, 3 - x - y]])
    assert row * Matrix.from_rows(ring, [[1], [1], [1]]) == eye(ring, 1)
    with pytest.raises(UnimodularCompletionUnavailable):
        complete_unimodular_row_basic(row)


@pytest.mark.parametrize("p", [2, 3, 5])
def test_rank_one_stable_frame_cancels_noncoordinate_extra_direction(p):
    ring = Ring(p, 3)
    x, y, z = ring.variables
    f = x + y * z ** -2
    P = Matrix.from_rows(ring, [[1 + f], [f]])
    Q = Matrix.from_rows(ring, [[1 - f, f]])
    E = P * Q
    Hinv = Matrix.from_rows(ring, [[1, -1], [f, 1 - f]])
    H = Matrix.from_rows(ring, [[1 - f, 1], [-f, 1]])
    A, B = H * direct_sum(Q, eye(ring, 1)), direct_sum(P, eye(ring, 1)) * Hinv
    original_A, original_B = A.copy(), B.copy()
    assert not any(f.is_unit for f in B[-1:, :].entries.values())
    result = cancel_stable_frame(E, A, B)
    assert result["rank"] == 1 and result["odd_free_rank"] == 0
    assert result["basis"].shape == (2, 1)
    assert result["basis"] * result["coordinates"] == E
    assert result["coordinates"] * result["basis"] == eye(ring, 1)
    assert result["cancelled_free_rank"] == 1 and len(result["steps"]) == 1
    assert A == original_A and B == original_B


def test_multiple_extra_directions_are_all_removed():
    ring = Ring(5, 2)
    x, y = ring.variables
    W = Matrix.from_rows(ring, [[1, x, 0], [0, 1, y], [0, 0, 1]])
    Winv = Matrix.from_rows(ring, [[1, -x, x * y], [0, 1, -y], [0, 0, 1]])
    P, Q = W[:, :2], Winv[:2, :]
    E = P * Q
    H = eye(ring, 5)
    Hinv = eye(ring, 5)
    for i, j, f in [(1, 0, x - 1), (3, 2, y + 1), (4, 3, x ** -1 * y)]:
        entry = Matrix(ring, 5, 5, {(i, j): f})
        H, Hinv = (eye(ring, 5) + entry) * H, Hinv * (eye(ring, 5) - entry)
    A, B = H * direct_sum(Q, eye(ring, 3)), direct_sum(P, eye(ring, 3)) * Hinv
    result = cancel_stable_frame(E, A, B)
    assert [(s["extra_before"], s["extra_after"]) for s in result["steps"]] == [(3, 2), (2, 1), (1, 0)]
    assert result["basis"].shape == (3, 2)
    assert result["basis"] * result["coordinates"] == E
    assert result["coordinates"] * result["basis"] == eye(ring, 2)


def test_oracle_is_external_and_every_completion_is_verified():
    ring = Ring(3, 2)
    E = eye(ring, 2)
    A, B = eye(ring, 3), eye(ring, 3)
    calls = []
    def completion(row):
        calls.append(row.copy())
        return complete_unimodular_row_basic(row)
    result = cancel_stable_frame(E, A, B, completion)
    assert len(calls) == 1 and result["rank"] == 2
    with pytest.raises(ValueError, match="e_first"):
        cancel_stable_frame(E, A, B, lambda row: (eye(ring, 3), eye(ring, 3)))
    with pytest.raises(ValueError, match="inverses"):
        cancel_stable_frame(E, A, B, lambda row: (zeros(ring, 3, 3), eye(ring, 3)))


def test_oracle_failure_does_not_return_a_still_stabilized_result():
    ring = Ring(5, 2)
    E, A, B = eye(ring, 1), eye(ring, 3), eye(ring, 3)
    calls = []
    def stop_on_second(row):
        calls.append(1)
        if len(calls) == 2:
            raise UnimodularCompletionUnavailable("deliberate oracle limitation")
        return complete_unimodular_row_basic(row)
    with pytest.raises(UnimodularCompletionUnavailable, match="deliberate"):
        cancel_stable_frame(E, A, B, stop_on_second)
    assert len(calls) == 2
    assert A == eye(ring, 3) and B == eye(ring, 3)


def test_zero_rank_and_already_unstabilized_frames():
    ring = Ring(3, 2)
    E = zeros(ring, 2, 2)
    A = zeros(ring, 2, 2).hstack(eye(ring, 2))
    B = zeros(ring, 2, 2).vstack(eye(ring, 2))
    result = cancel_stable_frame(E, A, B)
    assert result["basis"].shape == (2, 0) and result["coordinates"].shape == (0, 2)
    assert result["cancelled_free_rank"] == 2
    empty = cancel_stable_frame(E, zeros(ring, 0, 2), zeros(ring, 2, 0))
    assert empty["steps"] == []


def test_reject_corrupted_initial_stable_frame():
    ring = Ring(3, 2)
    with pytest.raises(ValueError, match="identities"):
        cancel_stable_frame(eye(ring, 1), eye(ring, 2), zeros(ring, 2, 2))



def test_unit_directions_cancel_and_nonunit_directions_remain():
    from css_decoupling.algebra.cancellation import cancel_unit_directions
    R = Ring(3, 2)
    x, y = R.variables
    # Q=0: both extra rows of B are units, so the whole frame cancels.
    A = Matrix.from_rows(R, [[x, 0], [0, 1]])
    B = Matrix.from_rows(R, [[x ** -1, 0], [0, 1]])
    left_A, left_B, cut = cancel_unit_directions(zeros(R, 0, 0), A, B)
    assert sorted(cut) == [0, 1] and left_A.shape == (0, 0) and left_B.shape == (0, 0)
    # Q=R: the extra row (1+x, 2+x+y+xy) has no Laurent unit and is kept.
    B = Matrix.from_rows(R, [[1, 0], [1 + x, 1]]) * Matrix.from_rows(R, [[1, 1 + y], [0, 1]])
    A = B.inverse()
    assert not any(B[1, j].is_unit for j in range(2))
    left_A, left_B, cut = cancel_unit_directions(eye(R, 1), A, B)
    assert cut == [] and left_A == A and left_B == B
