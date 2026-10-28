"""Exact matrix lifts must not replicate independent toric-copy operators."""
import pytest

from css_decoupling import Ring, Matrix, eye
from css_decoupling.algebra.modules import solve
from css_decoupling.decoupling.splitting import splitting_map
from css_decoupling.topology.complexes import koszul, solve_matrix_equations


@pytest.mark.parametrize("prime", [2, 3, 5])
def test_left_matrix_equation_preserves_all_rhs_without_vectorization(prime):
    R = Ring(prime, 3)
    x, y, z = R.variables
    A = Matrix.from_rows(R, [[x - 1, y - 1]])
    X = Matrix.from_rows(R, [[x**-1, y, 2*z], [z, x*y, -1]])
    B = A * X
    calls = []

    def counted(a, b):
        calls.append((a.shape, b.shape))
        return solve(a, b)

    result = solve_matrix_equations(R, {"X": (2, 3)},
                                   [([(A, "X", eye(R, 3))], B)], counted)
    assert A * result["X"] == B
    assert calls == [(A.shape, B.shape)]


@pytest.mark.parametrize("prime", [2, 3, 5])
def test_multiple_toric_copies_keep_interleaved_basis_and_signed_chain_maps(prime, monkeypatch):
    # Exercise the joint-equation branch independently of the descending
    # positive shortcut, whose success would avoid vectorization entirely.
    monkeypatch.setattr("css_decoupling.decoupling.splitting._descending_lift", lambda *a: None)
    R = Ring(prime, 4)
    C = koszul(R, degree=2, copies=3)
    # A nontrivial invertible change of charge representatives, including
    # negative support and an odd-characteristic coefficient.
    reps = eye(R, 3)
    reps[0, 1] = R.variable(2)**-1
    reps[1, 2] = 2 * R.variable(3)
    widths = []

    def counted(A, B):
        widths.append(A.ncols)
        return solve(A, B)

    phi = splitting_map(C, {2: reps}, window=True, solver=counted)
    assert phi.at(2) == reps
    assert phi.validate()
    # Three identical smaller presentations, each with a genuine cochain
    # lift; the original copy-major vectorization had three times this width.
    one_copy_width = C.rank(0) * 6 + C.rank(1) * 4
    assert widths == [one_copy_width] * 3


def test_bad_direct_matrix_solution_is_still_rejected():
    R = Ring(2, 3)
    with pytest.raises(ValueError, match="Invalid simultaneous"):
        solve_matrix_equations(R, {0: (1, 2)},
                               [([(eye(R, 1), 0, eye(R, 2))], Matrix.from_rows(R, [[1, 1]]))],
                               lambda A, B: Matrix.from_rows(R, [[0, 0]]))


def test_direct_matrix_shortcut_does_not_accept_mixed_coefficient_fields():
    R, S = Ring(3, 3), Ring(5, 3)
    with pytest.raises(ValueError, match="common ring"):
        solve_matrix_equations(R, {0: (1, 1)},
                               [([(eye(S, 1), 0, eye(R, 1))], eye(S, 1))], solve)


def test_unextendable_greedy_choice_falls_back_to_joint_chain_equations():
    from css_decoupling.topology.complexes import CochainComplex, direct_sum_complexes, koszul_window
    R = Ring(3, 3)
    T = koszul_window(R, degree=2)
    C = direct_sum_complexes(T, CochainComplex(R, {1: 1}, {}))
    calls, injected = [], False

    def choose_bad_first_lift(A, B):
        nonlocal injected
        calls.append(A.shape)
        result = solve(A, B)
        if not injected and A == C.d(1):
            result[-1, 0] += 1  # A valid top lift with an unextendable free H^1 part.
            assert A * result == B
            injected = True
        return result

    phi = splitting_map(C, {2: eye(R, 1)}, window=True, solver=choose_bad_first_lift)
    assert injected and phi.validate()
    assert phi.at(2) == eye(R, 1)
    assert len(calls) == 3  # Top lift, failed lower lift, actual joint Laurent solve.


def test_descending_multicopy_lifts_use_only_original_differential_sizes():
    R = Ring(5, 4)
    C = koszul(R, degree=2, copies=3)
    calls = []

    def counted(A, B):
        calls.append(A.shape)
        return solve(A, B)

    phi = splitting_map(C, {2: eye(R, 3)}, window=True, solver=counted)
    assert phi.validate()
    assert calls == [C.d(1).shape, C.d(0).shape] * 3
