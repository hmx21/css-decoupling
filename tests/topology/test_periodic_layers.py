"""Adjacent Koszul counts checked against actual finite periodic CSS matrices."""
from itertools import combinations
from math import comb

import pytest

from css_decoupling import Matrix, Ring, eye, zeros
from css_decoupling.algebra.laurent import direct_sum
from css_decoupling.decoupling.analysis import analyze_profile
from css_decoupling.errors import ResourceLimit
from css_decoupling.lattice.retained import Lattice
from css_decoupling.models.examples import css_checks, glued_toric_chain, glued_css
from css_decoupling.topology.periodic import periodic_code, periodic_layer, periodic_profile


def diagonal(values):
    return [[v if i == j else 0 for j in range(len(values))] for i, v in enumerate(values)]


def action_css(actions, degree):
    """Independent free Koszul presentation of x_a acting as actions[a]."""
    ring, t = actions[0].ring, actions[0].nrows

    def boundary(k):
        source = list(combinations(range(ring.D), k))
        target = {J: n for n, J in enumerate(combinations(range(ring.D), k - 1))}
        out = zeros(ring, len(target) * t, len(source) * t)
        for col, I in enumerate(source):
            for position, axis in enumerate(I):
                row = target[I[:position] + I[position + 1:]]
                value = eye(ring, t) * ring.variable(axis) - actions[axis]
                for (a, b), f in value.entries.items():
                    out[row * t + a, col * t + b] = (-1) ** position * f
        return out

    return boundary(degree), boundary(degree + 1).dagger()


@pytest.mark.parametrize("p,D,j", [(2, 3, 1), (3, 4, 2), (5, 5, 3)])
def test_unipotent_layers_match_finite_css_for_signed_tilted_periods(p, D, j):
    ring = Ring(p, D)
    J = Matrix.from_rows(ring, [[1, 1], [0, 1]])
    actions = [J] + [eye(ring, 2) for _ in range(D - 1)]
    checks = action_css(actions, j)
    for multiple in (1, p):
        basis = diagonal([-multiple] + [1] * (D - 1))
        basis[1][0] = -1
        layer = periodic_layer(actions, j, basis)
        finite = periodic_code(*checks, basis)
        assert layer.logical_qudits == finite.logical_qudits == comb(D, j) * (2 if multiple == p else 1)
        assert layer.translation_trivial == (multiple == p)


@pytest.mark.parametrize("p", [3, 5])
def test_semisimple_and_unipotent_summands_with_mixed_relations(p):
    ring = Ring(p, 3)
    A = Matrix.from_rows(ring, [[1, 1, 0], [0, 1, 0], [0, 0, 2]])
    inverse = A.inverse()
    actions = [A, inverse, eye(ring, 3)]
    # First column has a mixed relation; second carries the remaining action.
    basis = [[1, -1, 0], [1, 0, 0], [0, 0, 1]]
    result = periodic_layer(actions, 1, basis)
    assert result.period_actions[0] == eye(ring, 3)
    assert result.logical_qudits == 3
    assert periodic_code(*action_css(actions, 1), basis).logical_qudits == result.logical_qudits


def test_original_nontrivial_single_layer_large_period_never_expands(monkeypatch):
    ring = Ring(3, 2)
    x, y = ring.variables
    HX = Matrix.from_rows(ring, [[x + 1, y + 1]])
    HZ = Matrix.from_rows(ring, [[-(y + 1)], [x + 1]]).dagger()
    analysis = analyze_profile(HX, HZ)

    original = Lattice.representatives.fget

    def guarded(lattice):
        if lattice.index > 100:
            raise AssertionError("Periodic-cell expansion is forbidden")
        return original(lattice)

    monkeypatch.setattr(Lattice, "representatives", property(guarded))
    for length in (10**80, 10**80 + 1):
        result = periodic_profile(analysis, diagonal([length, length]), max_qudits=1)
        assert result.method == "single-layer"
        assert result.logical_qudits == (2 if length % 2 == 0 else 0)
        assert result.translation_loss == 2 - result.logical_qudits
        assert result.gluing_loss == 0


def test_large_dimension_allocates_only_adjacent_degrees(monkeypatch):
    import css_decoupling.topology.periodic as module
    ring = Ring(3, 80)
    actions = [eye(ring, 1) for _ in range(ring.D)]

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Zero differentials do not need exterior cell lists")

    monkeypatch.setattr(module, "combinations", forbidden)
    result = periodic_layer(actions, 1, diagonal([1] * ring.D))
    assert result.logical_qudits == 80
    with pytest.raises(ResourceLimit, match="bases"):
        periodic_layer(actions, 40, diagonal([1] * ring.D))


def test_zero_dimensional_layer_has_zero_contribution():
    ring = Ring(5, 4)
    result = periodic_layer([zeros(ring, 0, 0)] * 4, 2, diagonal([-1, 2, 3, 4]))
    assert result.logical_qudits == result.incoming_rank == result.outgoing_rank == 0
    assert result.translation_trivial


def test_two_layer_diagnostics_keep_nonzero_gluing_loss():
    analysis = analyze_profile(*glued_css(Ring(3, 3)))
    small = periodic_profile(analysis, diagonal([1, 1, 1]))
    assert (small.translation_loss, small.gluing_loss, small.logical_qudits) == (0, 2, 4)
    assert [layer["logical_qudits"] for layer in small.layer_contributions] == [3, 3]
    killed = periodic_profile(analysis, diagonal([3, 1, 1]))
    assert (killed.translation_loss, killed.gluing_loss, killed.logical_qudits) == (0, 0, 6)


def test_multilayer_count_uses_whole_complex_and_reports_combined_loss():
    checks = css_checks(glued_toric_chain(Ring(3, 4), [1, 2, 3]))
    analysis = analyze_profile(*checks)
    basis = diagonal([1] * 4)
    result = periodic_profile(analysis, basis)
    assert result.method == "finite"
    assert result.logical_qudits == periodic_code(*checks, basis).logical_qudits
    assert sum(c["logical_qudits"] for c in result.layer_contributions) == 14
    assert result.gluing_loss == 14 - result.logical_qudits > 0
    assert result.translation_loss == 0


def test_multilayer_translation_loss_is_separate_from_gluing():
    ring = Ring(3, 3)
    scalar_actions = [Matrix.from_rows(ring, [[2]])] * 3
    first = action_css(scalar_actions, 1)
    second = action_css([eye(ring, 1)] * 3, 2)
    checks = direct_sum(first[0], second[0]), direct_sum(first[1], second[1])
    analysis = analyze_profile(*checks)
    result = periodic_profile(analysis, diagonal([1] * 3))
    assert result.method == "finite"
    assert (result.logical_qudits, result.translation_loss, result.gluing_loss) == (3, 3, 0)
    assert [c["logical_qudits"] for c in result.layer_contributions] == [0, 3]


def test_empty_profile_fast_count_and_forced_finite_agree():
    ring = Ring(3, 2)
    analysis = analyze_profile(eye(ring, 1), zeros(ring, 0, 1))
    fast = periodic_profile(analysis, diagonal([10**40, 1]), max_qudits=1)
    assert fast.logical_qudits == 0 and fast.layer_contributions == []
    assert fast.translation_loss == fast.gluing_loss == 0
    assert periodic_profile(analysis, diagonal([1, 1]), prefer_fast=False).logical_qudits == 0


def test_invalid_action_representations_and_resources():
    ring = Ring(3, 2)
    I = eye(ring, 2)
    A = Matrix.from_rows(ring, [[1, 1], [0, 1]])
    B = Matrix.from_rows(ring, [[1, 0], [1, 1]])
    basis = diagonal([1, 1])
    for actions in ([A], [A, B], [I, zeros(ring, 2, 2)],
                    [I, Matrix.from_rows(ring, [[ring.variable(0), 0], [0, 1]])],
                    [I, eye(Ring(5, 2), 2)], [I, eye(ring, 1)]):
        with pytest.raises(ValueError):
            periodic_layer(actions, 1, basis)
    for degree in (0, 2, True):
        with pytest.raises(ValueError):
            periodic_layer([I, I], degree, basis)
    for option in (dict(max_matrix_entries=1), dict(max_rank_work=1), dict(max_exterior_entries=1)):
        with pytest.raises(ResourceLimit):
            periodic_layer([A, I], 1, basis, **option)


@pytest.mark.parametrize("p", [3, 5])
def test_independent_nilpotent_axes_have_correct_odd_characteristic_signs(p):
    ring = Ring(p, 4)
    U = eye(ring, 4)
    V = eye(ring, 4)
    U[0, 2] = U[1, 3] = 1
    V[0, 1] = V[2, 3] = 1
    actions = [U, V, eye(ring, 4), eye(ring, 4)]
    basis = [[-1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    result = periodic_layer(actions, 2, basis)
    assert result.logical_qudits == periodic_code(*action_css(actions, 2), basis).logical_qudits == 6
