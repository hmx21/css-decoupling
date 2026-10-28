"""Resolve, trivialize, then completely cancel the original projective image.

The explicit resolutions below are independent test fixtures. They supply
generators and differentials, never a basis of im(E), to the production API.
The forced general route must produce the final basis itself.
"""
import pytest

from css_decoupling.algebra.laurent import Matrix, Ring, direct_sum, eye, zeros
from css_decoupling.algebra.projective import free_basis
from css_decoupling.errors import NoLaurentSolution, ResourceLimit


def _conjugated_projector(ring):
    """A rank-two projector whose resolution forces a nonunit three-entry row."""
    x, y = ring.variables[:2]
    a, b = x - 1, y - 1
    c = a * a - a
    conjugator = Matrix.from_rows(ring, [
        [1 - a * b, 0, b * b],
        [-a * a, 0, 1 + a * b],
        [0, 1, c],
    ])
    inverse = Matrix.from_rows(ring, [
        [1 + a * b, -b * b, 0],
        [-c * a * a, -c * (1 - a * b), 1],
        [a * a, 1 - a * b, 0],
    ])
    assert conjugator * inverse == inverse * conjugator == eye(ring, 3)
    coordinate_projector = Matrix.from_rows(ring, [[1, 0, 0], [0, 1, 0], [0, 0, 0]])
    E = conjugator * coordinate_projector * inverse
    kernel_generator = conjugator[:, 2:]
    assert E * E == E
    assert E * kernel_generator == zeros(ring, 3, 1)
    return E, kernel_generator


def _check_original_image(E, result, rank):
    P, Q = result["basis"], result["coordinates"]
    assert P.shape == (E.nrows, rank)
    assert Q.shape == (rank, E.nrows)
    assert P * Q == E
    assert Q * P == eye(E.ring, rank)
    assert result["rank"] == result["even_free_rank"] == rank
    assert result["odd_free_rank"] == 0


def _check_all_cancellations(E, construction, extra):
    frame, cancellation = construction["stable_frame"], construction["cancellation"]
    A, B = frame["to_free"], frame["from_free"]
    assert frame["odd_free_rank"] == extra > 0
    assert A * B == eye(E.ring, A.nrows)
    assert B * A == direct_sum(E, eye(E.ring, extra))
    assert cancellation["cancelled_free_rank"] == extra
    assert cancellation["odd_free_rank"] == 0
    assert [(s["extra_before"], s["extra_after"]) for s in cancellation["steps"]] == [
        (s, s - 1) for s in range(extra, 0, -1)
    ]
    for step in cancellation["steps"]:
        v, u = step["row"], step["dual_column"]
        assert v == B[-1:, :]
        assert u == A[:, -1:]
        assert v * u == eye(E.ring, 1)
        target = zeros(E.ring, 1, v.ncols)
        target[0, 0] = 1
        assert v * step["completion"] == target
        W, Winv = step["alignment"], step["alignment_inverse"]
        assert W * Winv == Winv * W == eye(E.ring, W.nrows)
        assert v * W == target
        assert W * target.T == u
        A, B = (Winv * A)[1:, :-1], (B * W)[:-1, 1:]
    assert cancellation["basis"] == B
    assert cancellation["coordinates"] == A
    assert B * A == E
    assert A * B == eye(E.ring, A.nrows)


@pytest.mark.parametrize("p", [2, 3, 5])
def test_forced_general_basis_runs_local_global_completion(p):
    E, kernel_generator = _conjugated_projector(Ring(p, 2))
    originals = [E.copy(), kernel_generator.copy()]
    requested = []

    def resolution(projector):
        requested.append(projector.copy())
        return [projector, kernel_generator]

    trace = []
    result = free_basis(E, method="quillen-suslin", image_resolution=resolution,
                        qs_options={"trace": trace, "timeout": 30})
    assert requested == [E]
    assert [s["method"] for s in result["steps"]] == ["quillen-suslin-complete-cancellation"]
    assert len(result["general_constructions"]) == 1
    construction = result["general_constructions"][0]
    _check_all_cancellations(E, construction, extra=1)
    row = construction["cancellation"]["steps"][0]["row"]
    assert row.ncols == 3
    assert not any(f.is_unit for f in row.entries.values())
    # The trace establishes that a genuine local-global QS calculation ran;
    # neither unit elimination nor a two-entry base case completes this call.
    assert trace and any(len(t["patch_denominators"]) > 1 for t in trace)
    assert any(len(t["axes"]) == 2 for t in trace)
    _check_original_image(E, result, rank=2)
    assert [E, kernel_generator] == originals


@pytest.mark.parametrize("p,D", [(2, 3), (5, 4)])
def test_longer_exact_resolution_cancels_every_odd_free_summand(p, D):
    ring = Ring(p, D)
    E, kernel_generator = _conjugated_projector(ring)
    # im(d1)=ker(E), ker(d1)=<e2>, ker(d2)=<e2>, d3 injective.
    # F0,F1,F2,F3 have ranks 3,2,2,1: the intermediate stabilization has
    # three odd free directions, all of which must disappear from the result.
    maps = [E, kernel_generator.hstack(zeros(ring, 3, 1)),
            Matrix.from_rows(ring, [[0, 0], [1, 0]]),
            Matrix.from_rows(ring, [[0], [1]])]
    original_maps = [m.copy() for m in maps]
    trace = []
    result = free_basis(E, method="quillen-suslin", image_resolution=lambda _: maps,
                        qs_options={"trace": trace, "timeout": 30})
    _check_all_cancellations(E, result["general_constructions"][0], extra=3)
    assert trace
    _check_original_image(E, result, rank=2)
    assert maps == original_maps


@pytest.mark.parametrize("p,D", [(2, 2), (5, 4)])
def test_general_basis_uses_native_resolution_without_supplied_basis(p, D):
    ring = Ring(p, D)
    x, y = ring.variables[:2]
    conjugator, inverse = eye(ring, 3), eye(ring, 3)
    for i, j, coefficient in [(0, 1, x - 1), (1, 2, y - 1),
                               (2, 0, x + y), (0, 2, x ** -1)]:
        entry = Matrix(ring, 3, 3, {(i, j): coefficient})
        conjugator = conjugator * (eye(ring, 3) + entry)
        inverse = (eye(ring, 3) - entry) * inverse
    coordinate_projector = Matrix.from_rows(ring, [[1, 0, 0], [0, 1, 0], [0, 0, 0]])
    E = conjugator * coordinate_projector * inverse
    result = free_basis(E, method="quillen-suslin")
    construction = result["general_constructions"][0]
    assert len(construction["resolution"]) > 1
    _check_all_cancellations(E, construction, extra=construction["stable_frame"]["odd_free_rank"])
    _check_original_image(E, result, rank=2)


def test_incomplete_resolution_cannot_be_used_to_claim_a_basis():
    E, _ = _conjugated_projector(Ring(3, 2))
    with pytest.raises(NoLaurentSolution):
        free_basis(E, method="quillen-suslin", image_resolution=lambda F: [F])


def test_completion_resource_failure_never_returns_the_stabilized_module():
    E, kernel_generator = _conjugated_projector(Ring(5, 2))
    original = E.copy()
    # This row needs a second local patch; the cap stops before a full cover.
    with pytest.raises(ResourceLimit, match="local-patch cap"):
        free_basis(E, method="quillen-suslin",
                   image_resolution=lambda F: [F, kernel_generator],
                   qs_options={"max_patches": 1})
    assert E == original


@pytest.mark.parametrize("rank", [0, 2])
def test_general_zero_and_full_images(rank):
    ring = Ring(5, 3)
    E = eye(ring, 2) if rank else zeros(ring, 2, 2)
    result = free_basis(E, method="quillen-suslin")
    _check_original_image(E, result, rank)
