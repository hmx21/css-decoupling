"""Lattice queries and subspace lifts do not allocate their coset volume."""
import pytest

from css_decoupling.lattice.retained import Lattice
from css_decoupling.lattice.subspaces import subspace_preimage


def test_huge_integer_lattice_keeps_basis_queries_lazy():
    N = 10 ** 60
    parent = Lattice([[N, 1], [0, 1]], max_index=None)
    child = Lattice([[N, 3], [0, 3]], max_index=None)
    assert parent.index == N and child.index == 3 * N
    assert parent.contains(child)
    assert not child.contains(parent)
    assert parent.relative_basis(child) == ((1, 0), (0, 3))
    with pytest.raises(ValueError, match="not contained"):
        child.relative_basis(parent)
    index, shift = parent.reduce((-7, 13))
    assert index == N - 20
    assert shift == (-1, 13)
    assert parent._representatives is parent._lookup is None
    assert child._representatives is child._lookup is None


def test_explicit_coset_access_remains_backward_compatible():
    lattice = Lattice([[2, 1], [0, 3]])
    assert lattice._representatives is None
    reps = lattice.representatives
    assert reps == ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2))
    assert lattice.lookup == {rep: j for j, rep in enumerate(reps)}
    for j, rep in enumerate(reps):
        assert lattice.reduce(rep) == (j, (0, 0))
    assert lattice.metadata()["representatives"] == [list(rep) for rep in reps]


@pytest.mark.parametrize("p", [2, 5, 2147483659])
def test_subspace_preimage_is_exact_with_redundant_generators(p):
    W = [[1, 0, 1], [0, 1, 1], [0, -1, -1]]
    lattice = subspace_preimage(W, p=p)
    assert lattice.index == p
    assert all((lattice.basis[1][j] + lattice.basis[2][j]) % p == 0 for j in range(3))
    assert lattice.contains([[p, 0, 0], [0, p, 0], [0, 0, p]])
    assert lattice._representatives is None
    assert subspace_preimage([[], [], []], p=p).index == p ** 3
    assert subspace_preimage([[1, 0], [0, 1]], p=p).index == 1


def test_subspace_and_containment_input_validation():
    for dimension in (True, 2.0):
        with pytest.raises(ValueError, match="positive integer"):
            subspace_preimage([[1], [0]], p=3, D=dimension)
    with pytest.raises(ValueError):
        subspace_preimage([[1], [True]], p=3)
    with pytest.raises(ValueError):
        subspace_preimage([[1], [0]], p=4)
    with pytest.raises(ValueError, match="dimensions"):
        Lattice([[1]]).contains(Lattice([[1, 0], [0, 1]]))
