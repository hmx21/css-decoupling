"""Optimal tilted D3 cells, general sufficient choices, and hostile witnesses."""
from copy import deepcopy
from math import comb

import pytest

from css_decoupling.algebra.laurent import Ring, Matrix
from css_decoupling.decoupling.gluing import GluingClass
from css_decoupling.decoupling.lattice_selection import select_lattice
from css_decoupling.certificates.lattice_selection import verify_lattice_selection
from css_decoupling.errors import BackendUnavailable
from css_decoupling.lattice.retained import Lattice


@pytest.mark.parametrize("p", [2, 3, 5, 2147483659])
@pytest.mark.parametrize("rows,exponent,rank", [
    ([[0, 0, 0]], 0, 0),
    ([[1, 1, 1]], 1, 1),
    ([[1, 0, 1], [0, 1, 1]], 1, 2),
    ([[1, 0, 0], [0, 1, 0], [0, 0, 1]], 2, 3),
])
def test_complete_d3_rank_classification_for_any_prime(p, rows, exponent, rank):
    ring = Ring(p, 3)
    cls = GluingClass(2, 1, 1, len(rows), Matrix.from_rows(ring, rows))
    result = select_lattice([cls], D=3, p=p, strategy="optimal")
    assert result.relative_index == p ** exponent
    assert result.optimality == "family-optimal"
    if rank:
        assert result.witness["coefficient_rank"] == rank
    assert cls.restrict(result.relative_basis).is_zero
    assert verify_lattice_selection([cls], result.certificate()).valid
    lattice = Lattice(result.relative_basis, max_index=None)
    assert lattice._representatives is None


def test_matrix_multiplicities_use_rank_of_values_not_rank_of_each_matrix():
    ring = Ring(5, 3)
    # Two independent 2x2 coefficient matrices, with their sum as the third.
    cls = GluingClass(2, 1, 2, 2, Matrix.from_rows(ring,
        [[1, 0, 0, 1, 1, 1], [0, 1, 1, 0, 1, 1]]))
    result = select_lattice([cls], D=3, p=5)
    assert result.witness["coefficient_rank"] == 2
    assert result.relative_index == 5
    assert verify_lattice_selection([cls], result.certificate()).valid


def test_multiple_classes_share_one_isotropic_subspace():
    ring = Ring(3, 3)
    family = [GluingClass(2, 1, 1, 1, Matrix.from_rows(ring, [row]))
              for row in [[1, 0, 0], [0, 1, 0], [0, 0, 1]]]
    result = select_lattice(family, D=3, p=3)
    assert result.relative_index == 9
    assert result.witness["coefficient_rank"] == 3
    assert verify_lattice_selection(family, result.certificate()).valid


def test_tilted_cell_beats_every_one_axis_coordinate_choice():
    ring = Ring(5, 3)
    cls = GluingClass(2, 1, 1, 1, Matrix.from_rows(ring, [[1, 1, 1]]))
    automatic = select_lattice([cls], D=3, p=5)
    coordinate = select_lattice([cls], D=3, p=5, strategy="coordinate")
    assert automatic.relative_index == 5
    assert coordinate.relative_index == 25
    assert coordinate.optimality == "sufficient"
    assert verify_lattice_selection([cls], coordinate.certificate()).valid
    original = Lattice([[2, 1, 0], [0, 3, 1], [0, 0, 1]])
    from sympy import Matrix as IntegerMatrix
    absolute = Lattice((original.B * IntegerMatrix(automatic.relative_basis)).tolist())
    assert absolute.index == original.index * automatic.relative_index
    assert original.relative_basis(absolute) == automatic.relative_basis


def test_general_dimensions_and_ext_degrees_do_not_claim_an_unproved_optimum():
    ring = Ring(3, 5)
    # Ext^2 and Ext^3 in the same already-defined family.
    family = [GluingClass(2, 1, 1, 1, Matrix(ring, 1, comb(5, 2), {(0, 0): 1, (0, 9): 2})),
              GluingClass(3, 1, 1, 1, Matrix(ring, 1, comb(5, 3), {(0, 3): 2}))]
    result = select_lattice(family, D=5, p=3)
    assert result.optimality == "sufficient"
    assert all(c.restrict(result.relative_basis).is_zero for c in family)
    assert verify_lattice_selection(family, result.certificate()).valid
    with pytest.raises(BackendUnavailable, match="minimum killing lattice"):
        select_lattice(family, D=5, p=3, strategy="optimal")


def test_sparse_high_degree_coordinate_choice_never_enumerates_exterior_basis():
    ring = Ring(3, 32)
    cls = GluingClass(16, 1, 1, 1, Matrix(ring, 1, comb(32, 16), {(0, 0): 1}))
    result = select_lattice([cls], D=32, p=3, max_exterior_entries=1)
    assert result.relative_index == 3
    assert result.witness["axes"] == [0]
    assert verify_lattice_selection([cls], result.certificate(), max_exterior_entries=1).valid


def test_empty_family_is_certified_without_a_coset_expansion():
    result = select_lattice([], D=5, p=2147483659, strategy="optimal")
    assert result.relative_index == 1
    assert verify_lattice_selection([], result.certificate()).valid


@pytest.mark.parametrize("mutation", ["rank", "relation", "index", "basis", "proper-sublattice", "coefficients",
                                     "dimension", "prime", "boolean", "scope", "omit-class", "optimality"])
def test_tampered_selection_certificates_are_rejected(mutation):
    ring = Ring(5, 3)
    cls = GluingClass(2, 1, 1, 1, Matrix.from_rows(ring, [[1, 1, 1]]))
    data = deepcopy(select_lattice([cls], D=3, p=5).certificate())
    if mutation == "rank":
        data["witness"]["coefficient_rank"] = 2
    elif mutation == "relation":
        data["witness"]["kernel_vector"] = [1, 0, 0]
    elif mutation == "index":
        data["relative_index"] *= 5
    elif mutation == "basis":
        data["relative_basis"] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    elif mutation == "proper-sublattice":
        for row in data["relative_basis"]:
            row[0] *= 5
        data["relative_index"] *= 5
    elif mutation == "coefficients":
        data["classes"][0]["coefficients"]["entries"] = []
    elif mutation == "dimension":
        data["ring"]["dimension"] = 4
    elif mutation == "prime":
        data["ring"]["prime"] = 3
    elif mutation == "boolean":
        data["witness"]["coefficient_rank"] = True
    elif mutation == "scope":
        data["scope"] = "entire-unsplit-tower"
    elif mutation == "omit-class":
        data["classes"] = []
    else:
        data["optimality"] = "global"
    assert not verify_lattice_selection([cls], data).valid


def test_coordinate_choice_cannot_promote_itself_to_an_optimum():
    ring = Ring(5, 3)
    cls = GluingClass(2, 1, 1, 1, Matrix.from_rows(ring, [[1, 1, 1]]))
    data = select_lattice([cls], D=3, p=5, strategy="coordinate").certificate()
    data["optimality"] = "family-optimal"
    assert not verify_lattice_selection([cls], data).valid


def test_wrong_declared_ring_is_rejected():
    cls = GluingClass(2, 1, 1, 1, Matrix.from_rows(Ring(3, 3), [[1, 0, 0]]))
    with pytest.raises(ValueError, match="declared ring"):
        select_lattice([cls], D=3, p=5)
