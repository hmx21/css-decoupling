"""Independent signed boundary/homotopy checks for explicit Koszul lifts."""
from itertools import combinations
from math import comb
from random import Random
from unittest.mock import patch

import pytest

from css_decoupling.algebra.laurent import Matrix, Ring, zeros
from css_decoupling.errors import NoLaurentSolution, ResourceLimit
from css_decoupling.topology.complexes import koszul
from css_decoupling.topology.koszul_lifts import (
    koszul_boundary, koszul_homotopy, koszul_preimage,
)


def boundary_matrix(ring, degree, copies):
    """Independent construction with explicitly enumerated wedge bases."""
    D = ring.D
    if degree == 0:
        return zeros(ring, 0, copies)
    if degree == D+1:
        return zeros(ring, copies, 0)
    sources = tuple(combinations(range(D), degree))
    targets = {indices:i for i, indices in enumerate(combinations(range(D), degree-1))}
    matrix = zeros(ring, len(targets)*copies, len(sources)*copies)
    for j, indices in enumerate(sources):
        for position, axis in enumerate(indices):
            i = targets[indices[:position]+indices[position+1:]]
            for copy in range(copies):
                matrix[i*copies+copy,j*copies+copy] = (-1)**position*(ring.variable(axis)-1)
    return matrix


@pytest.mark.parametrize("dimension,prime,copies", [(1, 2, 1), (2, 3, 2), (3, 5, 3), (4, 3, 2), (5, 5, 1)])
def test_homotopy_identity_in_every_exterior_degree_with_negative_powers(dimension, prime, copies):
    ring = Ring(prime, dimension)
    rng = Random(31831+dimension+prime+copies)
    for k in range(dimension+1):
        B = zeros(ring, comb(dimension,k)*copies, 3)
        for _ in range(18):
            row, column = rng.randrange(B.nrows), rng.randrange(B.ncols)
            exponent = tuple(rng.randrange(-3,4) for _ in range(dimension))
            B[row,column] += ring.monomial(exponent,rng.randrange(1,prime))
        before = B.key()
        dB = boundary_matrix(ring,k,copies)*B
        assert koszul_boundary(B,k,copies=copies) == dB
        hB = koszul_homotopy(B,k,copies=copies)
        left = boundary_matrix(ring,k+1,copies)*hB
        if k:
            left += koszul_homotopy(dB,k-1,copies=copies)
            assert left == B
        else:
            augmentation = Matrix(ring,B.nrows,B.ncols,
                                  {at:sum(f.terms.values()) for at,f in B.entries.items()})
            assert left == B-augmentation
        assert B.key() == before


@pytest.mark.parametrize("prime,copies", [(2, 2), (3, 3), (5, 2)])
def test_exact_boundary_lifts_match_existing_koszul_copy_convention(prime,copies):
    ring=Ring(prime,4)
    complex=koszul(ring,degree=2,copies=copies)
    x,y,z,w=ring.variables
    for k in range(4):
        source=zeros(ring,comb(4,k+1)*copies,2)
        source[0,0]=2*x**-2-y*w
        source[source.nrows-1,1]=z**3*w**-1+1
        B=complex.d(2-k-1)*source
        Y=koszul_preimage(B,k,copies=copies)
        assert complex.d(2-k-1)*Y==B
        assert koszul_boundary(Y,k+1,copies=copies)==B


def test_negative_geometric_sum_and_prime_field_sign_are_explicit():
    ring=Ring(5,2);x,y=ring.variables
    B=Matrix.from_rows(ring,[[x**-3*y**2]])
    H=koszul_homotopy(B,0)
    assert H[0,0]==-(x**-3+x**-2+x**-1)*y**2
    assert H[1,0]==1+y
    assert koszul_boundary(H,1)==B-Matrix.from_rows(ring,[[1]])


def test_augmentation_of_earlier_variables_precedes_each_divided_difference():
    ring=Ring(3,3);x,y,z=ring.variables
    # K_1 row 2 is e_2.  Only axes 0 and 1 can be inserted before it.
    B=Matrix.from_rows(ring,[[0],[0],[(x-1)*y**-2*z]])
    H=koszul_homotopy(B,1)
    # Wedge rows are (0,1),(0,2),(1,2); the second contribution vanishes
    # because x has already been set to one before differentiating y.
    assert H==Matrix.from_rows(ring,[[0],[y**-2*z],[0]])


def test_homotopy_is_prime_field_linear_but_not_laurent_linear():
    ring=Ring(3,2);x,_=ring.variables
    one=Matrix.from_rows(ring,[[1]])
    assert not koszul_homotopy(one,0)
    assert koszul_homotopy(x*one,0)!=x*koszul_homotopy(one,0)


def test_noncycle_and_nonzero_augmentation_are_exactly_rejected():
    ring=Ring(5,2);x,y=ring.variables
    with pytest.raises(NoLaurentSolution,match="augmentation"):
        koszul_preimage(Matrix.from_rows(ring,[[x+y]]),0)
    with pytest.raises(NoLaurentSolution,match="not a cycle"):
        koszul_preimage(Matrix.from_rows(ring,[[1],[0]]),1)
    assert koszul_preimage(zeros(ring,1,3),2).shape==(0,3)


def test_huge_laurent_power_is_rejected_before_expanding_any_geometric_terms():
    ring=Ring(3,2);x,_=ring.variables
    B=Matrix.from_rows(ring,[[x**-(10**60)-1]])
    with patch('css_decoupling.topology.koszul_lifts._add',side_effect=AssertionError('premature expansion')):
        with pytest.raises(ResourceLimit):
            koszul_homotopy(B,0,max_terms=10)
    # A huge exponent is harmless when the exterior index excludes that axis
    # from H; there is no enclosing exponent-box allocation.
    B=Matrix.from_rows(ring,[[x**(10**60)],[0]])
    assert not koszul_homotopy(B,1,max_terms=1)


def test_sparse_exterior_layer_does_not_enumerate_all_wedge_basis_vectors():
    ring=Ring(3,40);k=20;copies=2
    # The last lexicographic subset is (20,...,39), so only the one active
    # source row needs decoding even though its ambient rank is enormous.
    B=Matrix(ring,comb(40,k)*copies,1,{(comb(40,k)*copies-1,0):ring.variable(0)-1})
    H=koszul_homotopy(B,k,copies=copies,max_terms=10)
    assert len(H.entries)==1
    assert H.nrows==comb(40,k+1)*copies
    assert all(f==ring.one() for f in H.entries.values())


@pytest.mark.parametrize("copies,cols", [(0,4),(2,0)])
def test_empty_copy_or_rhs_spaces_are_supported(copies,cols):
    ring=Ring(3,3)
    B=zeros(ring,comb(3,1)*copies,cols)
    assert koszul_homotopy(B,1,copies=copies).shape==(comb(3,2)*copies,cols)
    assert koszul_preimage(B,1,copies=copies).shape==(comb(3,2)*copies,cols)


@pytest.mark.parametrize("kwargs", [{"exterior_degree":True},{"exterior_degree":-1},
    {"exterior_degree":3},{"exterior_degree":0,"copies":True},
    {"exterior_degree":0,"copies":-1},{"exterior_degree":0,"max_terms":0}])
def test_invalid_degrees_copies_and_caps_are_rejected(kwargs):
    with pytest.raises(ValueError):
        koszul_homotopy(zeros(Ring(3,2),1,1),**kwargs)


def test_preimage_verification_detects_a_corrupted_homotopy_result():
    ring=Ring(3,2);x,_=ring.variables
    B=Matrix.from_rows(ring,[[x-1]])
    with patch('css_decoupling.topology.koszul_lifts.koszul_homotopy',return_value=zeros(ring,2,1)):
        with pytest.raises(ArithmeticError,match="boundary identity"):
            koszul_preimage(B,0)
