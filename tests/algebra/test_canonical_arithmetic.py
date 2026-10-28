"""Independent exact checks for internal canonical arithmetic fast paths."""
from random import Random
from types import MappingProxyType

import pytest

import css_decoupling.algebra.laurent as laurent
from css_decoupling.algebra.laurent import Matrix, Poly, Ring, block, direct_sum, hstack, vstack, zeros


def add_terms(left, right, p, sign=1):
    keys = set(left) | set(right)
    return {e:value for e in keys if (value := (left.get(e,0)+sign*right.get(e,0)) % p)}


def multiply_terms(left, right, p):
    result = {}
    for a,c in left.items():
        for b,d in right.items():
            exponent = tuple(a[i]+b[i] for i in range(len(a)))
            result[exponent] = result.get(exponent,0)+c*d
    return {e:c % p for e,c in result.items() if c % p}


@pytest.mark.parametrize("prime,dimension", [(2,2),(3,4),(5,5)])
def test_poly_arithmetic_matches_independent_integer_tuple_accumulation(prime,dimension):
    ring=Ring(prime,dimension);rng=Random(5912+prime+dimension)
    for _ in range(12):
        left=ring({tuple(rng.randrange(-5,6) for _ in range(dimension)):rng.randrange(-prime,2*prime)
                   for __ in range(14)})
        right=ring({tuple(rng.randrange(-5,6) for _ in range(dimension)):rng.randrange(-prime,2*prime)
                    for __ in range(14)})
        before=(dict(left.terms),dict(right.terms))
        assert dict((left+right).terms)==add_terms(*before,prime)
        assert dict((left-right).terms)==add_terms(*before,prime,sign=-1)
        assert dict((-left).terms)=={e:(-c)%prime for e,c in before[0].items()}
        assert dict((left*right).terms)==multiply_terms(*before,prime)
        assert dict(left.involution().terms)=={tuple(-x for x in e):c for e,c in before[0].items()}
        assert (dict(left.terms),dict(right.terms))==before
        for result in (left+right,left-right,-left,left*right,left.involution()):
            assert isinstance(result.terms,MappingProxyType)
            assert all(type(c) is int and 0<c<prime and len(e)==dimension
                       and all(type(x) is int for x in e) for e,c in result.terms.items())


def test_public_input_ownership_and_lazy_keys_preserve_hash_equality_and_repr():
    ring=Ring(5,2)
    external={(4,-2):7,(-3,1):-1,(0,0):5}
    f=Poly(ring,external)
    external[(4,-2)]=99
    external.clear()
    assert dict(f.terms)=={(4,-2):2,(-3,1):4}
    assert f._key is None
    summed=f+ring.variable(0)
    assert summed._key is None and f._key is None
    equal=ring({(-3,1):4,(4,-2):2})
    assert f==equal
    assert f.key()==(((-3,1),4),((4,-2),2))
    assert hash(f)==hash(equal)
    assert {f:"value"}[equal]=="value"
    assert repr(f)==repr(equal)
    with pytest.raises(TypeError):
        summed.terms[(0,0)]=1


def dense_product(A,B):
    result={}
    for i in range(A.nrows):
        for j in range(B.ncols):
            terms={}
            for k in range(A.ncols):
                terms=add_terms(terms,multiply_terms(A[i,k].terms,B[k,j].terms,A.ring.p),A.ring.p)
            if terms:result[i,j]=terms
    return Matrix(A.ring,A.nrows,B.ncols,result)


@pytest.mark.parametrize("prime", [2,3,5])
@pytest.mark.parametrize("encoded", [False,True])
def test_matrix_arithmetic_agrees_with_dense_tuple_oracle(prime,encoded,monkeypatch):
    monkeypatch.setattr(laurent,"_CONVOLUTION_MIN_PRODUCTS",0 if encoded else 10**30)
    monkeypatch.setattr(laurent,"_CONVOLUTION_REUSE_FACTOR",0 if encoded else 10**30)
    ring=Ring(prime,3);rng=Random(18971+prime)
    def matrix(rows,cols):
        return Matrix(ring,rows,cols,{
            (i,j):{tuple(rng.randrange(-3,4) for _ in range(3)):rng.randrange(1,prime)
                   for __ in range(7)}
            for i in range(rows) for j in range(cols) if rng.random()<.7})
    A,B,C=matrix(3,4),matrix(3,4),matrix(4,2)
    before=(A.key(),B.key(),C.key())
    plus=Matrix(ring,3,4,{(i,j):add_terms(A[i,j].terms,B[i,j].terms,prime)
                          for i in range(3) for j in range(4)})
    minus=Matrix(ring,3,4,{(i,j):add_terms(A[i,j].terms,B[i,j].terms,prime,-1)
                           for i in range(3) for j in range(4)})
    assert A+B==plus and A-B==minus
    assert A*C==dense_product(A,C)
    assert (A*C).dagger()==C.dagger()*A.dagger()
    assert not A-A and not A+(-A)
    assert A.transpose().transpose()==A
    assert A*0==zeros(ring,3,4)
    assert (A.key(),B.key(),C.key())==before
    for result in (A+B,A-B,A*C,-A,A.dagger(),A.copy()):
        assert all(value for value in result.entries.values())
        assert result.entries is not A.entries


def test_block_assembly_preserves_unequal_partition_counts_and_empty_blocks():
    ring=Ring(3,2);x,y=ring.variables
    A=Matrix.from_rows(ring,[[x],[y]])
    B=Matrix.from_rows(ring,[[1,0],[0,2]])
    C=Matrix.from_rows(ring,[[x,y,1]])
    original=[A.key(),B.key(),C.key()]
    result=block([[A,zeros(ring,2,0),B],[C]])
    assert result==Matrix.from_rows(ring,[[x,1,0],[y,0,2],[x,y,1]])
    assert hstack(A,B)==result[:2,:]
    assert vstack(hstack(A,B),C)==result
    assert direct_sum(A,C)==block([[A,zeros(ring,2,3)],[zeros(ring,1,1),C]])
    result[0,0]=2
    assert [A.key(),B.key(),C.key()]==original
    assert block([[zeros(ring,0,2),zeros(ring,0,0)],[zeros(ring,3,2)]]).shape==(3,2)
    assert block([[zeros(ring,2,0)],[zeros(ring,1,0)]]).shape==(3,0)


def test_slice_and_copy_containers_do_not_alias_source_or_external_entries():
    ring=Ring(5,2);x,y=ring.variables
    raw={(0,0):x,(0,1):y,(1,0):x+y}
    A=Matrix(ring,2,2,raw);raw.clear()
    original=A.copy()
    for result in (A.copy(),-A,A.T,A.dagger(),A[:,:],A+0,A-0,A*1):
        result[0,0]=0
        assert A==original
    repeated=A[[1,0,1],[1,0,0]]
    assert repeated==Matrix.from_rows(ring,[[0,x+y,x+y],[y,x,x],[0,x+y,x+y]])
    A[:,:]=A[::-1,::-1]
    assert A==original[::-1,::-1]


@pytest.mark.parametrize("terms,error", [
    ({(0,):1},ValueError),({(0,0,0):1},ValueError),
    ({(0.5,0):1},TypeError),({("0",0):1},TypeError),
    ({(0,0):1.5},TypeError),({(0,0):"1"},TypeError),
])
def test_malformed_public_polynomials_still_rejected(terms,error):
    ring=Ring(3,2)
    with pytest.raises(error):Poly(ring,terms)
    with pytest.raises(error):ring(terms)


def test_malformed_public_matrices_and_cross_ring_operations_still_rejected():
    ring=Ring(3,2);other=Ring(5,2)
    with pytest.raises(TypeError):Matrix(ring,1.5,2)
    with pytest.raises(ValueError):Matrix(ring,-1,2)
    with pytest.raises(IndexError):Matrix(ring,1,1,{(1,0):1})
    with pytest.raises(TypeError):Matrix(ring,1,1,{(0.5,0):1})
    with pytest.raises(ValueError):Matrix(ring,1,1,{(0,0):other.one()})
    with pytest.raises(ValueError):Matrix.from_rows(ring,[[1],[1,2]])
    A=Matrix.from_rows(ring,[[1]])
    with pytest.raises(ValueError):A[0,0]=other.one()
    with pytest.raises(TypeError):A[0,0]=1.1
    with pytest.raises(ValueError):_=A+Matrix.from_rows(other,[[1]])
    with pytest.raises(ValueError):_=A-Matrix.from_rows(other,[[1]])
    with pytest.raises(ValueError):block([[A,zeros(ring,2,1)]])
    with pytest.raises(ValueError):block([[A],[zeros(ring,1,2)]])
    with pytest.raises(ValueError):block([[A],[zeros(other,1,1)]])
    with pytest.raises(ValueError):hstack(A,zeros(other,1,1))
    with pytest.raises(ValueError):vstack(A,zeros(other,1,1))
