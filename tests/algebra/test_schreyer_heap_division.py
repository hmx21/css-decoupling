"""Compare heap division with an independent copy of the old max loop."""
from random import Random

import pytest

from css_decoupling.algebra.schreyer import _Budget, _Order, _divide
from css_decoupling.errors import ResourceLimit


def reference_divide(vector, basis, order, prime, *, track):
    """Original ordered reduction with independently written term updates."""
    work,remainder,quotient=dict(vector),{},{}
    leading=[max(g,key=order) for g in basis]
    while work:
        head=max(work,key=order)
        for i,divisor in enumerate(leading):
            if divisor[0]!=head[0] or any(a>b for a,b in zip(divisor[1:],head[1:])):
                continue
            shift=tuple(a-b for a,b in zip(head[1:],divisor[1:]))
            factor=work[head]*pow(basis[i][divisor],-1,prime)%prime
            for monomial,coefficient in basis[i].items():
                key=(monomial[0],)+tuple(a+b for a,b in zip(monomial[1:],shift))
                value=(work.get(key,0)-factor*coefficient)%prime
                if value:work[key]=value
                else:work.pop(key,None)
            if track:
                key=(i,)+shift;value=(quotient.get(key,0)+factor)%prime
                if value:quotient[key]=value
                else:quotient.pop(key,None)
            break
        else:
            remainder[head]=work.pop(head)
    return remainder,quotient


def order_for(kind):
    base=_Order()
    if kind==0:return base
    induced=_Order(base,[(0,2,0),(0,0,2),(1,1,1)])
    if kind==1:return induced
    return _Order(induced,[(0,1,0),(1,0,2),(2,1,1)])


def budget(**changes):
    options=dict(timeout=None,max_basis=10000,max_terms=100000,max_pairs=100000)
    options.update(changes)
    return _Budget(**options)


def evaluate_quotient(basis,quotient,prime):
    result={}
    for (index,*exponent),coefficient in quotient.items():
        for (component,*power),value in basis[index].items():
            key=(component,)+tuple(a+b for a,b in zip(exponent,power))
            residue=(result.get(key,0)+coefficient*value)%prime
            if residue:result[key]=residue
            else:result.pop(key,None)
    return result


@pytest.mark.parametrize("prime", [2,3,5])
@pytest.mark.parametrize("kind", [0,1,2])
@pytest.mark.parametrize("track", [False,True])
def test_random_remainders_and_quotients_match_old_loop_in_exact_order(prime,kind,track):
    rng=Random(5133+prime+31*kind)
    order=order_for(kind)
    for _ in range(12):
        basis=[]
        for __ in range(4):
            polynomial={(rng.randrange(3),rng.randrange(3),rng.randrange(3)):rng.randrange(1,prime)
                        for ___ in range(5)}
            basis.append(polynomial)
        vector={(rng.randrange(3),rng.randrange(6),rng.randrange(6)):rng.randrange(1,prime)
                for __ in range(17)}
        original=vector.copy();original_basis=[g.copy() for g in basis]
        expected=reference_divide(vector,basis,order,prime,track=track)
        actual=_divide(vector,basis,order,prime,budget(),track=track)
        assert tuple(actual[0].items())==tuple(expected[0].items())
        assert tuple(actual[1].items())==tuple(expected[1].items())
        assert vector==original and basis==original_basis
        if track:
            reconstructed=evaluate_quotient(basis,actual[1],prime)
            for key,value in actual[0].items():
                coefficient=(reconstructed.get(key,0)+value)%prime
                if coefficient:reconstructed[key]=coefficient
                else:reconstructed.pop(key,None)
            assert reconstructed==vector


def test_cancelled_heap_entry_can_be_revived_before_it_is_popped():
    # y^2 is initially present, deleted by reducing x^2*y, then recreated
    # when x*y is reduced. A stale heap entry must become live exactly once.
    vector={(0,3,0):1,(0,2,1):1,(0,0,2):1}
    basis=[{(0,2,0):1,(0,0,1):1},{(0,1,1):1,(0,0,2):1}]
    remainder,quotient=_divide(vector,basis,_Order(),2,budget(),track=True)
    assert remainder=={(0,0,2):1}
    assert (remainder,quotient)==reference_divide(vector,basis,_Order(),2,track=True)


def test_large_cancelled_tail_is_compacted_without_losing_the_last_term():
    vector={(0,1,i):1 for i in range(200)}
    vector.update({(0,0,i):1 for i in range(200)})
    vector[(1,0,0)]=1
    basis=[{(0,1,0):1,(0,0,0):1}]
    result=_divide(vector,basis,_Order(),2,budget(),track=True)
    assert result==reference_divide(vector,basis,_Order(),2,track=True)
    assert result[0]=={(1,0,0):1}


def test_empty_vector_and_empty_basis_have_the_same_exact_results():
    order=order_for(2)
    assert _divide({},[],order,3,budget(),track=True)==({},{})
    vector={(2,0,1):2,(0,3,1):1,(1,1,0):2}
    result=_divide(vector,[],order,3,budget(),track=True)
    assert tuple(result[0].items())==tuple(reference_divide(vector,[],order,3,track=True)[0].items())
    assert not result[1]


def test_division_resource_failures_remain_resource_failures():
    vector={(0,i,0):1 for i in range(4)}
    with pytest.raises(ResourceLimit):
        _divide(vector,[],_Order(),3,budget(max_terms=1),track=True)
    with pytest.raises(ResourceLimit):
        _divide(vector,[],_Order(),3,budget(timeout=1e-12),track=True)
