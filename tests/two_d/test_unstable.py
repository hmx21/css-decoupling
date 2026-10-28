"""Independent matrix products for original-register Suslin realization."""
import random
from unittest.mock import patch

import pytest

from css_decoupling.two_d.algebra import Ring,Matrix,eye,direct_sum
from css_decoupling.two_d.unstable import (
    factor_unstabilized,normality_factors,UnstableFactorizationIncomplete,
    _Budget,_patch_data,_patch_word,_replay,
    _monic_geometry,
)
from css_decoupling.two_d.unstable_local import local_sl3_word
from css_decoupling.algebra.local_completion import RationalContext
from css_decoupling.algebra.closed_points import closed_point
from css_decoupling.algebra.closed_points import apply_shear
from css_decoupling.errors import ResourceLimit


def e(r,n,i,j,c):
    E,Ei=eye(r,n),eye(r,n)
    E[i,j],Ei[i,j]=c,-r(c)
    return E,Ei


def cohn(r):
    x,y=r.variables;a=x+1;b=y+1
    return (Matrix.from_rows(r,[[1+a*b,a*a],[-b*b,1-a*b]]),
            Matrix.from_rows(r,[[1-a*b,-a*a],[b*b,1+a*b]]))


def check(U,V,data):
    r,n=U.ring,U.nrows
    assert data['added_ancillas']==0
    assert data['ancilla_action']==data['ancilla_inverse']==eye(r,0)
    P,Pi=eye(r,n),eye(r,n)
    for f in data['factors']:
        i,j=f['row'],f['column']
        assert 0<=i<n and 0<=j<n and i!=j
        E,Ei=e(r,n,i,j,f['coefficient'][0,0])
        P,Pi=E*P,Pi*Ei
    units=[r.monomial(a,c) for a,c in zip(data['shift_exponents'],data['transport_coefficients'])]
    assert len(units)==n
    S=Matrix(r,n,n,{(i,i):u for i,u in enumerate(units)})
    Si=Matrix(r,n,n,{(i,i):u**-1 for i,u in enumerate(units)})
    assert S*P==U
    assert Pi*Si==V
    assert P*Pi==Pi*P==eye(r,n)
    assert data['input_inverse']==V
    assert data['statistics']['physical_matrix_shape']==[n,n]
    assert data['statistics']['stable_factorization_used'] is False


@pytest.mark.parametrize('p',[2,3,5])
@pytest.mark.parametrize('n',[3,4])
def test_normality_conjugation_has_actual_fixed_rank_factors(p,n):
    r=Ring(p,2);x,y=r.variables
    B,Bi=eye(r,n),eye(r,n)
    for i,j,c in [(0,1,x+y**-1),(1,2,1+x),(2,0,2+y)]:
        E,Ei=e(r,n,i,j,c);B,Bi=E*B,Bi*Ei
    E,Ei=e(r,n,1,0,2*x**-1+y)
    factors=normality_factors(B,Bi,1,0,E[1,0])
    P,Pi=eye(r,n),eye(r,n)
    for f in factors:
        A,Ai=e(r,n,f['row'],f['column'],f['coefficient'][0,0])
        P,Pi=A*P,Pi*Ai
    assert P==B*E*Bi and Pi==B*Ei*Bi


@pytest.mark.parametrize('p,alpha,beta',[(2,1,1),(3,1,0),(5,2,0)])
def test_murthy_local_word_at_degree_two_closed_point_and_exact_dilation(p,alpha,beta):
    r=Ring(p,2);x,y=r.variables
    u=alpha*x+beta;a=y+u;b=x
    M=Matrix.from_rows(r,[[a*a,-1-a*b],[1-a*b,b*b]])
    point=closed_point(r,[1+u*x],axes=(0,))
    assert point.field.absolute_degree==2
    budget=_Budget(timeout=20)
    ctx=RationalContext(r,budget)
    word=local_sl3_word([[ctx.convert(M[i,j]) for j in range(2)] for i in range(2)],1,point,context=ctx)
    radius,data=_patch_data(word,1,ctx,point)
    # Nonconstant c and g exercise the proof that the radius is independent
    # of the later Bezout-cover coefficients, not just a numeric specialization.
    c=x+1;h=radius*(x*x+2)
    patchword=_patch_word(data,c,h,1,ctx,budget)
    def at(t):
        return Matrix(r,2,2,{ij:ctx.polynomial(ctx.substitute(ctx.convert(f),1,ctx.convert(t)*ctx.variables[1]))
                             for ij,f in M.entries.items()})
    old,new=at(c),at(c+h)
    oldi=Matrix.from_rows(r,[[old[1,1],-old[0,1]],[-old[1,0],old[0,0]]])
    assert _replay(patchword,r,3,budget)==direct_sum(oldi*new,eye(r,1))


@pytest.mark.parametrize('p',[2,3,5])
def test_forced_general_cohn_uses_existing_third_register(p):
    r=Ring(p,2);C,Ci=cohn(r)
    U,V=direct_sum(C,eye(r,1)),direct_sum(Ci,eye(r,1))
    with patch('css_decoupling.two_d.suslin.factor_suslin',side_effect=AssertionError('stable forbidden')):
        result=factor_unstabilized(U,V,method='suslin',timeout=20)
    check(U,V,result)
    assert result['complete_general_suslin_algorithm']
    if p==2:
        assert result['statistics']['local_patches']
    else:
        assert result['statistics']['structural_cancellations']
    assert all(f['row']<3 and f['column']<3 for f in result['factors'])


@pytest.mark.parametrize('p',[2,3,5])
def test_laurent_transport_and_scalar_units_preserve_original_q(p):
    r=Ring(p,2);x,y=r.variables;C,Ci=cohn(r)
    U,V=direct_sum(C,eye(r,2)),direct_sum(Ci,eye(r,2))
    D,Di=eye(r,4),eye(r,4)
    D[0,0]=r(p-1)*x**-2*y
    Di[0,0]=D[0,0]**-1
    U,V=D*U,V*Di
    result=factor_unstabilized(U,V,method='suslin',timeout=20)
    check(U,V,result)
    assert result['determinant'][0,0]==D[0,0]
    assert result['transport_coefficients'][-1]==p-1


@pytest.mark.parametrize('p',[2,3,5])
def test_zero_one_two_registers_and_auto_fast_are_truthful(p):
    r=Ring(p,2);x,y=r.variables
    for n in [0,1,2,3]:
        U,V=eye(r,n),eye(r,n)
        if n:
            U[0,0]=r(p-1)*x*y**-2;V[0,0]=U[0,0]**-1
        if n>1:
            E,Ei=e(r,n,0,1,1+y+x**-1);U,V=E*U,V*Ei
        result=factor_unstabilized(U,V,method='auto')
        check(U,V,result)
    C,Ci=cohn(r)
    if p==2:
        with pytest.raises(UnstableFactorizationIncomplete):
            factor_unstabilized(C,Ci,method='fast',max_factors=10000)
    else:
        result=factor_unstabilized(C,Ci,method='fast',max_factors=10000)
        check(C,Ci,result)
        assert not result['complete_general_suslin_algorithm']


def test_bad_inputs_resource_caps_and_altered_factors_are_not_successes():
    r=Ring(3,2);C,Ci=cohn(r);U,V=direct_sum(C,eye(r,1)),direct_sum(Ci,eye(r,1))
    for kwargs in [{'max_steps':1},{'max_terms':1},{'max_products':1},{'max_factors':1}]:
        with pytest.raises(ResourceLimit):factor_unstabilized(U,V,method='suslin',**kwargs)
    for kwargs in [{'max_steps':True},{'max_patches':0},{'timeout':float('nan')},{'method':'stable'}]:
        with pytest.raises(ValueError):factor_unstabilized(U,V,**kwargs)
    with pytest.raises(ValueError):factor_unstabilized(U,eye(r,3))
    result=factor_unstabilized(U,V,method='suslin')
    result['factors'][0]['coefficient'][0,0]+=1
    with pytest.raises(AssertionError):check(U,V,result)


def test_auto_incomplete_fixed_rank_probe_reaches_general_without_stabilization():
    r=Ring(2,2);C,Ci=cohn(r)
    U,V=direct_sum(C,eye(r,1)),direct_sum(Ci,eye(r,1))
    with patch('css_decoupling.two_d.suslin.factor_suslin',side_effect=AssertionError('stable forbidden')):
        result=factor_unstabilized(U,V,method='auto')
    check(U,V,result)
    assert result['complete_general_suslin_algorithm']
    assert result['statistics']['fast_probe_operation_charge']>0
    assert result['statistics']['fast_probe_outcome']=='incomplete'
    assert result['statistics']['local_patches']


@pytest.mark.parametrize('p',[2,3,5,7])
@pytest.mark.parametrize('method',['auto','suslin'])
def test_inverse_geometry_retains_original_determinant_and_exact_action(p,method):
    r=Ring(p,2);x,y=r.variables
    lower,upper=eye(r,3),eye(r,3)
    lower[1,0],lower[2,0]=x,y
    upper[0,1],upper[0,2]=x+1,y+1
    loweri,upperi=eye(r,3),eye(r,3)
    loweri[1,0],loweri[2,0]=-x,-y
    upperi[0,1],upperi[0,2]=-x-1,-y-1
    D,Di=eye(r,3),eye(r,3)
    D[0,0]=r(p-1)*x**-2*y;Di[0,0]=D[0,0]**-1
    U,V=D*lower*upper,upperi*loweri*Di
    if method=='auto':
        from css_decoupling.two_d.elementary import FactorizationIncomplete
        with patch('css_decoupling.two_d.elementary.factor_unit_reduction',
                   side_effect=FactorizationIncomplete('controlled stopped probe')):
            result=factor_unstabilized(U,V,method=method)
    else:
        result=factor_unstabilized(U,V,method=method)
    assert result['statistics']['factor_direction']=='inverse'
    assert result['determinant'][0,0]==D[0,0]
    check(U,V,result)


def test_auto_completed_fixed_rank_probe_skips_general_and_charges_actual_steps():
    r=Ring(5,2);x,y=r.variables
    U,V=e(r,3,0,1,2*x**-1+y)
    with patch('css_decoupling.two_d.unstable._general_sl_word',side_effect=AssertionError('general should be skipped')):
        result=factor_unstabilized(U,V,method='auto')
    check(U,V,result)
    assert result['statistics']['fast_probe_outcome']=='complete'
    assert result['statistics']['fast_probe_operation_charge']==0
    assert result['statistics']['operations']>0


def test_stopped_auto_probe_consumes_shared_steps_before_general_construction():
    r=Ring(2,2);C,Ci=cohn(r)
    U,V=direct_sum(C,eye(r,1)),direct_sum(Ci,eye(r,1))
    direct=factor_unstabilized(U,V,method='suslin')
    needed=direct['statistics']['operations']
    exact=factor_unstabilized(U,V,method='suslin',max_steps=needed)
    check(U,V,exact)
    # The same cap no longer fits once auto has consumed its stopped probe
    # quota. A reset budget at the general entry would incorrectly succeed.
    with pytest.raises(ResourceLimit):
        factor_unstabilized(U,V,method='auto',max_steps=needed)


@pytest.mark.parametrize('p',[2,3,5])
@pytest.mark.parametrize('axis',[0,1])
def test_monic_geometry_uses_strict_support_bound_and_preserves_existing_monic(p,axis):
    r=Ring(p,2);v=r.variable(1-axis);t=r.variable(axis)
    f=v**4*t+v*t**3+1
    degree,weights,scalar=_monic_geometry(f,axis)
    assert degree==5 and weights=={1-axis:1}
    changed=apply_shear(f,axis,weights)
    top=[(e,c) for e,c in changed.terms.items() if e[axis]==degree]
    assert len(top)==1 and top[0][0][1-axis]==0 and top[0][1]==scalar
    degree,weights,scalar=_monic_geometry(t**3+v+1,axis)
    assert (degree,weights,scalar)==(3,{},1)
