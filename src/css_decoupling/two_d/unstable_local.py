"""Murthy's fixed-three-coordinate local factorization (Park--Woodburn §5).

Internal coefficients are in F_p(x)[y], with every denominator checked to be
a unit at the supplied closed point. Returned triples form a WRITTEN product.
No matrix dimension is enlarged, and no rational answer is exported as Laurent.
"""
from ..algebra.local_completion import _identity, _product, _equal


def _unit_pair(i, j, u, one):
    return [(i,j,u),(j,i,-(u**-1)),(i,j,u),(i,j,-one),(j,i,one),(i,j,-one)]


def _joined(ctx, *words):
    count=sum(len(word) for word in words)
    if count>ctx.budget.max_factors:
        ctx.budget.fail('local elementary factor cap reached')
    result=[item for word in words for item in word if item[2]]
    ctx.budget.word(result)
    return result


def _at_zero(ctx, value, axis):
    return ctx.coefficients(value,axis).get(0,ctx.field.zero)


def _local_unit(ctx, point, value):
    num=ctx.ring({e:int(c) for e,c in value.numer.items()})
    den=ctx.ring({e:int(c) for e,c in value.denom.items()})
    if point.evaluate(den)==point.field.zero:
        raise ArithmeticError('Local factorization denominator vanishes at its closed point')
    return point.evaluate(num)!=point.field.zero


def _bezout(ctx,p,q,axis):
    """Degree-bounded Bezout pair in the coefficient fraction field."""
    F=ctx.field
    old,r=p,q
    a,b,c,d=F.one,F.zero,F.zero,F.one
    while r:
        quotient,remainder=ctx.remainder(old,r,axis)
        old,r=r,remainder
        a,b=b,a-quotient*b
        c,d=d,c-quotient*d
        ctx.budget.check(old,r,a,b,c,d)
    if not old or ctx.degree(old,axis)!=0:
        raise ValueError('The local polynomial pair is not unimodular')
    a,c=a/old,c/old
    if a*p+c*q-F.one:
        raise ArithmeticError('Local Bezout identity failed')
    return a,c


def local_sl3_word(block,axis,point,*,context):
    """Factor diag(block,1), where block is SL2 and block[0][0] is monic."""
    ctx=context
    F=ctx.field
    one=F.one

    def factor(p,q,r,s,depth=0):
        ctx.budget.check(p,q,r,s)
        if depth>256:
            ctx.budget.fail('local degree-recursion cap reached')
        if p*s-q*r-one:
            raise ArithmeticError('Murthy recursion lost determinant one')
        for value in (p,q,r,s):
            ctx.coefficients(value,axis)
        # A local constant pivot has a direct rank-two elementary word.
        if p and ctx.degree(p,axis)==0 and _local_unit(ctx,point,p):
            return _joined(ctx,[(1,0,r/p)],_unit_pair(0,1,p,one),[(0,1,q/p)])
        if q and ctx.degree(q,axis)==0 and _local_unit(ctx,point,q):
            return _joined(ctx,factor(q,-p,s,-r,depth+1),
                           [(0,1,one),(1,0,-one),(0,1,one)])
        if not p or ctx.lead(p,axis)-one:
            raise ValueError('Murthy local factorization requires a monic first entry')
        degree=ctx.degree(p,axis)
        quotient,remainder=ctx.remainder(q,p,axis)
        if quotient:
            # A E12(-quotient)=A_reduced.
            return _joined(ctx,factor(p,remainder,r,s-quotient*r,depth+1),[(0,1,quotient)])
        if ctx.degree(q,axis)>=degree:
            raise ArithmeticError('Local polynomial remainder did not lower degree')
        q0=_at_zero(ctx,q,axis)
        if q0 and _local_unit(ctx,point,q0):
            shift=_at_zero(ctx,p,axis)/q0
            pp,rr=p-shift*q,r-shift*s
            if _at_zero(ctx,pp,axis):
                raise ArithmeticError('Murthy constant-term elimination failed')
            a=ctx.variables[axis]
            ap=pp/a
            if ctx.degree(ap,axis)!=degree-1 or ctx.lead(ap,axis)-one:
                raise ArithmeticError('Murthy monic degree did not decrease')
            c,d,b=rr,s,q
            c1=c2=c
            d1,d2=ap*d,a*d
            # Park--Woodburn (52), an equality rather than a K1 congruence.
            correction=c*d1*d2-d*(c2+ap*c1*d2)
            return _joined(ctx,
                [(1,0,correction),(1,2,d2-one),(2,1,one),(1,2,-one)],
                factor(a,b,c1,d1,depth+1),
                [(1,2,one),(2,1,-one),(1,2,one)],
                factor(ap,b,c2,d2,depth+1),
                [(1,2,-one),(2,1,one),(1,2,a-one),(2,0,-ap*c1),(2,1,-d1)],
                [(1,0,shift)])
        f,g=_bezout(ctx,p,q,axis)
        qp=-g
        # The unique bounded-degree Bezout pair has only local-unit
        # denominators, because the monic pair has unit resultant.
        for value in (f,qp):
            for coefficient in ctx.coefficients(value,axis).values():
                _local_unit(ctx,point,coefficient)
        if ctx.degree(f,axis)>=ctx.degree(q,axis) or ctx.degree(qp,axis)>=degree:
            raise ArithmeticError('Local Bezout pair lost its strict degree bounds')
        if not _local_unit(ctx,point,q0+_at_zero(ctx,f,axis)):
            raise ArithmeticError('Murthy second case did not produce a unit constant')
        correction=r*f-s*qp
        return _joined(ctx,[(1,0,correction),(0,1,-one)],
                       factor(p+qp,q+f,qp,f,depth+1))

    p,q=block[0]
    r,s=block[1]
    result=factor(p,q,r,s)
    product=_identity(ctx,3)
    for i,j,c in result:
        # Written right multiplication: column j += c*column i.
        for k in range(3):
            product[k][j]+=product[k][i]*c
            ctx.budget.check(product[k][j])
        for coefficient in ctx.coefficients(c,axis).values():
            _local_unit(ctx,point,coefficient)
    expected=[[p,q,F.zero],[r,s,F.zero],[F.zero,F.zero,F.one]]
    if not _equal(product,expected):
        raise ArithmeticError('Murthy local factor word failed exact three-coordinate replay')
    return result
