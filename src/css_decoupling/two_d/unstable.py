"""Fixed-register elementary factorization over prime two-variable Laurent rings.

The general rank >= 3 route uses elementary column reduction, Murthy's local
SL3 degree descent, and denominator-clearing local/global patching. The third
coordinate used by an SL2 residual is an EXISTING register. No stabilization
or doubling of the physical register occurs. Internal words are written
products; public elementary records are in chronological execution order.
"""
from __future__ import annotations

from time import perf_counter

from .algebra import Matrix, Poly, eye
from ..errors import ResourceLimit
from ..algebra.local_completion import RationalContext, _identity, _equal
from ..algebra.closed_points import closed_point, apply_shear
from ..algebra.polynomial import solve_polynomial, NoPolynomialSolution
from .suslin import _pid_factor, _transport_word, _mm, _row_add, _col_add
from .unstable_local import local_sl3_word


class UnstableFactorizationIncomplete(RuntimeError):
    """No fixed-register word was obtained; this is not nonexistence evidence."""


class _Budget:
    def __init__(self,timeout=120,max_steps=1000000,max_terms=1000000,
                 max_factors=200000,max_products=30000000):
        if type(timeout) not in (int,float) or not 0<timeout<float('inf'):
            raise ValueError('timeout must be finite and positive')
        for name,value in [('max_steps',max_steps),('max_terms',max_terms),
                           ('max_factors',max_factors),('max_products',max_products)]:
            if type(value) is not int or value<1:
                raise ValueError(name+' must be a positive integer')
            setattr(self,name,value)
        self.deadline=perf_counter()+timeout
        self.steps=self.products=0
        self.max_exponent=1000000
        self.kernel_splits=[]
        self.local_patches=[]
        self.phase='validation'
        self.column_choices=[]
        self.structural_cancellations=0
        self.factor_direction='forward'
        self.fast_probe_operation_charge=0
        self.fast_probe_outcome='not-requested'

    def fail(self,message):
        raise ResourceLimit('Fixed-register factorization ['+self.phase+']: '+message)

    def remaining(self):
        self.check()
        return max(1e-12,self.deadline-perf_counter())

    def check(self,*values,terms=0):
        self.steps+=1
        if self.steps>self.max_steps or perf_counter()>self.deadline:
            self.fail('cooperative time/operation cap reached')
        for value in values:
            if isinstance(value,Matrix):
                terms+=sum(len(f.terms) for f in value.entries.values())
            elif isinstance(value,Poly):
                terms+=len(value.terms)
            elif hasattr(value,'numer'):
                terms+=len(value.numer)+len(value.denom)
        if terms>self.max_terms:
            self.fail('retained term cap reached')

    def pairs(self,count):
        self.products+=count
        if self.products>self.max_products:
            self.fail('convolution-product cap reached')
        self.check()

    def word(self,word):
        if len(word)>self.max_factors:
            self.fail('elementary factor cap reached')
        self.check(*(item[-1] for item in word))

    def multiply(self,a,b):
        self.pairs(len(a.terms)*len(b.terms))
        value=a*b
        self.check(value)
        return value

    def power(self,value,exponent):
        result=value.ring.one()
        while exponent:
            if exponent&1:result=self.multiply(result,value)
            exponent>>=1
            if exponent:value=self.multiply(value,value)
        return result

    def absorb(self,stats):
        self.steps+=stats.get('operations',0)
        self.products+=stats.get('products',stats.get('convolution_pairs',0))
        self.pairs(0)


def _join(budget,*words):
    if sum(len(w) for w in words)>budget.max_factors:
        budget.fail('elementary factor cap reached before concatenation')
    answer=[item for word in words for item in word if item[2]]
    budget.word(answer)
    return answer


def _inverse_word(word):
    return [(i,j,-c) for i,j,c in reversed(word)]


def _records(word,ring):
    return [dict(row=i,column=j,coefficient=Matrix.from_rows(ring,[[c]]))
            for i,j,c in reversed(word) if c]


def _from_records(records):
    return [(f['row'],f['column'],f['coefficient'][0,0]) for f in reversed(records)]


def _replay(word,ring,n,budget):
    P=eye(ring,n)
    for i,j,c in word:
        if i==j or not 0<=i<n or not 0<=j<n:
            raise ArithmeticError('A factor leaves the original physical register')
        _col_add(P,j,i,c,budget)
    return P


def _cohn_word(v,a,i,j,budget):
    """Park--Woodburn (7), valid over any commutative coefficient ring."""
    n=len(v)
    if n<3 or i==j:
        raise ValueError('Cohn factorization requires three existing coordinates')
    t=next(k for k in range(n) if k not in (i,j))
    vi,vj=v[i],v[j]
    avi,avj=budget.multiply(a,vi),budget.multiply(a,vj)
    result=[(i,t,-vi),(j,t,-vj),(t,i,-avj),(t,j,avi),
            (i,t,vi),(j,t,vj),(t,i,avj),(t,j,-avi)]
    for k in range(n):
        if k not in (i,j):
            result.extend(((k,i,budget.multiply(v[k],avj)),(k,j,-budget.multiply(v[k],avi))))
    return _join(budget,result)


def normality_factors(B,Binv,i,j,coefficient,*,budget=None):
    """Chronological fixed-rank factors of B E_ij(coefficient) B^-1."""
    budget=budget or _Budget()
    ring,n=B.ring,B.nrows
    if B.shape!=(n,n) or Binv.shape!=B.shape or Binv.ring!=ring or n<3 or i==j:
        raise ValueError('Normality requires inverse square matrices of rank at least three')
    if _mm(B,Binv,budget)!=eye(ring,n) or _mm(Binv,B,budget)!=eye(ring,n):
        raise ValueError('Normality matrices are not actual inverses')
    v=[B[k,i] for k in range(n)]
    g=[Binv[i,k] for k in range(n)]
    w=[budget.multiply(ring(coefficient),Binv[j,k]) for k in range(n)]
    word=[]
    for a in range(n):
        for b in range(a+1,n):
            scalar=budget.multiply(w[a],g[b])-budget.multiply(w[b],g[a])
            if scalar:
                word=_join(budget,word,_cohn_word(v,scalar,a,b,budget))
    E=eye(ring,n);E[i,j]=coefficient
    if _replay(word,ring,n,budget)!=_mm(_mm(B,E,budget),Binv,budget):
        raise ArithmeticError('Normality word failed actual conjugation replay')
    return _records(word,ring)


def _rational_right(ctx,A,i,j,c):
    for k in range(len(A)):
        A[k][j]+=A[k][i]*c
        ctx.budget.check(A[k][j])


def _rational_left(ctx,A,i,j,c):
    for k in range(len(A)):
        A[i][k]+=c*A[j][k]
        ctx.budget.check(A[i][k])


def _lcm_denominator(ctx,denominators):
    result=ctx.field.ring.one
    for value in denominators:
        result=result.lcm(value)
        ctx.budget.check(ctx.field(result))
    return result


def _patch_data(word,axis,ctx,point):
    """Finite denominator radius from actual suffixes, without trial dilation.

    For A=E1...Em, A(cX)^-1 A((c+h)X) is the product of conjugates
    Cj E_j(delta_j) Cj^-1, where Cj=(E_(j+1)...Em)^-1 at cX.
    Scale each Cj column by its common denominator d. The corresponding
    Cohn coefficient is delta*a/d^2. Its displayed denominator, including
    that of E_j's coefficient, divides the returned radius. Thus h=radius*g
    clears every coefficient, since each numerator difference is divisible h.
    """
    F=ctx.field
    C,Ci=_identity(ctx,3),_identity(ctx,3)
    radius=F.ring.one
    data=[]
    for i,j,f in reversed(word):
        if ctx.degree(f,axis)>0:
            v=[C[k][i] for k in range(3)]
            denominator=_lcm_denominator(ctx,[z.denom for z in v])
            scale=F(denominator)
            scaled=[ctx.polynomial(z*scale) for z in v]
            pairs=[]
            for a in range(3):
                for b in range(a+1,3):
                    coefficient=(Ci[j][a]*Ci[i][b]-Ci[j][b]*Ci[i][a])/(scale*scale)
                    if coefficient:
                        required=(coefficient/F(f.denom)).denom
                        radius=radius.lcm(required)
                        pairs.append((a,b,coefficient))
                        ctx.budget.check(coefficient,F(radius))
            data.append((f,scaled,pairs))
        _rational_right(ctx,C,i,j,-f)
        _rational_left(ctx,Ci,i,j,f)
    radius_poly=ctx.ring({e:int(c) for e,c in radius.items()})
    if any(e[axis] for e in radius_poly.terms):
        raise ArithmeticError('A local patch radius depends on the main variable')
    if point.evaluate(radius_poly)==point.field.zero:
        raise ArithmeticError('A patch radius vanishes at its construction point')
    return radius_poly,list(reversed(data))


def _patch_word(data,c,h,axis,ctx,budget):
    y=ctx.variables[axis]
    cy=ctx.convert(c)*y
    shifted=ctx.convert(c+h)*y
    word=[]
    for f,v,pairs in data:
        delta=ctx.substitute(f,axis,shifted)-ctx.substitute(f,axis,cy)
        if not delta:
            continue
        vector=[ctx.polynomial(ctx.substitute(ctx.convert(z),axis,cy)) for z in v]
        for i,j,a in pairs:
            scalar=ctx.polynomial(delta*ctx.substitute(a,axis,cy))
            word=_join(budget,word,_cohn_word(vector,scalar,i,j,budget))
    return word


def _univariate_word(A,axis,budget):
    """Fixed-size PID base case; the low-level routine never stabilizes."""
    atoms,_=_pid_factor(A,axis,budget)
    records,shifts,coefficients,determinant,extra=_transport_word(atoms,A.ring,A.nrows,'separate',budget)
    if determinant!=1 or extra or any(any(e) for e in shifts) or any(c!=1 for c in coefficients):
        raise ArithmeticError('SL matrix acquired determinant transport in the PID base case')
    return _from_records(records)


def _monic_geometry(f,axis):
    """One strict support-vertex bound, with no trial shear or field search."""
    degree=max(e[axis] for e in f.terms)
    top=[(e,c) for e,c in f.terms.items() if e[axis]==degree]
    if len(top)==1 and top[0][0][1-axis]==0:
        return degree,{},top[0][1]
    support=[(e[1-axis],e[axis],c) for e,c in f.terms.items()]
    A,B,coefficient=max(support,key=lambda t:(t[0],t[1]))
    # w(A-a)>b-B for every other coefficient degree a<A. For a=A,
    # choosing maximal B already makes the desired point strictly leading.
    weight=max([1]+[(b-B)//(A-a)+1 for a,b,_ in support if a<A])
    return weight*A+B,{1-axis:weight},coefficient


def _polynomial_sl2_word(A,budget,*,max_patches,max_field_trials,seed):
    """Written factors of diag(A,1), for an ordinary-polynomial SL2 matrix."""
    ring=A.ring
    if any(v<0 for f in A.entries.values() for e in f.terms for v in e):
        raise ValueError('Polynomial realization received negative exponents')
    if A.nrows!=2 or A.ncols!=2 or A[0,0]*A[1,1]-A[0,1]*A[1,0]!=1:
        raise ValueError('Expected a determinant-one two-by-two polynomial block')
    active={k for f in A.entries.values() for e in f.terms for k,a in enumerate(e) if a}
    if len(active)<=1:
        return _univariate_word(A,next(iter(active),0),budget)
    # Choose once from actual support. A scalar-leading entry uses no shear;
    # every other entry gets one proven strict-support bound, not a trial run.
    candidates=[]
    for (i,j),f in A.entries.items():
        for axis in range(2):
            degree,weights,_=_monic_geometry(f,axis)
            candidates.append((degree,len(f.terms),len(weights),axis,i,j))
    _,_,_,axis,row,col=min(candidates)
    M=A.copy();left=[];right=[]
    one=ring.one()
    signed=[(0,1,one),(1,0,-one),(0,1,one)]
    if row:
        for i,j,c in reversed(signed):
            _row_add(M,i,j,c,budget)
        left=signed
    if col:
        for i,j,c in signed:
            _col_add(M,j,i,c,budget)
        right=signed
    predicted,weights,scalar=_monic_geometry(M[0,0],axis)
    M=apply_shear(M,axis,weights,max_terms=budget.max_terms)
    if max(e[axis] for e in M[0,0].terms)!=predicted:
        raise ArithmeticError('Monic support bound did not predict the actual degree')
    # Constant diagonal normalization is itself elementary inside this pair.
    from .unstable_local import _unit_pair
    scale=ring(pow(scalar,-1,ring.p))
    norm=_unit_pair(0,1,scale,one)
    for i,j,c in reversed(norm):
        _row_add(M,i,j,c,budget)
    ctx=RationalContext(ring,budget)
    localblock=[[ctx.convert(M[i,j]) for j in range(2)] for i in range(2)]
    radii,patches=[],[]
    for attempt in range(max_patches):
        budget.phase='local-SL3-factorization'
        point=closed_point(ring,radii,axes=(1-axis,),seed=seed+attempt,
                           max_trials=max_field_trials,timeout=budget.remaining(),max_terms=budget.max_terms)
        local=local_sl3_word(localblock,axis,point,context=ctx)
        radius,data=_patch_data(local,axis,ctx,point)
        budget.local_patches.append({'radius_terms':len(radius.terms),
                                    'residue_degree':point.field.absolute_degree,
                                    'local_factors':len(local)})
        radii.append(radius);patches.append(data)
        try:
            cover=solve_polynomial(Matrix.from_rows(ring,[radii]),eye(ring,1),max_terms=budget.max_terms)
        except NoPolynomialSolution:
            continue
        break
    else:
        budget.fail('local patch count reached max_patches')
    zero=Matrix(ring,2,2,{ij:ctx.polynomial(ctx.substitute(ctx.convert(f),axis,ctx.field.zero))
                        for ij,f in M.entries.items()})
    word=_univariate_word(zero,1-axis,budget)
    c=ring.zero()
    for k,data in enumerate(patches):
        budget.phase='local-global-patching'
        h=radii[k]*cover[k,0]
        word=_join(budget,word,_patch_word(data,c,h,axis,ctx,budget))
        c+=h
    if c!=1:
        raise ArithmeticError('Local patch radii do not generate the unit ideal')
    # Undo the scalar normalization and the monic polynomial coordinate shear.
    word=_join(budget,_inverse_word(norm),word)
    word=[(i,j,apply_shear(f,axis,weights,sign=-1,max_terms=budget.max_terms)) for i,j,f in word]
    word=_join(budget,_inverse_word(left),word,_inverse_word(right))
    expected=eye(ring,3)
    for i in range(2):
        for j in range(2):expected[i,j]=A[i,j]
    if _replay(word,ring,3,budget)!=expected:
        raise ArithmeticError('Polynomial local/global word failed fixed-rank replay')
    return word


def _coordinate(value,B):
    """An integer unimodular monomial substitution; physical rank is unchanged."""
    ring=value.ring
    def poly(f):
        return ring({(B[0][0]*a+B[0][1]*b,B[1][0]*a+B[1][1]*b):c
                     for (a,b),c in f.terms.items()})
    if isinstance(value,Poly):return poly(value)
    return Matrix(ring,value.nrows,value.ncols,{ij:poly(f) for ij,f in value.entries.items()})


def _inverse_coordinates(B):
    a,b=B[0];c,d=B[1]
    if a*d-b*c!=1:raise ArithmeticError('Coordinate map is not unimodular')
    return ((d,-b),(-c,a))


def _unit_sl2_word(A,budget):
    """Exact rank-two Gaussian base case whenever any Laurent entry is a unit."""
    positions=[ij for ij,f in A.entries.items() if f.is_unit]
    if not positions:return None
    ring=A.ring;one=ring.one();i,j=min(positions)
    signed=[(0,1,one),(1,0,-one),(0,1,one)]
    M=A.copy();left=[];right=[]
    if i:
        for a,b,c in reversed(signed):_row_add(M,a,b,c,budget)
        left=signed
    if j:
        for a,b,c in signed:_col_add(M,b,a,c,budget)
        right=signed
    from .unstable_local import _unit_pair
    p,q,r=M[0,0],M[0,1],M[1,0]
    pi=p**-1
    return _join(budget,_inverse_word(left),[(1,0,r*pi)],
                 _unit_pair(0,1,p,one),[(0,1,q*pi)],_inverse_word(right))


def _laurent_sl2_word(A,budget,*,max_patches,max_field_trials,seed):
    """Park thesis §11.3: a Laurent SL2 block using an existing third slot."""
    options=dict(max_patches=max_patches,max_field_trials=max_field_trials,seed=seed)
    unit_word=_unit_sl2_word(A,budget)
    if unit_word is not None:return unit_word
    if all(v>=0 for f in A.entries.values() for e in f.terms for v in e):
        return _polynomial_sl2_word(A,budget,**options)
    # A finite system of valuation inequalities can clear many Laurent blocks
    # by row/column monomial gauges with total determinant one. This is an
    # exact optional simplification; infeasibility proceeds to Park's general
    # column normalization, never to a degree/support search.
    column_powers=[[],[]];row_second=[]
    possible=True
    for axis in range(2):
        low={ij:min(e[axis] for e in f.terms) for ij,f in A.entries.items()}
        choices=[]
        for lower_key,upper_key in (((0,0),(1,1)),((1,0),(0,1))):
            lower=-low[lower_key] if lower_key in low else None
            upper=low.get(upper_key)
            if lower is not None and upper is not None and lower>upper:
                possible=False;break
            choices.append(lower if lower is not None else min(0,upper) if upper is not None else 0)
        if not possible:break
        t,v=choices
        row_second.append(v-t)
        column_powers[0].append(t);column_powers[1].append(-v)
    if possible:
        ring=A.ring
        L=[ring.one(),ring.monomial(tuple(row_second))]
        R=[ring.monomial(tuple(e)) for e in column_powers]
        M=Matrix(ring,2,2,{(i,j):L[i]*f*R[j] for (i,j),f in A.entries.items()})
        if any(a<0 for f in M.entries.values() for e in f.terms for a in e):
            raise ArithmeticError('Monomial gauge violated its valuation inequalities')
        word=_polynomial_sl2_word(M,budget,**options)
        units=L+[ring.one()]
        word=[(i,j,units[i]**-1*c*units[j]) for i,j,c in word]
        from .unstable_local import _unit_pair
        return _join(budget,word,_unit_pair(0,1,(L[0]*R[0])**-1,ring.one()))
    from .laurent_to_polynomial import _laurent_column_to_polynomial
    budget.phase='Laurent-polynomialization'
    ring=A.ring
    bezout=Matrix.from_rows(ring,[[A[1,1],-A[0,1]]])
    prepared=_laurent_column_to_polynomial(A[:,0],bezout,budget)
    prefix=[(f['row'],f['column'],f['coefficient'][0,0]) for f in prepared['factors']]
    M=A.copy()
    for i,j,c in prefix:_row_add(M,i,j,c,budget)
    B=prepared['coordinate_map']
    Bi=prepared['coordinate_inverse']
    M=_coordinate(M,B)
    right=[]
    while M[0,1] and min(e[0] for e in M[0,1].terms)<=0:
        old=min(e[0] for e in M[0,1].terms)
        c=-ring({e:v for e,v in M[0,1].terms.items() if e[0]==old})
        _col_add(M,1,0,c,budget)
        right.append((0,1,_coordinate(c,Bi)))
        if M[0,1] and min(e[0] for e in M[0,1].terms)<=old:
            raise ArithmeticError('Laurent second-column valuation did not increase')
        budget.word(right)
    if any(e[0]<0 for f in M.entries.values() for e in f.terms):
        raise ArithmeticError('The determinant relation did not clear the second row valuation')
    if any(e[0]==0 and e[1]!=0 for f in M.entries.values() for e in f.terms):
        raise ArithmeticError('Laurent boundary normalization was lost')
    power=max([0]+[(-b+a-1)//a for f in M.entries.values() for a,b in f.terms if a>0])
    C=((1,0),(power,1))
    M=_coordinate(M,C)
    total=((B[0][0],B[0][1]),(power*B[0][0]+B[1][0],power*B[0][1]+B[1][1]))
    inverse=_inverse_coordinates(total)
    word=_polynomial_sl2_word(M,budget,**options)
    word=[(i,j,_coordinate(c,inverse)) for i,j,c in word]
    # prefix is chronological, whereas right contains a written product.
    word=_join(budget,[(i,j,-c) for i,j,c in prefix],word,_inverse_word(right))
    expected=eye(ring,3)
    for i in range(2):
        for j in range(2):expected[i,j]=A[i,j]
    if _replay(word,ring,3,budget)!=expected:
        raise ArithmeticError('Laurent-to-polynomial pullback failed fixed-rank replay')
    return word


class _CancellationBudget:
    """Finite actual-support correlations sharing the outer resource counters."""
    def __init__(self,budget):
        self.budget=budget
        self.used=0
        self.cap=min(2000000,max(0,budget.max_products-budget.products))

    def check(self,terms=None,correlations=0):
        from .elementary import _CorrelationLimit
        if self.used+correlations>self.cap:
            raise _CorrelationLimit('Fixed-register correlation probe cap reached')
        self.used+=correlations
        if correlations:self.budget.pairs(correlations)
        self.budget.check(terms=0 if terms is None else terms)


def _structural_column(current,inverse,last,budget,left,right,*,destination=None):
    """Keep exact decreasing eliminations, then choose one actual column.

    There is no trial factorization of alternate columns. A support-decreasing
    monomial operation is kept immediately (or exposes a unit pivot), and its
    actual inverse operation is recorded. Afterwards choose a unit-containing
    column, then a column with an existing zero slot, then the smallest support.
    """
    from .elementary import _best_cancellation,_CorrelationLimit
    from ..algebra.modules import _UnitReductionMatrix
    size=last+1
    destination=last if destination is None else destination
    probe=_CancellationBudget(budget)
    while not any(f.is_unit for (i,j),f in current.entries.items() if i<size and j<size):
        active=current[:size,:size]
        try:
            candidate=_best_cancellation(_UnitReductionMatrix(active),0,probe)
        except _CorrelationLimit:
            break
        if candidate is None:break
        _,side,target,source,shift,scalar=candidate
        coefficient=current.ring.monomial(shift,scalar)
        if side==0:
            _row_add(current,target,source,coefficient,budget)
            _col_add(inverse,source,target,-coefficient,budget)
            left.append((target,source,coefficient))
        else:
            _col_add(current,target,source,coefficient,budget)
            _row_add(inverse,source,target,-coefficient,budget)
            right.append((source,target,coefficient))
        budget.structural_cancellations+=1
        budget.word(left+right)
    choices=[]
    for j in range(size):
        values=[current[i,j] for i in range(size)]
        occupied=sum(bool(f) for f in values)
        units=sum(f.is_unit for f in values)
        terms=sum(len(f.terms) for f in values)
        bezout_terms=sum(len(inverse[j,i].terms) for i in range(size))
        category=0 if units else 1 if occupied<size else 2
        choices.append(((category,terms,bezout_terms,j),j,occupied,units,terms,bezout_terms))
    _,chosen,occupied,units,terms,bezout_terms=min(choices)
    budget.column_choices.append(dict(active_rank=size,column=chosen,occupied=occupied,
        destination=destination,unit_entries=units,column_terms=terms,bezout_terms=bezout_terms))
    if chosen!=destination:
        one=current.ring.one()
        for i,j,c in [(chosen,destination,one),(destination,chosen,-one),(chosen,destination,one)]:
            _col_add(current,j,i,c,budget)
            _row_add(inverse,i,j,-c,budget)
            right.append((i,j,c))
        budget.word(left+right)


def _general_sl_word(A,Ainv,budget,*,max_patches,max_field_trials,seed,progress):
    from .elementary_columns import elementary_column_reduce
    ring,n=A.ring,A.nrows
    def geometry(M):
        unit_columns=zero_columns=0
        for j in range(n):
            values=[M[i,j] for i in range(n)]
            unit_columns+=any(f.is_unit for f in values)
            zero_columns+=any(not f for f in values)
        return (-unit_columns,-zero_columns,
                sum(len(f.terms) for f in M.entries.values()),len(M.entries))
    forward,inverse_geometry=geometry(A),geometry(Ainv)
    reversed_direction=inverse_geometry<forward
    if reversed_direction:
        A,Ainv=Ainv,A
        budget.factor_direction='inverse'
    if progress:progress('structural_direction',{'direction':budget.factor_direction,
        'forward_geometry':list(forward),'inverse_geometry':list(inverse_geometry)})
    current,inverse=A.copy(),Ainv.copy()
    left,right=[],[]
    for last in range(n-1,1,-1):
        budget.phase='elementary-column'
        _structural_column(current,inverse,last,budget,left,right)
        if progress:progress('elementary_column',{'registers':n,'active_rank':last+1})
        column=current[:last+1,last]
        bezout=inverse[last,:last+1]
        result=elementary_column_reduce(column,bezout,timeout=budget.remaining(),
            max_steps=max(1,budget.max_steps-budget.steps),max_terms=budget.max_terms,
            max_products=max(1,budget.max_products-budget.products),max_factors=budget.max_factors,
            max_patches=max_patches,max_field_trials=max_field_trials)
        budget.absorb(result['statistics'])
        chronological=[(f['row'],f['column'],f['coefficient'][0,0]) for f in result['factors']]
        one=ring.one()
        chronological.extend(((last,0,one),(0,last,-one),(last,0,one)))
        for i,j,c in chronological:
            _row_add(current,i,j,c,budget)
            _col_add(inverse,j,i,-c,budget)
            left.append((i,j,c))
        for j in range(last):
            c=-current[last,j]
            if c:
                _col_add(current,j,last,c,budget)
                _row_add(inverse,last,j,-c,budget)
                right.append((last,j,c))
        budget.word(left+right)
    # Preserve further exact support reductions inside the final SL2 block;
    # an E2 failure still proceeds to the existing third-coordinate SL3 route.
    _structural_column(current,inverse,1,budget,left,right,destination=0)
    target=eye(ring,n)
    for i in range(2):
        for j in range(2):target[i,j]=current[i,j]
    if current!=target:
        raise ArithmeticError('Column reductions did not isolate the SL2 residual')
    if progress:progress('sl3_residual',{'registers':n,'workspace_registers':[0,1,2]})
    residual=_laurent_sl2_word(current[:2,:2],budget,max_patches=max_patches,
                              max_field_trials=max_field_trials,seed=seed)
    word=_join(budget,[(i,j,-c) for i,j,c in left],residual,_inverse_word(right))
    if _replay(word,ring,n,budget)!=A:
        raise ArithmeticError('General fixed-register word failed exact physical action')
    return _inverse_word(word) if reversed_direction else word


def factor_unstabilized(U,V=None,*,method='auto',timeout=120,max_steps=1000000,
                       max_terms=1000000,max_factors=200000,max_products=30000000,
                       max_patches=128,max_field_trials=128,seed=0,progress=None):
    """Return U=S F with all elementary factors on the original q registers.

    ``auto`` first tries a bounded fixed-rank unit/cancellation probe; if it
    stops, the general route retains exact unit/support reductions and chooses
    the forward or inverse SL matrix once from actual sparse geometry.
    ``suslin`` enters that general route directly. No probe uses stabilization.
    ``fast`` is an explicitly incomplete unit/cancellation reduction. Ranks
    zero/one have only determinant transport; rank two uses the fast route
    and never claims general E2 generation. Resource exhaustion is distinct
    from algebraic nonmembership and never publishes a partial factorization.
    ``max_products`` bounds instrumented sparse convolution pairs, including
    normality and matrix replay, not every internal rational/SymPy operation.
    All time/operation limits are cooperative, not process-level hard limits.
    """
    if not isinstance(U,Matrix) or U.nrows!=U.ncols or U.ring.D!=2:
        raise ValueError('Expected a square prime-field bivariate Laurent matrix')
    if method not in ('auto','suslin','fast'):
        raise ValueError('method must be auto, suslin or fast')
    if V is not None and (not isinstance(V,Matrix) or V.ring!=U.ring or V.shape!=U.shape):
        raise ValueError('Inverse has incompatible ring or shape')
    for name,value in [('max_patches',max_patches),('max_field_trials',max_field_trials)]:
        if type(value) is not int or value<1:raise ValueError(name+' must be a positive integer')
    if type(seed) is not int:raise ValueError('seed must be an integer')
    if progress is not None and not callable(progress):raise ValueError('progress must be callable')
    budget=_Budget(timeout,max_steps,max_terms,max_factors,max_products)
    ring,n=U.ring,U.nrows
    if progress:progress('validation',{'registers':n})
    budget.check(U)
    if V is None:
        from .algebra import inverse
        V=inverse(U)
        budget.check(V)
    if _mm(U,V,budget)!=eye(ring,n) or _mm(V,U,budget)!=eye(ring,n):
        raise ValueError('Supplied inverse failed both exact Laurent identities')
    fast_data=None
    if method=='auto' and n>=3:
        budget.phase='fixed-rank-fast-probe'
        from .elementary import factor_unit_reduction,FactorizationIncomplete
        # A stopped child has no complete operation count. Charge its whole
        # finite quota rather than omit that work from the shared step cap.
        quota=min(50000,max(1,(max_steps-budget.steps)//4))
        try:
            fast_data=factor_unit_reduction(U,V,transport='separate',residual='stop',
                timeout=min(5.0,budget.remaining()),max_steps=quota,
                max_terms=max_terms,max_factors=max_factors,
                max_correlations=min(2000000,max_products))
        except (FactorizationIncomplete,ResourceLimit) as exc:
            budget.fast_probe_outcome='incomplete' if isinstance(exc,FactorizationIncomplete) else 'resource-stop'
            budget.steps+=quota
            budget.fast_probe_operation_charge=quota
            budget.check()
        else:
            budget.fast_probe_outcome='complete'
            budget.absorb(fast_data['reduction_statistics'])
    if method=='fast' or n<3 or fast_data is not None:
        from .elementary import factor_unit_reduction,FactorizationIncomplete
        data=fast_data
        if data is None:
            try:
                data=factor_unit_reduction(U,V,transport='separate',residual='stop',
                    timeout=budget.remaining(),max_steps=max(1,max_steps-budget.steps),
                    max_terms=max_terms,max_factors=max_factors,
                    max_correlations=min(2000000,max_products))
            except FactorizationIncomplete as exc:
                raise UnstableFactorizationIncomplete(
                    'The fixed-register fast path stalled; general E2 generation is not claimed'
                    if n<3 else 'The fixed-register fast path stalled; select the general suslin method') from exc
            budget.absorb(data['reduction_statistics'])
        if data['added_ancillas']!=0:raise ArithmeticError('A fast factorization enlarged the input register')
        word=_from_records(data['factors'])
        coefficients=data.get('transport_coefficients',[1]*n)
        shifts=data['shift_exponents']
        determinant=data['determinant'][0,0]
        algorithm='fixed-register-unit-reduction-fp-v1'
        complete=False
    else:
        from .circuits import _unit_determinant_verified
        determinant=_unit_determinant_verified(U)
        budget.check(determinant)
        (exponent,coefficient),=determinant.terms.items()
        shifts=[(0,0)]*n;coefficients=[1]*n
        shifts[-1]=exponent;coefficients[-1]=coefficient
        S=eye(ring,n);Si=eye(ring,n)
        S[n-1,n-1]=determinant;Si[n-1,n-1]=determinant**-1
        A,Ai=_mm(Si,U,budget),_mm(V,S,budget)
        word=_general_sl_word(A,Ai,budget,max_patches=max_patches,
                             max_field_trials=max_field_trials,seed=seed,progress=progress)
        algorithm='park-woodburn-fixed-register-laurent-fp-v1'
        complete=True
    budget.phase='final-original-register-replay'
    P=_replay(word,ring,n,budget)
    Pi=_replay(_inverse_word(word),ring,n,budget)
    S=Matrix(ring,n,n,{(i,i):ring.monomial(e,c) for i,(e,c) in enumerate(zip(shifts,coefficients))})
    Si=Matrix(ring,n,n,{(i,i):ring.monomial(e,c)**-1 for i,(e,c) in enumerate(zip(shifts,coefficients))})
    if (_mm(S,P,budget)!=U or _mm(Pi,Si,budget)!=V or
        _mm(P,Pi,budget)!=eye(ring,n) or _mm(Pi,P,budget)!=eye(ring,n)):
        raise ArithmeticError('Final original-register elementary action/inverse check failed')
    if progress:progress('complete',{'registers':n,'factors':len(word),'added_ancillas':0})
    return {'factors':_records(word,ring),'shift_exponents':shifts,
            'transport_coefficients':coefficients,'added_ancillas':0,
            'ancilla_action':eye(ring,0),'ancilla_inverse':eye(ring,0),'input_inverse':V,
            'determinant':Matrix.from_rows(ring,[[determinant]]),'algorithm':algorithm,
            'complete':True,'complete_general_suslin_algorithm':complete,
            'statistics':{'registers':n,'physical_matrix_shape':[n,n],
                'operations':budget.steps,'convolution_pairs':budget.products,
                'factors':len(word),'input_ancillas':0,'stable_factorization_used':False,
                'fast_probe_operation_charge':budget.fast_probe_operation_charge,
                'fast_probe_outcome':budget.fast_probe_outcome,
                'operation_accounting':'shared cooperative checks; completed child counts absorbed; stopped probe charged its full quota',
                'convolution_pair_scope':'instrumented sparse products; rational internals checked cooperatively',
                'column_choices':budget.column_choices,
                'structural_cancellations':budget.structural_cancellations,
                'factor_direction':budget.factor_direction,
                'local_patches':budget.local_patches}}
