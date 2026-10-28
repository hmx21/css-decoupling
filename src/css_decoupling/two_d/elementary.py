"""Tracked elementary reduction over Fp[x^+-1,y^+-1].

The unit/cancellation fast path alone is not complete. The explicit
``residual='higman'`` option invokes the constructive stable PID algorithm on
its residual block, retaining the exact surrounding elementary transformations.
A resource stop never claims that no circuit exists. Diagonal transport is
separate or balanced by one declared Z-product ancilla.
"""
from __future__ import annotations

from time import perf_counter

from .algebra import Matrix, eye, direct_sum
from ..algebra.modules import _UnitReductionMatrix
from ..errors import ResourceLimit


class FactorizationIncomplete(RuntimeError):
    """The implemented elementary reductions stopped before a full answer."""
    def __init__(self, message, *, residual=None, pivot_count=0):
        super().__init__(message)
        self.residual = residual
        self.pivot_count = pivot_count


class _CorrelationLimit(ResourceLimit):
    pass


def _limit(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return value


class _Budget:
    def __init__(self, timeout, max_steps, max_terms, max_correlations):
        if type(timeout) not in (int, float) or not 0 < timeout < float('inf'):
            raise ValueError('timeout must be finite and positive')
        self.deadline = perf_counter() + timeout
        self.max_steps = _limit(max_steps, 'max_steps')
        self.max_terms = _limit(max_terms, 'max_terms')
        self.max_correlations = _limit(max_correlations, 'max_correlations')
        self.steps = self.correlations = 0

    def check(self, terms=None, correlations=0):
        self.steps += 1
        if perf_counter() > self.deadline:
            raise ResourceLimit('Elementary reduction exceeded its cooperative timeout')
        if self.steps > self.max_steps:
            raise ResourceLimit('Elementary reduction exceeded its operation cap')
        if correlations and self.correlations+correlations > self.max_correlations:
            raise _CorrelationLimit('Elementary cancellation exceeded its correlation cap')
        self.correlations += correlations
        if terms is not None and terms > self.max_terms:
            raise ResourceLimit('Elementary reduction exceeded its Laurent term cap')

    def subcall(self, options=None):
        """Share the remaining wall clock and operation cap with a constructor."""
        self.check()
        remaining_steps=self.max_steps-self.steps
        remaining_time=self.deadline-perf_counter()
        if remaining_steps<=0 or remaining_time<=0:
            raise ResourceLimit('No remaining operation/time budget for a sub-construction')
        opts=dict(options or {})
        requested=opts.pop('max_steps',remaining_steps)
        _limit(requested,'Suslin max_steps')
        opts['max_steps']=min(requested,remaining_steps)
        return remaining_time,opts

    def absorb(self, operations):
        if type(operations) is not int or operations<0:
            raise ArithmeticError('Invalid constructor operation count')
        self.steps+=operations
        self.check()


def _factor(ring, row, column, coefficient):
    return {'row': row, 'column': column,
            'coefficient': Matrix.from_rows(ring, [[coefficient]])}


def _swap_factors(ring, a, b):
    # Signed swap [[0,1],[-1,0]] has determinant one in every characteristic.
    return [] if a == b else [_factor(ring, a, b, 1),
                              _factor(ring, b, a, -1),
                              _factor(ring, a, b, 1)]


def _unit_pair(ring, i, j, unit):
    """Chronological elementary word for diag(unit,unit^-1) on i,j."""
    # Written product w(unit) w(1)^-1; reverse it for chronological factors.
    written = [(i, j, unit), (j, i, -unit**-1), (i, j, unit),
               (i, j, -ring.one()), (j, i, ring.one()), (i, j, -ring.one())]
    return [_factor(ring, a, b, c) for a, b, c in reversed(written)]


def _best_cancellation(A, start, budget):
    """Choose a strictly term-reducing monomial row/column operation.

    For a scalar-weighted shift its exact improvement is overlap plus
    cancellations minus source_terms. These counts coincide only over F2.
    This terminating heuristic is not a completeness test for elementary groups.
    """
    n = len(A.rows)
    rows = [{j: f for j, f in A.rows[i].items() if j >= start}
            for i in range(start, n)]
    columns = [{i: A.rows[i][j] for i in A.columns[j] if i >= start}
               for j in range(start, n)]
    # Exact monomial associates give promising operations without enumerating
    # every pair of terms. A new unit advances the pivot rank immediately;
    # otherwise require strict support decrease. Neither condition is a search
    # completeness claim. Full correlation remains a separately capped fallback.
    for side,vectors in enumerate((rows,columns)):
        groups={}
        shapes={}
        for index,vector in enumerate(vectors):
            for coordinate,f in sorted(vector.items()):
                if f not in shapes:
                    anchor=min(f.terms)
                    leading=f.terms[anchor]
                    reciprocal=pow(leading,-1,A.ring.p)
                    shape=tuple(sorted((e[0]-anchor[0],e[1]-anchor[1],c*reciprocal % A.ring.p)
                                       for e,c in f.terms.items()))
                    shapes[f]=(anchor,leading,shape)
                anchor,leading,shape=shapes[f]
                groups.setdefault((coordinate,shape),[]).append((index,anchor,leading))
        seen=set()
        for group in groups.values():
            for source,se,sc in group:
                for target,te,tc in group:
                    if source==target:continue
                    scalar=-tc*pow(sc,-1,A.ring.p) % A.ring.p
                    shift=(te[0]-se[0],te[1]-se[1]);key=(source,target,shift,scalar)
                    if key in seen:continue
                    seen.add(key);budget.check()
                    c=A.ring.monomial(shift,scalar)
                    old=vectors[target];src=vectors[source]
                    total=sum(len(f.terms) for f in old.values())
                    delta=0;new_unit=False
                    for coordinate,value in src.items():
                        previous=old.get(coordinate,A.ring.zero())
                        updated=previous+c*value
                        delta+=len(updated.terms)-len(previous.terms)
                        new_unit=new_unit or updated.is_unit
                    budget.check(total+delta)
                    if delta<0 or new_unit:
                        return (delta,side,target+start,source+start,shift,scalar)
    best = None
    for side, vectors in enumerate((rows, columns)):
        for source, vector in enumerate(vectors):
            source_terms = sum(len(f.terms) for f in vector.values())
            for target, other in enumerate(vectors):
                if target == source:
                    continue
                correlations, overlaps = {}, {}
                for index in vector.keys() & other.keys():
                    f, g = vector[index], other[index]
                    budget.check(correlations=len(f.terms)*len(g.terms))
                    for e,sc in f.terms.items():
                        reciprocal=pow(sc,-1,A.ring.p)
                        for d,tc in g.terms.items():
                            shift = (d[0]-e[0], d[1]-e[1])
                            scalar=-tc*reciprocal % A.ring.p
                            key=(shift,scalar)
                            correlations[key] = correlations.get(key, 0)+1
                            overlaps[shift] = overlaps.get(shift,0)+1
                for (shift,scalar), cancelled in correlations.items():
                    improvement = overlaps[shift]+cancelled-source_terms
                    if improvement > 0:
                        key = (-improvement, side, target+start, source+start, shift,scalar)
                        if best is None or key < best:
                            best = key
                # Any proven decrease is useful; a global maximum requires a
                # quadratic scan and can exhaust the budget despite already
                # having a valid elementary step. The potential still falls.
                if best is not None:
                    return best
    return best


def _linearization_geometry(M):
    bounds=[]
    for axis in range(2):
        low=high=None
        for f in M.entries.values():
            for e in f.terms:
                low=e[axis] if low is None else min(low,e[axis])
                high=e[axis] if high is None else max(high,e[axis])
        bounds.append(0 if low is None else high-low)
    axis=min((0,1),key=lambda a:(bounds[a],-a))
    return {'rank':M.nrows,'axis':axis,'degree':bounds[axis],
            'linearized_rank':M.nrows*max(1,bounds[axis])}


def _invert_factor_data(data, inverse, budget=None):
    """If Vext=S F, return Uext=S^-1 (S F^-1 S^-1), without a solver."""
    ring=inverse.ring
    coefficients=data.get('transport_coefficients',[1]*len(data['shift_exponents']))
    units=[ring.monomial(e,c) for e,c in zip(data['shift_exponents'],coefficients)]
    factors=[]
    for f in reversed(data['factors']):
        i,j=f['row'],f['column']
        factors.append(_factor(ring,i,j,-f['coefficient'][0,0]*units[i]*units[j]**-1))
        if budget is not None:budget.check(len(f['coefficient'][0,0].terms))
    result = {**data,'factors':factors,'input_inverse':inverse,
        'shift_exponents':[tuple(-a for a in e) for e in data['shift_exponents']],
        'ancilla_action':data['ancilla_inverse'],'ancilla_inverse':data['ancilla_action'],
        'determinant':Matrix.from_rows(ring,[[data['determinant'][0,0]**-1]]),
        'statistics':{**data.get('statistics',{}),'direction_reversed':True}}
    if ring.p != 2:
        result['transport_coefficients']=[pow(c,-1,ring.p) for c in coefficients]
    return result


def _verify_factor_data(U,V,data,budget):
    """Check the actual returned word, inverse, and full scalar/shift transport."""
    ring=U.ring
    q=U.nrows+data['added_ancillas']
    product,inverse=_UnitReductionMatrix(eye(ring,q)),_UnitReductionMatrix(eye(ring,q))
    for f in data['factors']:
        product.add_row(f['row'],f['column'],f['coefficient'][0,0])
        budget.check(sum(product.row_weight))
    for f in reversed(data['factors']):
        inverse.add_row(f['row'],f['column'],-f['coefficient'][0,0])
        budget.check(sum(inverse.row_weight))
    units=[ring.monomial(e,c) for e,c in zip(data['shift_exponents'],
           data.get('transport_coefficients',[1]*q))]
    S=Matrix(ring,q,q,{(i,i):u for i,u in enumerate(units)})
    Si=Matrix(ring,q,q,{(i,i):u**-1 for i,u in enumerate(units)})
    P,Pi=product.matrix(),inverse.matrix()
    if (S*P!=direct_sum(U,data['ancilla_action']) or
        Pi*Si!=direct_sum(V,data['ancilla_inverse']) or
        P*Pi!=eye(ring,q) or Pi*P!=eye(ring,q)):
        raise ArithmeticError('Elementary factor/transport/inverse replay failed')
    budget.check(sum(len(f.terms) for M in (P,Pi) for f in M.entries.values()))
    if 'reduction_statistics' in data:
        data['reduction_statistics']['operations']=budget.steps
    return data


def factor_unit_reduction(U, V, *, transport='separate', timeout=120,
                          max_steps=1000000, max_terms=1000000,
                          max_correlations=2000000, max_factors=200000,
                          cancellation=True,residual='stop',suslin_options=None,
                          _try_inverse=True):
    """Return actual factors and explicit transport data for a verified QCA.

    ``transport='separate'`` gives U=S*F on the original species, with S a
    single-species determinant shift and F an actual elementary product.
    ``transport='balance'`` instead realizes diag(U,det(U)^-1) with one added
    Z-product species when the determinant is not one. If det(U)=1 it adds none.
    With residual='stop', a stalled block raises FactorizationIncomplete.
    With residual='higman', it is solved by actual stable elementary factors,
    not an unexpanded QS completion matrix. Additional identity ancillas are
    counted separately from determinant balancing in the returned statistics.
    """
    if (not isinstance(U, Matrix) or not isinstance(V, Matrix) or U.ring != V.ring
            or U.shape != V.shape or U.nrows != U.ncols):
        raise ValueError('Expected matching square Laurent matrices U and V')
    ring, n = U.ring, U.nrows
    if ring.D != 2:
        raise ValueError('Physical elementary compiler requires a prime-field ring with D=2')
    if (transport not in ('separate', 'balance') or type(cancellation) is not bool or
        residual not in ('stop','higman') or type(_try_inverse) is not bool):
        raise ValueError('Invalid transport or cancellation policy')
    _limit(max_factors, 'max_factors')
    budget = _Budget(timeout, max_steps, max_terms, max_correlations)
    budget.check(sum(len(f.terms) for f in U.entries.values()))
    budget.check(sum(len(f.terms) for f in V.entries.values()))
    if U*V != eye(ring, n) or V*U != eye(ring, n):
        raise ValueError('QCA inverse failed its two exact Laurent identities')
    budget.check()
    A = _UnitReductionMatrix(U)
    left, right = [], []
    k = 0
    cancellations = 0
    residual_data=None
    while k < n:
        budget.check()
        pivot = A.choose_unit(k)
        if pivot is None:
            try:
                candidate = _best_cancellation(A, k, budget) if cancellation else None
            except _CorrelationLimit:
                if residual=='stop':raise
                candidate=None
            if candidate is None:
                if residual=='higman':
                    from .suslin import factor_suslin
                    block=A.matrix()[k:,k:]
                    geometries={name:_linearization_geometry(M)
                                for name,M in [('residual',block),('original',U),('inverse',V)]}
                    chosen=min(geometries,key=lambda name:geometries[name]['linearized_rank'])
                    remaining=budget.deadline-perf_counter()
                    if remaining<=0:raise ResourceLimit('No time remains for Higman factorization')
                    options=dict(suslin_options or {})
                    for reserved in ('transport','timeout','max_terms','max_factors'):
                        if reserved in options:raise ValueError('Reserved Suslin option: '+reserved)
                    notify=options.get('progress')
                    if notify:notify('higman_route_selection',{'selected':chosen,'candidates':geometries})
                    # An inverse often retains sparse unit pivots lost by the
                    # forward reduction. Probe that direction with unit pivots
                    # only, then solve its much smaller residual. Recursion is
                    # explicitly disabled in the child and the same wall-clock
                    # deadline/remaining step cap is retained.
                    if _try_inverse and geometries[chosen]['linearized_rank']>32:
                        if notify:notify('inverse_unit_reduction',{'rank':n,'remaining_seconds':remaining})
                        remaining,child_caps=budget.subcall()
                        child=factor_unit_reduction(V,U,transport=transport,timeout=remaining,
                            max_steps=child_caps['max_steps'],max_terms=max_terms,
                            max_correlations=max_correlations,max_factors=max_factors,
                            cancellation=False,residual='higman',suslin_options=options,
                            _try_inverse=False)
                        child_stats=child.get('reduction_statistics',{})
                        budget.absorb(child_stats['operations'])
                        data=_invert_factor_data(child,V,budget)
                        budget.check()
                        data['algorithm']=('unit-reduction-with-higman-pid-v1' if ring.p==2 else
                                           'unit-reduction-with-higman-pid-fp-v1')
                        data['reduction_statistics']={'unit_pivots':k,'cancellations':cancellations,
                            'row_factors':len(left),'column_factors':len(right),
                            'correlations':budget.correlations,'operations':budget.steps,
                            'higman_used':child_stats.get('higman_used',False),
                            'inverse_unit_direction':True,'candidate_geometry':geometries,
                            'discarded_fast_trace':True,'inverse_reduction':child_stats}
                        return _verify_factor_data(U,V,data,budget)
                    if chosen!='residual':
                        original,inverse=(U,V) if chosen=='original' else (V,U)
                        remaining,call_options=budget.subcall(options)
                        data=factor_suslin(original,inverse,transport=transport,
                            timeout=remaining,max_terms=max_terms,max_factors=max_factors,**call_options)
                        budget.absorb(data['statistics']['operations'])
                        if chosen=='inverse':data=_invert_factor_data(data,V,budget)
                        budget.check()
                        data['algorithm']=('unit-reduction-with-higman-pid-v1' if ring.p==2 else
                                           'unit-reduction-with-higman-pid-fp-v1')
                        data['reduction_statistics']={'unit_pivots':k,'cancellations':cancellations,
                            'row_factors':len(left),'column_factors':len(right),
                            'correlations':budget.correlations,'operations':budget.steps,
                            'higman_used':True,'higman_input':chosen,'candidate_geometry':geometries,
                            'discarded_fast_trace':True,'higman':data.get('statistics',{})}
                        return _verify_factor_data(U,V,data,budget)
                    # From D=L U R obtain its inverse R^-1 V L^-1 by exact
                    # sparse operations, then restrict its residual block.
                    inverse_work=_UnitReductionMatrix(V)
                    for f in right:
                        inverse_work.add_row(f['row'],f['column'],-f['coefficient'][0,0])
                        budget.check(sum(inverse_work.row_weight))
                    for f in left:
                        inverse_work.add_column(f['column'],f['row'],-f['coefficient'][0,0])
                        budget.check(sum(inverse_work.row_weight))
                    inverse_block=inverse_work.matrix()[k:,k:]
                    remaining,call_options=budget.subcall(options)
                    residual_data=factor_suslin(block,inverse_block,
                        transport='separate',timeout=remaining,max_terms=max_terms,
                        max_factors=max_factors,**call_options)
                    budget.absorb(residual_data['statistics']['operations'])
                    break
                raise FactorizationIncomplete(
                    'No unit pivot or strictly decreasing elementary cancellation; '
                    'the general Laurent Suslin residual remains unsolved',
                    residual=A.matrix()[k:, k:], pivot_count=k)
            _, side, target, source, shift,scalar = candidate
            c = ring.monomial(shift,scalar)
            if side == 0:
                A.add_row(target, source, c)
                left.append(_factor(ring, target, source, c))
            else:
                A.add_column(target, source, c)
                right.append(_factor(ring, source, target, c))
            cancellations += 1
            budget.check(sum(A.row_weight))
            if len(left)+len(right)>max_factors:
                raise ResourceLimit('Elementary cancellation exceeded its factor cap')
            continue
        i, j = pivot
        if k!=i:
            for f in _swap_factors(ring,k,i):
                A.add_row(f['row'],f['column'],f['coefficient'][0,0]);left.append(f)
        if k!=j:
            for f in _swap_factors(ring,k,j):
                A.add_column(f['column'],f['row'],f['coefficient'][0,0]);right.append(f)
        if len(left)+len(right)>max_factors:
            raise ResourceLimit('Elementary swap exceeded its factor cap')
        unit = A.rows[k][k]
        reciprocal = unit**-1
        for i in sorted(A.columns[k]):
            if i != k:
                c = -A.rows[i][k]*reciprocal
                A.add_row(i, k, c); left.append(_factor(ring, i, k, c))
                budget.check(sum(A.row_weight))
                if len(left)+len(right)>max_factors:
                    raise ResourceLimit('Elementary row reduction exceeded its factor cap')
        for j, f in sorted(A.rows[k].items()):
            if j != k:
                c = -f*reciprocal
                A.add_column(j, k, c); right.append(_factor(ring, k, j, c))
                budget.check(sum(A.row_weight))
                if len(left)+len(right)>max_factors:
                    raise ResourceLimit('Elementary column reduction exceeded its factor cap')
        k += 1
        budget.check(sum(A.row_weight))
        if len(left)+len(right) > max_factors:
            raise ResourceLimit('Elementary reduction exceeded its factor cap')
    units = [A.rows[i][i] for i in range(k)]
    residual_factors=[];stabilizing_ancillas=0
    if residual_data is not None:
        stabilizing_ancillas=residual_data['added_ancillas']
        if (residual_data['ancilla_action']!=eye(ring,stabilizing_ancillas) or
            residual_data['ancilla_inverse']!=eye(ring,stabilizing_ancillas)):
            raise ArithmeticError('Higman separate-transport stabilization must be identity')
        units += [ring.monomial(e,c) for e,c in zip(residual_data['shift_exponents'],
                   residual_data.get('transport_coefficients',[1]*len(residual_data['shift_exponents'])))]
        residual_factors=[_factor(ring,k+f['row'],k+f['column'],f['coefficient'][0,0])
                          for f in residual_data['factors']]
    determinant = ring.one()
    for unit in units:
        determinant *= unit
    if residual_data is None and A.matrix() != Matrix(ring,n,n,{(i,i):unit for i,unit in enumerate(units)}):
        raise ArithmeticError('Elementary elimination did not produce a unit diagonal')
    balancing = int(transport == 'balance' and determinant != 1)
    a = stabilizing_ancillas+balancing
    q = n+a
    diagonal = units+[determinant**-1] if balancing else units
    shifts = [(0,0)]*q
    coefficients=[1]*q
    anchor = q-1
    if q and transport == 'separate':
        shifts[anchor] = next(iter(determinant.terms))
        coefficients[anchor] = next(iter(determinant.terms.values()))
    S_units = [ring.monomial(e,c) for e,c in zip(shifts,coefficients)]
    # A=L U R=D. Chronological word: R^-1, det-one D factors, S^-1 L^-1 S.
    factors = [_factor(ring,f['row'],f['column'],-f['coefficient'][0,0]) for f in right]
    factors.extend(residual_factors)
    for i in range(max(0,q-1)):
        if diagonal[i] != 1:
            factors.extend(_unit_pair(ring, i, anchor, diagonal[i]))
    for f in reversed(left):
        i,j = f['row'],f['column']
        coefficient = -f['coefficient'][0,0]*S_units[j]*S_units[i]**-1
        factors.append(_factor(ring,i,j,coefficient))
    if len(factors)>max_factors:
        raise ResourceLimit('Final elementary word exceeded its factor cap')
    budget.check(sum(len(f['coefficient'][0,0].terms) for f in factors))
    B,Binv=eye(ring,a),eye(ring,a)
    if balancing:B[a-1,a-1],Binv[a-1,a-1]=determinant**-1,determinant
    result = {'factors':factors, 'added_ancillas':a, 'ancilla_action':B,
            'ancilla_inverse':Binv, 'shift_exponents':shifts,
            'algorithm':('unit-reduction-with-higman-pid-v1' if residual=='higman' else
                         'unit-pivots-and-decreasing-monomial-cancellations-v1') if ring.p==2 else
                        ('unit-reduction-with-higman-pid-fp-v1' if residual=='higman' else
                         'unit-pivots-and-decreasing-monomial-cancellations-fp-v1'),
            'complete_general_suslin_algorithm':residual=='higman',
            'reduction_statistics':{'unit_pivots':k,'cancellations':cancellations,
                'row_factors':len(left),'column_factors':len(right),
                'correlations':budget.correlations,'operations':budget.steps,
                'higman_used':residual_data is not None,
                'identity_ancillas':stabilizing_ancillas,'balance_ancillas':balancing,
                'higman':residual_data.get('statistics',{}) if residual_data else {}},
            'determinant':Matrix.from_rows(ring,[[determinant]])}
    if ring.p!=2:
        result['transport_coefficients']=coefficients
    return _verify_factor_data(U,V,result,budget)
