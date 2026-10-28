"""Separate certificate for an elementary circuit bound to a complete QCA.

The original QCA schema and the doubled-register compiler are unchanged.
Nonzero determinant transport is either explicit or balanced by one ancilla.
"""
from __future__ import annotations

from .algebra import direct_sum
from .elementary import factor_unit_reduction
from .pipeline import stabilized_checks
from .serialization import digest
from .verify import verify, verify_generation

SCHEMA = 'css-decoupling-2d/direct-decoupling/v1'
PRIME_SCHEMA = 'css-decoupling-2d/direct-decoupling/v2'
_METHODS={'unit-pivots-and-decreasing-monomial-cancellations-v1':False,
          'unit-reduction-with-higman-pid-v1':True,
          'higman-pid-f2-laurent-v1':True,
          'unit-pivots-and-decreasing-monomial-cancellations-fp-v1':False,
          'unit-reduction-with-higman-pid-fp-v1':True,
          'higman-pid-fp-laurent-v1':True}
_PRIME_METHODS = {'unit-pivots-and-decreasing-monomial-cancellations-fp-v1',
                  'unit-reduction-with-higman-pid-fp-v1', 'higman-pid-fp-laurent-v1'}


def compile_qca_direct(result, *, transport='balance', timeout=120,
                       max_factors=200000,max_families=200000,
                       max_matrix_terms=1000000,max_steps=1000000,
                       max_correlations=2000000, verify_source=True,
                       factorization='auto',max_linearized_size=128,
                       max_algebra_products=30000000,progress=None):
    """Construct a separate direct certificate; never mutate the QCA input.

    Auto first uses exact elementary reductions, then the constructive stable
    Higman/PID algorithm on its residual. Fast alone is deliberately incomplete.
    On a resource stop the caller retains the separately serialized QCA.
    """
    from .direct_circuits import compile_direct_circuit
    if type(verify_source) is not bool:
        raise ValueError('verify_source must be boolean')
    if progress is not None and not callable(progress):
        raise ValueError('progress must be callable or None')
    if type(max_steps) is not int or max_steps<1:
        raise ValueError('max_steps must be a positive integer')
    if verify_source:
        verify(result,max_families=max_families,max_matrix_terms=max_matrix_terms)
    Q=result['qca'];U,V=Q['psi'][1],Q['psi_inverse'][1]
    if factorization not in ('fast','auto','higman'):
        raise ValueError('Unknown elementary factorization strategy')
    if type(max_linearized_size) is not int or max_linearized_size<1:
        raise ValueError('max_linearized_size must be a positive integer')
    if type(max_algebra_products) is not int or max_algebra_products<1:
        raise ValueError('max_algebra_products must be a positive integer')
    if factorization=='higman':
        from .suslin import factor_suslin
        data=factor_suslin(U,V,transport=transport,timeout=timeout,
            max_steps=max_steps,max_terms=max_matrix_terms,max_factors=max_factors,
            max_linearized_size=max_linearized_size,max_products=max_algebra_products,progress=progress)
    else:
        data=factor_unit_reduction(U,V,transport=transport,timeout=timeout,
              max_steps=max_steps,max_terms=max_matrix_terms,
              max_correlations=max_correlations,max_factors=max_factors,
              residual='higman' if factorization=='auto' else 'stop',
              suslin_options={'max_linearized_size':max_linearized_size,
                              'max_products':max_algebra_products,'progress':progress})
    if progress:
        progress('elementary_factorization_complete',{'method':data['algorithm'],
             'statistics':data.get('reduction_statistics',data.get('statistics',{})),
             'added_ancillas':data['added_ancillas'],'factor_count':len(data['factors'])})
    C=compile_direct_circuit(U,V,data['factors'],added_ancillas=data['added_ancillas'],
         ancilla_action=data['ancilla_action'],ancilla_inverse=data['ancilla_inverse'],
         shift_exponents=[list(e) for e in data['shift_exponents']],max_factors=max_factors,
         transport_coefficients=data.get('transport_coefficients'),
         max_families=max_families,max_matrix_terms=max_matrix_terms)
    a=data['added_ancillas']
    OX,OZ=stabilized_checks(result['coarse_original_HX'],result['coarse_original_HZ'],a)
    TX,TZ=stabilized_checks(Q['standard_HX'],Q['standard_HZ'],a)
    # The actual circuit is F=S^-1 W, so its exact standard target is shifted.
    S,Sinv=C['transport']['z_action'],C['transport']['z_inverse']
    targetX,targetZ=TX*S,TZ*Sinv.dagger()
    old=result['qca_stabilizers']
    generation={
        'Z':{'source_in_target':direct_sum(old['Z']['source_in_target'],data['ancilla_action']),
             'target_in_source':direct_sum(old['Z']['target_in_source'],data['ancilla_inverse'])},
        'X':dict(old['X'])}
    binding={'source_HX':OX,'source_HZ':OZ,'standard_HX':TX,'standard_HZ':TZ,
             'actual_target_HX':targetX,'actual_target_HZ':targetZ,
             'stabilizers':generation,'retained_lattice':result['lattice']['basis_columns'],
             'original_product_X':Q['product_x'],'original_product_Z':Q['product_z'],
             'added_Z_zero_ancillas':a}
    artifact={'schema':SCHEMA if U.ring.p == 2 else PRIME_SCHEMA,
            'qca_result':result,'direct_circuit':C,'code_binding':binding,
            'construction':{'method':data['algorithm'],
                            'general_suslin_complete':data['complete_general_suslin_algorithm'],
                            'transport_policy':transport}}
    # compile_direct_circuit has just verified the actual core. No mutation of
    # that core occurs while attaching the code-level data above. Check the new
    # binding here; a separate reader still performs both full checks afresh.
    if progress:progress('code_stabilizer_binding',{'equation':'Both actual circuit stabilizer modules generate their declared targets'})
    _verify_code_binding(artifact)
    return artifact


def verify_direct_binding(artifact, *, max_factors=200000,max_families=200000,
                          max_matrix_terms=1000000):
    """Check direct circuit and source/target binding after source QCA verification."""
    from .direct_circuits import verify_direct_circuit
    _direct_parts(artifact)
    verify_direct_circuit(artifact['direct_circuit'],max_factors=max_factors,max_families=max_families,
                          max_matrix_terms=max_matrix_terms)
    return _verify_code_binding(artifact)


def _direct_parts(artifact):
    if (not isinstance(artifact,dict) or
        set(artifact)!={'schema','qca_result','direct_circuit','code_binding','construction'} or
            artifact['schema'] not in (SCHEMA, PRIME_SCHEMA)):
        raise ValueError('Invalid direct-decoupling schema')
    R=artifact['qca_result'];Q=R['qca'];C=artifact['direct_circuit'];B=artifact['code_binding']
    prime = Q['psi'][1].ring.p
    if artifact['schema'] != (SCHEMA if prime == 2 else PRIME_SCHEMA):
        raise ValueError('Direct wrapper schema does not match its prime field')
    construction=artifact['construction']
    if (set(construction)!={'method','general_suslin_complete','transport_policy'} or
        construction['method'] not in _METHODS or
        construction['general_suslin_complete'] is not _METHODS[construction['method']] or
        construction['transport_policy'] not in ('balance','separate')):
        raise ValueError('Unsupported or altered elementary construction declaration')
    if (construction['method'] in _PRIME_METHODS) != (prime != 2):
        raise ValueError('Elementary construction declaration has the wrong coefficient field')
    return R,Q,C,B,construction


def _verify_code_binding(artifact):
    """Private code-level check, used only after an actual core verification."""
    R,Q,C,B,construction=_direct_parts(artifact)
    U,V=Q['psi'][1],Q['psi_inverse'][1]
    if C['original_qca']!=U or C['original_inverse']!=V:
        raise ValueError('Direct circuit is not bound to the saved physical QCA')
    # Read actual dimensions and shifts from the verified core certificate.
    a=C['ancillas']['count'];n=U.nrows+a
    S=C['transport']['z_action'];Sinv=C['transport']['z_inverse']
    OX,OZ=stabilized_checks(R['coarse_original_HX'],R['coarse_original_HZ'],a)
    TX,TZ=stabilized_checks(Q['standard_HX'],Q['standard_HZ'],a)
    expected={'source_HX':OX,'source_HZ':OZ,'standard_HX':TX,'standard_HZ':TZ,
        'actual_target_HX':TX*S,'actual_target_HZ':TZ*Sinv.dagger(),
        'retained_lattice':R['lattice']['basis_columns'],
        'original_product_X':Q['product_x'],'original_product_Z':Q['product_z'],
        'added_Z_zero_ancillas':a}
    if set(B)!=set(expected)|{'stabilizers'} or digest({k:B[k] for k in expected})!=digest(expected):
        raise ValueError('Direct source, target, lattice or ancilla metadata mismatch')
    if set(B['stabilizers'])!={'X','Z'}:
        raise ValueError('Both stabilizer-generation families are required')
    verify_generation(C['z_action']*OZ.dagger(),B['actual_target_HZ'].dagger(),B['stabilizers']['Z'])
    verify_generation(C['x_action']*OX.dagger(),B['actual_target_HX'].dagger(),B['stabilizers']['X'])
    if B['actual_target_HX']*B['actual_target_HZ'].dagger():
        raise ValueError('Actual target checks fail Pauli commutation')
    if construction['transport_policy']=='balance' and C['mode']!='exact':
        raise ValueError('Balanced circuit cannot retain unimplemented transport')
    return True


def verify_direct_result(artifact, *, expected_HX=None,expected_HZ=None,
                         max_factors=200000,max_families=200000,
                         max_matrix_terms=1000000):
    """Independent full input/QCA/factor/gate/target reader; no constructor call."""
    if not isinstance(artifact,dict) or artifact.get('schema') not in (SCHEMA, PRIME_SCHEMA):
        raise ValueError('Invalid direct-decoupling schema')
    R=artifact['qca_result']
    report=verify(R,expected_HX=expected_HX,expected_HZ=expected_HZ,
                  max_families=max_families,max_matrix_terms=max_matrix_terms)
    verify_direct_binding(artifact,max_factors=max_factors,max_families=max_families,
                          max_matrix_terms=max_matrix_terms)
    answer = {**report,'direct_circuit':True,'circuit':True,
            'direct_mode':artifact['direct_circuit']['mode'],
            'added_ancillas':artifact['direct_circuit']['ancillas']['count'],
            'original_qca_implemented_by_circuit':artifact['direct_circuit']['equivalence']['original_qca_implemented_by_circuit'],
            'extended_qca_implemented_by_circuit':artifact['direct_circuit']['equivalence']['extended_qca_implemented_by_circuit'],
            'remaining_transport':artifact['direct_circuit']['transport']['exponents'],
            'general_suslin_complete':artifact['construction']['general_suslin_complete']}
    if artifact['schema'] == PRIME_SCHEMA:
        answer['remaining_transport_coefficients'] = artifact['direct_circuit']['transport']['coefficients']
    return answer
