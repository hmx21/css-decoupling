"""Independent verification of Appendix C4 cone and Schanuel witnesses.

The verifier performs only exact matrix algebra.  It neither calls the
constructor, computes a new resolution, nor invokes module solves or QS.
The scope is a stable check-cokernel isomorphism for the recorded bounded
comparison; a raw CSS/topological input must be bound by its outer pipeline.
"""
from __future__ import annotations

from css_decoupling.algebra.laurent import Ring, block, direct_sum, eye, zeros
from css_decoupling.certificates.comparison import (
    VerificationReport, _int, _keys, _list, _matrix_dict, _matrix_from_dict,
)
from css_decoupling.certificates.cone_decoupling import (
    _complex_dict, _read_complex, _degree_matrices, _read_degree_matrices,
)
from css_decoupling.topology.complexes import CochainMap, identity_map, validate_homotopy

SCHEMA = 'css-decoupling.schanuel-cokernel'
SCOPE = 'bounded-comparison-to-stable-check-cokernel'
CONVENTIONS = {
    'vectors': 'columns', 'phi': 'T-to-C',
    'cone': 'Cone(rho)[n]=C_le1[n]+(T_le1+C2_at1)[n+1]',
    'homotopy': 'I-phi*psi=d*h+h*d',
    'parity': 'Q+sum(cone_odd)->sum(cone_even)',
    'free_T': 'rank(C2)+sum(cone_odd)',
}
# Version 2 removes the frame's unit directions before forming the lifts.
CONVENTIONS_V2 = dict(CONVENTIONS, free_C='sum(cone_even)-cancelled',
                      free_T='rank(C2)+sum(cone_odd)-cancelled',
                      cancellation='unit rows of extra directions, elementary completion')
LIFTS = ('dC', 'dT', 'phi0', 'phi1', 'psi0', 'psi1', 'etaC', 'etaT')


def schanuel_certificate_dict(result):
    """Serialize current matrices and counts, without trusting success flags."""
    f = result['schanuel']
    phi, psi = f['input_phi'], f['input_psi']
    R = phi.source.ring
    reduced = result['unit_cancellation']
    return {
        'schema': SCHEMA, 'version': 2, 'scope': SCOPE,
        'conventions': dict(CONVENTIONS_V2),
        'ring': {'prime': R.p, 'dimension': R.D},
        'input': {'C': _complex_dict(phi.target), 'T': _complex_dict(phi.source),
                  'phi': _degree_matrices({n: phi.at(n) for n in phi.degrees}),
                  'psi': _degree_matrices({n: psi.at(n) for n in psi.degrees}),
                  'hC': _degree_matrices(f['input_hC']), 'hT': _degree_matrices(f['input_hT'])},
        'rho': _degree_matrices(f['rho'].components), 'cone': _complex_dict(f['cone']),
        'frame': {name: _matrix_dict(f[name]) for name in
                  ('idempotent', 'section', 'projection', 'to_free', 'from_free')},
        'contraction': _degree_matrices(f['contraction']),
        'ranks': {'even': f['even_free_rank'], 'odd': f['odd_free_rank'],
                  'cancelled': len(reduced['directions']),
                  'free_C': result['free_C'], 'free_T': result['free_T']},
        'unit_cancellation': {'directions': list(reduced['directions']),
                              'to_free': _matrix_dict(reduced['to_free']),
                              'from_free': _matrix_dict(reduced['from_free'])},
        'lifts': {name: _matrix_dict(result[name]) for name in LIFTS},
    }


def verify_schanuel(value):
    """Check input homotopies, rho cone, degree splittings, parity and lifts."""
    checks, errors = [], []
    try:
        data = schanuel_certificate_dict(value) if isinstance(value, dict) and 'schanuel' in value else value
        version = _int(data.get('version') if isinstance(data, dict) else None, 'version')
        names = ('schema', 'version', 'scope', 'conventions', 'ring', 'input',
                 'rho', 'cone', 'frame', 'contraction', 'ranks', 'lifts')
        _keys(data, names + (('unit_cancellation',) if version == 2 else ()), 'Schanuel certificate')
        if data['schema'] != SCHEMA or version not in (1, 2) or data['scope'] != SCOPE:
            raise ValueError('Unsupported Schanuel certificate')
        if data['conventions'] != (CONVENTIONS if version == 1 else CONVENTIONS_V2):
            raise ValueError('Unsupported Schanuel conventions')
        _keys(data['ring'], ('prime', 'dimension'), 'ring')
        R = Ring(_int(data['ring']['prime'], 'prime', 2), _int(data['ring']['dimension'], 'dimension', 2))
        inp = _keys(data['input'], ('C', 'T', 'phi', 'psi', 'hC', 'hT'), 'input')
        C, T = _read_complex(inp['C'], R, 'C'), _read_complex(inp['T'], R, 'T')
        phi = CochainMap(T, C, _read_degree_matrices(inp['phi'], R, 'phi'))
        psi = CochainMap(C, T, _read_degree_matrices(inp['psi'], R, 'psi'))
        degrees = set(C.degrees) | set(T.degrees)
        if set(phi.components) != degrees or set(psi.components) != degrees:
            raise ValueError('Comparison must record every degree')
        hC, hT = (_read_degree_matrices(inp[name], R, name) for name in ('hC', 'hT'))
        for X, h in ((C, hC), (T, hT)):
            for n, m in h.items():
                if m.shape != (X.rank(n - 1), X.rank(n)):
                    raise ValueError('Homotopy component shape differs from its full complex')
        validate_homotopy(C, C, identity_map(C), CochainMap(C, C, {
            n: phi.at(n) * psi.at(n) for n in C.degrees}), hC)
        validate_homotopy(T, T, identity_map(T), CochainMap(T, T, {
            n: psi.at(n) * phi.at(n) for n in T.degrees}), hT)
        checks.append('full bounded input complexes, cochain maps and inverse homotopies')
        z = hC.get(2, zeros(R, C.rank(1), C.rank(2)))
        zp = hT.get(2, zeros(R, T.rank(1), T.rank(2)))
        i = psi.at(1).vstack(C.d(1))
        r = phi.at(1).hstack(z)
        j = direct_sum(T.d(1), eye(R, C.rank(2)))
        v = block([[zp, -psi.at(1) * z],
                   [zeros(R, C.rank(2), T.rank(2)), eye(R, C.rank(2))]])
        pr = eye(R, r.ncols) - i * r
        K, J = pr * v, j * pr
        E = j * K
        if E * E != E or E * J != J:
            raise ValueError('Input does not define the C4 complement')

        low = min((0, *C.degrees, *T.degrees))
        rho = _read_degree_matrices(data['rho'], R, 'rho')
        if set(rho) != set(range(low, 2)):
            raise ValueError('rho must include all negative resolution degrees')
        for n in range(low, 2):
            if rho[n] != (r if n == 1 else phi.at(n)):
                raise ValueError('rho lift does not use the prescribed input comparison')
        cone = _read_complex(data['cone'], R, 'cone')
        if set(cone.degrees) != set(range(low - 1, 2)):
            raise ValueError('rho cone has missing or extraneous degrees')
        def arank(n):
            return C.rank(n) if low <= n <= 1 else 0
        def brank(n):
            return (T.rank(n) + (C.rank(2) if n == 1 else 0)) if low <= n <= 1 else 0
        def da(n):
            return C.d(n) if low <= n < 1 else zeros(R, arank(n + 1), arank(n))
        def db(n):
            if n == 0:
                return T.d(0).vstack(zeros(R, C.rank(2), T.rank(0)))
            return T.d(n) if low <= n < 0 else zeros(R, brank(n + 1), brank(n))
        for n in cone.degrees:
            if cone.rank(n) != arank(n) + brank(n + 1):
                raise ValueError('Incorrect cone rank')
            rn = rho.get(n + 1, zeros(R, arank(n + 1), brank(n + 1)))
            expected = block([[da(n), rn],
                              [zeros(R, brank(n + 2), arank(n)), -db(n + 1)]])
            if cone.d(n) != expected:
                raise ValueError('Cone differential is not the rho mapping cone')
        checks.append('rho lift and its mapping cone use the existing negative complexes')

        names = ('idempotent', 'section', 'projection', 'to_free', 'from_free')
        _keys(data['frame'], names, 'frame')
        frame = {name: _matrix_from_dict(data['frame'][name], R, name) for name in names}
        if frame['idempotent'] != E:
            raise ValueError('Stored idempotent is not the input complement')
        s, q = frame['section'], frame['projection']
        if s.shape != (cone.rank(0), E.nrows) or q.shape != (E.nrows, cone.rank(0)):
            raise ValueError('Incorrect homology section/projection shapes')
        if q != zeros(R, E.nrows, C.rank(0)).hstack(J):
            raise ValueError('Homology projection is not bound to the cokernel complement')
        if s[C.rank(0):, :] != K * E:
            raise ValueError('Homology section does not lift the cokernel complement')
        if q * s != E or s * E != s or E * q != q or cone.d(0) * s or q * cone.d(-1):
            raise ValueError('Invalid cone homology retraction')
        h = _read_degree_matrices(data['contraction'], R, 'contraction')
        if set(h) != set(cone.degrees):
            raise ValueError('Degree splitting must cover the entire cone')
        for n in cone.degrees:
            if h[n].shape != (cone.rank(n - 1), cone.rank(n)):
                raise ValueError('Invalid cone splitting shape')
            hp = h.get(n + 1, zeros(R, cone.rank(n), cone.rank(n + 1)))
            hm = h.get(n - 1, zeros(R, cone.rank(n - 2), cone.rank(n - 1)))
            P = s * q if n == 0 else zeros(R, cone.rank(n), cone.rank(n))
            if cone.d(n - 1) * h[n] + hp * cone.d(n) != eye(R, cone.rank(n)) - P or hm * h[n]:
                raise ValueError(f'Invalid split cone equation in degree {n}')
        if h[0] * s or q * h[1]:
            raise ValueError('Cone contraction fails homology normalization')
        checks.append('degree-by-degree split contraction onto the exact complement')

        even = [n for n in cone.degrees if n % 2 == 0]
        odd = [n for n in cone.degrees if n % 2]
        a, k = sum(cone.rank(n) for n in even), sum(cone.rank(n) for n in odd)
        rank_names = ('even', 'odd', 'free_C', 'free_T') + (('cancelled',) if version == 2 else ())
        ranks = _keys(data['ranks'], rank_names, 'ranks')
        actual = {name: _int(ranks[name], name, 0) for name in ranks}
        cancelled = actual.get('cancelled', 0)
        if actual != dict(even=a, odd=k, free_C=a - cancelled, free_T=C.rank(2) + k - cancelled,
                          **({'cancelled': cancelled} if version == 2 else {})):
            raise ValueError('Stable free ranks do not match the Schanuel parity terms and cancellations')
        def off(n, m):
            if n == m + 1:
                return cone.d(m)
            if n == m - 1:
                return h[m]
            return zeros(R, cone.rank(n), cone.rank(m))
        expected_to = block([[s if n == 0 else zeros(R, cone.rank(n), E.nrows),
                              *(off(n, m) for m in odd)] for n in even])
        expected_from = block([[q if n == 0 else zeros(R, E.nrows, cone.rank(n)) for n in even],
                               *[[off(m, n) for n in even] for m in odd]])
        P, Q = frame['to_free'], frame['from_free']
        if P != expected_to or Q != expected_from:
            raise ValueError('Frame is not the parity map of the supplied split cone')
        if P * Q != eye(R, a) or Q * P != direct_sum(E, eye(R, k)):
            raise ValueError('Schanuel frame lacks exact two-sided inverses')
        checks.append('even/odd Schanuel frame and both inverses, including all stable directions')
        if version == 2:
            reduced = _keys(data['unit_cancellation'], ('directions', 'to_free', 'from_free'),
                            'unit_cancellation')
            directions = _list(reduced['directions'], 'cancelled directions')
            if (len(directions) != cancelled or len(set(map(str, directions))) != cancelled
                    or any(isinstance(d, bool) or not isinstance(d, int) or not 0 <= d < k
                           for d in directions)):
                raise ValueError('Cancelled directions must be distinct extra coordinates')
            P = _matrix_from_dict(reduced['to_free'], R, 'reduced to_free')
            Q = _matrix_from_dict(reduced['from_free'], R, 'reduced from_free')
            k, a = k - cancelled, a - cancelled
            if (P.shape != (a, E.nrows + k) or Q.shape != (E.nrows + k, a)
                    or P * Q != eye(R, a) or Q * P != direct_sum(E, eye(R, k))):
                raise ValueError('Reduced Schanuel frame lacks exact two-sided inverses')
            checks.append(f'unit-cancelled frame Q+R^{k}<->R^{a} and both inverses')

        _keys(data['lifts'], LIFTS, 'lifts')
        lifts = {name: _matrix_from_dict(data['lifts'][name], R, name) for name in LIFTS}
        expected_phi = block([[r, zeros(R, C.rank(1), k)], [P[:, :E.nrows] * J, P[:, E.nrows:]]])
        expected_psi = block([[i, K * Q[:E.nrows, :]], [zeros(R, k, C.rank(1)), Q[E.nrows:, :]]])
        dC = C.d(0).vstack(zeros(R, a, C.rank(0)))
        dT = T.d(0).vstack(zeros(R, C.rank(2) + k, T.rank(0)))
        if (lifts['phi1'] != expected_phi or lifts['psi1'] != expected_psi
                or lifts['dC'] != dC or lifts['dT'] != dT):
            raise ValueError('Comparison lifts are not the Schanuel cokernel maps')
        for A, X, B in ((dC, lifts['phi0'], expected_phi * dT),
                        (dT, lifts['psi0'], expected_psi * dC),
                        (dC, lifts['etaC'], eye(R, dC.nrows) - expected_phi * expected_psi),
                        (dT, lifts['etaT'], eye(R, dT.nrows) - expected_psi * expected_phi)):
            if A * X != B:
                raise ValueError('Stable check-cokernel lift equation failed')
        checks.append('all four stable check-cokernel comparison and inverse-error lifts')
    except (ValueError, TypeError, KeyError, AttributeError, IndexError) as exc:
        errors.append(str(exc))
    return VerificationReport(not errors, tuple(checks), tuple(errors), SCOPE)
