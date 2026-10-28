"""Independent positive witnesses for a single, source-bound C4 section.

All four original Laurent constraints are multiplied. The returned X proves
the boundary inclusion; neither a success flag nor bounded-search failure
establishes it. These witnesses are not whole-complex/circuit certificates.
"""
from css_decoupling.algebra.laurent import Ring
from .comparison import VerificationReport, _keys, _int, _matrix_dict, _matrix_from_dict

SCHEMA = "css-decoupling.structural-c4-section"
FACTOR_NAMES = ('d', 'R', 'hnext', 's', 'q', 'Pprev')
METHODS = ('verified-seed', 'annihilator-shared-coefficients', 'factorized-coefficients')


def _bound(factors):
    d = factors['d']
    D, m = d.ring.D, d.nrows
    H = max((abs(v) for A in factors.values() for f in A.entries.values()
             for e in f.terms for v in e), default=0)
    delta = max((2*D-1)*H, D+1)
    return dict(dimension=D, target_rank=m, actual_half_width=H, delta=delta,
                base=max(3, m*delta), power=D+1,
                additive=(m-1)*delta+(4*D-2)*H if m else 0,
                zero_degree=not H or not m,
                formula="0 if zero_degree else base**power+additive",
                qualification="returned-section-proves-boundary-image",
                support_guarantee="some solution has canonical lift weight <= B; box [-B,B]^D suffices")


def _resources(X):
    return dict(monomials=sum(len(f.terms) for f in X.entries.values()),
                max_l1_radius=max((sum(map(abs, e)) for f in X.entries.values() for e in f.terms), default=0),
                max_canonical_weight=max((sum(e)+(X.ring.D+1)*max(0, -min(e))
                                          for f in X.entries.values() for e in f.terms), default=0))


def structural_section_dict(d, R, hnext, s, q, Pprev, X, *, method):
    factors = dict(zip(FACTOR_NAMES, (d, R, hnext, s, q, Pprev)))
    return dict(schema=SCHEMA, version=1, ring=dict(prime=d.ring.p, dimension=d.ring.D),
                factors={name: _matrix_dict(A) for name, A in factors.items()},
                solution=_matrix_dict(X), degree_bound=_bound(factors), resources=_resources(X),
                search=dict(method=method, complete_search=False))


def verify_structural_section(certificate, *, expected=None):
    checks, errors = [], []
    try:
        _keys(certificate, ('schema', 'version', 'ring', 'factors', 'solution', 'degree_bound', 'resources', 'search'), 'C4 section')
        if certificate['schema'] != SCHEMA or _int(certificate['version'], 'version') != 1:
            raise ValueError('Unsupported structural C4 witness')
        _keys(certificate['ring'], ('prime', 'dimension'), 'ring')
        Rg = Ring(_int(certificate['ring']['prime'], 'prime', 2),
                  _int(certificate['ring']['dimension'], 'dimension', 1))
        _keys(certificate['factors'], FACTOR_NAMES, 'section factors')
        factors = {name: _matrix_from_dict(certificate['factors'][name], Rg, name) for name in FACTOR_NAMES}
        d, R, hn, s, q, P = (factors[name] for name in FACTOR_NAMES)
        X = _matrix_from_dict(certificate['solution'], Rg, 'section')
        m, r = d.shape
        if (R.shape != (m, m) or hn.nrows != m or s.nrows != m
                or q.ncols != r or P.shape != (r, r) or X.shape != (r, m)):
            raise ValueError('Malformed original C4 section dimensions')
        if expected is not None:
            if not isinstance(expected, dict) or set(expected) != set(FACTOR_NAMES):
                raise ValueError('Expected factors must specify every original input')
            if any(factors[name] != expected[name] for name in FACTOR_NAMES):
                raise ValueError('Structural witness changed the bound source equation')
        if R*d != d or d*P or q*P != q or P*P != P:
            raise ValueError('Invalid structural projector relations')
        if d*X != R or X*hn or X*s or q*X:
            raise ValueError('A structural section fails an original Laurent constraint')
        # dX=R and Rd=d imply R^2=R and im(R)=im(d). They also imply
        # Rhnext=Rs=0. These algebraic consequences avoid huge redundant
        # zero products without weakening any original section constraint.
        bound, resources = _bound(factors), _resources(X)
        _keys(certificate['degree_bound'], tuple(bound), 'structural degree metadata')
        for name in ('dimension', 'target_rank', 'actual_half_width', 'delta', 'base', 'power', 'additive'):
            _int(certificate['degree_bound'][name], name, 0)
        if type(certificate['degree_bound']['zero_degree']) is not bool:
            raise ValueError('Malformed zero-degree flag')
        if certificate['degree_bound'] != bound:
            raise ValueError('False structural degree bound or qualification')
        if certificate['resources'] != resources:
            raise ValueError('Incorrect actual section support statistics')
        for name, value in certificate['resources'].items():
            _int(value, name, 0)
        value = resources['max_canonical_weight']
        if bound['zero_degree']:
            within = value == 0
        else:
            # Independent bounded exponentiation, never constructing E.
            target, acc = max(1, value-bound['additive']), 1
            for _ in range(bound['power']):
                acc = min(target, acc*bound['base'])
                if acc >= target:
                    break
            within = acc >= target
        if not within:
            raise ValueError('Section exceeds its claimed structural degree ceiling')
        _keys(certificate['search'], ('method', 'complete_search'), 'positive search metadata')
        if certificate['search']['method'] not in METHODS or certificate['search']['complete_search'] is not False:
            raise ValueError('Finite positive search is not a complete nonmembership proof')
        checks.extend(('original factor equations and source binding', 'actual four linear constraints',
                       'boundary projection completeness from dX=R and Rd=d', 'structural degree and actual support'))
    except (ValueError, TypeError, KeyError, IndexError, AttributeError, ArithmeticError) as exc:
        errors.append(str(exc))
    return VerificationReport(not errors, tuple(checks), tuple(errors), 'single-C4-section-only')
