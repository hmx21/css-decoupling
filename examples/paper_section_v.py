"""Reproduce Sec. V of the manuscript: removing the gluing of the glued code.

The checks are typed from Eq. (13). Every claim of Sec. V is recomputed and
compared with the stated value; finite torus sizes are a finite check only.
"""
import argparse
import itertools
import json
import sys
from css_decoupling import (Ring, Matrix, coarse_grain, check_lattice, analyze_profile,
                            decouple, verify_result, periodic_code)


def paper_glued(R):
    """H_X(GC) and H_Z(GC) of Eq. (13), with w_i = x_i - 1."""
    x1,x2,x3=R.variables
    w1,w2,w3=x1-1,x2-1,x3-1
    o,l=R(0),R(1)
    HX=Matrix.from_rows(R,[[w1,w2,w3,-l,o,o],
                           [o,o,o,-w2,-w3,o],
                           [o,o,o,w1,o,-w3],
                           [o,o,o,o,w1,w2]])
    HZ_dagger=Matrix.from_rows(R,[[-w2,-w3,o,o],
                                  [w1,o,-w3,o],
                                  [o,w1,w2,l],
                                  [o,o,o,w3],
                                  [o,o,o,-w2],
                                  [o,o,o,w1]])
    return HX,HZ_dagger.dagger()


def diagonal(*entries):
    return [[entries[i] if i==j else 0 for j in range(3)] for i in range(3)]


def plaquette_values(R,HX,HZ):
    """Values (f_12, f_13, f_23) of the gluing class on Z^3, from check_lattice."""
    decision=check_lattice(HX,HZ,diagonal(1,1,1))
    if decision.reason!='nonzero-gluing':
        return decision.reason,None
    cls,=decision.gluing[-1].classes
    unit=[[int(i==a) for i in range(3)] for a in range(3)]
    values=[]
    for a,b in ((0,1),(0,2),(1,2)):
        entry=cls.evaluate([unit[a],unit[b]]).entries.get((0,0))  # a 1x1 constant matrix
        values.append(0 if entry is None else entry.terms[(0,0,0)])
    return decision.reason,values


def reproduce(p,*,uniform=False,torus_sizes=range(1,4)):
    R=Ring(p,3)
    HX,HZ=paper_glued(R)
    claims={}
    coarse=coarse_grain(HX,HZ)
    claims['two nonzero layers, t_1=t_2=1']=coarse.multiplicities=={1:1,2:1}
    claims['Lambda_triv = Z^3']=[list(r) for r in coarse.translation_kernel.basis]==diagonal(1,1,1)
    claims['Algorithm 1 gives Lambda_suff = pZ^3']=[list(r) for r in coarse.lattice.basis]==diagonal(p,p,p)
    reason,values=plaquette_values(R,HX,HZ)
    claims['gluing nonzero on Z^3']=reason=='nonzero-gluing'
    claims['plaquette values (-1,0,0) up to sign']=values is not None and any(
        values==[s*v%p for v in (-1,0,0)] for s in (1,-1))
    for name,basis,expected in (('x_1',diagonal(p,1,1),True),('x_2',diagonal(1,p,1),True),
                                ('x_3',diagonal(1,1,p),False)):
        decision=check_lattice(HX,HZ,basis)
        claims[f'coarse-graining along {name} alone {"suffices" if expected else "does not suffice"}']=(
            decision.decouplable is expected)
    analysis=analyze_profile(HX,HZ)
    claims['smallest superlattice has index p']=(analysis.lattice.index==p
                                                 and analysis.lattice_guarantee=='minimum-cell')
    modes=[('adaptive',{})]+([('uniform',{'coarse_graining':'uniform'})] if uniform else [])
    circuits={}
    for mode,options in modes:
        result=decouple(HX,HZ,3,p,algorithm='general',output='circuit',options=options)
        valid=verify_result(result,expected_HX=HX,expected_HZ=HZ).valid
        claims[f'{mode} circuit onto TC_1 + TC_2 verified']=(
            valid and result.status=='circuit_ready'
            and result.target['toric_multiplicities']==[[1,1],[2,1]])
        circuits[mode]={'lattice_columns':result.lattice['basis_columns'],
                        'registers_per_cell':result.metadata['qca_registers'],
                        'input_ancillas':result.metadata['qca_input_ancillas'],
                        'gate_layers':len(result.circuit.to_dict()['layers'])}
    torus={}
    for L in itertools.product(torus_sizes,repeat=3):
        k=periodic_code(HX,HZ,diagonal(*L)).logical_qudits
        torus['x'.join(map(str,L))]=k
    claims['torus: 6 logical qudits iff p | L_1 L_2, else 4']=all(
        k==(6 if int(L.split('x')[0])*int(L.split('x')[1])%p==0 else 4) for L,k in torus.items())
    return {'p':p,'claims':claims,'circuits':circuits,'torus':torus}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prime',type=int,action='append',help='Prime p (repeatable; default 2 and 3)')
    parser.add_argument('--uniform',action='store_true',help='Also build the circuit on pZ^3')
    parser.add_argument('--torus-max',type=int,default=3,help='Largest torus period checked')
    parser.add_argument('--output',help='New JSON report path')
    args=parser.parse_args()
    reports=[reproduce(p,uniform=args.uniform,torus_sizes=range(1,args.torus_max+1))
             for p in (args.prime or [2,3])]
    for report in reports:
        print(f"p = {report['p']}")
        for claim,ok in report['claims'].items():
            print(f"  [{'ok' if ok else 'FAIL'}] {claim}")
        for mode,data in report['circuits'].items():
            print(f"  {mode}: {data}")
    if args.output:
        with open(args.output,'x') as handle:
            json.dump(reports,handle,indent=1)
    return 0 if all(all(r['claims'].values()) for r in reports) else 1


if __name__=='__main__':
    raise SystemExit(main())
