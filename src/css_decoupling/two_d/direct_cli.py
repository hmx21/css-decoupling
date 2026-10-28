"""Opt-in raw-input/compile/verify CLI for the elementary circuit route."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import traceback

from .algebra import ResourceLimit
from .cli import example, peak_memory_bytes, runtime_record, raw_input as _raw_input
from .direct_pipeline import compile_qca_direct, verify_direct_result
from .elementary import FactorizationIncomplete
from .pipeline import decouple, InvalidCSS, LatticeObstruction, check_shape, lattice_policy
from .serialization import save, load
from .verify import verify
from ..algebra.modules import get_backend


def main(argv=None):
    parser=argparse.ArgumentParser(prog='css-decouple-2d-direct')
    commands=parser.add_subparsers(dest='command',required=True)
    for command in ('run','example','compile','verify'):
        sub=commands.add_parser(command)
        sub.add_argument('source')
        if command=='verify':
            sub.add_argument('--input');sub.add_argument('--report')
        else:
            sub.add_argument('--output',required=True,help='New output directory')
        sub.add_argument('--max-terms',type=int,default=1000000)
        sub.add_argument('--max-factors',type=int,default=200000)
        sub.add_argument('--max-families',type=int,default=200000)
        if command!='verify':
            sub.add_argument('--transport',choices=('balance','separate'),default='balance')
            sub.add_argument('--factor-timeout',type=float,default=120)
            sub.add_argument('--max-correlations',type=int,default=2000000)
            sub.add_argument('--max-linearized-size',type=int,default=128)
            sub.add_argument('--max-algebra-products',type=int,default=30000000)
            sub.add_argument('--max-steps',type=int,default=1000000)
            sub.add_argument('--factorization',choices=('fast','auto','higman'),default='auto')
            if command!='compile':
                sub.add_argument('--max-index',type=int,default=4096)
                sub.add_argument('--max-matrix-size',type=int,default=3000)
                sub.add_argument('--lattice')
                sub.add_argument('--lattice-mode',choices=('auto','fixed'))
                if command=='example':
                    sub.add_argument('--prime',type=int,default=2)
    args=parser.parse_args(argv);start=time.perf_counter();output=None;output_owned=False;phase='initialization'
    limits={'max_factors':args.max_factors,'max_families':args.max_families,
            'max_matrix_terms':args.max_terms}
    def progress(stage,details):
        if output is not None:
            with (output/'progress.jsonl').open('a',encoding='utf-8',newline='\n') as stream:
                stream.write(json.dumps({'stage':stage,'details':details,
                    'wall_seconds':time.perf_counter()-start,'peak_memory_bytes':peak_memory_bytes()})+'\n')
    try:
        if args.command=='verify':
            phase='independent full direct verification'
            raw=load(args.input) if args.input else {}
            if args.input:_raw_input(raw)
            report=verify_direct_result(load(args.source),expected_HX=raw.get('HX'),
                                       expected_HZ=raw.get('HZ'),**limits)
            report.update(wall_seconds=time.perf_counter()-start,peak_memory_bytes=peak_memory_bytes())
            if args.report:save(report,args.report)
            print(json.dumps(report));return 0
        output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
        output_owned=True
        save(runtime_record(),output/'environment.json')
        if args.command=='compile':
            result=load(args.source)
            raw=result['original'];HX,HZ=raw['HX'],raw['HZ']
        else:
            if args.command=='example':
                HX,HZ=example(args.source,prime=args.prime);raw={'HX':HX,'HZ':HZ,'p':HX.ring.p,'D':2}
            else:
                raw=load(args.source)
                HX,HZ=_raw_input(raw)
            phase='raw QCA construction'
            save(raw,output/'input.json')
            basis=json.loads(args.lattice) if args.lattice else None
            result=decouple(HX,HZ,compile_circuit=False,
               lattice_basis=basis,lattice_mode=lattice_policy(args.lattice_mode,basis),
               max_index=args.max_index,max_matrix_size=args.max_matrix_size,
               max_terms=args.max_terms,progress=progress)
        if not (output/'input.json').exists():save(raw,output/'input.json')
        get_backend(HX.ring).close()
        phase='QCA serialization and verification';save(result,output/'qca.json')
        qca_report=verify(load(output/'qca.json'),expected_HX=HX,expected_HZ=HZ,
                           max_families=args.max_families,max_matrix_terms=args.max_terms)
        save(qca_report,output/'qca-verification.json');get_backend(HX.ring).close()
        phase='elementary factorization and actual gates';progress(phase,{'transport':args.transport})
        artifact=compile_qca_direct(result,transport=args.transport,timeout=args.factor_timeout,
                max_correlations=args.max_correlations,verify_source=False,
                factorization=args.factorization,max_linearized_size=args.max_linearized_size,
                max_algebra_products=args.max_algebra_products,max_steps=args.max_steps,
                progress=progress,**limits)
        # The constructor has checked its actual core once and the newly added
        # code binding. The independent CLI reader replays everything separately.
        phase='direct serialization';save(artifact,output/'direct-result.json')
        C=artifact['direct_circuit']
        status={'status':'complete' if C['mode']=='exact' else
                'complete_shift_separated' if HX.ring.p==2 else 'complete_transport_separated',
            'qca_verified':True,'direct_circuit_verified':True,
            'mode':C['mode'],'added_ancillas':C['ancillas']['count'],
            'original_qca_implemented_by_circuit':C['equivalence']['original_qca_implemented_by_circuit'],
            'extended_qca_implemented_by_circuit':C['equivalence']['extended_qca_implemented_by_circuit'],
            'remaining_transport':C['transport']['exponents'],
            'remaining_transport_coefficients':C['transport'].get('coefficients',[1]*len(C['transport']['exponents'])),
            'general_suslin_complete':artifact['construction']['general_suslin_complete'],
            'factorization_method':artifact['construction']['method'],'resources':C['resources'],
            'wall_seconds':time.perf_counter()-start,'peak_memory_bytes':peak_memory_bytes()}
        save(status,output/'status.json');print(json.dumps(status));return 0
    except Exception as exc:
        status={'status':'lattice_obstruction' if isinstance(exc,LatticeObstruction) else
                'algorithm_incomplete' if isinstance(exc,FactorizationIncomplete) else
                'resource_incomplete' if isinstance(exc,(ResourceLimit,MemoryError)) else
                'invalid_input' if isinstance(exc,InvalidCSS) else 'failed',
                'phase':phase,'exception':type(exc).__name__,'error':str(exc),
                'wall_seconds':time.perf_counter()-start,'peak_memory_bytes':peak_memory_bytes(),
                'traceback':traceback.format_exc()}
        if output_owned and output is not None and output.is_dir():
            if isinstance(exc,FactorizationIncomplete):
                save({'equation':'factor the residual Laurent matrix into elementary factors',
                      'residual':exc.residual,'pivot_count':exc.pivot_count},output/'remaining-equation.json')
            elif isinstance(exc,(ResourceLimit,MemoryError)) and (output/'qca.json').exists():
                save({'equation':'Find actual elementary factors F with diag(psi_1,B)=S*F',
                      'qca_artifact':'qca.json','phase':phase,'resource_reason':str(exc),
                      'no_mathematical_nonexistence_claim':True},output/'remaining-equation.json')
            if not (output/'status.json').exists():save(status,output/'status.json')
        print(json.dumps(status),file=sys.stderr);return 2


if __name__=='__main__':raise SystemExit(main())
