"""QCA to a circuit on the original physical register, with no input ancillas."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
import traceback

from .algebra import ResourceLimit
from .cli import example, peak_memory_bytes, runtime_record, raw_input
from .elementary import FactorizationIncomplete
from .unstable import UnstableFactorizationIncomplete
from .pipeline import decouple, InvalidCSS, LatticeObstruction, lattice_policy
from .serialization import save, load
from .verify import verify
from ..algebra.modules import get_backend


def main(argv=None):
    parser=argparse.ArgumentParser(prog='css-decouple-2d-circuit',description=__doc__)
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
            sub.add_argument('--factorization',choices=('auto','suslin','fast'),default='auto')
            sub.add_argument('--factor-timeout',type=float,default=120)
            sub.add_argument('--max-steps',type=int,default=1000000)
            sub.add_argument('--max-products',type=int,default=30000000)
            sub.add_argument('--max-patches',type=int,default=128)
            sub.add_argument('--max-field-trials',type=int,default=128)
            sub.add_argument('--seed',type=int,default=0)
            if command!='compile':
                sub.add_argument('--max-index',type=int,default=4096)
                sub.add_argument('--max-matrix-size',type=int,default=3000)
                sub.add_argument('--lattice')
                sub.add_argument('--lattice-mode',choices=('auto','fixed'))
                sub.add_argument('--qca-only',action='store_true')
                if command=='example':sub.add_argument('--prime',type=int,default=2)
    args=parser.parse_args(argv)
    start=time.perf_counter();output=None;output_owned=False;phase='initialization';qca_verified=False
    limits=dict(max_factors=args.max_factors,max_families=args.max_families,max_matrix_terms=args.max_terms)
    def progress(stage,details):
        if output_owned:
            with (output/'progress.jsonl').open('a',encoding='utf-8',newline='\n') as stream:
                stream.write(json.dumps({'stage':stage,'details':details,
                    'wall_seconds':time.perf_counter()-start,'peak_memory_bytes':peak_memory_bytes()})+'\n')
    try:
        from .code_circuits import compile_qca_no_ancilla, verify_no_ancilla_result
        if args.command=='verify':
            phase='independent no-input-ancilla verification'
            raw=load(args.input) if args.input else {}
            if args.input:raw_input(raw)
            report=verify_no_ancilla_result(load(args.source),expected_HX=raw.get('HX'),
                                           expected_HZ=raw.get('HZ'),**limits)
            report.update(wall_seconds=time.perf_counter()-start,peak_memory_bytes=peak_memory_bytes())
            if args.report:save(report,args.report)
            print(json.dumps(report));return 0
        output=Path(args.output);output.mkdir(parents=True,exist_ok=False);output_owned=True
        save(runtime_record(),output/'environment.json')
        if args.command=='compile':
            result=load(args.source);raw=result['original'];HX,HZ=raw_input(raw)
        else:
            if args.command=='example':
                HX,HZ=example(args.source,prime=args.prime)
                raw={'HX':HX,'HZ':HZ,'p':HX.ring.p,'D':2}
            else:
                raw=load(args.source);HX,HZ=raw_input(raw)
            save(raw,output/'input.json')
            phase='raw checks to original-register QCA'
            basis=json.loads(args.lattice) if args.lattice else None
            result=decouple(HX,HZ,compile_circuit=False,
                lattice_basis=basis,lattice_mode=lattice_policy(args.lattice_mode,basis),
                max_index=args.max_index,max_matrix_size=args.max_matrix_size,
                max_terms=args.max_terms,progress=progress)
        if not (output/'input.json').exists():save(raw,output/'input.json')
        get_backend(HX.ring).close()
        phase='QCA serialization and independent verification'
        save(result,output/'qca.json')
        qca_report=verify(load(output/'qca.json'),expected_HX=HX,expected_HZ=HZ,
                          max_families=args.max_families,max_matrix_terms=args.max_terms)
        qca_verified=True
        save(qca_report,output/'qca-verification.json');get_backend(HX.ring).close()
        if args.command in ('run','example') and args.qca_only:
            status={'status':'qca_complete','qca_verified':True,'circuit_requested':False,
                    'input_ancillas':0,'wall_seconds':time.perf_counter()-start}
            save(status,output/'status.json');print(json.dumps(status));return 0
        phase='fixed-register elementary factorization and target correction'
        artifact=compile_qca_no_ancilla(result,factorization=args.factorization,
            timeout=args.factor_timeout,max_steps=args.max_steps,max_products=args.max_products,
            max_patches=args.max_patches,max_field_trials=args.max_field_trials,seed=args.seed,
            verify_source=False,progress=progress,**limits)
        phase='circuit serialization and independent verification'
        save(artifact,output/'circuit-result.json')
        report=verify_no_ancilla_result(load(output/'circuit-result.json'),
                                       expected_HX=HX,expected_HZ=HZ,**limits)
        if not report['verified'] or report['input_ancillas']!=0:
            raise ArithmeticError('No-input-ancilla contract failed')
        save(report,output/'verification.json')
        status={**report,'status':'complete','wall_seconds':time.perf_counter()-start,
                'peak_memory_bytes':peak_memory_bytes(),'factorization':args.factorization}
        save(status,output/'status.json');print(json.dumps(status));return 0
    except Exception as exc:
        status={'status':'lattice_obstruction' if isinstance(exc,LatticeObstruction) else
                'factorization_incomplete' if isinstance(exc,(FactorizationIncomplete,UnstableFactorizationIncomplete)) else
                'resource_incomplete' if isinstance(exc,(ResourceLimit,MemoryError)) else
                'invalid_input' if isinstance(exc,InvalidCSS) else 'failed',
                'phase':phase,'exception':type(exc).__name__,'error':str(exc),
                'qca_verified':qca_verified,
                'wall_seconds':time.perf_counter()-start,'peak_memory_bytes':peak_memory_bytes(),
                'traceback':traceback.format_exc()}
        if output_owned:
            if (output/'qca.json').exists():
                save({'qca_artifact':'qca.json','required_input_ancillas':0,
                      'qca_verified':qca_verified,
                      'remaining_task':('factor on the original register and verify the standard-code target'
                                        if qca_verified else 'validate or reconstruct the saved QCA before factorization'),
                      'reason':str(exc),'no_nonexistence_claim':True},output/'remaining-equation.json')
            if not (output/'status.json').exists():save(status,output/'status.json')
        print(json.dumps(status),file=sys.stderr);return 2


if __name__=='__main__':raise SystemExit(main())
