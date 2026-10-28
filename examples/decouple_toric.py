"""Run either algorithm on a prime-qudit toric code and save exact evidence."""
import argparse
from css_decoupling import Ring, decouple, save_result, verify_result
from css_decoupling.models.examples import toric_css


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dimension',type=int,default=2)
    parser.add_argument('--prime',type=int,default=3)
    parser.add_argument('--algorithm',choices=('auto','two_d','general'),default='auto')
    parser.add_argument('--circuit',action='store_true')
    parser.add_argument('--output',required=True,help='New JSON result path')
    args=parser.parse_args()
    HX,HZ=toric_css(Ring(args.prime,args.dimension),{1:1})
    result=decouple(HX,HZ,args.dimension,args.prime,algorithm=args.algorithm,
                   output='circuit' if args.circuit else 'qca')
    report=verify_result(result,expected_HX=HX,expected_HZ=HZ)
    if not report.valid:
        raise RuntimeError('; '.join(report.errors))
    save_result(result,args.output)
    print(result.algorithm,result.status,result.qca.qudits)
    if result.status=='circuit_incomplete':
        print(result.diagnostics)
        return 3
    return 0


if __name__=='__main__':
    raise SystemExit(main())
