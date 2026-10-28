"""General D=3, p=3 example with nonzero gluing and source-bound verification."""
import argparse
from css_decoupling import Ring, decouple, save_result, verify_result
from css_decoupling.models.examples import glued_css


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,help='New JSON result path')
    args=parser.parse_args()
    HX,HZ=glued_css(Ring(3,3))
    result=decouple(HX,HZ,3,3,algorithm='general',output='circuit')
    report=verify_result(result,expected_HX=HX,expected_HZ=HZ)
    if not report.valid:
        raise RuntimeError('; '.join(report.errors))
    save_result(result,args.output)
    print(result.algorithm,result.status,result.qca.qudits)


if __name__=='__main__':
    main()
