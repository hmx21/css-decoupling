"""Independent command line construction and read-back verification."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import platform
import sys
import time
import traceback
from pathlib import Path

from .serialization import save, load, digest
from .pipeline import decouple, compile_result_circuit, InvalidCSS, LatticeObstruction, lattice_policy, check_shape
from .verify import verify, verify_circuit_binding
from ..errors import ResourceLimit
from ..algebra.modules import get_backend


def peak_memory_bytes():
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [("cb",wintypes.DWORD),("PageFaultCount",wintypes.DWORD)] + [
                (name,ctypes.c_size_t) for name in ("PeakWorkingSetSize","WorkingSetSize",
                    "QuotaPeakPagedPoolUsage","QuotaPagedPoolUsage","QuotaPeakNonPagedPoolUsage",
                    "QuotaNonPagedPoolUsage","PagefileUsage","PeakPagefileUsage")]
        counters = Counters(); counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.WinDLL("kernel32",use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi",use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
        if psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(counters),counters.cb):
            return counters.PeakWorkingSetSize
        return None
    import resource
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == "darwin" else value*1024


def example(name, *, prime=2):
    from .models import standard_code, COLOR_CODE_666, COLOR_CODE_488, PUBLISHED_BB_CODE_INSTANCES
    from .algebra import Ring
    ring = Ring(prime, 2)
    if prime != 2 and name not in ("toric", "stack", "product", "empty"):
        raise ValueError("The published color/BB catalog is qubit data; supply signed prime-field checks with run")
    if name in ("666","488"):
        return (COLOR_CODE_666 if name == "666" else COLOR_CODE_488).checks()
    if name == "toric":
        return standard_code(1, ring=ring)
    if name == "stack":
        return standard_code(3,product_x=1,product_z=2,ring=ring)
    if name == "product":
        return standard_code(0,product_x=2,product_z=3,ring=ring)
    if name == "empty":
        return standard_code(0,ring=ring)
    cases = {case.identifier:case for case in PUBLISHED_BB_CODE_INSTANCES}
    if name not in cases:
        from .catalog import additional_inputs, checks
        records=additional_inputs()
        if name in records:
            return checks(records[name])
        raise ValueError(f"Unknown example {name}")
    return cases[name].checks()


def runtime_record():
    return {"python":sys.version,"executable":sys.executable,"platform":platform.platform(),
            "installed_distributions":{d.metadata["Name"]:d.version for d in importlib.metadata.distributions()},
            "external_CAS_required":False}


def raw_input(raw):
    """Validate the entire raw-input contract, including external binding data."""
    if not isinstance(raw,dict) or not {"HX","HZ"} <= set(raw) or set(raw)-{"HX","HZ","p","D"}:
        raise ValueError("Unknown or missing raw-input parameter")
    HX,HZ=raw["HX"],raw["HZ"]
    check_shape(HX,HZ)
    if (type(raw.get("p",HX.ring.p)) is not int or type(raw.get("D",HX.ring.D)) is not int or
            raw.get("p",HX.ring.p) != HX.ring.p or raw.get("D",HX.ring.D) != HX.ring.D):
        raise ValueError("Raw domain metadata disagrees with matrices")
    return HX,HZ


def main(argv=None):
    parser = argparse.ArgumentParser(prog="css-decouple-2d")
    commands = parser.add_subparsers(dest="command",required=True)
    for command in ("example","run"):
        sub = commands.add_parser(command)
        sub.add_argument("source")
        sub.add_argument("--output",required=True,help="New output directory; existing directories are rejected")
        sub.add_argument("--qca-only",action="store_true")
        sub.add_argument("--lattice",help="JSON array of rows of the integer column-basis matrix")
        sub.add_argument("--lattice-mode",choices=("auto","fixed"),
                         help="auto selects the full translation kernel; fixed preserves --lattice exactly")
        if command == "example":
            sub.add_argument("--prime",type=int,default=2)
        sub.add_argument("--max-index",type=int,default=4096)
        sub.add_argument("--max-matrix-size",type=int,default=2048)
        sub.add_argument("--max-terms",type=int,default=2_000_000)
        sub.add_argument("--max-families",type=int,default=200000)
    sub = commands.add_parser("verify")
    sub.add_argument("artifact");sub.add_argument("--input");sub.add_argument("--report")
    sub.add_argument("--max-families",type=int,default=200000)
    sub.add_argument("--max-terms",type=int,default=1_000_000)
    sub = commands.add_parser("compile")
    sub.add_argument("artifact");sub.add_argument("--output",required=True)
    sub.add_argument("--max-families",type=int,default=200000)
    sub.add_argument("--max-terms",type=int,default=1_000_000)
    sub=commands.add_parser("catalog");sub.add_argument("--all",action="store_true")
    commands.add_parser("environment")
    args = parser.parse_args(argv)
    start = time.perf_counter()
    output = None
    output_owned = False
    phase = "initialization"
    def progress(stage, details):
        if output is not None:
            with (output/"progress.jsonl").open("a",encoding="utf8",newline="\n") as stream:
                stream.write(json.dumps({"stage":stage,"details":details,"wall_seconds":time.perf_counter()-start,
                                         "peak_memory_bytes":peak_memory_bytes()})+"\n")
    try:
        if args.command == "environment":
            print(json.dumps(runtime_record(),indent=2));return 0
        if args.command == "catalog":
            if args.all:
                from .catalog import table_records, identifier
                print(json.dumps([{**r,"identifier":identifier(r)} for r in table_records()],indent=2));return 0
            from .models import PUBLISHED_BB_CODE_INSTANCES
            print(json.dumps([case.as_reference_record() for case in PUBLISHED_BB_CODE_INSTANCES],indent=2));return 0
        if args.command == "verify":
            phase = "verification"
            expected = load(args.input) if args.input else {}
            if args.input:
                raw_input(expected)
            report = verify(load(args.artifact),expected_HX=expected.get("HX"),expected_HZ=expected.get("HZ"),
                            max_families=args.max_families,max_matrix_terms=args.max_terms)
            report.update(wall_seconds=time.perf_counter()-start,peak_memory_bytes=peak_memory_bytes())
            if args.report:
                save(report,args.report)
            print(json.dumps(report));return 0
        if args.command == "compile":
            result = load(args.artifact)
            verify(result,max_families=args.max_families,max_matrix_terms=args.max_terms)
            compile_result_circuit(result,max_families=args.max_families,max_matrix_terms=args.max_terms)
            verify(result,max_families=args.max_families,max_matrix_terms=args.max_terms)
            save(result,args.output)
            print(json.dumps({"circuit_status":result["circuit_status"]}));return 0
        output = Path(args.output)
        output.mkdir(parents=True,exist_ok=False)
        output_owned = True
        save(runtime_record(),output/"environment.json")
        if args.command == "example":
            HX,HZ = example(args.source,prime=args.prime)
        else:
            source = load(args.source)
            HX,HZ = raw_input(source)
        save({"HX":HX,"HZ":HZ,"p":HX.ring.p,"D":HX.ring.D},output/"input.json")
        phase = "QCA construction"
        basis = json.loads(args.lattice) if args.lattice else None
        result = decouple(HX,HZ,lattice_basis=basis,lattice_mode=lattice_policy(args.lattice_mode,basis),
            compile_circuit=False,max_index=args.max_index,max_matrix_size=args.max_matrix_size,max_terms=args.max_terms,
            progress=progress)
        # Large inverse elimination witnesses are construction scratch data.
        # The returned certificate owns every matrix it needs. Release caches
        # before serialization and force the QCA read-back to rebuild algebra.
        get_backend(HX.ring).close()
        phase="QCA serialization"
        progress(phase,{"artifact":"qca.json","transport":"matrix-table-v1"})
        save(result,output/"qca.json")
        phase = "QCA verification"
        progress(phase,{"artifact":"qca.json"})
        qca_report = verify(load(output/"qca.json"),expected_HX=HX,expected_HZ=HZ)
        get_backend(HX.ring).close()
        save(qca_report,output/"qca-verification.json")
        if not args.qca_only:
            phase = "circuit compilation"
            progress(phase,{"equation":"gate product = diag(psi_1,psi_inverse_1)","size":HX.ncols*result["lattice"]["index"]})
            compile_result_circuit(result,max_families=args.max_families,max_matrix_terms=args.max_terms)
            phase = "circuit verification"
            progress(phase,{"equation":"actual gates, inverse, Pauli and both stabilizer generation witnesses"})
            if result["circuit"] is not None:
                verify_circuit_binding(result,max_families=args.max_families,max_matrix_terms=args.max_terms)
        phase="result serialization"
        progress(phase,{"artifact":"result.json","transport":"matrix-table-v1"})
        save(result,output/"result.json")
        status = {"status":"complete" if result["circuit"] is not None else result["circuit_status"],
            "qca_verified":True,"circuit_verified":result["circuit"] is not None,
            "input_sha256":digest(result["original"]),"timings_seconds":result["timings_seconds"],
            "wall_seconds":time.perf_counter()-start,"peak_memory_bytes":peak_memory_bytes(),
            "resources":result["resources"],"circuit_resources":result["circuit"]["resources"] if result["circuit"] else None}
        save(status,output/"status.json");print(json.dumps(status));return 0
    except Exception as exc:
        status = {"status":"lattice_obstruction" if isinstance(exc,LatticeObstruction) else
                  "resource_incomplete" if isinstance(exc,(ResourceLimit,MemoryError)) else
                  "invalid_input" if isinstance(exc,InvalidCSS) else "failed",
                  "phase":phase,"exception":type(exc).__name__,"error":str(exc),
                  "wall_seconds":time.perf_counter()-start,"peak_memory_bytes":peak_memory_bytes(),
                  "traceback":traceback.format_exc()}
        if output_owned and output is not None and output.is_dir() and not (output/"status.json").exists():
            save(status,output/"status.json")
        print(json.dumps(status),file=sys.stderr);return 2


if __name__ == "__main__":
    raise SystemExit(main())
