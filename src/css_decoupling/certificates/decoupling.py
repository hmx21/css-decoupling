"""Verify complete certificates independently of the construction pipeline."""
from css_decoupling.algebra.laurent import direct_sum, eye, zeros
from css_decoupling.algebra.modules import GeneralBackend, get_backend
from css_decoupling.topology.complexes import toric_window
from css_decoupling.lattice.retained import Lattice
from .comparison import (certificate_from_dict, verify_certificate, _matrix_from_dict,
                         _keys, _int, VerificationReport)
from .formats import COMPLETE_SCHEMA as SCHEMA


def verify_decoupling(result):
    """Authenticate original CSS input, lattice, toric target and actual circuit.

    This verifier needs no saved profile-success flag or construction cache.
    The final square and exactness independently imply the decoupling even
    without replaying the profile computation or projective stabilization.
    """
    if ((isinstance(result, dict) and result.get("schema") == "css-decoupling.general.dual")
            or getattr(result, "sector", None) == "Z"):
        from css_decoupling.certificates.dual import verify_dual_decoupling
        return verify_dual_decoupling(result)
    checks, errors = [], []
    try:
        if isinstance(result, dict):
            data = _keys(result, ("schema", "version", "input", "lattice", "multiplicities",
                                  "identity_checks", "stabilization"), "complete certificate")
            if data["schema"] != SCHEMA or _int(data["version"], "version") != 1:
                raise ValueError("Unsupported complete certificate schema")
            stable = certificate_from_dict(data["stabilization"])
            ring = stable.lifts.dC.ring
            inp = _keys(data["input"], ("HX", "HZ"), "input")
            HX = _matrix_from_dict(inp["HX"], ring, "input.HX")
            HZ = _matrix_from_dict(inp["HZ"], ring, "input.HZ")
            metadata = _keys(data["lattice"], ("basis_columns", "index", "representatives"), "lattice")
            lattice = Lattice(metadata["basis_columns"], max_index=None)
            _int(metadata["index"], "lattice index", 1)
            if not isinstance(metadata["representatives"], list):
                raise ValueError("Lattice representatives must be a list")
            for representative in metadata["representatives"]:
                if not isinstance(representative, list) or len(representative) != lattice.D:
                    raise ValueError("Lattice representative has an incompatible dimension")
                for coordinate in representative:
                    _int(coordinate, "lattice representative coordinate", 0)
            if lattice.metadata() != metadata:
                raise ValueError("Inconsistent lattice metadata")
            multiplicities = {}
            for pair in data["multiplicities"]:
                if not isinstance(pair, list) or len(pair) != 2:
                    raise ValueError("Malformed multiplicity")
                j, t = _int(pair[0], "degree", 1), _int(pair[1], "multiplicity", 1)
                if j in multiplicities:
                    raise ValueError("Duplicate profile degree")
                multiplicities[j] = t
            identity = _keys(data["identity_checks"], ("source", "target"), "identity checks")
            free_C = _int(identity["source"], "source identity checks", 0)
            free_T = _int(identity["target"], "target identity checks", 0)
        else:
            HX, HZ, lattice = result.HX, result.HZ, result.lattice
            multiplicities, free_C, free_T = result.multiplicities, result.free_C, result.free_T
            stable = result.stabilization
            ring = HX.ring
        if HX.ring != HZ.ring or HX.ncols != HZ.ncols or lattice.D != ring.D:
            raise ValueError("Input rings, qudit counts or lattice dimensions disagree")
        backend = get_backend(ring)
        if lattice.index > 1 and type(backend) is GeneralBackend:
            # This recognizes restrictions by complete matrix reconstruction,
            # not by a saved construction flag. Both original and expanded
            # equations are still verified; unknown/custom backends keep
            # their selected semantics, including strict bounded arithmetic.
            from css_decoupling.lattice.restriction_backend import RestrictionBackend
            backend = RestrictionBackend(backend, lattice)
        solve, kernel = backend.solve, backend.kernel
        if HX * HZ.dagger() != zeros(ring, HX.nrows, HZ.nrows):
            raise ValueError("Input checks fail commutation")
        solve(HZ.dagger(), kernel(HX))
        solve(HX.dagger(), kernel(HZ))
        checks.append("original CSS commutation and both exactness conditions")
        source = lattice.restrict(HX)
        target = toric_window(ring, multiplicities)
        if stable.lifts.dC != source.vstack(zeros(ring, free_C, source.ncols)):
            raise ValueError("Source differential is not the restricted input plus identity checks")
        if stable.lifts.dT != target.d(0).vstack(zeros(ring, free_T, target.rank(0))):
            raise ValueError("Target differential is not the declared toric stack plus identity checks")
        checks.append("restriction of original checks and canonical toric target")
        local = verify_certificate(stable)
        checks.extend(local.checks)
        errors.extend(local.errors)
        # Verify actual Z generators in both directions as an additional
        # check of the disk ordering and connection to the original HZ.
        c1, c0 = stable.lifts.dC.shape
        t1, t0 = stable.lifts.dT.shape
        ZC = direct_sum(lattice.restrict(HZ.dagger()), zeros(ring, t1, 0), eye(ring, t0 + c1))
        ZT = direct_sum(target.d(-1), zeros(ring, c1, 0), eye(ring, c0 + t1))
        solve(ZT, stable.U * ZC)
        solve(ZC, stable.Uinv * ZT)
        checks.append("Z stabilizer generators mapped onto one another in both directions")
    except (ValueError, TypeError, KeyError, AttributeError, IndexError) as exc:
        errors.append(str(exc))
    return VerificationReport(not errors, tuple(checks), tuple(errors), "original-CSS-to-toric-stack-circuit")
