"""Z-sector decoupling by exact Fourier conjugation of an X-sector run.

The output qudit basis is the dual of the inner canonical toric stack.
A degree-j inner block is labeled TC_(D-j), but this module does not silently
identify its matrices with the primal canonical complement-subset basis.
Fourier conjugation exchanges X-fixed |+> and Z-fixed |0> disk ancillas.
Only SUM and SWAP layers remain after conjugating the complete circuit.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter

from css_decoupling.certificates.comparison import circuit_dict, _matrix_dict
from css_decoupling.certificates.dual import verify_dual_certificate
from css_decoupling.circuits.gates import Circuit
from css_decoupling.circuits.duality import dualize_circuit
from css_decoupling.algebra.modules import using_backend

from css_decoupling.certificates.formats import (
    DUAL_SCHEMA as SCHEMA, DUAL_TARGET_BASIS as TARGET_BASIS, dual_ancillas,
)


@dataclass
class DualDecouplingResult:
    HX: object
    HZ: object
    inner: object
    circuit: Circuit
    timings: dict

    @property
    def sector(self):
        return "Z"

    @property
    def lattice(self):
        return self.inner.lattice

    @property
    def qca(self):
        """Outer physical action, Fourier-conjugated like this result's circuit."""
        return self.inner.qca.fourier_dual()

    @property
    def profile_analysis(self):
        return replace(self.inner.profile_analysis, HX=self.HX, HZ=self.HZ, sector="Z")

    @property
    def coarse_graining_result(self):
        coarse = self.inner.coarse_graining_result
        return None if coarse is None else replace(coarse, HX=self.HX, HZ=self.HZ, sector="Z")

    @property
    def multiplicities(self):
        return dict(sorted((self.HX.ring.D - j, t) for j, t in self.inner.multiplicities.items()))

    @property
    def target_basis(self):
        return TARGET_BASIS

    @property
    def ancillas(self):
        return dual_ancillas(self.inner.stabilization.ancillas)

    @property
    def identity_checks(self):
        """Redundant identity generators now lie in the original Z sector."""
        return {"sector": "Z", "source": self.inner.free_C, "target": self.inner.free_T}

    @property
    def trace(self):
        return {"sector": "Z", "target_basis": TARGET_BASIS, "inner": self.inner.trace}

    def certificate(self):
        return {
            "schema": SCHEMA, "version": 1,
            "input": {"HX": _matrix_dict(self.HX), "HZ": _matrix_dict(self.HZ)},
            "target": {"basis": TARGET_BASIS,
                       "multiplicities": [[j, t] for j, t in self.multiplicities.items()]},
            "ancillas": self.ancillas,
            "identity_checks": self.identity_checks,
            "inner": self.inner.certificate(),
            "circuit": circuit_dict(self.circuit),
        }


def dual_decouple(HX, HZ, *, verify=True, **options):
    """Run the ordinary algorithm on (HZ,HX), then conjugate its actual circuit."""
    from css_decoupling.decoupling.pipeline import decouple

    # Keep an explicitly chosen backend active through the wrapper's final
    # verification, not just during the nested construction.
    if options.get("backend") is not None:
        options = dict(options)
        backend = options.pop("backend")
        with using_backend(backend, timeout=options.get("resolution_timeout", 120)):
            return dual_decouple(HX, HZ, verify=verify, **options)
    started = perf_counter()
    # The final dual verifier independently runs the complete inner verifier.
    # Repeating it here adds cost without checking an intervening mutation.
    from css_decoupling.errors import BackendUnavailable, ResourceLimit, InfiniteLayer
    try:
        inner = decouple(HZ, HX, sector="X", verify=False, **options)
    except (BackendUnavailable, ResourceLimit, InfiniteLayer) as exc:
        if hasattr(exc, "profile_analysis"):
            exc.profile_analysis = replace(exc.profile_analysis, HX=HX, HZ=HZ, sector="Z")
        if hasattr(exc, "coarse_graining_result"):
            exc.coarse_graining_result = replace(exc.coarse_graining_result, HX=HX, HZ=HZ, sector="Z")
        if hasattr(exc, "coarse_graining_partial"):
            exc.coarse_graining_partial = {**exc.coarse_graining_partial, "sector": "Z"}
        raise
    tick = perf_counter()
    circuit = dualize_circuit(inner.circuit)
    timings = dict(inner.timings)
    timings["dual_conjugation"] = perf_counter() - tick
    result = DualDecouplingResult(HX, HZ, inner, circuit, timings)
    if verify:
        tick = perf_counter()
        try:
            if inner.coarse_graining_result is not None:
                # The physical dual certificate does not include the complete
                # Algorithm 1/2 source and Schanuel certificate.
                from css_decoupling.certificates.construction import verify_construction
                construction = inner.trace["construction"]
                full = verify_construction(construction, linear_backend=construction.linear_backend,
                                           bounded_options=options.get("bounded_options"))
                if not full.valid:
                    raise ArithmeticError("Uniform dual construction failed verification: " + "; ".join(full.errors))
            report = verify_dual_certificate(result)
        except (BackendUnavailable, ResourceLimit) as exc:
            if result.coarse_graining_result is not None:
                exc.coarse_graining_result = result.coarse_graining_result
            raise
        if not report.valid:
            raise ArithmeticError("Dual decoupling failed verification: " + "; ".join(report.errors))
        timings["dual_verification"] = perf_counter() - tick
    timings["total"] = perf_counter() - started
    return result
