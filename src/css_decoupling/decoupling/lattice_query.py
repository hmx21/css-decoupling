"""Lightweight two-layer lattice queries, with the source analysis attached."""
from dataclasses import dataclass
from copy import deepcopy

from css_decoupling.errors import BackendUnavailable
from css_decoupling.lattice.retained import Lattice


@dataclass
class LatticeDecision:
    analysis: object
    lattice: Lattice
    decouplable: bool
    reason: str
    restricted_classes: tuple

    def certificate(self):
        return {
            "schema": "css-decoupling.lattice-decision", "version": 1,
            "analysis": self.analysis.certificate() if hasattr(self.analysis, "certificate") else deepcopy(self.analysis),
            "basis": [list(row) for row in self.lattice.basis],
            "decouplable": self.decouplable, "reason": self.reason,
        }


def check_two_layer_lattice(analysis, lattice_basis, *, max_exterior_entries=2000000):
    """Decide Corollary 2 without expanding the candidate cell.

    The finite-field calculation uses the first attachment in a fully verified
    source analysis. The candidate must lie inside the lattice on which that
    attachment was computed. For arbitrary multilevel profiles use
    ``analyze_profile(..., lattice_basis=...)``, which checks stages on that
    exact lattice and raises ``NoLaurentSolution`` at the first obstruction.
    """
    from css_decoupling.certificates.analysis import verify_analysis
    from css_decoupling.certificates.comparison import _matrix_from_dict
    from css_decoupling.algebra.laurent import Ring, eye
    from css_decoupling.lattice.retained import matrix_power
    from .gluing import GluingClass
    data = analysis.certificate() if hasattr(analysis, "certificate") else deepcopy(analysis)
    report = verify_analysis(data)
    if not report.valid:
        raise ValueError("Invalid source analysis: " + "; ".join(report.errors))
    layers = data["layers"]
    ring = Ring(data["ring"]["prime"], data["ring"]["dimension"])
    if len(layers) > 2:
        raise BackendUnavailable("The lightweight criterion requires at most two nonzero layers")
    lattice = Lattice(lattice_basis, max_index=None)
    if lattice.D != ring.D:
        raise ValueError("Candidate lattice dimension differs from the CSS input")
    for layer in layers:
        actions = [_matrix_from_dict(a, ring, "action") for a in layer["actions"]]
        for vector in zip(*lattice.basis):
            action = eye(ring, actions[0].nrows)
            for a, exponent in zip(actions, vector):
                action = action * matrix_power(a, exponent)
            if action != eye(ring, action.nrows):
                return LatticeDecision(data, lattice, False, "nontrivial-layer-translation", ())
    if len(layers) < 2:
        return LatticeDecision(data, lattice, True, "trivial-layers-without-gluing", ())
    first = data["attachments"][0]
    base = Lattice(first["basis"], max_index=None)
    if not base.contains(lattice):
        raise BackendUnavailable("Candidate is outside the recorded gluing lattice; analyze on Lambda0 first")
    transition = base.relative_basis(lattice)
    dimensions = {l["degree"]: _matrix_from_dict(l["representatives"], ring, "representative").ncols for l in layers}
    upper = first["gluing"]["upper"]
    original = tuple(GluingClass(upper, c["lower"], dimensions[upper], dimensions[c["lower"]],
                                 _matrix_from_dict(c["coefficients"], ring, "coefficient"))
                     for c in first["gluing"]["classes"])
    classes = tuple(c.restrict(transition, max_exterior_entries=max_exterior_entries)
                    for c in original)
    vanishes = all(c.is_zero for c in classes)
    return LatticeDecision(data, lattice, vanishes,
                          "vanishing-gluing" if vanishes else "nonzero-gluing", classes)
