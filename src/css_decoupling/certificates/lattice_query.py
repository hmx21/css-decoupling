"""Independent input-bound verification of the two-layer lattice criterion."""
from css_decoupling.algebra.laurent import Ring
from css_decoupling.lattice.retained import Lattice
from css_decoupling.decoupling.gluing import GluingClass
from .analysis import verify_analysis
from .comparison import _keys, _int, _matrix_from_dict, VerificationReport


def verify_lattice_decision(data):
    if hasattr(data, "certificate"):
        data = data.certificate()
    checks, errors = [], []
    try:
        _keys(data, ("schema", "version", "analysis", "basis", "decouplable", "reason"), "lattice decision")
        if data["schema"] != "css-decoupling.lattice-decision" or _int(data["version"], "version") != 1:
            raise ValueError("Unsupported lattice decision schema")
        if not isinstance(data["decouplable"], bool):
            raise ValueError("Decouplable must be boolean")
        source = data["analysis"]
        report = verify_analysis(source)
        if not report.valid:
            raise ValueError("Invalid source analysis: " + "; ".join(report.errors))
        ring = Ring(source["ring"]["prime"], source["ring"]["dimension"])
        layers = source["layers"]
        if len(layers) > 2:
            raise ValueError("At most two layers are allowed for this criterion")
        candidate = Lattice(data["basis"], max_index=None)
        if candidate.D != ring.D:
            raise ValueError("Candidate dimension mismatch")
        # Verify the actual layer actions instead of trusting kernel metadata.
        from css_decoupling.lattice.retained import matrix_power
        from css_decoupling.algebra.laurent import eye
        trivial = True
        for layer in layers:
            actions = [_matrix_from_dict(a, ring, "action") for a in layer["actions"]]
            for column in zip(*candidate.basis):
                action = eye(ring, actions[0].nrows)
                for a, exponent in zip(actions, column):
                    action = action * matrix_power(a, exponent)
                trivial = trivial and action == eye(ring, action.nrows)
        if not trivial:
            expected, reason = False, "nontrivial-layer-translation"
        elif len(layers) < 2:
            expected, reason = True, "trivial-layers-without-gluing"
        else:
            attachment = source["attachments"][0]
            base = Lattice(attachment["basis"], max_index=None)
            transition = base.relative_basis(candidate)
            upper = attachment["gluing"]["upper"]
            dimensions = {l["degree"]: _matrix_from_dict(l["representatives"], ring, "representative").ncols
                          for l in layers}
            classes = [GluingClass(upper, c["lower"], dimensions[upper], dimensions[c["lower"]],
                                   _matrix_from_dict(c["coefficients"], ring, "coefficient"))
                       for c in attachment["gluing"]["classes"]]
            expected = all(c.restrict(transition).is_zero for c in classes)
            reason = "vanishing-gluing" if expected else "nonzero-gluing"
        if data["decouplable"] != expected or data["reason"] != reason:
            raise ValueError("Claim disagrees with the actual translation/gluing obstruction")
        checks.append("input-bound two-layer translation and exterior restriction criterion")
    except (ValueError, TypeError, KeyError, IndexError, AttributeError) as exc:
        errors.append(str(exc))
    return VerificationReport(not errors, tuple(checks), tuple(errors), "two-layer-lattice-decision")
