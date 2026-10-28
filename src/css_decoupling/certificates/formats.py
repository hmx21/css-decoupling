"""Stable format identifiers; Python module moves do not change saved JSON."""
COMPLETE_SCHEMA = "css-decoupling.general.complete"
DUAL_SCHEMA = "css-decoupling.general.dual"
DUAL_TARGET_BASIS = "dual-of-inner-canonical-toric-stack"
DUAL_SCOPE = "original-CSS-to-dual-toric-stack-circuit"
DUAL_ANCILLA_KEYS = ("source_X_fixed", "source_Z_fixed", "target_X_fixed", "target_Z_fixed")


def dual_ancillas(ancillas):
    return {
        "source_X_fixed": ancillas.source_degree_minus_one,
        "source_Z_fixed": ancillas.source_degree_zero,
        "target_X_fixed": ancillas.target_degree_minus_one,
        "target_Z_fixed": ancillas.target_degree_zero,
    }
