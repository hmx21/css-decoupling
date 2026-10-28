"""Raw CSS inputs for examples; model names never select solver paths."""

from .color_codes import (bcc_color_css, fcc_color_css, freudenthal_color_css,
                          freudenthal_color_geometry)
from .tricycle import tricycle_css, tricycle_case, TRICYCLE_CASES
from .stress import (glued_stress_css, rank_four_glued_css, stress_case,
                     lattice_relabel_css, redundant_css_checks)

__all__ = ["bcc_color_css", "fcc_color_css", "freudenthal_color_css",
           "freudenthal_color_geometry", "tricycle_css", "tricycle_case",
           "TRICYCLE_CASES", "glued_stress_css", "rank_four_glued_css", "stress_case",
           "lattice_relabel_css", "redundant_css_checks"]
