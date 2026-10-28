"""Exact stable decoupling of prime-qudit CSS codes in arbitrary dimension."""
from .algebra.laurent import Ring, Poly, Matrix, zeros, eye, block, direct_sum
from .topology.complexes import CochainComplex, CochainMap, koszul, toric_stack, koszul_window, toric_window
from .decoupling.pipeline import decouple as decouple_general, DecouplingResult
from .api import decouple
from .result import UnifiedResult, CircuitView, save_result, load_result, verify_result
from .raw_input import save_css_input, load_css_input, css_input_dict
from .certificates.decoupling import verify_decoupling
from .certificates.io import save_input, save_decoupling, load_decoupling
from .circuits.qca import QCA
from .certificates.outputs import export_outputs, verify_qca_output, verify_circuit_output, verify_output_pair
from .certificates.output_io import save_decoupling_outputs, save_exported_outputs
from .decoupling.analysis import analyze_profile, ProfileAnalysis
from .certificates.analysis import save_analysis, verify_analysis
from .decoupling.gluing import gluing_classes, GluingClass, GluingStep
from .certificates.gluing import verify_gluing
from .lattice.kernel import translation_kernel
from .certificates.translation_kernel import verify_translation_kernel
from .algebra.projective import ProjectiveBasisUnavailable, free_basis
from .certificates.projective import verify_projective_basis
from .decoupling.lattice_selection import select_lattice, LatticeSelection
from .certificates.lattice_selection import verify_lattice_selection
from .decoupling.lattice_query import check_two_layer_lattice, LatticeDecision
from .certificates.lattice_query import verify_lattice_decision
from .topology.periodic import periodic_code, periodic_profile, periodic_from_gluing, PeriodicResult
from .certificates.periodic import verify_periodic
from .decoupling.cone_decoupling import ConePresentation, ConeDecouplingResult, decouple_cone
from .certificates.cone_decoupling import verify_cone_decoupling
from .lattice.retained import Lattice
from .lattice.subspaces import subspace_preimage
from .decoupling.criterion import check_lattice, LatticeCriterion
from .certificates.criterion import verify_lattice_criterion
from .topology.periodic import periodic_layer, PeriodicLayerResult
from .certificates.periodic import verify_periodic_layer
from .lattice.koszul_comparison import koszul_comparison, KoszulComparison
from .certificates.koszul_comparison import verify_koszul_comparison
from .decoupling.coarse_graining import coarse_grain, CoarseGrainingResult
from .certificates.coarse_graining import verify_coarse_graining
from .decoupling.construction import decouple_complex, CircuitConstructionResult, construction_size
from .certificates.construction import verify_construction
from .topology.cohomology_handoff import build_cohomology_handoff, handoff_from_coarse_certificate
from .certificates.cohomology_handoff import verify_cohomology_handoff
from .certificates.schanuel import verify_schanuel, schanuel_certificate_dict
from .algebra.bounded_linear import BoundedLinearBackend
from .certificates.bounded_linear import verify_bounded_linear, verify_trivial_cohomology
from .topology.trivial_cohomology import trivial_cohomology, trivial_cohomology_layer
from .errors import ResourceLimit, BackendUnavailable, NoLaurentSolution, InfiniteLayer

__version__ = "0.1.0"
__all__ = ["Ring", "Poly", "Matrix", "zeros", "eye", "block", "direct_sum",
           "CochainComplex", "CochainMap", "koszul", "toric_stack", "koszul_window", "toric_window", "decouple",
           "decouple_general", "UnifiedResult", "CircuitView", "save_result", "load_result", "verify_result",
           "save_css_input", "load_css_input", "css_input_dict",
           "verify_decoupling", "DecouplingResult", "save_input", "save_decoupling", "load_decoupling",
           "QCA", "export_outputs", "verify_qca_output", "verify_circuit_output", "verify_output_pair",
           "save_decoupling_outputs", "save_exported_outputs",
           "ResourceLimit", "BackendUnavailable", "NoLaurentSolution", "InfiniteLayer",
           "analyze_profile", "ProfileAnalysis", "save_analysis", "verify_analysis",
           "gluing_classes", "GluingClass", "GluingStep", "verify_gluing",
           "translation_kernel", "verify_translation_kernel",
           "ProjectiveBasisUnavailable", "free_basis", "verify_projective_basis",
           "Lattice", "subspace_preimage", "select_lattice", "LatticeSelection", "verify_lattice_selection",
           "check_two_layer_lattice", "LatticeDecision", "verify_lattice_decision",
           "periodic_code", "periodic_profile", "periodic_from_gluing", "PeriodicResult", "verify_periodic",
           "ConePresentation", "ConeDecouplingResult", "decouple_cone", "verify_cone_decoupling",
           "check_lattice", "LatticeCriterion", "verify_lattice_criterion",
           "periodic_layer", "PeriodicLayerResult", "verify_periodic_layer",
           "koszul_comparison", "KoszulComparison", "verify_koszul_comparison",
           "coarse_grain", "CoarseGrainingResult", "verify_coarse_graining",
           "decouple_complex", "CircuitConstructionResult", "construction_size", "verify_construction",
           "build_cohomology_handoff", "handoff_from_coarse_certificate", "verify_cohomology_handoff",
           "verify_schanuel", "schanuel_certificate_dict", "BoundedLinearBackend",
           "verify_bounded_linear", "verify_trivial_cohomology", "trivial_cohomology", "trivial_cohomology_layer"]
