"""Native two-dimensional CSS decoupling, with a signed prime-field extension."""
from .algebra import Ring, Poly, Matrix
from .pipeline import decouple, compile_result_circuit, InvalidCSS, LatticeObstruction
from .serialization import save, load
from .verify import verify, verify_file
from .direct_pipeline import compile_qca_direct, verify_direct_result
from .code_circuits import compile_qca_no_ancilla, verify_no_ancilla_result
from .unstable import factor_unstabilized

__all__ = ["Ring","Poly","Matrix","decouple","compile_result_circuit",
           "InvalidCSS","LatticeObstruction","save","load","verify","verify_file",
           "compile_qca_direct","verify_direct_result",
           "compile_qca_no_ancilla","verify_no_ancilla_result","factor_unstabilized"]
__version__ = "0.1.0"
