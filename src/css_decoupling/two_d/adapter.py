"""Retained explicit 2D wire adapter; the public dispatcher is css_decoupling.api."""
from .pipeline import decouple
from .serialization import decode
from sympy import isprime


def dispatch_contract(payload):
    """Strict explicit opt-in conversion; never silently discard parameters.

    Cross-package matrices use the canonical JSON wire representation. This
    function does not import or modify the high-dimensional public API.
    """
    allowed = {"algorithm","D","p","HX","HZ","lattice_basis","lattice_mode","options"}
    if set(payload)-allowed:
        raise ValueError(f"Unsupported contract fields: {sorted(set(payload)-allowed)}")
    algorithm = payload.get("algorithm")
    if algorithm not in ("arxiv-2608.09915v1-native-stabilized", "two_d"):
        raise ValueError("Two-dimensional dispatch requires explicit algorithm selection")
    if (type(payload.get("D")) is not int or payload["D"] != 2 or
            type(payload.get("p")) is not int or not isprime(payload["p"])):
        raise ValueError("The two-dimensional adapter requires D=2 and prime p")
    if algorithm == "arxiv-2608.09915v1-native-stabilized" and payload["p"] != 2:
        raise ValueError("The legacy paper adapter is qubit-only; use algorithm='two_d'")
    HX,HZ = decode(payload["HX"]),decode(payload["HZ"])
    if HX.ring.p != payload["p"] or HX.ring.D != payload["D"]:
        raise ValueError("Adapter domain metadata disagrees with matrices")
    options=dict(payload.get("options",{}))
    if algorithm == "arxiv-2608.09915v1-native-stabilized":
        options.setdefault("compile_circuit",True)
    return decouple(HX,HZ,lattice_basis=payload.get("lattice_basis"),
                    lattice_mode=payload.get("lattice_mode"),**options)
