"""Construct the invertible lifts of Lemma 11 (Appendix C2) from exact comparison data.

This module is the certified last construction stage.  It does not obtain
comparison maps from arbitrary CSS checks or certify topological exactness.
"""
from __future__ import annotations

from dataclasses import dataclass

from css_decoupling.algebra.laurent import Matrix, block, direct_sum, eye, zeros
from css_decoupling.circuits.gates import Circuit, compile_theta


@dataclass(frozen=True)
class LiftData:
    """phi:T->C, psi:C->T and degree-one inverse-defect witnesses."""
    dC: Matrix
    dT: Matrix
    phi0: Matrix
    psi0: Matrix
    phi1: Matrix
    psi1: Matrix
    etaC: Matrix
    etaT: Matrix

    def validate(self) -> None:
        c1, c0 = self.dC.nrows, self.dC.ncols
        t1, t0 = self.dT.nrows, self.dT.ncols
        ring = self.dC.ring
        shapes = {"dT": (t1, t0), "phi0": (c0, t0), "psi0": (t0, c0),
                  "phi1": (c1, t1), "psi1": (t1, c1),
                  "etaC": (c0, c1), "etaT": (t0, t1)}
        for name, shape in shapes.items():
            value = getattr(self, name)
            if value.ring != ring or (value.nrows, value.ncols) != shape:
                raise ValueError(f"{name} has an incompatible ring or shape")
        identities = {
            "phi chain relation": (self.dC * self.phi0, self.phi1 * self.dT),
            "psi chain relation": (self.dT * self.psi0, self.psi1 * self.dC),
            "C inverse defect": (self.dC * self.etaC,
                                 eye(ring, c1) - self.phi1 * self.psi1),
            "T inverse defect": (self.dT * self.etaT,
                                 eye(ring, t1) - self.psi1 * self.phi1),
        }
        for name, (left, right) in identities.items():
            if left != right:
                raise ValueError(f"Invalid comparison data: {name}")


@dataclass(frozen=True)
class DiskAncillas:
    """Ranks of added D^0 (X-fixed) and D^-1 (Z-fixed) disk complexes."""
    source_degree_zero: int
    source_degree_minus_one: int
    target_degree_zero: int
    target_degree_minus_one: int


@dataclass(frozen=True)
class Stabilization:
    lifts: LiftData
    phihat: Matrix
    psihat: Matrix
    dChat: Matrix
    dThat: Matrix
    U: Matrix
    Uinv: Matrix
    V: Matrix
    Vinv: Matrix
    circuit: Circuit
    ancillas: DiskAncillas


def theta(f: Matrix, g: Matrix) -> Matrix:
    """Theta(f,g): A+B -> B+A with explicit inverse Theta(g,f)."""
    if f.ring != g.ring or (f.nrows, f.ncols) != (g.ncols, g.nrows):
        raise ValueError("Theta factors must have opposite shapes over one ring")
    return block([[f, eye(f.ring, f.nrows) - f * g],
                  [eye(f.ring, f.ncols), -g]])


def stabilize(lifts: LiftData) -> Stabilization:
    """Validate witnesses and perform the paper's rectangular block construction."""
    lifts.validate()
    r = lifts.dC.ring
    c1, c0 = lifts.dC.nrows, lifts.dC.ncols
    t1, t0 = lifts.dT.nrows, lifts.dT.ncols
    phihat = block([[lifts.phi0, lifts.etaC], [lifts.dT, -lifts.psi1]])
    psihat = block([[lifts.psi0, lifts.etaT], [lifts.dC, -lifts.phi1]])
    dCp = direct_sum(lifts.dC, eye(r, t1))
    dTp = direct_sum(lifts.dT, eye(r, c1))
    dChat = dCp.hstack(zeros(r, c1 + t1, t0 + c1))
    dThat = dTp.hstack(zeros(r, t1 + c1, c0 + t1))
    return Stabilization(
        lifts=lifts, phihat=phihat, psihat=psihat, dChat=dChat, dThat=dThat,
        U=theta(psihat, phihat), Uinv=theta(phihat, psihat),
        V=theta(lifts.psi1, lifts.phi1), Vinv=theta(lifts.phi1, lifts.psi1),
        circuit=compile_theta(psihat, phihat),
        ancillas=DiskAncillas(t1, t0 + c1, c1, c0 + t1),
    )
