"""Verify a complete translation-kernel certificate without discrete logs.

Membership of the returned basis is insufficient.  Orthogonal cyclic
components give *equivalences* with integer congruences; the unipotent ideal
filtration gives exact successive kernels.  Reconstructing their intersection
therefore checks the full kernel and its index, not just a contained lattice.
"""
from math import gcd, prod
from numbers import Integral

from sympy import Matrix as ZMatrix, isprime
from sympy.matrices.normalforms import hermite_normal_form

from css_decoupling.lattice.finite_algebra import identity, multiply, power, add, sub
from css_decoupling.lattice.kernel import (
    SCHEMA, _actions, _positive_integer, _intersect_character, _intersect_unipotent,
)


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _decode(data, n, p):
    if not isinstance(data, list):
        raise ValueError("Certificate matrix must be a sparse-entry list")
    out, occupied = [0] * (n * n), set()
    for entry in data:
        if not isinstance(entry, list) or len(entry) != 3:
            raise ValueError("Malformed certificate matrix entry")
        i, j, c = [_integer(value, "matrix entry") for value in entry]
        if not (0 <= i < n and 0 <= j < n and 0 < c < p) or (i, j) in occupied:
            raise ValueError("Noncanonical certificate matrix entry")
        occupied.add((i, j))
        out[i * n + j] = c
    return tuple(out)


def _verify_prime(q, proof):
    if q < 2 ** 64:
        if not isprime(q):
            raise ValueError("Invalid cyclic-order prime factor")
        return
    if not isinstance(proof, dict) or _integer(proof.get("prime"), "certified prime") != q:
        raise ValueError("Large cyclic-order factor has no exact primality certificate")
    raw, children = proof.get("factors"), proof.get("children")
    if not isinstance(raw, list) or not isinstance(children, list):
        raise ValueError("Malformed primality certificate")
    factors, child_map = {}, {}
    for child in children:
        if not isinstance(child, dict):
            raise ValueError("Malformed primality subcertificate")
        r = _positive_integer(child.get("prime"), "certified prime")
        if r >= q or r in child_map:
            raise ValueError("Invalid primality-certificate recursion")
        child_map[r] = child
    for pair in raw:
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError("Malformed primality factor")
        r, e = [_positive_integer(value, "primality factor") for value in pair]
        if r in factors or r >= q:
            raise ValueError("Invalid primality factor")
        _verify_prime(r, child_map.get(r))
        factors[r] = e
    if prod(r ** e for r, e in factors.items()) != q - 1:
        raise ValueError("Incomplete primality-certificate factorization")
    a = _positive_integer(proof.get("base"), "primality witness")
    if not 1 < a < q or pow(a, q - 1, q) != 1 or any(pow(a, (q - 1) // r, q) == 1 for r in factors):
        raise ValueError("Primality order witness failed")


def verify_translation_kernel(layers, D, certificate, *, max_linear_entries=2_000_000):
    """Raise on a malformed/incomplete kernel witness; return True on success.

    ``layers`` bind the witness to actual translation matrices.  Unlike the
    constructor this function performs neither field factorization nor DLP.
    Standard exact matrix and GF(p) echelon routines are shared arithmetic.
    """
    D = _positive_integer(D, "D")
    max_linear_entries = _positive_integer(max_linear_entries, "max_linear_entries")
    p, source = _actions(layers, D, max_linear_entries=max_linear_entries)
    if not isinstance(certificate, dict) or certificate.get("schema") != SCHEMA:
        raise ValueError("Unknown translation-kernel certificate schema")
    if _integer(certificate.get("D"), "D") != D:
        raise ValueError("Translation-kernel dimension does not match input")
    claimed_p = certificate.get("p")
    if (p is None and claimed_p is not None) or (p is not None and _integer(claimed_p, "p") != p):
        raise ValueError("Translation-kernel characteristic does not match input")
    records = certificate.get("layers")
    if not isinstance(records, list) or len(records) != len(source):
        raise ValueError("Translation-kernel certificate omits or adds a layer")
    B = ZMatrix.eye(D)
    for (degree, n, actions), record in zip(source, records):
        if not isinstance(record, dict) or _integer(record.get("degree"), "degree") != degree:
            raise ValueError("Translation-kernel layer degree mismatch")
        if _integer(record.get("dimension"), "dimension") != n:
            raise ValueError("Translation-kernel layer dimension mismatch")
        components = record.get("components")
        if not isinstance(components, list) or not components:
            raise ValueError("Cyclic-component decomposition is missing")
        one, zero = identity(n), (0,) * (n * n)
        semisimple, projectors = [zero] * D, []
        for component in components:
            if not isinstance(component, dict):
                raise ValueError("Malformed cyclic component")
            projector = _decode(component.get("projector"), n, p)
            generator = _decode(component.get("generator"), n, p)
            if not any(projector) or multiply(projector, projector, n, p) != projector:
                raise ValueError("Cyclic component does not have a nonzero idempotent")
            if any(multiply(projector, other, n, p) != zero or
                   multiply(other, projector, n, p) != zero for other in projectors):
                raise ValueError("Cyclic-component projectors are not orthogonal")
            projectors.append(projector)
            if multiply(projector, generator, n, p) != generator or multiply(generator, projector, n, p) != generator:
                raise ValueError("Cyclic generator is not supported on its component")
            order = _positive_integer(component.get("order"), "cyclic order")
            if gcd(order, p) != 1:
                raise ValueError("Semisimple cyclic order is divisible by the characteristic")
            raw_factors = component.get("prime_factors")
            if not isinstance(raw_factors, list):
                raise ValueError("Missing cyclic-order factorization")
            factors = {}
            proofs = component.get("prime_certificates", [])
            if not isinstance(proofs, list):
                raise ValueError("Malformed cyclic-order primality certificates")
            proof_map = {}
            for proof in proofs:
                if not isinstance(proof, dict):
                    raise ValueError("Malformed cyclic-order primality certificate")
                q = _positive_integer(proof.get("prime"), "certified prime")
                if q in proof_map:
                    raise ValueError("Duplicate cyclic-order primality certificate")
                proof_map[q] = proof
            for pair in raw_factors:
                if not isinstance(pair, list) or len(pair) != 2:
                    raise ValueError("Malformed cyclic-order factor")
                q, exponent = [_positive_integer(value, "prime factor") for value in pair]
                if q in factors:
                    raise ValueError("Invalid cyclic-order prime factor")
                _verify_prime(q, proof_map.get(q))
                factors[q] = exponent
            if prod(q ** exponent for q, exponent in factors.items()) != order:
                raise ValueError("Incomplete cyclic-order prime factorization")
            if power(generator, order, n, p, one=projector) != projector or any(
                    power(generator, order // q, n, p, one=projector) == projector for q in factors):
                raise ValueError("Cyclic generator does not have the certified exact order")
            logs = component.get("logs")
            if not isinstance(logs, list) or len(logs) != D:
                raise ValueError("Wrong number of cyclic-component logarithms")
            logs = [_integer(value, "discrete logarithm") for value in logs]
            if any(value < 0 or value >= order for value in logs):
                raise ValueError("Noncanonical discrete logarithm")
            for i in range(D):
                semisimple[i] = add(semisimple[i], power(generator, logs[i], n, p, one=projector), p)
            B = _intersect_character(B, logs, order)
        total = zero
        for projector in projectors:
            total = add(total, projector, p)
        if total != one:
            raise ValueError("Cyclic-component projectors do not sum to identity")
        raw_unipotent = record.get("unipotent_actions")
        if not isinstance(raw_unipotent, list) or len(raw_unipotent) != D:
            raise ValueError("Missing unipotent actions")
        unipotent = [_decode(matrix, n, p) for matrix in raw_unipotent]
        for i in range(D):
            if any(power(sub(unipotent[i], one, p), n, n, p)):
                raise ValueError("Certified unipotent action is not unipotent")
            if multiply(semisimple[i], unipotent[i], n, p) != actions[i]:
                raise ValueError("Cyclic/unipotent factors do not reconstruct the input")
            for j in range(D):
                if multiply(semisimple[i], unipotent[j], n, p) != multiply(unipotent[j], semisimple[i], n, p):
                    raise ValueError("Semisimple and unipotent factors do not commute")
                if multiply(unipotent[i], unipotent[j], n, p) != multiply(unipotent[j], unipotent[i], n, p):
                    raise ValueError("Unipotent factors do not commute")
        B, _ = _intersect_unipotent(B, unipotent, n, p, max_linear_entries=max_linear_entries)
    raw_basis = certificate.get("basis_columns")
    if not isinstance(raw_basis, list) or len(raw_basis) != D or any(
            not isinstance(row, list) or len(row) != D for row in raw_basis):
        raise ValueError("Malformed kernel lattice basis")
    claimed = ZMatrix([[_integer(value, "lattice entry") for value in row] for row in raw_basis])
    if claimed.det() == 0 or hermite_normal_form(claimed) != B:
        raise ValueError("Claimed lattice is not the complete translation kernel")
    if _positive_integer(certificate.get("index"), "index") != abs(int(B.det())):
        raise ValueError("Claimed translation-kernel index is incorrect")
    return True
