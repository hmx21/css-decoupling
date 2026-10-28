"""CSS free models, using the independently reusable algebra resolution layer.

``image_resolution`` and the historical exception imports are re-exported for
compatibility; their implementation lives in ``algebra.resolutions``.
"""
from css_decoupling.algebra.laurent import zeros
from css_decoupling.algebra.modules import kernel, solve
from css_decoupling.algebra.resolutions import image_resolution
from css_decoupling.errors import BackendUnavailable, ResourceLimit


def css_complex(HX, HZ, *, timeout=120, complete_negative=True, resolution_method="native"):
    """Free C with the original physical window unchanged.

    ``complete_negative=False`` retains degree -1 but avoids resolving lower
    stabilizer relations: the window construction only contracts its cone in
    nonnegative degrees and never needs those additional terms.
    """
    from css_decoupling.topology.complexes import CochainComplex
    if resolution_method != "native":
        raise ValueError("resolution_method must be 'native'")
    if HX.ring != HZ.ring or HX.ncols != HZ.ncols:
        raise ValueError("CSS checks must share their ring and physical columns")
    ring = HX.ring
    if HX * HZ.dagger() != zeros(ring, HX.nrows, HZ.nrows):
        raise ValueError("CSS checks do not commute")
    # Image equality, not a rank count over the rational function field.
    solve(HZ.dagger(), kernel(HX))
    solve(HX.dagger(), kernel(HZ))
    ranks = {-1: HZ.nrows, 0: HX.ncols, 1: HX.nrows}
    differentials = {-1: HZ.dagger(), 0: HX}
    positive = image_resolution(kernel(HX.dagger()), timeout=timeout, method=resolution_method)
    for k, differential in enumerate(positive, start=1):
        differentials[k] = differential.dagger()
        ranks[k + 1] = differential.ncols
    if complete_negative:
        negative = image_resolution(kernel(HZ.dagger()), timeout=timeout, method=resolution_method)
        for k, differential in enumerate(negative, start=2):
            differentials[-k] = differential
            ranks[-k] = differential.ncols
    C = CochainComplex(ring, ranks, differentials)
    C.validate()
    return C
