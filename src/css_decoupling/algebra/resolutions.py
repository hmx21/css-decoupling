"""Native finite polynomial free resolutions, then exact Laurent localization.

The Schreyer algorithm uses only Python prime-field arithmetic. Its first map
may use different generators; both actual image-change matrices are retained
and checked after localization. Every successive kernel/image equality and
terminal injectivity is independently checked with the native Laurent backend.
"""
from css_decoupling.algebra.laurent import Matrix, zeros
from css_decoupling.algebra.modules import kernel, solve


def image_resolution(A, *, method="native", timeout=120, verify=True,
                     max_basis=100000, max_terms=2000000, max_pairs=1000000):
    """Return F0 -> ambient, F1 -> F0, ... resolving im(A).

    Zero terminal maps are omitted. Clearing each column by a Laurent unit
    preserves its image after localization. There is no assumed free basis
    for an arbitrary syzygy or projective summand. The only method is the
    native arbitrary-prime Schreyer construction. Limits cover construction;
    verification uses the selected native Laurent backend and its limits.
    """
    if method != "native":
        raise ValueError("Resolution method must be 'native'")
    if not A.entries:
        return [zeros(A.ring, A.nrows, 0)]
    polynomial = A.copy()
    columns, units = {}, {}
    for (i, j), value in A.entries.items():
        columns.setdefault(j, []).append((i, value))
    for j, entries in columns.items():
        support = [e for _, f in entries for e in f.terms]
        shift = tuple(max(0, -min((e[k] for e in support), default=0)) for k in range(A.ring.D))
        unit = A.ring.monomial(shift)
        units[j] = unit
        for i, value in entries:
            polynomial[i, j] = unit * value
    from css_decoupling.algebra.schreyer import polynomial_image_resolution
    native = polynomial_image_resolution(
        polynomial, timeout=timeout, max_basis=max_basis,
        max_terms=max_terms, max_pairs=max_pairs,
        return_image_witness=verify)
    maps = native["maps"] if verify else native
    if verify:
        # P=A*D for the diagonal column-unit matrix D. The native polynomial
        # identities P*T=G and G*S=P give Laurent identities A*(D*T)=G and
        # G*(S*D^-1)=A. Check products rather than trusting success flags or
        # re-solving these already constructed image-change witnesses.
        T, S = native["basis_in_input"], native["input_in_basis"]
        if T.ring != A.ring or S.ring != A.ring or T.shape != (A.ncols, maps[0].ncols) or (
                S.shape != (maps[0].ncols, A.ncols)):
            raise ArithmeticError("Native resolution image witness has wrong dimensions")
        T = Matrix(A.ring, T.nrows, T.ncols,
                   {(i, j): units.get(i, A.ring.one()) * f for (i, j), f in T.entries.items()})
        S = Matrix(A.ring, S.nrows, S.ncols,
                   {(i, j): f * units.get(j, A.ring.one()) ** -1 for (i, j), f in S.entries.items()})
        if A * T != maps[0] or maps[0] * S != A:
            raise ArithmeticError("Native resolution image witnesses failed their exact identities")
        for previous, following in zip(maps, maps[1:]):
            if previous * following != zeros(A.ring, previous.nrows, following.ncols):
                raise ArithmeticError("Free resolution is not a complex")
            solve(following, kernel(previous))
        if kernel(maps[-1]).entries:
            raise ArithmeticError("Free resolution does not terminate in an injection")
    return maps
