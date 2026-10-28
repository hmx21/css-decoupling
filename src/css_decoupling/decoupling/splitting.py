"""Construct a splitting as a finite Laurent module equation.

After the profile splits, each chosen basis of H^j lifts to a cochain map
T_j -> C. Descending lifts can yield a much smaller exact solution. If a
chosen intermediate lift cannot extend, solve all lower lifts simultaneously;
failure of the greedy choice is never treated as a splitting obstruction.
Unknowns are Laurent matrix entries, not a searched degree box.
"""
from css_decoupling.algebra.laurent import eye, zeros
from css_decoupling.topology.complexes import (CochainComplex, CochainMap, koszul, toric_stack,
                        solve_matrix_equations)
from css_decoupling.algebra.modules import solve
from css_decoupling.errors import NoLaurentSolution, ResourceLimit


def _descending_lift(C, T, degree, reps, solver):
    """A positive shortcut; an unextendable choice requires the joint solve."""
    components = {degree: reps}
    for n in reversed(T.degrees):
        if n >= degree:
            continue
        rhs = components[n + 1] * T.d(n)
        try:
            value = solver(C.d(n), rhs)
        except (NoLaurentSolution, ResourceLimit):
            return None
        if value.ring != C.ring or value.shape != (C.rank(n), T.rank(n)) or C.d(n) * value != rhs:
            raise ArithmeticError("Descending lift failed its exact chain equation")
        components[n] = value
    CochainMap(T, C, components).validate()
    return {n: value for n, value in components.items() if n < degree}


def restrict_complex(C, lattice):
    return CochainComplex(C.ring,
                          {n: r * lattice.index for n, r in C.ranks.items()},
                          {n: lattice.restrict(d) for n, d in C.differentials.items()})


def splitting_map(C, representatives, *, max_unknowns=250000, window=False, solver=None,
                  max_total_rank=100000):
    """Return Phi:T->C inducing the supplied bases in all nonzero layers.

    Failure of the module equation is an obstruction to this splitting, not
    an instruction to enumerate polynomial degrees.  A resource exception is
    kept distinct from algebraic nonmembership.
    """
    if window:
        from css_decoupling.topology.complexes import koszul_window, toric_window
        C = CochainComplex(C.ring, {n: r for n, r in C.ranks.items() if n >= 0},
                           {n: d for n, d in C.differentials.items() if n >= 0})
    pieces = []
    for j, reps in sorted(representatives.items()):
        Tj = (koszul_window(C.ring, degree=j, copies=reps.ncols, lowest_degree=0, max_total_rank=max_total_rank)
              if window else koszul(C.ring, degree=j, copies=reps.ncols, max_total_rank=max_total_rank))
        if reps.shape != (C.rank(j), Tj.rank(j)) or C.d(j) * reps != zeros(C.ring, C.rank(j + 1), reps.ncols):
            raise ValueError("Layer representatives must be cycles of the stated rank")
        shapes = {n: (C.rank(n), Tj.rank(n)) for n in Tj.degrees if n < j}
        unknowns = sum(a * b for a, b in shapes.values())
        if max_unknowns is not None and unknowns > max_unknowns:
            raise ResourceLimit(f"Splitting map has {unknowns} Laurent unknowns; cap is {max_unknowns}")
        # Different copies of one toric layer have independent lift
        # equations, with exactly the same operator. Solve that smaller
        # presentation per copy so the module backend can reuse its basis;
        # do not build a block-diagonal Kronecker presentation t times larger.
        # A degree-one lift is just A X = B and already batches its RHS.
        copies = reps.ncols
        separate = copies > 1 and j > 1
        unit = (koszul_window(C.ring, degree=j, copies=1, lowest_degree=0,
                              max_total_rank=max_total_rank) if window else
                koszul(C.ring, degree=j, copies=1, max_total_rank=max_total_rank)) if separate else Tj
        unit_shapes = {n: (C.rank(n), unit.rank(n)) for n in unit.degrees if n < j}
        components = {n: zeros(C.ring, C.rank(n), Tj.rank(n)) for n in shapes}
        for copy in range(copies if separate else 1):
            equations = []
            copy_representatives = reps[:, copy:copy + 1] if separate else reps
            chosen_solver = solve if solver is None else solver
            lifted = _descending_lift(C, unit, j, copy_representatives, chosen_solver)
            for n in unit.degrees:
                if n >= j:
                    continue
                terms = [(C.d(n), n, eye(C.ring, unit.rank(n)))]
                rhs = zeros(C.ring, C.rank(n + 1), unit.rank(n))
                if n + 1 == j:
                    rhs = copy_representatives * unit.d(n)
                else:
                    terms.append((-eye(C.ring, C.rank(n + 1)), n + 1, unit.d(n)))
                equations.append((terms, rhs))
            if lifted is None:
                lifted = solve_matrix_equations(C.ring, unit_shapes, equations, chosen_solver)
            if separate:
                for n, matrix in lifted.items():
                    for (row, column), value in matrix.entries.items():
                        components[n][row, column * copies + copy] = value
            else:
                components = lifted
        components[j] = reps
        pieces.append(CochainMap(Tj, C, components))
    multiplicities = {j: reps.ncols for j, reps in representatives.items()}
    T = (toric_window(C.ring, multiplicities, lowest_degree=0, max_total_rank=max_total_rank) if window
         else toric_stack(C.ring, multiplicities, max_total_rank=max_total_rank))
    maps = {}
    for n in sorted(set(T.degrees) | set(C.degrees)):
        maps[n] = zeros(C.ring, C.rank(n), 0)
        for piece in pieces:
            maps[n] = maps[n].hstack(piece.at(n))
    return CochainMap(T, C, maps)
