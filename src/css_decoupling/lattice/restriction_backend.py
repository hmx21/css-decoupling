"""Exact algebra for matrices recognized as restrictions of scalars.

The source ring R and retained-lattice ring S share the computational type
``Ring(p,D)``.  Their distinct meanings are fixed by the explicit lattice:
the S variable y_j acts as x raised to column j of its integer basis.  No
model name, saved comparison map, or claimed provenance establishes that
a matrix is a restriction: the entire reconstructed matrix must match.
"""
from bisect import bisect_left, bisect_right

from css_decoupling.algebra.laurent import Matrix, Poly, eye, zeros
from css_decoupling.algebra.modules import GeneralBackend
from css_decoupling.errors import NoLaurentSolution
from css_decoupling.lattice.retained import Lattice


def unpack_columns(A, lattice):
    """Inverse of ``lattice.pack`` for arbitrary columns over S.

    Rows use species-major, coset-minor order.  An entry y**e in coset r
    becomes x**(r+B*e); negative and skew basis directions are exact integer
    operations.  Columns are not replicated, so their count need not be a
    multiple of the lattice index.
    """
    if not isinstance(A, Matrix) or not isinstance(lattice, Lattice):
        raise TypeError("Unpacking requires a Laurent matrix and a Lattice")
    if A.ring.D != lattice.D or A.nrows % lattice.index:
        raise ValueError("Packed rows must match the ring and lattice index")
    basis = tuple(tuple(int(value) for value in row) for row in lattice.basis)
    entries = {}
    for (row, column), value in A.entries.items():
        species, coset = divmod(row, lattice.index)
        representative = lattice.representatives[coset]
        target = entries.setdefault((species, column), {})
        for exponent, coefficient in value.terms.items():
            old_exponent = tuple(
                representative[i] + sum(basis[i][j] * exponent[j] for j in range(lattice.D))
                for i in range(lattice.D))
            target[old_exponent] = (target.get(old_exponent, 0) + coefficient) % A.ring.p
    return Matrix(A.ring, A.nrows // lattice.index, A.ncols,
                  {position: Poly(A.ring, terms) for position, terms in entries.items()})


class RestrictionBackend(GeneralBackend):
    """Use an exact smaller source operator only after full reconstruction.

    Only the practical native ``GeneralBackend`` is supported.  In
    particular this adapter does not silently replace the strictly bounded
    backend, an external backend, or an unknown custom kernel algorithm.
    Its base kernel already verifies the complete polynomial Schreyer
    module and generator transfers, not just A*K=0.  Restriction of scalars
    is exact, so restricting that *complete* kernel yields the complete
    S-kernel; every source equation and expanded equation is also checked.

    Normal-form and quotient operations remain ordinary S-ring operations
    in the selected base backend.  They must not discard zero rows, since
    those rows represent free cokernel summands.
    """
    name = "restriction-exact-prime-laurent"

    def __init__(self, base_backend, lattice):
        if type(base_backend) is not GeneralBackend:
            raise TypeError("RestrictionBackend requires the native practical GeneralBackend")
        if not isinstance(lattice, Lattice):
            raise TypeError("RestrictionBackend requires an explicit Lattice")
        if lattice.D != base_backend.ring.D:
            raise ValueError("Backend ring and restriction lattice dimensions differ")
        super().__init__(base_backend.ring)
        self.base_backend = base_backend
        # Snapshot the basis, so later changes to the caller's Lattice do
        # not change the embedding or invalidate recognition caches.
        self.lattice = Lattice([[int(x) for x in row] for row in lattice.basis],
                               max_index=None)
        self._source_matrices = {}

    def source_matrix(self, A):
        """Return A0 only if the actual identity Res(A0)=A holds; else None."""
        self._check(A)
        index = self.lattice.index
        if A.nrows % index or A.ncols % index:
            return None
        key = A.key()
        if key not in self._source_matrices:
            # The first representative is zero for the HNF-box convention.
            # This extracts exactly the coset-zero column of each species.
            columns = A[:, range(0, A.ncols, index)]
            source = unpack_columns(columns, self.lattice)
            if self.lattice.pack(source) != columns:
                raise ArithmeticError("Restriction reconstruction failed the pack/unpack identity")
            self._source_matrices[key] = (
                source.copy() if self.lattice.restrict(source) == A else None)
        source = self._source_matrices[key]
        return None if source is None else source.copy()

    def _trim_zero_rows(self, A, B=None):
        """Delete only literal zero constraints, never zero cokernel rows."""
        active = sorted({row for row, _ in A.entries})
        if len(active) == A.nrows:
            return A, B
        positions = {row: index for index, row in enumerate(active)}
        if B is not None and any(row not in positions for row, _ in B.entries):
            raise NoLaurentSolution("A nonzero RHS occurs in a zero constraint row")
        operator = Matrix(self.ring, len(active), A.ncols,
                          {(positions[row], column): value
                           for (row, column), value in A.entries.items()})
        right = None if B is None else Matrix(
            self.ring, len(active), B.ncols,
            {(positions[row], column): value for (row, column), value in B.entries.items()})
        return operator, right

    def _split_isolated_units(self, A, B=None):
        """Separate only independent one-by-one Laurent-unit blocks.

        Both the row and the column must contain exactly this one entry.
        No row or column combination is made, so removing these blocks
        cannot create elimination fill in the remaining code matrix.
        The returned gaps lift sparse remaining-domain indices without
        allocating a list for every (possibly unused) domain coordinate.
        """
        row_counts, column_counts = {}, {}
        for row, column in A.entries:
            row_counts[row] = row_counts.get(row, 0) + 1
            column_counts[column] = column_counts.get(column, 0) + 1
        disks = tuple((row, column, value) for (row, column), value in sorted(A.entries.items())
                      if row_counts[row] == column_counts[column] == 1 and value.is_unit)
        if not disks:
            return disks, A, B, ()
        removed_rows = sorted(row for row, _, _ in disks)
        removed_columns = sorted(column for _, column, _ in disks)
        row_set, column_set = set(removed_rows), set(removed_columns)
        operator = Matrix(
            self.ring, A.nrows - len(disks), A.ncols - len(disks),
            {(row - bisect_left(removed_rows, row), column - bisect_left(removed_columns, column)): value
             for (row, column), value in A.entries.items()
             if row not in row_set and column not in column_set})
        right = None if B is None else Matrix(
            self.ring, B.nrows - len(disks), B.ncols,
            {(row - bisect_left(removed_rows, row), column): value
             for (row, column), value in B.entries.items() if row not in row_set})
        gaps = tuple(column - index for index, column in enumerate(removed_columns))
        return disks, operator, right, gaps

    def solve(self, A, B):
        self._check(A, B)
        if A.nrows != B.nrows:
            raise ValueError("Solve requires matching row counts")
        # A small solution in the retained-cell coordinates can be much
        # sparser than a particular source-ring Groebner lift. This matters
        # when later homotopies multiply that lift. Try one finite support
        # system in the actual coordinates first; failure is only a request
        # for the complete restriction/native fallback below.
        from css_decoupling.algebra.sparse_linear import sparse_binary_solution
        candidate = sparse_binary_solution(A, B)
        if candidate is not None:
            if (candidate.ring != self.ring or candidate.shape != (A.ncols, B.ncols)
                    or A * candidate != B):
                raise ArithmeticError("Expanded sparse solution failed its exact equation")
            return candidate
        operator, right = self._trim_zero_rows(A, B)
        if not right:
            return zeros(self.ring, A.ncols, B.ncols)
        disks, remaining, remaining_rhs, gaps = self._split_isolated_units(operator, right)
        if disks:
            smaller = self.solve(remaining, remaining_rhs)
            entries = {(row + bisect_right(gaps, row), column): value
                       for (row, column), value in smaller.entries.items()}
            rhs_rows = {}
            for (row, column), value in right.entries.items():
                rhs_rows.setdefault(row, []).append((column, value))
            for row, coordinate, unit in disks:
                inverse = unit ** -1
                for column, value in rhs_rows.get(row, ()):
                    entries[coordinate, column] = inverse * value
            answer = Matrix(self.ring, A.ncols, B.ncols, entries)
        else:
            source = self.source_matrix(operator)
            if source is None:
                answer = self.base_backend.solve(operator, right)
            else:
                source_rhs = unpack_columns(right, self.lattice)
                if self.lattice.pack(source_rhs) != right:
                    raise ArithmeticError("RHS unpacking changed its retained-lattice coordinates")
                source_answer = self.base_backend.solve(source, source_rhs)
                if source * source_answer != source_rhs:
                    raise ArithmeticError("Source-ring solution failed its exact identity")
                answer = self.lattice.pack(source_answer)
        if answer.shape != (A.ncols, B.ncols) or A * answer != B:
            raise ArithmeticError("Restriction-aware solution failed its expanded equation")
        return answer

    def kernel(self, A):
        self._check(A)
        key = A.key()
        if key in self._kernels:
            return self._kernels[key].copy()
        if not A:
            return eye(self.ring, A.ncols)
        operator, _ = self._trim_zero_rows(A)
        disks, remaining, _, gaps = self._split_isolated_units(operator)
        if disks:
            smaller = self.kernel(remaining)
            # Each isolated unit equation forces its domain coordinate to
            # zero; the other coordinates have exactly the remaining kernel.
            answer = Matrix(self.ring, A.ncols, smaller.ncols,
                            {(row + bisect_right(gaps, row), column): value
                             for (row, column), value in smaller.entries.items()})
        else:
            source = self.source_matrix(operator)
            if source is None:
                answer = self.base_backend.kernel(operator)
            else:
                # This call executes the base backend's complete kernel
                # algorithm, including its source completeness verification.
                source_kernel = self.base_backend.kernel(source)
                if (source_kernel.ring != self.ring or source_kernel.nrows != source.ncols
                        or source * source_kernel):
                    raise ArithmeticError("Complete source kernel failed its actual source equation")
                # pack alone would omit the translated generators and would NOT
                # establish a complete S-kernel.  Res retains every coset column.
                answer = self.lattice.restrict(source_kernel)
        if answer.nrows != A.ncols or A * answer:
            raise ArithmeticError("Restricted complete kernel failed its expanded equation")
        self._kernels[key] = answer.copy()
        return answer

    kernel_generators = kernel

    def quotient_groebner(self, A):
        return self.base_backend.quotient_groebner(A)

    def reduce_quotient(self, A, B):
        return self.base_backend.reduce_quotient(A, B)

    def normal_form(self, A, B):
        return self.base_backend.normal_form(A, B)

    def close(self):
        super().close()
        self._source_matrices.clear()
