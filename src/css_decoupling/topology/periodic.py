"""Exact periodic CSS counts and volume-independent layer Koszul counts.

The finite backend scales with the actual number of cells, never with the
number of quantum states.  Only ``periodic_profile`` attaches a physical
upper bound/decoupling conclusion, after verification of its source profile.
An abstract gluing input describes a trivial two-layer derived object; it is
not, by itself, a certificate of a topological CSS realization.
"""
from copy import deepcopy
from dataclasses import dataclass
from itertools import combinations
from math import comb
from numbers import Integral

from sympy import Matrix as IntegerMatrix

from css_decoupling.algebra.laurent import Matrix, Ring, eye
from css_decoupling.errors import ResourceLimit
from css_decoupling.lattice.retained import Lattice, matrix_power


DEFAULT_QUDIT_LIMIT = 100_000
DEFAULT_ENTRY_LIMIT = 2_000_000
DEFAULT_WORK_LIMIT = 20_000_000


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _limit(value, name):
    return None if value is None else _integer(value, name, 1)


def _check_size(size, cap, name):
    if cap is not None and size > cap:
        raise ResourceLimit(f"{name} has size {size}; cap is {cap}")


def _basis(value, D=None):
    rows = tuple(tuple(row) for row in value)
    if not rows or any(len(row) != len(rows) for row in rows):
        raise ValueError("Period basis must be a nonempty square integer matrix")
    if D is not None and len(rows) != D:
        raise ValueError("Period basis dimension differs from the coefficient ring")
    if any(isinstance(a, bool) or not isinstance(a, Integral) for row in rows for a in row):
        raise ValueError("Period basis must have integer entries")
    rows = tuple(tuple(map(int, row)) for row in rows)
    if not IntegerMatrix(rows).det():
        raise ValueError("Period lattice must have full rank")
    return rows


def _relative_basis(base, period):
    coordinates = IntegerMatrix(base).inv() * IntegerMatrix(period)
    if any(a.q != 1 for a in coordinates):
        raise ValueError("Period lattice is not contained in the gluing base lattice")
    return tuple(tuple(int(coordinates[i, j]) for j in range(coordinates.cols))
                 for i in range(coordinates.rows))


@dataclass
class FiniteMatrix:
    """Sparse prime-field matrix, with no Laurent variables or float entries."""
    prime: int
    nrows: int
    ncols: int
    entries: dict

    @property
    def shape(self):
        return self.nrows, self.ncols

    def dense(self):
        """Explicit dense copy for small checks; main algorithms stay sparse."""
        return [[self.entries.get((i, j), 0) for j in range(self.ncols)]
                for i in range(self.nrows)]


def _add(entries, position, coefficient, p, cap):
    value = (entries.get(position, 0) + coefficient) % p
    if value:
        entries[position] = value
    else:
        entries.pop(position, None)
    _check_size(len(entries), cap, "Sparse finite-field matrix")


def periodic_matrix(A, lattice_basis, *, max_qudits=DEFAULT_QUDIT_LIMIT,
                    max_matrix_entries=DEFAULT_ENTRY_LIMIT,
                    max_rank_work=DEFAULT_WORK_LIMIT):
    """Regular-representation matrix on F_p[Z^D/L], in HNF coset order.

    ``max_qudits`` bounds the number of columns, i.e. check-matrix qudits.
    The lattice columns are expressed in the original Laurent coordinates.
    """
    cap = _limit(max_matrix_entries, "max_matrix_entries")
    work_cap = _limit(max_rank_work, "max_rank_work")
    qudit_cap = _limit(max_qudits, "max_qudits")
    basis = _basis(lattice_basis, A.ring.D)
    index = abs(int(IntegerMatrix(basis).det()))
    _check_size(index * max(A.ncols, 1), qudit_cap, "Periodic matrix columns/cells")
    terms = sum(len(f.terms) for f in A.entries.values())
    _check_size(terms * index, work_cap, "Periodic matrix construction work")
    lattice = Lattice(basis, max_index=None)
    entries = {}
    for (i, j), polynomial in A.entries.items():
        for exponent, coefficient in polynomial.terms.items():
            for column, representative in enumerate(lattice.representatives):
                row, _ = lattice.reduce(tuple(a + b for a, b in zip(exponent, representative)))
                _add(entries, (i * index + row, j * index + column),
                     coefficient, A.ring.p, cap)
    return FiniteMatrix(A.ring.p, A.nrows * index, A.ncols * index, entries)


def finite_rank(A, *, max_matrix_entries=DEFAULT_ENTRY_LIMIT,
                max_rank_work=DEFAULT_WORK_LIMIT):
    """Sparse forward row elimination over F_p, with fill/work limits."""
    cap = _limit(max_matrix_entries, "max_matrix_entries")
    work_cap = _limit(max_rank_work, "max_rank_work")
    _check_size(len(A.entries), cap, "Rank input")
    rows = {}
    for (i, j), coefficient in A.entries.items():
        rows.setdefault(i, {})[j] = coefficient % A.prime
    pivots, work, retained = {}, 0, 0
    for i in sorted(rows):
        row = rows[i]
        while row:
            pivot = min(row)
            if pivot not in pivots:
                inverse = pow(row[pivot], -1, A.prime)
                row = {j: coefficient * inverse % A.prime for j, coefficient in row.items()}
                pivots[pivot] = row
                retained += len(row)
                _check_size(retained, cap, "Rank echelon fill")
                break
            coefficient = row[pivot]
            source = pivots[pivot]
            work += len(source)
            _check_size(work, work_cap, "Rank elimination work")
            for j, value in source.items():
                updated = (row.get(j, 0) - coefficient * value) % A.prime
                if updated:
                    row[j] = updated
                else:
                    row.pop(j, None)
            _check_size(retained + len(row), cap, "Rank elimination fill")
    return len(pivots)


class _ActionArithmetic:
    """Sparse constant matrix arithmetic with one cumulative work budget."""

    def __init__(self, ring, size, entry_cap, work_cap):
        self.ring, self.size = ring, size
        self.entry_cap, self.work_cap, self.work = entry_cap, work_cap, 0
        _check_size(size, entry_cap, "Translation action dimension")
        self.identity = {(i, i): 1 for i in range(size)}

    def charge(self, amount):
        self.work += amount
        _check_size(self.work, self.work_cap, "Translation action arithmetic work")

    def multiply(self, A, B):
        right = {}
        for (i, j), value in B.items():
            right.setdefault(i, []).append((j, value))
        out = {}
        for (i, k), a in A.items():
            self.charge(len(right.get(k, ())))
            for j, b in right.get(k, ()):
                _add(out, (i, j), a * b, self.ring.p, self.entry_cap)
        return out

    def inverse(self, A):
        if A == self.identity:
            return self.identity
        rows = [{self.size + i: 1} for i in range(self.size)]
        for (i, j), value in A.items():
            rows[i][j] = value
        retained = sum(map(len, rows))
        _check_size(retained, self.entry_cap, "Translation inverse augmented entries")
        for column in range(self.size):
            pivot = None
            for row in range(column, self.size):
                self.charge(1)
                if rows[row].get(column):
                    pivot = row
                    break
            if pivot is None:
                raise ValueError("Translation actions must be invertible")
            rows[column], rows[pivot] = rows[pivot], rows[column]
            inv = pow(rows[column][column], -1, self.ring.p)
            rows[column] = {j: a * inv % self.ring.p for j, a in rows[column].items()}
            for row in range(self.size):
                self.charge(1)
                if row == column or not rows[row].get(column):
                    continue
                factor = rows[row][column]
                self.charge(len(rows[column]))
                retained -= len(rows[row])
                for j, a in rows[column].items():
                    value = (rows[row].get(j, 0) - factor * a) % self.ring.p
                    if value:
                        rows[row][j] = value
                    else:
                        rows[row].pop(j, None)
                retained += len(rows[row])
                _check_size(retained, self.entry_cap, "Translation inverse fill")
        return {(i, j - self.size): a for i, row in enumerate(rows)
                for j, a in row.items() if j >= self.size}

    def power(self, A, exponent):
        result = self.identity
        while exponent:
            if exponent & 1:
                result = self.multiply(result, A)
            exponent >>= 1
            if exponent:
                A = self.multiply(A, A)
        return result


def _layer_period_actions(actions, basis, entry_cap, work_cap):
    actions = tuple(actions)
    if not actions or not all(isinstance(A, Matrix) for A in actions):
        raise ValueError("Supply the original D translation actions as constant matrices")
    ring, size = actions[0].ring, actions[0].nrows
    if len(actions) != ring.D or any(A.ring != ring or A.shape != (size, size) for A in actions):
        raise ValueError("Translation actions must be D square matrices of one ring and size")
    if any(not f.is_constant for A in actions for f in A.entries.values()):
        raise ValueError("Translation actions must have constant finite-field entries")
    _check_size(sum(len(A.entries) for A in actions), entry_cap, "Input translation entries")
    arithmetic = _ActionArithmetic(ring, size, entry_cap, work_cap)
    raw = [{key: value.constant() for key, value in A.entries.items()} for A in actions]
    inverse = [arithmetic.inverse(A) for A in raw]
    for a in range(ring.D):
        for b in range(a):
            if arithmetic.multiply(raw[a], raw[b]) != arithmetic.multiply(raw[b], raw[a]):
                raise ValueError("Translation actions must commute")
    period = []
    for vector in zip(*basis):
        product = arithmetic.identity
        for axis, exponent in enumerate(vector):
            if exponent:
                power = arithmetic.power(raw[axis] if exponent > 0 else inverse[axis], abs(exponent))
                product = arithmetic.multiply(product, power)
        period.append(Matrix(ring, size, size, product))
        _check_size(sum(len(A.entries) for A in period), entry_cap, "Period action entries")
    return actions, tuple(period)


def _layer_differential(actions, degree, exterior_cap, entry_cap, work_cap):
    """Only the requested adjacent exterior degree, never the full algebra."""
    D, size, ring = len(actions), actions[0].nrows, actions[0].ring
    ncols, nrows = comb(D, degree) * size, comb(D, degree - 1) * size
    _check_size(ncols + nrows, exterior_cap, "Layer Koszul basis")
    differences = tuple(A - eye(ring, size) for A in actions)
    if not size or not any(differences):
        return FiniteMatrix(ring.p, nrows, ncols, {})
    targets = {I: n for n, I in enumerate(combinations(range(D), degree - 1))}
    entries, work = {}, 0
    for source, I in enumerate(combinations(range(D), degree)):
        for position, axis in enumerate(I):
            target = targets[I[:position] + I[position + 1:]]
            work += len(differences[axis].entries)
            _check_size(work, work_cap, "Layer Koszul assembly work")
            for (a, b), value in differences[axis].entries.items():
                _add(entries, (target * size + a, source * size + b),
                     (-1 if position % 2 else 1) * value.constant(), ring.p, entry_cap)
    return FiniteMatrix(ring.p, nrows, ncols, entries)


@dataclass
class PeriodicLayerResult:
    """One finite translation module's contribution, without a CSS claim."""

    actions: tuple
    degree: int
    lattice_basis: tuple
    period_actions: tuple
    dimension: int
    logical_qudits: int
    model_total: int
    incoming_rank: int
    outgoing_rank: int
    translation_trivial: bool

    def contribution(self):
        return {name: getattr(self, name) for name in (
            "degree", "dimension", "logical_qudits", "model_total", "incoming_rank",
            "outgoing_rank", "translation_trivial")}

    def certificate(self):
        from css_decoupling.certificates.comparison import _matrix_dict
        ring = self.actions[0].ring
        return {"schema": "css-decoupling.periodic-layer", "version": 1,
                "ring": {"prime": ring.p, "dimension": ring.D},
                "lattice_basis": [list(row) for row in self.lattice_basis],
                "actions": [_matrix_dict(A) for A in self.actions],
                "period_actions": [_matrix_dict(A) for A in self.period_actions],
                "contribution": self.contribution()}


def periodic_layer(actions, degree, lattice_basis, *,
                   max_exterior_entries=DEFAULT_ENTRY_LIMIT,
                   max_matrix_entries=DEFAULT_ENTRY_LIMIT,
                   max_rank_work=DEFAULT_WORK_LIMIT):
    """Compute h_j(L) from commuting invertible finite translation matrices.

    Costs depend on the layer dimension, adjacent exterior degrees and the
    bit lengths of the period vectors, not the number of cells in Z^D/L.
    Lattice columns are vectors in the original D translation coordinates.
    """
    entry_cap = _limit(max_matrix_entries, "max_matrix_entries")
    exterior_cap = _limit(max_exterior_entries, "max_exterior_entries")
    work_cap = _limit(max_rank_work, "max_rank_work")
    actions = tuple(actions)
    if not actions or not isinstance(actions[0], Matrix):
        raise ValueError("Supply D constant translation action matrices")
    D = actions[0].ring.D
    degree = _integer(degree, "layer degree", 1)
    if degree >= D:
        raise ValueError("Layer degree is outside 1,...,D-1")
    basis = _basis(lattice_basis, D)
    # Check both required degrees before creating any combination basis.
    size = actions[0].nrows
    _check_size((comb(D, degree - 1) + comb(D, degree) + comb(D, degree + 1)) * size,
                exterior_cap, "Adjacent layer Koszul bases")
    actions, period = _layer_period_actions(actions, basis, entry_cap, work_cap)
    options = dict(max_matrix_entries=entry_cap, max_rank_work=work_cap)
    outgoing = finite_rank(_layer_differential(period, degree, exterior_cap, entry_cap, work_cap), **options)
    incoming = finite_rank(_layer_differential(period, degree + 1, exterior_cap, entry_cap, work_cap), **options)
    total = comb(D, degree) * size
    return PeriodicLayerResult(actions, degree, basis, period, size, total - incoming - outgoing,
                               total, incoming, outgoing, all(A == eye(A.ring, size) for A in period))


@dataclass
class PeriodicResult:
    prime: int
    dimension: int
    lattice_basis: tuple
    index: int
    logical_qudits: int
    method: str
    physical_qudits: int | None = None
    rank_x: int | None = None
    rank_z: int | None = None
    incoming_rank: int | None = None
    outgoing_rank: int | None = None
    model_total: int | None = None
    upper_bound: int | None = None
    decouplable: bool | None = None
    HX_periodic: FiniteMatrix | None = None
    HZ_periodic: FiniteMatrix | None = None
    _certificate: dict | None = None
    layer_contributions: list | None = None
    translation_loss: int | None = None
    gluing_loss: int | None = None

    def certificate(self):
        # Keep source evidence as a snapshot, but verify the claims currently
        # exposed by this mutable result, not stale cached scalar values.
        data = deepcopy(self._certificate)
        data.update(ring={"prime": self.prime, "dimension": self.dimension},
                    lattice_basis=[list(row) for row in self.lattice_basis],
                    method=self.method, count=_count_dict(self))
        if data["version"] == 2:
            data["diagnostics"] = _diagnostics(self)
        return data

    @property
    def degeneracy(self):
        """Return the exact p**k integer only when explicitly requested."""
        return self.prime ** self.logical_qudits


def _count_dict(result):
    return {name: getattr(result, name) for name in (
        "index", "logical_qudits", "physical_qudits", "rank_x", "rank_z",
        "incoming_rank", "outgoing_rank", "model_total", "upper_bound", "decouplable")}


def _diagnostics(result):
    return {"layers": deepcopy(result.layer_contributions),
            "translation_loss": result.translation_loss, "gluing_loss": result.gluing_loss}


def _certificate(result, *, source=None, model=None, analysis=None):
    data = {"schema": "css-decoupling.periodic", "version": 1,
            "ring": {"prime": result.prime, "dimension": result.dimension},
            "lattice_basis": [list(row) for row in result.lattice_basis],
            "method": result.method, "source": source, "model": model,
            "profile_analysis": analysis, "count": _count_dict(result)}
    if result.layer_contributions is not None:
        data.update(version=2, diagnostics=_diagnostics(result))
    return data


def periodic_code(HX, HZ, lattice_basis, *, max_qudits=DEFAULT_QUDIT_LIMIT,
                  max_matrix_entries=DEFAULT_ENTRY_LIMIT,
                  max_rank_work=DEFAULT_WORK_LIMIT):
    """Count an arbitrary finite periodic CSS code, without topology claims."""
    if HX.ring != HZ.ring or HX.ncols != HZ.ncols:
        raise ValueError("CSS checks must have the same ring and physical columns")
    if HX.ring.D < 2:
        raise ValueError("CSS periodic counting requires dimension >= 2")
    if HX * HZ.dagger():
        raise ValueError("CSS checks do not commute")
    basis = _basis(lattice_basis, HX.ring.D)
    options = dict(max_qudits=max_qudits, max_matrix_entries=max_matrix_entries,
                   max_rank_work=max_rank_work)
    X, Z = periodic_matrix(HX, basis, **options), periodic_matrix(HZ, basis, **options)
    rank_options = dict(max_matrix_entries=max_matrix_entries, max_rank_work=max_rank_work)
    rx, rz = finite_rank(X, **rank_options), finite_rank(Z, **rank_options)
    index = abs(int(IntegerMatrix(basis).det()))
    qudits = HX.ncols * index
    k = qudits - rx - rz
    if k < 0:
        raise ArithmeticError("Periodic commuting CSS ranks exceed the qudit count")
    result = PeriodicResult(HX.ring.p, HX.ring.D, basis, index, k, "finite",
                            qudits, rx, rz, HX_periodic=X, HZ_periodic=Z)
    from css_decoupling.certificates.io import input_dict
    result._certificate = _certificate(result, source=input_dict(HX, HZ))
    return result


def _profile_model(classes, multiplicities, lattice_basis, prime, base_lattice_basis):
    from css_decoupling.certificates.comparison import _matrix_dict
    classes = tuple(classes)
    basis = _basis(lattice_basis)
    D = len(basis)
    if prime is None:
        if not classes:
            raise ValueError("prime is required when the class list is empty")
        prime = classes[0].coefficients.ring.p
    ring = Ring(_integer(prime, "prime", 2), D)
    if D < 2:
        raise ValueError("CSS profile counting requires dimension >= 2")
    multiplicities = {_integer(j, "layer degree", 1): _integer(t, "multiplicity")
                      for j, t in multiplicities.items()}
    multiplicities = {j: t for j, t in multiplicities.items() if t}
    if any(j >= D for j in multiplicities):
        raise ValueError("Layer degree is outside the topological CSS range 1,...,D-1")
    if len(multiplicities) > 2:
        raise ValueError("Pairwise contraction counts require at most two nonzero layers")
    if len(classes) != int(len(multiplicities) == 2):
        raise ValueError("Supply exactly the sole two-layer class, including a zero class")
    if classes:
        cls = classes[0]
        i, j = sorted(multiplicities)
        if (cls.lower, cls.upper, cls.target_dimension, cls.source_dimension) != (
                i, j, multiplicities[i], multiplicities[j]):
            raise ValueError("Gluing endpoints do not match the supplied profile")
        if cls.coefficients.ring != ring or cls.coefficients.shape != (
                multiplicities[i], comb(D, j - i + 1) * multiplicities[j]):
            raise ValueError("Wrong ring or exterior coefficient shape")
        if any(not f.is_constant for f in cls.coefficients.entries.values()):
            raise ValueError("Exterior coefficients must be constant prime-field matrices")
    base = _basis(base_lattice_basis if base_lattice_basis is not None else
                  [[int(i == j) for j in range(D)] for i in range(D)], D)
    relative = _relative_basis(base, basis)
    model = {"base_lattice_basis": [list(row) for row in base],
             "multiplicities": [[j, t] for j, t in sorted(multiplicities.items())],
             "classes": [{"upper": cls.upper, "lower": cls.lower,
                           "coefficients": _matrix_dict(cls.coefficients)} for cls in classes]}
    return ring, basis, relative, classes, multiplicities, model


def contraction_matrix(cls, degree, *, max_exterior_entries=DEFAULT_ENTRY_LIMIT):
    """Interior product Λ^degree V⊗E_upper -> Λ^(degree-q) V⊗E_lower.

    Exterior axes precede layer-copy indices.  Removing I from J has sign
    (-1)**sum(position_of_I[a]-a), valid also in odd characteristic.
    """
    cap = _limit(max_exterior_entries, "max_exterior_entries")
    degree = _integer(degree, "exterior degree")
    D, p, q = cls.coefficients.ring.D, cls.coefficients.ring.p, cls.ext_degree
    target_degree = degree - q
    if not 0 <= target_degree <= degree <= D:
        raise ValueError("Invalid contraction degree")
    source_size = comb(D, degree) * cls.source_dimension
    target_size = comb(D, target_degree) * cls.target_dimension
    _check_size(source_size + target_size, cap, "Contraction basis")
    source_basis = {axes: n for n, axes in enumerate(combinations(range(D), degree))}
    target_basis = {axes: n for n, axes in enumerate(combinations(range(D), target_degree))}
    entries, work = {}, 0
    for I, target_copy, source_copy, coefficient in cls.terms():
        remaining = tuple(a for a in range(D) if a not in I)
        for K in combinations(remaining, target_degree):
            work += 1
            _check_size(work, cap, "Contraction assembly work")
            J = tuple(sorted(I + K))
            sign = -1 if sum(k < i for i in I for k in K) % 2 else 1
            row = target_basis[K] * cls.target_dimension + target_copy
            column = source_basis[J] * cls.source_dimension + source_copy
            _add(entries, (row, column), sign * coefficient, p, cap)
    return FiniteMatrix(p, target_size, source_size, entries)


def periodic_from_gluing(classes, multiplicities, lattice_basis, *, prime=None,
                         base_lattice_basis=None, max_exterior_entries=DEFAULT_ENTRY_LIMIT,
                         max_matrix_entries=DEFAULT_ENTRY_LIMIT,
                         max_rank_work=DEFAULT_WORK_LIMIT):
    """Count the derived torus homology of a trivial <=2-layer profile.

    This formal-profile API does not certify a physical CSS realization;
    ``upper_bound`` and ``decouplable`` remain None.  Use periodic_profile
    to bind those conclusions to independently verified physical input.
    """
    ring, basis, relative, classes, multiplicities, model = _profile_model(
        classes, multiplicities, lattice_basis, prime, base_lattice_basis)
    cap = _limit(max_exterior_entries, "max_exterior_entries")
    total = sum(comb(ring.D, j) * t for j, t in multiplicities.items())
    incoming = outgoing = 0
    if classes:
        cls = classes[0]
        _check_size(comb(ring.D, cls.ext_degree) * max(cls.source_dimension, cls.target_dimension),
                    cap, "Exterior restriction basis")
        restricted = cls.restrict(relative, max_exterior_entries=cap)
        options = dict(max_matrix_entries=max_matrix_entries, max_rank_work=max_rank_work)
        incoming = finite_rank(contraction_matrix(restricted, cls.upper + 1,
                               max_exterior_entries=cap), **options)
        outgoing = finite_rank(contraction_matrix(restricted, cls.upper,
                               max_exterior_entries=cap), **options)
    result = PeriodicResult(ring.p, ring.D, basis, abs(int(IntegerMatrix(basis).det())),
                            total - incoming - outgoing, "two-layer",
                            incoming_rank=incoming, outgoing_rank=outgoing, model_total=total)
    if result.logical_qudits < 0:
        raise ArithmeticError("Contraction losses exceed profile homology")
    result._certificate = _certificate(result, model=model)
    return result


def _analysis_data(analysis):
    from css_decoupling.certificates.analysis import verify_analysis
    from css_decoupling.certificates.io import parse_input
    from css_decoupling.certificates.comparison import _matrix_from_dict
    data = analysis.certificate() if hasattr(analysis, "certificate") else deepcopy(analysis)
    report = verify_analysis(data)
    if not report.valid:
        raise ValueError("Invalid source profile: " + "; ".join(report.errors))
    source = {"schema": "css-decoupling.general.input", "version": 1,
              "ring": data["ring"], **data["input"], "lattice_basis": None}
    HX, HZ, _ = parse_input(source)
    layers = {item["degree"]: _matrix_from_dict(item["representatives"], HX.ring,
                                              "representatives").ncols for item in data["layers"]}
    return data, source, HX, HZ, layers


def _fast_model_from_analysis(data, basis, ring, layers):
    from css_decoupling.certificates.comparison import _matrix_from_dict
    from css_decoupling.decoupling.gluing import GluingClass
    if len(layers) > 2:
        return None
    if len(layers) < 2:
        # A period outside the recorded initial cell can nevertheless act
        # trivially.  Test its actual original translation matrices.
        for raw in data["layers"]:
            actions = [_matrix_from_dict(a, ring, "action") for a in raw["actions"]]
            dimension = layers[raw["degree"]]
            for vector in zip(*basis):
                action = eye(ring, dimension)
                for A, exponent in zip(actions, vector):
                    action = action * matrix_power(A, exponent)
                if action != eye(ring, dimension):
                    return None
        return (), basis
    base = _basis(data["initial_basis"], ring.D)
    try:
        _relative_basis(base, basis)
    except ValueError:
        return None
    # Exactly two layers have a unique first attachment.  Later zero
    # records refer to a different coarse lattice and must not be reused.
    record = data["attachments"][0]
    if _basis(record["basis"], ring.D) != base:
        raise ValueError("Initial gluing record uses a different lattice")
    witness = record["gluing"]
    classes = tuple(GluingClass(witness["upper"], cls["lower"], layers[witness["upper"]],
                                layers[cls["lower"]], _matrix_from_dict(
                                    cls["coefficients"], ring, "coefficients"))
                    for cls in witness["classes"])
    return classes, base


def periodic_profile(analysis, lattice_basis, *, prefer_fast=True,
                     max_qudits=DEFAULT_QUDIT_LIMIT,
                     max_matrix_entries=DEFAULT_ENTRY_LIMIT,
                     max_rank_work=DEFAULT_WORK_LIMIT,
                     max_exterior_entries=DEFAULT_ENTRY_LIMIT):
    """Bind torus count and Corollary-3 conclusions to a verified analysis.

    Accepts a ProfileAnalysis or its JSON certificate. A single finite layer
    uses its two adjacent Koszul differentials, including nontrivial period
    actions. General multilayer totals use complete periodic matrices unless
    the existing normalized two-layer contraction route applies. Per-layer
    counts diagnose translation and gluing losses; they do not replace the
    full multilayer total.
    ``prefer_fast=False`` is useful for an independent small-cell comparison.
    """
    if not isinstance(prefer_fast, bool):
        raise ValueError("prefer_fast must be boolean")
    data, source, HX, HZ, layers = _analysis_data(analysis)
    basis = _basis(lattice_basis, HX.ring.D)
    from css_decoupling.certificates.comparison import _matrix_from_dict
    contributions = [periodic_layer(
        tuple(_matrix_from_dict(A, HX.ring, "action") for A in item["actions"]),
        item["degree"], basis, max_exterior_entries=max_exterior_entries,
        max_matrix_entries=max_matrix_entries, max_rank_work=max_rank_work).contribution()
        for item in sorted(data["layers"], key=lambda item: item["degree"])]
    fast = (_fast_model_from_analysis(data, basis, HX.ring, layers)
            if prefer_fast and len(layers) > 1 else None)
    if prefer_fast and len(layers) <= 1:
        result = PeriodicResult(HX.ring.p, HX.ring.D, basis,
            abs(int(IntegerMatrix(basis).det())),
            sum(c["logical_qudits"] for c in contributions), "single-layer",
            incoming_rank=sum(c["incoming_rank"] for c in contributions),
            outgoing_rank=sum(c["outgoing_rank"] for c in contributions),
            model_total=sum(c["model_total"] for c in contributions))
        result.physical_qudits = HX.ncols * result.index
        model = None
    elif fast is not None:
        classes, base = fast
        result = periodic_from_gluing(classes, layers, basis, prime=HX.ring.p,
                     base_lattice_basis=base, max_exterior_entries=max_exterior_entries,
                     max_matrix_entries=max_matrix_entries, max_rank_work=max_rank_work)
        result.physical_qudits = HX.ncols * result.index
        model = result._certificate["model"]
    else:
        result = periodic_code(HX, HZ, basis, max_qudits=max_qudits,
                     max_matrix_entries=max_matrix_entries, max_rank_work=max_rank_work)
        model = None
    result.upper_bound = sum(comb(HX.ring.D, j) * t for j, t in layers.items())
    if result.logical_qudits > result.upper_bound:
        raise ArithmeticError("Periodic count exceeds the verified profile bound")
    result.decouplable = result.logical_qudits == result.upper_bound
    result.layer_contributions = contributions
    result.translation_loss = result.upper_bound - sum(c["logical_qudits"] for c in contributions)
    result.gluing_loss = sum(c["logical_qudits"] for c in contributions) - result.logical_qudits
    if result.translation_loss < 0 or result.gluing_loss < 0:
        raise ArithmeticError("Periodic count violates the verified layer contribution bound")
    result._certificate = _certificate(result, source=source, model=model, analysis=data)
    return result
