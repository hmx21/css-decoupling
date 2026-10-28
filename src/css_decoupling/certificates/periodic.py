"""Independent verification of finite-periodic and two-layer count records.

Ranks are recomputed by sparse column elimination with the opposite pivot
order to the constructor. Finite matrices use restriction+augmentation;
gluing restriction uses explicit minors and contraction iterates source
cells. No saved success flag or claimed rank is used as an assumption.
"""
from itertools import combinations
from math import comb

from sympy import Matrix as IntegerMatrix

from css_decoupling.algebra.laurent import Matrix, Ring, eye
from css_decoupling.errors import ResourceLimit
from css_decoupling.lattice.retained import Lattice, matrix_power
from css_decoupling.topology.periodic import (
    FiniteMatrix, DEFAULT_QUDIT_LIMIT, DEFAULT_ENTRY_LIMIT, DEFAULT_WORK_LIMIT,
)
from .comparison import _int, _keys, _matrix_from_dict, VerificationReport
from .io import parse_input


def _cap(value, name):
    return None if value is None else _int(value, name, 1)


def _bound(size, cap, name):
    if cap is not None and size > cap:
        raise ResourceLimit(f"Periodic verifier {name} has size {size}; cap is {cap}")


def _lattice(raw, D):
    if not isinstance(raw, list) or len(raw) != D:
        raise ValueError("Wrong period basis dimension")
    if any(not isinstance(row, list) or len(row) != D for row in raw):
        raise ValueError("Lattice basis must be a square list of lists")
    for row in raw:
        for value in row:
            # _int's minimum is unsuitable for negative lattice coordinates.
            if type(value) is not int:
                raise ValueError("Lattice coordinates must be integers, not booleans")
    B = IntegerMatrix(raw)
    if not B.det():
        raise ValueError("Lattice basis is singular")
    return B


def _rank(A, entry_cap, work_cap):
    """Independent descending-pivot column elimination."""
    _bound(len(A.entries), entry_cap, "rank input entries")
    columns = {}
    for (i, j), value in A.entries.items():
        if value % A.prime:
            columns.setdefault(j, {})[i] = value % A.prime
    pivots, retained, work = {}, 0, 0
    for j in sorted(columns, reverse=True):
        column = columns[j]
        while column:
            pivot = max(column)
            if pivot not in pivots:
                inverse = pow(column[pivot], -1, A.prime)
                column = {i: a * inverse % A.prime for i, a in column.items()}
                pivots[pivot] = column
                retained += len(column)
                _bound(retained, entry_cap, "rank fill")
                break
            factor = column[pivot]
            work += len(pivots[pivot])
            _bound(work, work_cap, "rank work")
            for i, value in pivots[pivot].items():
                residue = (column.get(i, 0) - factor * value) % A.prime
                if residue:
                    column[i] = residue
                else:
                    column.pop(i, None)
            _bound(retained + len(column), entry_cap, "rank fill")
    return len(pivots)


def _verified_period_actions(actions, period, ring, entry_cap, work_cap):
    """Recompute group action by sparse modular products and column inverses."""
    if len(actions) != ring.D:
        raise ValueError("A layer needs the D original translation actions")
    size = actions[0].nrows
    if any(A.ring != ring or A.shape != (size, size) for A in actions):
        raise ValueError("Layer actions must share a square shape and coefficient ring")
    if any(not f.is_constant for A in actions for f in A.entries.values()):
        raise ValueError("Layer actions must be constant finite-field matrices")
    _bound(size, entry_cap, "action dimension")
    _bound(sum(len(A.entries) for A in actions), entry_cap, "input action entries")
    work = 0

    def charge(n):
        nonlocal work
        work += n
        _bound(work, work_cap, "translation arithmetic work")

    identity = {(i, i): 1 for i in range(size)}

    def multiply(A, B):
        left = {}
        for (i, k), a in A.items():
            left.setdefault(k, []).append((i, a))
        result = {}
        for (k, j), b in B.items():
            charge(len(left.get(k, ())))
            for i, a in left.get(k, ()):
                value = (result.get((i, j), 0) + a * b) % ring.p
                if value:
                    result[i, j] = value
                else:
                    result.pop((i, j), None)
                _bound(len(result), entry_cap, "action product entries")
        return result

    def inverse(A):
        if A == identity:
            return identity
        columns = [{size + j: 1} for j in range(size)]
        for (i, j), value in A.items():
            columns[j][i] = value
        fill = sum(map(len, columns))
        _bound(fill, entry_cap, "action inverse augmented entries")
        for row in range(size):
            found = None
            for j in range(size - 1, row - 1, -1):
                charge(1)
                if columns[j].get(row):
                    found = j
                    break
            if found is None:
                raise ValueError("Singular layer translation action")
            columns[row], columns[found] = columns[found], columns[row]
            scale = pow(columns[row][row], -1, ring.p)
            columns[row] = {i: a * scale % ring.p for i, a in columns[row].items()}
            for j in range(size - 1, -1, -1):
                charge(1)
                if j == row or not columns[j].get(row):
                    continue
                factor = columns[j][row]
                charge(len(columns[row]))
                fill -= len(columns[j])
                for i, value in columns[row].items():
                    residue = (columns[j].get(i, 0) - factor * value) % ring.p
                    if residue:
                        columns[j][i] = residue
                    else:
                        columns[j].pop(i, None)
                fill += len(columns[j])
                _bound(fill, entry_cap, "action inverse fill")
        return {(i - size, j): a for j, column in enumerate(columns)
                for i, a in column.items() if i >= size}

    raw = [{at: f.constant() for at, f in A.entries.items()} for A in actions]
    inverses = [inverse(A) for A in raw]
    for a in range(ring.D):
        for b in range(a):
            if multiply(raw[a], raw[b]) != multiply(raw[b], raw[a]):
                raise ValueError("Layer translations do not commute")
    outputs = []
    for column in range(ring.D):
        result = identity
        for axis in range(ring.D):
            exponent = int(period[axis, column])
            power = raw[axis] if exponent >= 0 else inverses[axis]
            # Left-to-right binary powering differs from the constructor's
            # right-to-left accumulation, and never enumerates group elements.
            accumulated = identity
            for bit in bin(abs(exponent))[2:] if exponent else ():
                accumulated = multiply(accumulated, accumulated)
                if bit == "1":
                    accumulated = multiply(power, accumulated)
            result = multiply(accumulated, result)
        outputs.append(result)
        _bound(sum(map(len, outputs)), entry_cap, "period action entries")
    return size, outputs


def _verified_layer(actions, degree, period, ring, exterior_cap, entry_cap, work_cap):
    degree = _int(degree, "layer degree", 1)
    if degree >= ring.D:
        raise ValueError("Layer degree must lie in 1,...,D-1")
    if len(actions) != ring.D:
        raise ValueError("A layer needs D actions")
    size = actions[0].nrows
    _bound((comb(ring.D, degree - 1) + comb(ring.D, degree) + comb(ring.D, degree + 1)) * size,
           exterior_cap, "adjacent layer Koszul bases")
    size, translations = _verified_period_actions(actions, period, ring, entry_cap, work_cap)
    identity = {(i, i): 1 for i in range(size)}
    differences = []
    for translation in translations:
        delta = dict(translation)
        for i in range(size):
            value = (delta.get((i, i), 0) - 1) % ring.p
            if value:
                delta[i, i] = value
            else:
                delta.pop((i, i), None)
        differences.append(delta)

    def differential(k):
        nrows, ncols = comb(ring.D, k - 1) * size, comb(ring.D, k) * size
        if not size or not any(differences):
            return FiniteMatrix(ring.p, nrows, ncols, {})
        sources = {I: n for n, I in enumerate(combinations(range(ring.D), k))}
        entries, work = {}, 0
        for target, J in enumerate(combinations(range(ring.D), k - 1)):
            for axis in range(ring.D):
                if axis in J:
                    continue
                I = tuple(sorted(J + (axis,)))
                column = sources[I]
                sign = -1 if sum(a < axis for a in J) % 2 else 1
                work += len(differences[axis])
                _bound(work, work_cap, "layer differential assembly")
                for (a, b), value in differences[axis].items():
                    entries[target * size + a, column * size + b] = sign * value % ring.p
                    _bound(len(entries), entry_cap, "layer differential entries")
        return FiniteMatrix(ring.p, nrows, ncols, entries)

    outgoing = _rank(differential(degree), entry_cap, work_cap)
    incoming = _rank(differential(degree + 1), entry_cap, work_cap)
    total = comb(ring.D, degree) * size
    contribution = dict(degree=degree, dimension=size, logical_qudits=total - incoming - outgoing,
                        model_total=total, incoming_rank=incoming, outgoing_rank=outgoing,
                        translation_trivial=all(A == identity for A in translations))
    return contribution, [Matrix(ring, size, size, A) for A in translations]


def _compare_contribution(raw, expected):
    _keys(raw, tuple(expected), "layer contribution")
    for key, value in raw.items():
        if key == "translation_trivial":
            if type(value) is not bool:
                raise ValueError("translation_trivial must be boolean")
        else:
            _int(value, key, 0)
        if value != expected[key]:
            raise ValueError(f"Incorrect layer contribution field: {key}")


def verify_periodic_layer(data, *, max_matrix_entries=DEFAULT_ENTRY_LIMIT,
                          max_rank_work=DEFAULT_WORK_LIMIT,
                          max_exterior_entries=DEFAULT_ENTRY_LIMIT):
    """Check a finite module contribution, without assuming a physical code."""
    checks, errors = [], []
    try:
        if hasattr(data, "certificate"):
            data = data.certificate()
        entry_cap, work_cap, exterior_cap = (_cap(max_matrix_entries, "max_matrix_entries"),
            _cap(max_rank_work, "max_rank_work"), _cap(max_exterior_entries, "max_exterior_entries"))
        _keys(data, ("schema", "version", "ring", "lattice_basis", "actions", "period_actions",
                     "contribution"), "periodic layer certificate")
        if data["schema"] != "css-decoupling.periodic-layer" or _int(data["version"], "version") != 1:
            raise ValueError("Unsupported periodic layer certificate")
        _keys(data["ring"], ("prime", "dimension"), "ring")
        ring = Ring(_int(data["ring"]["prime"], "prime", 2), _int(data["ring"]["dimension"], "dimension", 2))
        period = _lattice(data["lattice_basis"], ring.D)
        if not isinstance(data["actions"], list) or len(data["actions"]) != ring.D:
            raise ValueError("Wrong number of original layer actions")
        if not isinstance(data["period_actions"], list) or len(data["period_actions"]) != ring.D:
            raise ValueError("Wrong number of period actions")
        actions = [_matrix_from_dict(A, ring, "layer action") for A in data["actions"]]
        expected, translations = _verified_layer(actions, data["contribution"]["degree"],
            period, ring, exterior_cap, entry_cap, work_cap)
        _compare_contribution(data["contribution"], expected)
        if any(A != _matrix_from_dict(raw, ring, "period action") for A, raw in zip(translations, data["period_actions"])):
            raise ValueError("Claimed period actions differ from the original translation representation")
        checks.extend(("commuting invertible finite-field actions and integer lattice verified",
                       "period actions and both signed Koszul ranks independently recomputed"))
    except (ValueError, TypeError, KeyError, AttributeError, IndexError) as exc:
        errors.append(str(exc))
    return VerificationReport(not errors, tuple(checks), tuple(errors), "finite-module-periodic-contribution")


def _finite(A, lattice, qudit_cap, entry_cap, work_cap):
    _bound(lattice.index * max(A.ncols, 1), qudit_cap, "finite columns/cells")
    _bound(lattice.index * sum(len(f.terms) for f in A.entries.values()),
           work_cap, "finite assembly work")
    # Restriction temporarily retains Laurent shifts before augmentation.
    # Bound that potential allocation, including terms that later cancel.
    _bound(lattice.index * sum(len(f.terms) for f in A.entries.values()),
           entry_cap, "temporary restriction terms")
    # This independent route retains the lattice shifts until augmentation.
    restricted = lattice.restrict(A)
    _bound(sum(len(f.terms) for f in restricted.entries.values()), entry_cap,
           "restricted Laurent entries")
    entries = {position: sum(f.terms.values()) % A.ring.p
               for position, f in restricted.entries.items() if sum(f.terms.values()) % A.ring.p}
    _bound(len(entries), entry_cap, "finite entries")
    return FiniteMatrix(A.ring.p, restricted.nrows, restricted.ncols, entries)


def _read_model(raw, ring, period, exterior_cap):
    _keys(raw, ("base_lattice_basis", "multiplicities", "classes"), "trivial profile model")
    base = _lattice(raw["base_lattice_basis"], ring.D)
    relative = base.inv() * period
    if any(a.q != 1 for a in relative):
        raise ValueError("Period is not a sublattice of the model base")
    if not isinstance(raw["multiplicities"], list):
        raise ValueError("Multiplicities must be an ordered list")
    layers = {}
    for pair in raw["multiplicities"]:
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError("Malformed multiplicity")
        j, t = _int(pair[0], "degree", 1), _int(pair[1], "multiplicity", 1)
        if j >= ring.D or j in layers:
            raise ValueError("Duplicate or out-of-range profile degree")
        layers[j] = t
    if list(layers) != sorted(layers) or len(layers) > 2:
        raise ValueError("Only an ordered at-most-two-layer profile is supported")
    if not isinstance(raw["classes"], list) or len(raw["classes"]) != int(len(layers) == 2):
        raise ValueError("Wrong number of two-layer gluing classes")
    cls = None
    if raw["classes"]:
        item = raw["classes"][0]
        _keys(item, ("upper", "lower", "coefficients"), "model class")
        i, j = sorted(layers)
        if _int(item["lower"], "lower", 1) != i or _int(item["upper"], "upper", 2) != j:
            raise ValueError("Class endpoints do not match the model")
        A = _matrix_from_dict(item["coefficients"], ring, "coefficients")
        q = j - i + 1
        _bound(comb(ring.D, q) * max(layers[i], layers[j]), exterior_cap,
               "exterior coefficient basis")
        if A.shape != (layers[i], comb(ring.D, q) * layers[j]):
            raise ValueError("Wrong exterior coefficient matrix shape")
        if any(not f.is_constant for f in A.entries.values()):
            raise ValueError("Exterior coefficients must be constants")
        cls = (i, j, A)
    return layers, cls, relative


def _restricted_coefficients(cls, layers, relative, ring, cap):
    i, j, A = cls
    q = j - i + 1
    basis = tuple(combinations(range(ring.D), q))
    out, minors, work = {}, {}, 0
    for (row, column), coefficient in A.entries.items():
        I, copy = basis[column // layers[j]], column % layers[j]
        for J in basis:
            work += 1
            _bound(work, cap, "minor restriction work")
            key = I, J
            if key not in minors:
                minors[key] = int(relative.extract(I, J).det()) % ring.p
            a = minors[key] * coefficient.constant() % ring.p
            position = J, row, copy
            value = (out.get(position, 0) + a) % ring.p
            if value:
                out[position] = value
            else:
                out.pop(position, None)
    return out


def _contraction(coefficients, i, j, layers, degree, ring, cap):
    q, output_degree = j - i + 1, degree - (j - i + 1)
    ncols, nrows = comb(ring.D, degree) * layers[j], comb(ring.D, output_degree) * layers[i]
    _bound(ncols + nrows, cap, "contraction basis")
    targets = {K: index for index, K in enumerate(combinations(range(ring.D), output_degree))}
    terms = {}
    for (I, a, b), coefficient in coefficients.items():
        terms.setdefault(I, []).append((a, b, coefficient))
    entries, work = {}, 0
    for column_cell, J in enumerate(combinations(range(ring.D), degree)):
        for positions in combinations(range(degree), q):
            I = tuple(J[position] for position in positions)
            K = tuple(J[position] for position in range(degree) if position not in positions)
            sign = -1 if (sum(positions) - q * (q - 1) // 2) % 2 else 1
            for a, b, coefficient in terms.get(I, ()):
                work += 1
                _bound(work, cap, "contraction assembly")
                at = targets[K] * layers[i] + a, column_cell * layers[j] + b
                value = (entries.get(at, 0) + sign * coefficient) % ring.p
                if value:
                    entries[at] = value
                else:
                    entries.pop(at, None)
                _bound(len(entries), cap, "contraction entries")
    return FiniteMatrix(ring.p, nrows, ncols, entries)


def _verify_source_profile(analysis, source, ring):
    from .analysis import verify_analysis
    report = verify_analysis(analysis)
    if not report.valid:
        raise ValueError("Invalid bound profile analysis: " + "; ".join(report.errors))
    HX, HZ, supplied_basis = parse_input(source)
    if supplied_basis is not None or HX.ring != ring or analysis["ring"] != source["ring"]:
        raise ValueError("Profile/source ring or input convention mismatch")
    if (HX != _matrix_from_dict(analysis["input"]["HX"], ring, "analysis HX") or
            HZ != _matrix_from_dict(analysis["input"]["HZ"], ring, "analysis HZ")):
        raise ValueError("Profile analysis is not bound to the counted CSS input")
    layers = {item["degree"]: _matrix_from_dict(item["representatives"], ring,
                         "layer representatives").ncols for item in analysis["layers"]}
    return layers


def _bind_model(model, layers, analysis, period, ring):
    if len(layers) == 2:
        record = analysis["attachments"][0]
        if model["base_lattice_basis"] != analysis["initial_basis"] or record["basis"] != analysis["initial_basis"]:
            raise ValueError("Two-layer coefficients are not on the verified initial lattice")
        item = model["classes"][0]
        witness = record["gluing"]
        if len(witness["classes"]) != 1:
            raise ValueError("Expected exactly one verified two-layer attachment")
        original = witness["classes"][0]
        if (item["upper"] != witness["upper"] or item["lower"] != original["lower"] or
                _matrix_from_dict(item["coefficients"], ring, "model coefficients") !=
                _matrix_from_dict(original["coefficients"], ring, "verified coefficients")):
            raise ValueError("Model class differs from the actual verified gluing")
    else:
        # No gluing data is needed, but action triviality remains necessary.
        for item in analysis["layers"]:
            size = layers[item["degree"]]
            actions = [_matrix_from_dict(A, ring, "action") for A in item["actions"]]
            for column in range(ring.D):
                action = eye(ring, size)
                for A, exponent in zip(actions, period[:, column]):
                    action = action * matrix_power(A, int(exponent))
                if action != eye(ring, size):
                    raise ValueError("Fast single-layer period action is nontrivial")


def verify_periodic(data, *, max_qudits=DEFAULT_QUDIT_LIMIT,
                    max_matrix_entries=DEFAULT_ENTRY_LIMIT,
                    max_rank_work=DEFAULT_WORK_LIMIT,
                    max_exterior_entries=DEFAULT_ENTRY_LIMIT):
    """Verify input, period and actual ranks; resource exhaustion propagates."""
    checks, errors = [], []
    try:
        if hasattr(data, "certificate"):
            data = data.certificate()
        qudit_cap = _cap(max_qudits, "max_qudits")
        entry_cap = _cap(max_matrix_entries, "max_matrix_entries")
        work_cap = _cap(max_rank_work, "max_rank_work")
        exterior_cap = _cap(max_exterior_entries, "max_exterior_entries")
        if not isinstance(data, dict):
            raise ValueError("Periodic certificate must be a dictionary")
        version = _int(data["version"], "version")
        fields = ("schema", "version", "ring", "lattice_basis", "method", "source",
                  "model", "profile_analysis", "count")
        _keys(data, fields + (("diagnostics",) if version == 2 else ()), "periodic certificate")
        if data["schema"] != "css-decoupling.periodic" or version not in (1, 2):
            raise ValueError("Unsupported periodic certificate")
        _keys(data["ring"], ("prime", "dimension"), "ring")
        ring = Ring(_int(data["ring"]["prime"], "prime", 2),
                    _int(data["ring"]["dimension"], "dimension", 1))
        period = _lattice(data["lattice_basis"], ring.D)
        index = abs(int(period.det()))
        expected = dict(index=index, logical_qudits=0, physical_qudits=None,
                        rank_x=None, rank_z=None, incoming_rank=None, outgoing_rank=None,
                        model_total=None, upper_bound=None, decouplable=None)
        analysis_layers = None
        contributions = None
        if data["profile_analysis"] is not None:
            if data["source"] is None:
                raise ValueError("A physical profile count needs its original CSS source")
            analysis_layers = _verify_source_profile(data["profile_analysis"], data["source"], ring)
            checks.append("verified topological source model, all finite layers and recorded translations")
        if version == 2:
            if analysis_layers is None:
                raise ValueError("Layer diagnostics require a complete verified physical profile")
            contributions = []
            for item in sorted(data["profile_analysis"]["layers"], key=lambda item: item["degree"]):
                actions = [_matrix_from_dict(A, ring, "layer action") for A in item["actions"]]
                contribution, _ = _verified_layer(actions, item["degree"], period, ring,
                                                  exterior_cap, entry_cap, work_cap)
                contributions.append(contribution)
        if data["method"] == "finite":
            if data["source"] is None or data["model"] is not None:
                raise ValueError("Finite count requires CSS source and no abstract model")
            HX, HZ, supplied_basis = parse_input(data["source"])
            if HX.ring != ring or supplied_basis is not None or HX.ncols != HZ.ncols:
                raise ValueError("Incompatible finite source")
            if HX * HZ.dagger():
                raise ValueError("Source CSS checks do not commute")
            _bound(index * max(HX.ncols, 1), qudit_cap, "finite qudits/cells")
            lattice = Lattice(data["lattice_basis"], max_index=None)
            X = _finite(HX, lattice, qudit_cap, entry_cap, work_cap)
            Z = _finite(HZ, lattice, qudit_cap, entry_cap, work_cap)
            rx, rz = _rank(X, entry_cap, work_cap), _rank(Z, entry_cap, work_cap)
            expected.update(physical_qudits=index * HX.ncols, rank_x=rx, rank_z=rz,
                            logical_qudits=index * HX.ncols - rx - rz)
            checks.append("periodic check matrices reconstructed from input and integer lattice; independent column ranks")
        elif data["method"] == "single-layer":
            if version != 2 or analysis_layers is None or len(analysis_layers) > 1 or data["model"] is not None:
                raise ValueError("Single-layer fast count requires a verified profile with at most one layer")
            HX, _, _ = parse_input(data["source"])
            expected.update(physical_qudits=HX.ncols * index,
                            logical_qudits=sum(c["logical_qudits"] for c in contributions),
                            model_total=sum(c["model_total"] for c in contributions),
                            incoming_rank=sum(c["incoming_rank"] for c in contributions),
                            outgoing_rank=sum(c["outgoing_rank"] for c in contributions))
            checks.append("single-layer total from original period actions; no periodic-cell expansion")
        elif data["method"] == "two-layer":
            if ring.D < 2:
                raise ValueError("CSS profile dimension must be at least two")
            layers, cls, relative = _read_model(data["model"], ring, period, exterior_cap)
            total = sum(comb(ring.D, j) * t for j, t in layers.items())
            incoming = outgoing = 0
            if cls is not None:
                i, j, _ = cls
                coefficients = _restricted_coefficients(cls, layers, relative, ring, exterior_cap)
                incoming = _rank(_contraction(coefficients, i, j, layers, j + 1, ring, exterior_cap), entry_cap, work_cap)
                outgoing = _rank(_contraction(coefficients, i, j, layers, j, ring, exterior_cap), entry_cap, work_cap)
            expected.update(model_total=total, incoming_rank=incoming, outgoing_rank=outgoing,
                            logical_qudits=total - incoming - outgoing)
            if analysis_layers is not None:
                if analysis_layers != layers:
                    raise ValueError("Formal model differs from the verified physical layers")
                _bind_model(data["model"], layers, data["profile_analysis"], period, ring)
                HX, _, _ = parse_input(data["source"])
                expected["physical_qudits"] = HX.ncols * index
                checks.append("two-layer formal model bound to the actual initial profile and gluing")
            elif data["source"] is not None:
                raise ValueError("Unverified CSS source cannot certify a formal-model physical count")
            checks.append("minor pullback and both signed exterior contraction ranks independently recomputed")
        else:
            raise ValueError("Unknown periodic counting method")
        if analysis_layers is not None:
            upper = sum(comb(ring.D, j) * t for j, t in analysis_layers.items())
            if expected["logical_qudits"] > upper:
                raise ValueError("Computed count violates the verified profile bound")
            expected.update(upper_bound=upper, decouplable=expected["logical_qudits"] == upper)
            checks.append("Corollary-3 saturation/decoupling conclusion uses verified profile assumptions")
        if version == 2:
            diagnostics = data["diagnostics"]
            _keys(diagnostics, ("layers", "translation_loss", "gluing_loss"), "layer diagnostics")
            if not isinstance(diagnostics["layers"], list) or len(diagnostics["layers"]) != len(contributions):
                raise ValueError("Wrong number of layer contribution records")
            for raw, contribution in zip(diagnostics["layers"], contributions):
                _compare_contribution(raw, contribution)
            layer_total = sum(c["logical_qudits"] for c in contributions)
            losses = dict(translation_loss=expected["upper_bound"] - layer_total,
                          gluing_loss=layer_total - expected["logical_qudits"])
            for key, loss in losses.items():
                if _int(diagnostics[key], key, 0) != loss:
                    raise ValueError(f"Incorrect periodic layer diagnostic: {key}")
            checks.append("layer contributions and separate translation/gluing losses independently verified")
        _keys(data["count"], tuple(expected), "count")
        for key, value in data["count"].items():
            if key == "decouplable":
                if value is not None and type(value) is not bool:
                    raise ValueError("decouplable must be boolean or null")
            elif value is not None:
                _int(value, key, 0)
            if value != expected[key]:
                raise ValueError(f"Incorrect periodic count field: {key}")
    except (ValueError, TypeError, KeyError, AttributeError, IndexError) as exc:
        errors.append(str(exc))
    scope = ("verified-physical-periodic-count" if isinstance(data, dict) and data.get("profile_analysis") is not None
             else "periodic-count-without-topology-claim")
    return VerificationReport(not errors, tuple(checks), tuple(errors), scope)
