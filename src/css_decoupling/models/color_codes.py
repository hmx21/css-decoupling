"""Geometric, periodic qubit color codes, supplied as raw Laurent checks.

The Freudenthal triangulation of Z^D has one D-simplex for each coordinate
permutation in every unit cube. Its vertex coloring is sum(v) mod (D+1).
Qubits occupy D-simplices; an r-simplex check acts on its full D-star.
This is the simplicial definition of the color code, not a toric-stack
presentation: Kubica--Yoshida--Pastawski, *Unfolding the color code*,
https://arxiv.org/abs/1503.02065, Appendix (color-code definition and
general unfolding theorem). Our ``degree=j`` places X checks on
(j-1)-simplices and Z checks on (D-j-1)-simplices. In the paper's notation
this is CC_(D-j-1); our toric X/Z degree convention uses j.

Translations of the primitive cell can permute colors. Their kernel is
sum(v)=0 mod (D+1); no color-preserving supercell is expanded here.
For D=4 there are 24 qubits, with check shapes (1,24)/(50,24) at j=1
and (15,24)/(15,24) at j=2. These are geometric input facts, not a claim
that the decoupling pipeline has completed for every such input.

Only qubits and D>=3 are included; odd-prime oriented color codes require
additional conventions and must not be obtained by changing the modulus.
"""
from dataclasses import dataclass
from itertools import combinations, permutations
from math import comb

from css_decoupling.algebra.laurent import Matrix, Poly, Ring
from css_decoupling.errors import ResourceLimit


COLOR_CODE_REFERENCE = "https://arxiv.org/abs/1503.02065"
BCC_CHECKS_SOURCE = "3D color codes/color_codes/truncated_octahedral/checks.json"
BCC_CHECKS_SOURCE_SHA256 = "19144596df8e6b3a2839a83d5dd68f88894cbe6c8c444da7b9cb614a7d2858d4"
FCC_CHECKS_SOURCE = "3D color codes/color_codes/chamfered_cubic/checks.json"
FCC_CHECKS_SOURCE_SHA256 = "46c0f3580b397df495dfdf141724846339f102786f26eec65d3fa87145451229"
FCC_GEOMETRY_REFERENCE = (
    "https://github.com/larsesser/qcodeplot3d/tree/"
    "69acc69fc195bb5da2393a39d119b5cadd2e5b12"
)


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _parameters(dimension, degree, max_qudits, max_incidence_terms):
    _positive_int(dimension, "dimension")
    _positive_int(degree, "degree")
    if dimension < 3 or degree >= dimension:
        raise ValueError("Require dimension >= 3 and 1 <= degree < dimension")
    _positive_int(max_qudits, "max_qudits")
    _positive_int(max_incidence_terms, "max_incidence_terms")
    # Stop before allocating permutations, faces, or an enormous factorial.
    qudits = 1
    for factor in range(2, dimension + 1):
        qudits *= factor
        if qudits > max_qudits:
            raise ResourceLimit(f"Freudenthal color code exceeds {max_qudits} qudits per cell")
    terms = qudits * (comb(dimension + 1, degree) + comb(dimension + 1, dimension - degree))
    if terms > max_incidence_terms:
        raise ResourceLimit(f"Color-code star incidence needs {terms} terms; cap is {max_incidence_terms}")


def _normalize_face(vertices):
    anchor = vertices[0]
    return tuple(tuple(a - b for a, b in zip(vertex, anchor)) for vertex in vertices), anchor


@dataclass(frozen=True)
class FreudenthalColorGeometry:
    """Primitive-cell simplices and canonical translated face types.

    Coordinates are integers in the primitive Z^D basis. A simplex's
    vertices are ordered along its coordinate path. Every face is shifted
    so its first vertex is zero, then face types are lexicographically
    ordered. All tuples are immutable; no matrices or computed answers are
    cached in this geometry object.
    """

    dimension: int
    degree: int
    simplices: tuple
    x_faces: tuple
    z_faces: tuple

    @property
    def color_preserving_basis(self):
        """Column basis e_i-e_last and (D+1)e_last, of index D+1."""
        D = self.dimension
        return tuple(tuple((int(i == j) if i < D - 1 else
                            (-1 if j < D - 1 else D + 1))
                           for j in range(D)) for i in range(D))


def freudenthal_color_geometry(dimension=4, degree=1, *, max_qudits=720,
                               max_incidence_terms=100000):
    """Construct the actual periodic triangulation and the two face families.

    D! qubits and the sum of both star-incidence term counts are checked
    before allocation. The defaults admit D<=6, subject to the term cap;
    larger finite outputs can be requested by explicit larger limits.
    """
    _parameters(dimension, degree, max_qudits, max_incidence_terms)
    simplices = []
    for order in permutations(range(dimension)):
        vertex = [0] * dimension
        vertices = [tuple(vertex)]
        for axis in order:
            vertex[axis] = 1
            vertices.append(tuple(vertex))
        simplices.append(tuple(vertices))

    def faces(size):
        return tuple(sorted({_normalize_face(vertices)[0]
                             for simplex in simplices
                             for vertices in combinations(simplex, size)}))

    return FreudenthalColorGeometry(dimension, degree, tuple(simplices),
                                   faces(degree), faces(dimension - degree))


def _star_checks(ring, simplices, faces):
    rows = {face: row for row, face in enumerate(faces)}
    terms = {}
    size = len(faces[0])
    for column, simplex in enumerate(simplices):
        for vertices in combinations(simplex, size):
            face, anchor = _normalize_face(vertices)
            terms.setdefault((rows[face], column), {})[anchor] = 1
    # H maps a qubit at v to a check at v+anchor; H^dagger's columns
    # are the physical Pauli supports, hence the positive exponents here.
    return Matrix(ring, len(faces), len(simplices),
                  {position: Poly(ring, support) for position, support in terms.items()})


def freudenthal_color_css(dimension=4, degree=1, *, max_qudits=720,
                          max_incidence_terms=100000):
    """Return (H_X,H_Z) for the geometric qubit color code in dimension D.

    No resolution, comparison map, toric checks, or decoupling answer is
    used. ``degree=D-j`` exchanges X and Z relative to ``degree=j``.
    The checks commute on the infinite lattice, including primitive
    translations which permute colors.
    """
    geometry = freudenthal_color_geometry(dimension, degree, max_qudits=max_qudits,
                                         max_incidence_terms=max_incidence_terms)
    ring = Ring(2, dimension)
    return (_star_checks(ring, geometry.simplices, geometry.x_faces),
            _star_checks(ring, geometry.simplices, geometry.z_faces))


def bcc_color_css(*, dual=False):
    """Three-dimensional BCC/truncated-octahedral color-code checks.

    The Freudenthal dual is combinatorially the BCC Voronoi colex.
    Primitive physical translations are (1,1,-1), (1,-1,1), (-1,1,1).
    The six qubit columns agree with the project's geometric BCC example;
    the seven Z rows here use lexicographic normalized-edge order. Relative
    to BCC_CHECKS_SOURCE, the Z-row permutation is (2,1,5,0,4,3,6), while
    the X row is unchanged. The source digest is recorded for cross-checks;
    this model is generated from simplices and never reads that source.
    """
    if type(dual) is not bool:
        raise ValueError("dual must be boolean")
    return freudenthal_color_css(3, 2 if dual else 1)


def fcc_color_css(*, dual=False):
    """Exact cubic/chamfered-cubic color code, with ten qubits per cell.

    FCC translation columns are (1,1,0), (1,0,1), (0,1,1). X checks
    occupy the cube and chamfered-cube cells, Z checks their twelve face
    types. Constants below are the uncoarsened primal HX/HZ from the
    independently geometry-verified project source identified by
    FCC_CHECKS_SOURCE and FCC_CHECKS_SOURCE_SHA256. The wheel has no runtime
    dependence on that file, the archive, or qCodePlot3D. ``dual`` swaps
    only Pauli labels, preserving columns and primitive translations.
    """
    if type(dual) is not bool:
        raise ValueError("dual must be boolean")
    ring = Ring(2, 3)

    def matrix(rows, entries):
        return Matrix(ring, rows, 10, {(i, j): Poly(ring, {e: 1 for e in support})
                                       for i, j, support in entries})

    HX, HZ = matrix(2, _FCC_X), matrix(12, _FCC_Z)
    return (HZ, HX) if dual else (HX, HZ)


# Exact GF(2) Laurent supports; the source JSON digest is recorded above.
_FCC_X = (
    (0, 1, ((0, 0, 0),)), (0, 2, ((0, 0, 1),)), (0, 3, ((0, 1, 0),)),
    (0, 4, ((1, 0, 0),)), (0, 6, ((0, 1, 1),)), (0, 7, ((1, 0, 1),)),
    (0, 8, ((1, 1, 0),)), (0, 9, ((1, 1, 1),)),
    (1, 0, ((-1, 0, 1), (0, -1, 1), (0, 0, 0), (0, 0, 1))),
    (1, 1, ((-1, 0, 1), (0, -1, 1), (0, 0, 0))),
    (1, 2, ((-1, 0, 1), (0, -1, 1), (0, 0, 1))),
    (1, 3, ((-1, 0, 1), (0, 0, 0), (0, 0, 1))),
    (1, 4, ((0, -1, 1), (0, 0, 0), (0, 0, 1))),
    (1, 5, ((0, 0, 1), (0, 0, 2), (0, 1, 1), (1, 0, 1))),
    (1, 6, ((0, 0, 1), (0, 0, 2), (0, 1, 1))),
    (1, 7, ((0, 0, 1), (0, 0, 2), (1, 0, 1))),
    (1, 8, ((0, 0, 1), (0, 1, 1), (1, 0, 1))),
    (1, 9, ((0, 0, 2), (0, 1, 1), (1, 0, 1))),
)
_FCC_Z = (
    (0, 3, ((0, 1, 0),)), (0, 4, ((1, 0, 0),)), (0, 8, ((1, 1, 0),)), (0, 9, ((1, 1, 1),)),
    (1, 2, ((0, 0, 1),)), (1, 4, ((1, 0, 0),)), (1, 7, ((1, 0, 1),)), (1, 9, ((1, 1, 1),)),
    (2, 2, ((0, 0, 1),)), (2, 3, ((0, 1, 0),)), (2, 6, ((0, 1, 1),)), (2, 9, ((1, 1, 1),)),
    (3, 1, ((0, 0, 0),)), (3, 4, ((1, 0, 0),)), (3, 7, ((1, 0, 1),)), (3, 8, ((1, 1, 0),)),
    (4, 1, ((0, 0, 0),)), (4, 3, ((0, 1, 0),)), (4, 6, ((0, 1, 1),)), (4, 8, ((1, 1, 0),)),
    (5, 1, ((0, 0, 0),)), (5, 2, ((0, 0, 1),)), (5, 6, ((0, 1, 1),)), (5, 7, ((1, 0, 1),)),
    (6, 0, ((0, 0, 1),)), (6, 2, ((0, 0, 1),)), (6, 3, ((0, 0, 1),)),
    (6, 5, ((1, 0, 1),)), (6, 7, ((1, 0, 1),)), (6, 8, ((1, 0, 1),)),
    (7, 0, ((0, 0, 1),)), (7, 2, ((0, 0, 1),)), (7, 4, ((0, 0, 1),)),
    (7, 5, ((0, 1, 1),)), (7, 6, ((0, 1, 1),)), (7, 8, ((0, 1, 1),)),
    (8, 0, ((0, 0, 0),)), (8, 1, ((0, 0, 0),)), (8, 3, ((0, 0, 0),)),
    (8, 5, ((1, 0, 1),)), (8, 7, ((1, 0, 1),)), (8, 9, ((1, 0, 1),)),
    (9, 0, ((0, 0, 0),)), (9, 1, ((0, 0, 0),)), (9, 4, ((0, 0, 0),)),
    (9, 5, ((0, 1, 1),)), (9, 6, ((0, 1, 1),)), (9, 9, ((0, 1, 1),)),
    (10, 0, ((0, 0, 1),)), (10, 3, ((0, 0, 1),)), (10, 4, ((0, 0, 1),)),
    (10, 5, ((0, 0, 2),)), (10, 6, ((0, 0, 2),)), (10, 7, ((0, 0, 2),)),
    (11, 0, ((0, -1, 1),)), (11, 1, ((0, -1, 1),)), (11, 2, ((0, -1, 1),)),
    (11, 5, ((1, 0, 1),)), (11, 8, ((1, 0, 1),)), (11, 9, ((1, 0, 1),)),
)
