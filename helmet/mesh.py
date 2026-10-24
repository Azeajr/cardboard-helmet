"""Planar-faceted meshes and the grid ("column") surfaces they are built from."""

import math
import numpy as np

from . import geom2d as g2


class Mesh:
    """Vertices plus planar polygon faces, all oriented with outward normals."""

    def __init__(self, name):
        self.name = name
        self.verts = []
        self._key = {}
        self.faces = []      # dicts: id, vi, island

    def vid(self, p):
        p = np.asarray(p, float)
        k = tuple(np.round(p, 5))
        if k not in self._key:
            self._key[k] = len(self.verts)
            self.verts.append(p)
        return self._key[k]

    def add_face(self, pts, fid, outward=None, **attrs):
        vi = []
        for p in pts:
            i = self.vid(p)
            if not vi or vi[-1] != i:
                vi.append(i)
        if len(vi) > 1 and vi[0] == vi[-1]:
            vi.pop()
        if len(vi) < 3:
            return None
        f = dict(id=fid, vi=vi, **attrs)
        if outward is not None and np.dot(self.normal_of(vi), outward(self.center_of(vi))) < 0:
            f["vi"] = vi[::-1]
        self.faces.append(f)
        return f

    def pts(self, f):
        return np.array([self.verts[i] for i in f["vi"]])

    def center_of(self, vi):
        return np.mean([self.verts[i] for i in vi], axis=0)

    def normal_of(self, vi):
        p = np.array([self.verts[i] for i in vi])
        n = np.zeros(3)
        for a, b in zip(p, np.roll(p, -1, axis=0)):  # Newell
            n += np.cross(a, b)
        return n / (np.linalg.norm(n) + 1e-15)

    def normal(self, f):
        return self.normal_of(f["vi"])

    def edges(self):
        """edge key (a, b) with a < b -> list of (face index, i) where the face
        runs vi[i] -> vi[i+1] along that edge."""
        out = {}
        for fi, f in enumerate(self.faces):
            vi = f["vi"]
            for i in range(len(vi)):
                a, b = vi[i], vi[(i + 1) % len(vi)]
                out.setdefault((min(a, b), max(a, b)), []).append((fi, i))
        return out

    def planarity(self, f):
        p = self.pts(f)
        n = self.normal(f)
        d = (p - p.mean(axis=0)) @ n
        return float(np.abs(d).max())

    def triangles_of(self, f):
        """Ear-clip triangulation of one face, as vertex-index triples."""
        p = self.pts(f)
        n = self.normal(f)
        e1 = p[1] - p[0]
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(n, e1)
        q = [((v - p[0]) @ e1, (v - p[0]) @ e2) for v in p]
        return [(f["vi"][a], f["vi"][b], f["vi"][c]) for a, b, c in ear_clip(q)]

    def triangles(self):
        """Triangulation of every face (faces are simple polygons)."""
        return [t for f in self.faces for t in self.triangles_of(f)]


def ear_clip(poly):
    """Triangulate a simple CCW polygon; returns index triples."""
    idx = list(range(len(poly)))
    if g2.area(poly) < 0:
        idx.reverse()
    out = []
    guard = 0
    while len(idx) > 3 and guard < 10000:
        guard += 1
        for k in range(len(idx)):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
            a, b, c = np.asarray(poly[i0]), np.asarray(poly[i1]), np.asarray(poly[i2])
            if (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) <= 1e-12:
                continue
            tri = [a, b, c]
            if any(g2.point_in_poly(poly[j], tri) for j in idx if j not in (i0, i1, i2)):
                continue
            out.append((i0, i1, i2))
            idx.pop(k)
            break
        else:
            break
    if len(idx) == 3:
        out.append(tuple(idx))
    return out


class Grid:
    """Surface between fold lines x = xs[i] and rows at angles phis[j] about the
    pivot axis: vertex(i, j) = pivot + R[i, j] * dir(phis[j]).

    When neighbouring columns are scaled copies of one profile (the shell) every
    cell is a planar trapezoid. Otherwise (the faceplate) a cell is split along
    the diagonal that folds outward into two triangles, which are always flat.
    Param space is (x, phi): a cell's pieces map affinely (triangles) or
    bilinearly (planar quads) onto 3D."""

    def __init__(self, name, xs, phis, R, pivot):
        self.name = name
        self.xs = np.asarray(xs, float)
        self.phis = np.asarray(phis, float)
        self.R = np.asarray(R, float)
        self.pivot = pivot  # (y, z)
        self.split = {}
        for i in range(len(self.xs) - 1):
            for j in range(len(self.phis) - 1):
                self.split[(i, j)] = self._choose_split(i, j)

    @classmethod
    def scaled(cls, name, xs, scales, phis, radii, pivot):
        return cls(name, xs, phis, np.outer(scales, radii), pivot)

    def vertex(self, i, j):
        return point_at(self.pivot, self.xs[i], self.R[i, j], self.phis[j])

    def corners(self, i, j):
        return [self.vertex(i, j), self.vertex(i + 1, j),
                self.vertex(i + 1, j + 1), self.vertex(i, j + 1)]

    def _choose_split(self, i, j):
        A, B, C, D = self.corners(i, j)
        n = np.cross(B - A, C - A)
        n /= np.linalg.norm(n)
        off = float(np.dot(n, D - A))
        if abs(off) < 1e-7:
            return None
        out = np.array([0.0, A[1] - self.pivot[0], A[2] - self.pivot[1]])
        if np.dot(n, out) < 0:
            n, off = -n, -off
        return "AC" if off < 0 else "BD"   # the diagonal whose fold bulges outward

    # (s, t) corners of each piece of a cell: A=(0,0) B=(1,0) C=(1,1) D=(0,1)
    _PIECES = {None: [("q", [(0, 0), (1, 0), (1, 1), (0, 1)])],
               "AC": [("a", [(0, 0), (1, 0), (1, 1)]), ("b", [(0, 0), (1, 1), (0, 1)])],
               "BD": [("a", [(0, 0), (1, 0), (0, 1)]), ("b", [(1, 0), (1, 1), (0, 1)])]}

    def pieces(self, i, j):
        """[(key, param polygon in (x, phi), 3D polygon)] for cell (i, j)."""
        out = []
        for tag, st in self._PIECES[self.split[(i, j)]]:
            par = [self._xp(i, j, s, t) for s, t in st]
            out.append(((i, j, tag), par, [self._corner(i, j, s, t) for s, t in st]))
        return out

    def _xp(self, i, j, s, t):
        return np.array([self.xs[i] + s * (self.xs[i + 1] - self.xs[i]),
                         self.phis[j] + t * (self.phis[j + 1] - self.phis[j])])

    def _corner(self, i, j, s, t):
        return self.vertex(i + int(s), j + int(t))

    def cell_of(self, x, phi):
        i = int(np.clip(np.searchsorted(self.xs, x) - 1, 0, len(self.xs) - 2))
        ph = self.phis
        if ph[0] > ph[-1]:
            j = int(np.clip(np.searchsorted(-ph, -phi) - 1, 0, len(ph) - 2))
        else:
            j = int(np.clip(np.searchsorted(ph, phi) - 1, 0, len(ph) - 2))
        return i, j

    def st(self, i, j, x, phi):
        s = (x - self.xs[i]) / (self.xs[i + 1] - self.xs[i])
        t = (phi - self.phis[j]) / (self.phis[j + 1] - self.phis[j])
        return s, t

    def _tag(self, i, j, s, t):
        sp = self.split[(i, j)]
        if sp is None:
            return "q"
        if sp == "AC":
            return "a" if s >= t else "b"
        return "a" if s + t <= 1 else "b"

    def locate(self, x, phi, cell=None):
        i, j = cell if cell else self.cell_of(x, phi)
        s, t = self.st(i, j, x, phi)
        return (i, j, self._tag(i, j, s, t))

    def point(self, x, phi, key=None):
        if key is None or len(key) == 2:
            key = self.locate(x, phi, key)
        i, j, tag = key
        s, t = self.st(i, j, x, phi)
        A, B, C, D = self.corners(i, j)
        if tag == "q":
            return (1 - s) * (1 - t) * A + s * (1 - t) * B + s * t * C + (1 - s) * t * D
        if self.split[(i, j)] == "AC":
            # a: A + s(B-A) + t(C-B);  b: A + s(C-D) + t(D-A)
            return A + s * (B - A) + t * (C - B) if tag == "a" else A + s * (C - D) + t * (D - A)
        return A + s * (B - A) + t * (D - A) if tag == "a" else C + (1 - s) * (D - C) + (1 - t) * (B - C)

    def invert(self, key, p):
        """(x, phi) of a 3D point lying on piece `key`."""
        i, j, tag = key
        A, B, C, D = self.corners(i, j)
        s, t = 0.5, 0.5
        for _ in range(40):
            q = self.point(*self._xp(i, j, s, t), key)
            h = 1e-6
            ds = (self.point(*self._xp(i, j, s + h, t), key) - q) / h
            dt = (self.point(*self._xp(i, j, s, t + h), key) - q) / h
            step = np.linalg.lstsq(np.column_stack([ds, dt]), np.asarray(p) - q, rcond=None)[0]
            s, t = s + step[0], t + step[1]
            if np.linalg.norm(step) < 1e-11:
                break
        return tuple(self._xp(i, j, s, t))

    def cell_rect(self, i, j):
        lo, hi = sorted((self.phis[j], self.phis[j + 1]))
        return self.xs[i], self.xs[i + 1], lo, hi

    def clip(self, poly):
        """Clip a param-space polygon to every piece: [(key, param poly, 3D poly)]."""
        from . import geom2d as g2
        out = []
        poly = [np.asarray(p, float) for p in poly]
        for i in range(len(self.xs) - 1):
            for j in range(len(self.phis) - 1):
                x0, x1, p0, p1 = self.cell_rect(i, j)
                cell = g2.clip_rect(poly, x0, x1, p0, p1)
                if len(cell) < 3:
                    continue
                for key, par, _ in self.pieces(i, j):
                    c = cell
                    if key[2] != "q":
                        for k in range(3):     # clip by each triangle edge
                            a, b, o = par[k], par[(k + 1) % 3], par[(k + 2) % 3]
                            e = b - a
                            nrm = np.array([-e[1], e[0]])
                            if np.dot(o - a, nrm) < 0:
                                nrm = -nrm
                            c = g2.clip_halfplane(c, a, nrm)
                            if len(c) < 3:
                                break
                    if len(c) >= 3 and abs(g2.area(c)) > 1e-6:
                        out.append((key, c, [self.point(q[0], q[1], key) for q in c]))
        return out

    def split_segment(self, p, q):
        """Split a param segment where it crosses grid lines and diagonals:
        [(key, p0, q0)] with each sub-segment inside one piece."""
        p, q = np.asarray(p, float), np.asarray(q, float)
        ts = [0.0, 1.0]
        for x in self.xs:
            if (p[0] - x) * (q[0] - x) < 0:
                ts.append((x - p[0]) / (q[0] - p[0]))
        for ph in self.phis:
            if (p[1] - ph) * (q[1] - ph) < 0:
                ts.append((ph - p[1]) / (q[1] - p[1]))
        ts = sorted(ts)
        out = []
        for t0, t1 in zip(ts, ts[1:]):
            if t1 - t0 < 1e-12:
                continue
            a, b = p + (q - p) * t0, p + (q - p) * t1
            cell = self.cell_of(*((a + b) / 2))
            sp = self.split[cell]
            cuts = [0.0, 1.0]
            if sp is not None:
                i, j = cell
                sa, ta = self.st(i, j, *a)
                sb, tb = self.st(i, j, *b)
                fa = sa - ta if sp == "AC" else sa + ta - 1
                fb = sb - tb if sp == "AC" else sb + tb - 1
                if fa * fb < 0:
                    cuts.append(fa / (fa - fb))
            cuts = sorted(cuts)
            for u0, u1 in zip(cuts, cuts[1:]):
                a0, b0 = a + (b - a) * u0, a + (b - a) * u1
                out.append((self.locate(*((a0 + b0) / 2), cell), a0, b0))
        return out


def point_at(pivot, x, r, phi_deg):
    a = math.radians(phi_deg)
    return np.array([x, pivot[0] - r * math.cos(a), pivot[1] + r * math.sin(a)])


def polar(pts, pivot):
    """(r, phi) of 3D points about the pivot axis; phi in (-90, 270]."""
    pts = np.asarray(pts, float)
    f = pivot[0] - pts[:, 1]
    u = pts[:, 2] - pivot[1]
    r = np.hypot(f, u)
    phi = np.degrees(np.arctan2(u, f))
    phi = np.where(phi <= -90.0, phi + 360.0, phi)
    return r, phi


def profile_radius(phis, radii, phi):
    """Radius of the polygonal profile (vertices at phis/radii) along angle phi."""
    phis = np.asarray(phis, float)
    radii = np.asarray(radii, float)
    if phis[0] > phis[-1]:
        phis, radii = phis[::-1], radii[::-1]
    phi = np.atleast_1d(np.asarray(phi, float))
    k = np.clip(np.searchsorted(phis, phi) - 1, 0, len(phis) - 2)
    a0, a1 = np.radians(phis[k]), np.radians(phis[k + 1])
    r0, r1 = radii[k], radii[k + 1]
    A = np.stack([r0 * np.cos(a0), r0 * np.sin(a0)], -1)
    B = np.stack([r1 * np.cos(a1), r1 * np.sin(a1)], -1)
    ang = np.radians(phi)
    d = np.stack([np.cos(ang), np.sin(ang)], -1)
    e = B - A
    # solve A + s e = rho d  ->  rho = cross(A, e) / cross(d, e)
    num = A[:, 0] * e[:, 1] - A[:, 1] * e[:, 0]
    den = d[:, 0] * e[:, 1] - d[:, 1] * e[:, 0]
    return num / den


def chord_min_dist(phis, radii):
    """Smallest distance from the axis to any chord of the profile."""
    best = math.inf
    for j in range(len(phis) - 1):
        a = np.array([radii[j] * math.cos(math.radians(phis[j])), radii[j] * math.sin(math.radians(phis[j]))])
        b = np.array([radii[j + 1] * math.cos(math.radians(phis[j + 1])), radii[j + 1] * math.sin(math.radians(phis[j + 1]))])
        e = b - a
        t = np.clip(-np.dot(a, e) / np.dot(e, e), 0, 1)
        best = min(best, float(np.linalg.norm(a + t * e)))
    return best
