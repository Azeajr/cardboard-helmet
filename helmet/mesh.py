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
    pivot axis: vertex(i, j) = pivot + scale[i] * radius[j] * dir(phis[j]).
    Every cell is a planar trapezoid because neighbouring columns are scaled
    copies of the same profile about the same axis."""

    def __init__(self, name, xs, scales, phis, radii, pivot):
        self.name = name
        self.xs = np.asarray(xs, float)
        self.scales = np.asarray(scales, float)
        self.phis = np.asarray(phis, float)
        self.radii = np.asarray(radii, float)
        self.pivot = pivot  # (y, z)

    def vertex(self, i, j):
        return point_at(self.pivot, self.xs[i], self.scales[i] * self.radii[j], self.phis[j])

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

    def corners(self, i, j):
        return [self.vertex(i, j), self.vertex(i + 1, j),
                self.vertex(i + 1, j + 1), self.vertex(i, j + 1)]

    def point(self, x, phi, cell=None):
        i, j = cell if cell else self.cell_of(x, phi)
        s, t = self.st(i, j, x, phi)
        A, B, C, D = self.corners(i, j)
        return (1 - s) * (1 - t) * A + s * (1 - t) * B + s * t * C + (1 - s) * t * D

    def cell_rect(self, i, j):
        lo, hi = sorted((self.phis[j], self.phis[j + 1]))
        return self.xs[i], self.xs[i + 1], lo, hi


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
