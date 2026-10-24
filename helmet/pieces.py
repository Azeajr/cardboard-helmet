"""Cut the meshes into flat pieces: islands, unfolding, seam numbers, fold
lines and holes. Everything is computed in the outside view and mirrored at
the end, because the printed side of each piece is the inside.

Seams are butt joints: both edges are cut square at the outer surface, so the
outer corners meet and the board leaves a V on the inside for hot glue."""

import math
import numpy as np

from . import params as P
from . import geom2d as g2
from .mesh import Mesh

FLAT_FOLD = 0.2   # degrees; folds flatter than this are not drawn
SEAM_SIZE = 3.2   # seam numbers along each edge


class Piece:
    def __init__(self, name, body, kind, order, title=""):
        self.name, self.body, self.kind, self.order, self.title = name, body, kind, order, title
        self.faces = []
        self.poly = {}       # face index -> 2D polygon (printed view)
        self.cuts = []       # dict(a, b, seam, length)
        self.folds = []      # dict(a, b, kind, angle)
        self.holes = []      # (a, b) segments
        self.labels = []     # dict(text, pos, angle, size, bold)
        self.count = 1       # how many to cut

    def points(self):
        return np.array([p for poly in self.poly.values() for p in poly])

    def transform(self, R, t):
        f = lambda p: np.asarray(p) @ R.T + t
        self.poly = {k: f(v) for k, v in self.poly.items()}
        for c in self.cuts + self.folds:
            c["a"], c["b"] = f(c["a"]), f(c["b"])
        self.holes = [(f(a), f(b)) for a, b in self.holes]
        for lb in self.labels:
            lb["pos"] = f(lb["pos"])
            a = math.radians(lb["angle"])
            v = R @ np.array([math.cos(a), math.sin(a)])
            lb["angle"] = _readable(math.degrees(math.atan2(v[1], v[0])))


def _readable(a):
    a = (a + 180.0) % 360.0 - 180.0
    if a > 90.0:
        a -= 180.0
    elif a <= -90.0:
        a += 180.0
    return a


# --- unfolding ------------------------------------------------------------------
class Unfolder:
    """Lays one island flat by walking a spanning tree of its faces."""

    def __init__(self, mesh):
        self.m = mesh
        self.xf = {}

    def frame(self, fi):
        f = self.m.faces[fi]
        o = self.m.verts[f["vi"][0]]
        e1 = self.m.verts[f["vi"][1]] - o
        e1 = e1 / np.linalg.norm(e1)
        n = self.m.normal(f)
        e2 = np.cross(n, e1)
        return o, e1, e2

    @staticmethod
    def local(fr, p):
        o, e1, e2 = fr
        d = np.asarray(p) - o
        return np.array([d @ e1, d @ e2])

    def map(self, fi, p):
        fr, R, t = self.xf[fi]
        return R @ self.local(fr, p) + t

    def place(self, fi, a, b, Pa, Pb):
        """Transform for face fi putting 3D vertices a, b at 2D Pa, Pb."""
        fr = self.frame(fi)
        La, Lb = self.local(fr, self.m.verts[a]), self.local(fr, self.m.verts[b])
        ang = math.atan2(*(Pb - Pa)[::-1]) - math.atan2(*(Lb - La)[::-1])
        c, s = math.cos(ang), math.sin(ang)
        R = np.array([[c, -s], [s, c]])
        return fr, R, Pa - R @ La

    def unfold(self, faces, tree, root):
        fr = self.frame(root)
        self.xf[root] = (fr, np.eye(2), np.zeros(2))
        queue = [root]
        while queue:
            f = queue.pop(0)
            for g, a, b in tree.get(f, []):
                if g in self.xf:
                    continue
                Pa, Pb = self.map(f, self.m.verts[a]), self.map(f, self.m.verts[b])
                self.xf[g] = self.place(g, a, b, Pa, Pb)
                queue.append(g)
        missing = set(faces) - set(self.xf)
        if missing:
            raise RuntimeError(f"faces not reached by the fold tree: {sorted(missing)[:5]}")

    def poly2d(self, fi):
        return np.array([self.map(fi, self.m.verts[v]) for v in self.m.faces[fi]["vi"]])


def spanning_tree(mesh, faces, allow, root):
    """BFS tree over edges where allow(fa, fb) is true. Returns child lists
    {face: [(child, a, b)]} and the set of fold edge keys."""
    faces = set(faces)
    adj = {}
    for key, users in mesh.edges().items():
        if len(users) != 2:
            continue
        (f1, _), (f2, _) = users
        if f1 in faces and f2 in faces and allow(mesh.faces[f1], mesh.faces[f2]):
            adj.setdefault(f1, []).append((f2, key))
            adj.setdefault(f2, []).append((f1, key))
    tree, folds, seen, queue = {}, set(), {root}, [root]
    while queue:
        f = queue.pop(0)
        for g, key in sorted(adj.get(f, []), key=lambda e: e[0]):
            if g in seen:
                continue
            seen.add(g)
            folds.add(key)
            tree.setdefault(f, []).append((g, key[0], key[1]))
            queue.append(g)
    return tree, folds


# --- building pieces ---------------------------------------------------------------
class Body:
    """A mesh plus island assignment and holes, ready to be cut into pieces."""

    def __init__(self, mesh, grid=None):
        self.mesh = mesh
        self.grid = grid
        self.islands = []          # dict(name, faces, allow, root, order, title, kind)
        self.param_holes = []      # param-space polygons on the grid
        self.plate_holes = {}      # face index -> list of 3D polygons


def build_pieces(bodies, start_seam=1):
    """Unfold every island of every body; returns pieces and the seam table."""
    pieces = []
    seams = []
    seam_no = start_seam
    for body in bodies:
        m = body.mesh
        edges = m.edges()
        owner = {}
        for isl in body.islands:
            for f in isl["faces"]:
                owner[f] = isl["name"]
        unf = Unfolder(m)
        folds = set()
        isl_by_name = {}
        for isl in body.islands:
            tree, fk = spanning_tree(m, isl["faces"], isl["allow"], isl["root"])
            unf.unfold(isl["faces"], tree, isl["root"])
            folds |= fk
            pc = Piece(isl["name"], m.name, isl["kind"], isl["order"], isl.get("title", ""))
            pc.faces = list(isl["faces"])
            pc.count = isl.get("count", 1)
            for f in pc.faces:
                pc.poly[f] = unf.poly2d(f)
            isl_by_name[isl["name"]] = pc
            pieces.append(pc)

        vparam = _vertex_params(body) if body.grid is not None else {}
        hole_iv = lambda a, b: _hole_intervals(body, vparam, a, b)

        # cut edges: number the seams and size each joint
        cut_edges = []
        for key, users in edges.items():
            if len(users) == 2 and key in folds:
                continue
            if len(users) == 2 and owner.get(users[0][0]) and owner.get(users[1][0]):
                cut_edges.append((key, users))
        order_of = {isl["name"]: isl["order"] for isl in body.islands}

        def sort_key(item):
            key, users = item
            o = sorted(order_of[owner[u[0]]] for u in users)
            mid = (m.verts[key[0]] + m.verts[key[1]]) / 2
            return (o[1], o[0], -mid[2], mid[1])
        for key, users in sorted(cut_edges, key=sort_key):
            (f1, _), (f2, _) = users
            solid = g2.subtract_intervals([(0, 1)], hole_iv(*key))
            length = float(np.linalg.norm(m.verts[key[1]] - m.verts[key[0]]))
            n1, n2 = m.normal(m.faces[f1]), m.normal(m.faces[f2])
            angle = math.degrees(math.acos(np.clip(np.dot(n1, n2), -1, 1)))
            # Convex (seen from outside): square edges open into a V inside.
            # Concave would make the boards collide on the inside.
            c1, c2 = m.center_of(m.faces[f1]["vi"]), m.center_of(m.faces[f2]["vi"])
            convex = np.dot(n1, c2 - c1) <= 1e-9
            seams.append(dict(seam=seam_no, edge=key, length=length, pieces=(owner[f1], owner[f2]),
                              solid=solid, angle=angle, convex=bool(convex),
                              gap=2 * P.BOARD * math.sin(math.radians(angle) / 2)))
            for f in (f1, f2):
                pc = isl_by_name[owner[f]]
                a2, b2 = unf.map(f, m.verts[key[0]]), unf.map(f, m.verts[key[1]])
                for s0, s1 in solid:
                    pc.cuts.append(dict(a=a2 + (b2 - a2) * s0, b=a2 + (b2 - a2) * s1, seam=seam_no,
                                        length=length * (s1 - s0)))
                _seam_label(pc, f, a2, b2, seam_no, length)
            seam_no += 1

        # free edges and folds
        for key, users in edges.items():
            if len(users) == 1:
                f = users[0][0]
                if f not in owner:
                    continue
                pc = isl_by_name[owner[f]]
                a2, b2 = unf.map(f, m.verts[key[0]]), unf.map(f, m.verts[key[1]])
                length = float(np.linalg.norm(m.verts[key[1]] - m.verts[key[0]]))
                for s0, s1 in g2.subtract_intervals([(0, 1)], hole_iv(*key)):
                    pc.cuts.append(dict(a=a2 + (b2 - a2) * s0, b=a2 + (b2 - a2) * s1, seam=None,
                                        length=length * (s1 - s0)))
            elif key in folds:
                (f1, _), (f2, _) = users
                pc = isl_by_name[owner[f1]]
                n1 = m.normal(m.faces[f1])
                c2 = m.center_of(m.faces[f2]["vi"])
                c1 = m.center_of(m.faces[f1]["vi"])
                n2 = m.normal(m.faces[f2])
                angle = math.degrees(math.acos(np.clip(np.dot(n1, n2), -1, 1)))
                if angle < FLAT_FOLD:
                    continue
                # Convex seen from outside -> valley seen from the printed (inside) side.
                kind = "valley" if np.dot(n1, c2 - c1) < 0 else "mountain"
                a2, b2 = unf.map(f1, m.verts[key[0]]), unf.map(f1, m.verts[key[1]])
                for s0, s1 in g2.subtract_intervals([(0, 1)], hole_iv(*key)):
                    pc.folds.append(dict(a=a2 + (b2 - a2) * s0, b=a2 + (b2 - a2) * s1,
                                         kind=kind, angle=angle))

        # holes
        for pc in isl_by_name.values():
            pc.holes += _hole_segments(body, pc, unf)



    # printed side is the inside: mirror every piece
    M = np.array([[-1.0, 0.0], [0.0, 1.0]])
    for pc in pieces:
        pc.transform(M, np.zeros(2))
        pc.poly = {k: v[::-1] for k, v in pc.poly.items()}
        name_label(pc)
    return pieces, seams


def _ray_hit(segs, p, d):
    """Distance from p along d to the nearest segment (inf if none)."""
    best = math.inf
    for a, b in segs:
        e = b - a
        den = d[0] * e[1] - d[1] * e[0]
        if abs(den) < 1e-12:
            continue
        w = a - p
        t = (w[0] * e[1] - w[1] * e[0]) / den
        u = (w[0] * d[1] - w[1] * d[0]) / den
        if t > 1e-9 and -1e-9 <= u <= 1 + 1e-9:
            best = min(best, t)
    return best


def name_label(pc, max_size=4.2):
    """Put the piece name where it has the most room, sized to fit."""
    from .pdf import text_width
    segs = [(c["a"], c["b"]) for c in pc.cuts] + list(pc.holes)
    k = text_width(pc.name, 1.0, bold=True)
    cands = []
    for f in pc.faces:
        poly = pc.poly[f]
        c = g2.centroid(poly)
        cands.append(c)
        for v in poly:
            cands.append(c + (v - c) * 0.35)
    pts = np.vstack(list(pc.poly.values()))
    ev = np.linalg.eigh(np.cov((pts - pts.mean(0)).T))[1][:, -1]
    axis = math.degrees(math.atan2(ev[1], ev[0])) % 180.0
    best = None
    for c in cands:
        if not any(g2.point_in_poly(c, pc.poly[f]) for f in pc.faces):
            continue
        for deg in range(0, 180, 15):
            a = math.radians(deg)
            d = np.array([math.cos(a), math.sin(a)])
            n = np.array([-d[1], d[0]])
            half_len = min(_ray_hit(segs, c, d), _ray_hit(segs, c, -d))
            half_h = min(_ray_hit(segs, c, n), _ray_hit(segs, c, -n))
            h = min(max_size, 2 * (half_len - 1.5) / k, 2 * (half_h - 1.2))
            slack_len = half_len - (k * h / 2 + 1.5)
            slack_h = half_h - (h / 2 + 1.2)
            dev = abs((deg - axis + 90.0) % 180.0 - 90.0)
            key = (round(h, 1), -round(dev / 15.0), round(min(slack_len, slack_h), 0))
            if best is None or key > best[0]:
                best = (key, c, deg)
    (h, _, _), c, deg = best
    pc.labels.append(dict(text=pc.name, pos=c, angle=_readable(deg), size=max(h, 1.6),
                          bold=True, kind="name"))


def _span_along(poly, ang):
    a = math.radians(ang)
    d = np.array([math.cos(a), math.sin(a)])
    proj = poly @ d
    return float(proj.max() - proj.min())


def _seam_label(pc, f, a2, b2, seam, length):
    if length < 7.0:
        return
    d = b2 - a2
    L = np.linalg.norm(d)
    d = d / L
    m = np.array([-d[1], d[0]])
    poly = pc.poly[f]
    mid = (a2 + b2) / 2
    if not g2.point_in_poly(mid + m * 1.0, poly):
        m = -m
    pos = mid + m * (SEAM_SIZE * 0.5 + 1.6)
    pc.labels.append(dict(text=str(seam), pos=pos, angle=_readable(math.degrees(math.atan2(d[1], d[0]))),
                          size=SEAM_SIZE, bold=False, kind="seam"))


# --- holes ---------------------------------------------------------------------
def _vertex_params(body):
    g = body.grid
    out = {}
    for i in range(len(g.xs)):
        for j in range(len(g.phis)):
            v = body.mesh._key.get(tuple(np.round(g.vertex(i, j), 5)))
            if v is not None:
                out[v] = (g.xs[i], g.phis[j])
    return out


def _hole_intervals(body, vparam, a, b):
    if a not in vparam or b not in vparam:
        return []
    out = []
    for poly in body.param_holes:
        out += g2.seg_poly_intervals(vparam[a], vparam[b], poly)
    return out


def _hole_segments(body, pc, unf):
    segs = []
    m = body.mesh
    faces = set(pc.faces)
    if body.grid is not None and body.param_holes:
        g = body.grid
        cell_face = {}
        for fi, f in enumerate(m.faces):
            if f.get("cell") is not None:
                cell_face[tuple(f["cell"])] = fi
        for poly in body.param_holes:
            pts = g2.densify(poly, 0.4)
            pts = [np.array([p[0], p[1]]) for p in pts]
            for k in range(len(pts)):
                p, q = pts[k], pts[(k + 1) % len(pts)]
                for p0, q0 in _split_on_grid(g, p, q):
                    mid = (p0 + q0) / 2
                    cell = g.cell_of(*mid)
                    fi = cell_face.get(cell)
                    if fi is None or fi not in faces:
                        continue
                    a3 = g.point(p0[0], p0[1], cell)
                    b3 = g.point(q0[0], q0[1], cell)
                    segs.append((unf.map(fi, a3), unf.map(fi, b3)))
    for fi, polys in body.plate_holes.items():
        if fi not in faces:
            continue
        for poly in polys:
            for k in range(len(poly)):
                segs.append((unf.map(fi, poly[k]), unf.map(fi, poly[(k + 1) % len(poly)])))
    return segs


def _split_on_grid(g, p, q):
    ts = [0.0, 1.0]
    for x in g.xs:
        if (p[0] - x) * (q[0] - x) < 0:
            ts.append((x - p[0]) / (q[0] - p[0]))
    for ph in g.phis:
        if (p[1] - ph) * (q[1] - ph) < 0:
            ts.append((ph - p[1]) / (q[1] - p[1]))
    ts = sorted(ts)
    return [(p + (q - p) * t0, p + (q - p) * t1) for t0, t1 in zip(ts, ts[1:]) if t1 - t0 > 1e-12]


# --- flat parts (discs, rings) -------------------------------------------------
def flat_piece(name, order, outline, holes=(), count=1, title=""):
    pc = Piece(name, "flat", "flat", order, title)
    poly = np.asarray(outline, float)
    if g2.area(poly) < 0:
        poly = poly[::-1]
    pc.faces = [0]
    pc.poly = {0: poly}
    for k in range(len(poly)):
        pc.cuts.append(dict(a=poly[k], b=poly[(k + 1) % len(poly)], seam=None,
                            length=float(np.linalg.norm(poly[(k + 1) % len(poly)] - poly[k]))))
    for h in holes:
        h = np.asarray(h, float)
        for k in range(len(h)):
            pc.holes.append((h[k], h[(k + 1) % len(h)]))
    pc.count = count
    c = g2.centroid(poly)
    span = float(np.ptp(poly[:, 0]))
    off = 0.0
    if holes:
        hr = max(np.linalg.norm(np.asarray(h) - c, axis=1).max() for h in holes)
        off = hr + 3.5
    pc.labels.append(dict(text=name, pos=c + np.array([0.0, off]), angle=0.0,
                          size=min(3.6, span / 7.0), bold=True, kind="name"))
    return pc


def circle(c, r, n=72):
    t = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return np.column_stack([c[0] + r * np.cos(t), c[1] + r * np.sin(t)])
