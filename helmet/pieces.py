"""Cut the meshes into flat pieces: islands, unfolding, glue tabs, seam
numbers, fold lines and holes. Everything is computed in the outside view and
mirrored at the end, because the printed side of each piece is the inside."""

import math
import numpy as np

from . import params as P
from . import geom2d as g2
from .mesh import Mesh

FLAT_FOLD = 0.2   # degrees; folds flatter than this are not drawn


class Piece:
    def __init__(self, name, body, kind, order, title=""):
        self.name, self.body, self.kind, self.order, self.title = name, body, kind, order, title
        self.faces = []
        self.poly = {}       # face index -> 2D polygon (printed view)
        self.cuts = []       # dict(a, b, seam, length)
        self.folds = []      # dict(a, b, kind, angle)
        self.tabs = []       # dict(poly, seam)
        self.holes = []      # (a, b) segments
        self.labels = []     # dict(text, pos, angle, size, bold)
        self.count = 1       # how many to cut

    def points(self):
        pts = [p for poly in self.poly.values() for p in poly]
        pts += [p for t in self.tabs for p in t["poly"]]
        return np.array(pts)

    def transform(self, R, t):
        f = lambda p: np.asarray(p) @ R.T + t
        self.poly = {k: f(v) for k, v in self.poly.items()}
        for c in self.cuts + self.folds:
            c["a"], c["b"] = f(c["a"]), f(c["b"])
        for tb in self.tabs:
            tb["poly"] = f(tb["poly"])
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


# --- tabs ------------------------------------------------------------------------
def make_tab(A, B, other, width):
    """Glue tab along A-B (2D) on the side where `other` (the neighbour face
    unfolded across the edge) lies, kept inside `other` so it lands flat on it."""
    d = B - A
    L = np.linalg.norm(d)
    d = d / L
    m = np.array([-d[1], d[0]])
    if np.dot(g2.centroid(other) - A, m) < 0:
        m = -m

    def corner_angle(P0, P1):
        # interior angle of `other` at P0, between the shared edge and its next
        # edge; mid-edge ends (beside a hole) get a plain 45 degree chamfer
        dist = np.linalg.norm(other - P0, axis=1)
        k = int(np.argmin(dist))
        if dist[k] > 1e-6:
            return 90.0
        n = len(other)
        for q in (other[(k - 1) % n], other[(k + 1) % n]):
            if np.linalg.norm(q - P1) < 1e-6:
                continue
            v, e = q - P0, P1 - P0
            return math.degrees(math.acos(np.clip(np.dot(v, e) / (np.linalg.norm(v) * np.linalg.norm(e)), -1, 1)))
        return 90.0

    for w in (width, width * 0.75, width * 0.625, width * 0.5, width * 0.375):
        aA = math.radians(min(45.0, corner_angle(A, B) - 8.0))
        aB = math.radians(min(45.0, corner_angle(B, A) - 8.0))
        if aA <= math.radians(10) or aB <= math.radians(10):
            return None
        ia, ib = w / math.tan(aA), w / math.tan(aB)
        if ia + ib > L - 2.0:
            continue
        tab = np.array([A, B, B + m * w - d * ib, A + m * w + d * ia])
        inner = g2.offset_inward(tab, 0.05)
        if all(g2.point_in_poly(p, other) for p in inner) and not any(
                g2.seg_intersect(inner[i], inner[(i + 1) % 4], other[j], other[(j + 1) % len(other)])
                for i in range(4) for j in range(len(other))):
            return tab
    return None


# --- building pieces ---------------------------------------------------------------
class Body:
    """A mesh plus island assignment and holes, ready to be cut into pieces."""

    def __init__(self, mesh, grid=None):
        self.mesh = mesh
        self.grid = grid
        self.islands = []          # dict(name, faces, allow, root, order, title, kind)
        self.param_holes = []      # param-space polygons on the grid
        self.plate_holes = {}      # face index -> list of 3D polygons
        self.tabs = True           # False: internal cuts are slits, not seams


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

        # cut edges: decide tab side, number the seams
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
            i1, i2 = owner[f1], owner[f2]
            if i1 == i2 and not body.tabs:
                # A slit inside a glued-on overlay: it closes up when glued down.
                pc = isl_by_name[i1]
                length = float(np.linalg.norm(m.verts[key[1]] - m.verts[key[0]]))
                for f in (f1, f2):
                    a2, b2 = unf.map(f, m.verts[key[0]]), unf.map(f, m.verts[key[1]])
                    pc.cuts.append(dict(a=a2, b=b2, seam=None, length=length))
                continue
            # tab on the piece attached later; within one piece, on the first face
            cand = [(f2, f1), (f1, f2)] if order_of[i2] > order_of[i1] else [(f1, f2), (f2, f1)]
            solid = g2.subtract_intervals([(0, 1)], hole_iv(*key))
            length = float(np.linalg.norm(m.verts[key[1]] - m.verts[key[0]]))
            placed = None
            for tf, nf in cand:
                pc = isl_by_name[owner[tf]]
                tabs = []
                A3, B3 = m.verts[key[0]], m.verts[key[1]]
                A, B = unf.map(tf, A3), unf.map(tf, B3)
                fr, R, t = unf.place(nf, key[0], key[1], A, B)
                other = np.array([R @ Unfolder.local(fr, m.verts[v]) + t for v in m.faces[nf]["vi"]])
                ok = True
                for s0, s1 in solid:
                    if (s1 - s0) * length < (6.0 if len(solid) == 1 else 12.0):
                        continue
                    tab = make_tab(A + (B - A) * s0, A + (B - A) * s1, other, P.TAB_WIDTH)
                    if tab is None:
                        ok = False
                        break
                    if any(g2.polys_overlap(tab, poly) for poly in pc.poly.values()) or any(
                            g2.polys_overlap(tab, tb["poly"]) for tb in pc.tabs):
                        ok = False
                        break
                    tabs.append(tab)
                if ok:
                    placed = (tf, nf, tabs)
                    break
            if placed is None:
                raise RuntimeError(f"no room for a glue tab on edge {key} ({i1}/{i2}), length {length:.1f}, solid {solid}")
            tf, nf, tabs = placed
            pc_t = isl_by_name[owner[tf]]
            for tab in tabs:
                pc_t.tabs.append(dict(poly=tab, seam=seam_no))
            seams.append(dict(seam=seam_no, edge=key, length=length, tab_piece=owner[tf],
                              other_piece=owner[nf], tab_face=tf, other_face=nf, solid=solid))
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

        # paint marks: facets to paint red get a small "RED" in their middle
        for pc in isl_by_name.values():
            for f in pc.faces:
                if m.faces[f].get("paint") == "red":
                    poly = pc.poly[f]
                    c = g2.centroid(poly)
                    if all(g2.point_in_poly(c + d, poly) for d in ((4, 0), (-4, 0), (0, 2), (0, -2))):
                        pc.labels.append(dict(text="RED", pos=c, angle=0.0, size=2.2, bold=True,
                                              kind="paint"))



    # printed side is the inside: mirror every piece
    M = np.array([[-1.0, 0.0], [0.0, 1.0]])
    for pc in pieces:
        pc.transform(M, np.zeros(2))
        pc.poly = {k: v[::-1] for k, v in pc.poly.items()}
        # tab polygons keep their order: [A, B] is the fold, the rest is cut
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
    for lb in pc.labels:           # keep clear of paint marks already placed
        if lb["kind"] == "paint":
            c = np.asarray(lb["pos"])
            box = [c + (-5, -2), c + (5, -2), c + (5, 2), c + (-5, 2)]
            segs += [(box[k], box[(k + 1) % 4]) for k in range(4)]
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
    size = 2.6
    pos = mid + m * (size * 0.5 + 1.6)
    pc.labels.append(dict(text=str(seam), pos=pos, angle=_readable(math.degrees(math.atan2(d[1], d[0]))),
                          size=size, bold=False, kind="seam"))


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
        piece_face = {}
        for fi, f in enumerate(m.faces):
            if f.get("cell") is not None:
                piece_face[tuple(f["cell"]) + (f["tri"],)] = fi
        for poly in body.param_holes:
            pts = g2.densify(poly, 0.4)
            pts = [np.array([p[0], p[1]]) for p in pts]
            for k in range(len(pts)):
                p, q = pts[k], pts[(k + 1) % len(pts)]
                for key, p0, q0 in g.split_segment(p, q):
                    fi = piece_face.get(key)
                    if fi is None or fi not in faces:
                        continue
                    a3 = g.point(p0[0], p0[1], key)
                    b3 = g.point(q0[0], q0[1], key)
                    segs.append((unf.map(fi, a3), unf.map(fi, b3)))
    for fi, polys in body.plate_holes.items():
        if fi not in faces:
            continue
        for poly in polys:
            for k in range(len(poly)):
                segs.append((unf.map(fi, poly[k]), unf.map(fi, poly[(k + 1) % len(poly)])))
    return segs


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
