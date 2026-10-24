"""Physical checks against the head proxy, run inside Blender (mathutils BVH).

Every mesh is the OUTER surface of the board; the board itself lies BOARD mm
inside it, so clearances below are measured to the board's inner surface."""

import math
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from . import params as P
from . import geom2d as g2
from .mesh import Mesh, polar, point_at


def _bvh(verts, tris):
    return BVHTree.FromPolygons([Vector(v) for v in verts], [list(t) for t in tris], all_triangles=True)


def shell_parts(d):
    """Shell mesh plus the pivot discs glued outside the side plates."""
    m = d.shell
    verts = [np.asarray(v) for v in m.verts]
    tris = list(m.triangles())
    for sgn in (-1.0, 1.0):
        x = sgn * (d.side_x + P.BOARD)
        c = len(verts)
        ring = [point_at(d.pivot, x, P.SIDE_DISC_R, a) for a in np.linspace(0, 360, 36, endpoint=False)]
        verts += [np.array([x, d.pivot[0], d.pivot[1]])] + ring
        for k in range(36):
            a, b = c + 1 + k, c + 1 + (k + 1) % 36
            # ring runs with increasing phi, which winds clockwise seen from +x
            tris.append((c, b, a) if sgn > 0 else (c, a, b))
    return verts, tris


def rotate_open(pts, pivot, ang):
    """Swing points up about the pivot axis by ang degrees (phi increases)."""
    pts = np.asarray(pts, float)
    f = pivot[0] - pts[:, 1]
    u = pts[:, 2] - pivot[1]
    a = math.radians(ang)
    f2 = f * math.cos(a) - u * math.sin(a)
    u2 = f * math.sin(a) + u * math.cos(a)
    return np.column_stack([pts[:, 0], pivot[0] - f2, pivot[1] + u2])


def surface_samples(m, spacing=3.0, inset=0.0):
    """Points spread over every face, optionally pushed inside by `inset`."""
    out = []
    for f in m.faces:
        p = m.pts(f)
        n = m.normal(f)
        e1 = p[1] - p[0]
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(n, e1)
        q = np.array([[(v - p[0]) @ e1, (v - p[0]) @ e2] for v in p])
        lo, hi = q.min(0), q.max(0)
        for x in np.arange(lo[0], hi[0] + spacing, spacing):
            for y in np.arange(lo[1], hi[1] + spacing, spacing):
                pt = np.array([min(x, hi[0]), min(y, hi[1])])
                if g2.point_in_poly(pt, q):
                    out.append(p[0] + pt[0] * e1 + pt[1] * e2 - n * inset)
        for v in p:
            out.append(v - n * inset)
    return np.array(out)


def nearest(bvh, pts):
    """Unsigned and signed (positive = outside the surface) distances."""
    d = np.empty(len(pts))
    s = np.empty(len(pts))
    for k, p in enumerate(pts):
        loc, nrm, _, dist = bvh.find_nearest(Vector(p))
        if loc is None:
            d[k] = s[k] = math.inf
            continue
        d[k] = dist
        s[k] = dist if (Vector(p) - loc).dot(nrm) >= 0 else -dist
    return d, s


def clearance(d):
    h = d.head
    sv, st = shell_parts(d)
    shell = _bvh(sv, st)
    face = _bvh(d.faceplate.verts, d.faceplate.triangles())
    out = {}
    dc, _ = nearest(shell, h.cranium)
    df, _ = nearest(face, h.cranium)
    dmin = np.minimum(dc, df) - P.BOARD
    k = int(np.argmin(dmin))
    out["head_to_helmet_min"] = (float(dmin[k]), h.cranium[k].tolist())
    de, _ = nearest(shell, h.ears)
    out["ears_to_shell_min"] = (float(de.min() - P.BOARD), h.ears[int(np.argmin(de))].tolist())
    dfa, _ = nearest(face, h.face)
    k = int(np.argmin(dfa))
    out["face_to_faceplate_min"] = (float(dfa[k] - P.BOARD), h.face[k].tolist())
    for name in ("nose_tip", "pogonion", "menton", "mouth"):
        p = h.landmarks[name]
        out[f"{name}_to_faceplate"] = float(face.find_nearest(Vector(p))[3] - P.BOARD)
    out["crown_to_shell"] = float(shell.find_nearest(Vector(h.landmarks["vertex"]))[3] - P.BOARD)
    out["forehead_to_shell"] = float(min(shell.find_nearest(Vector(h.landmarks["glabella"]))[3],
                                         face.find_nearest(Vector(h.landmarks["glabella"]))[3]) - P.BOARD)
    return out


def swing(d, step=0.5):
    """Rotate the faceplate from closed to fully open: no intersection with the
    shell, the board's inner surface keeps its gap, and the head is never hit."""
    sv, st = shell_parts(d)
    shell = _bvh(sv, st)
    fm = d.faceplate
    ftris = fm.triangles()
    inner = surface_samples(fm, spacing=2.5, inset=P.BOARD)
    # The board's inner surface as its own triangles (each face offset inward).
    iv, it = [], []
    for f in fm.faces:
        n = fm.normal(f)
        pts = fm.pts(f) - n * P.BOARD
        base = len(iv)
        iv += list(pts)
        k = {v: i for i, v in enumerate(f["vi"])}
        for tri in fm.triangles_of(f):
            it.append(tuple(base + k[v] for v in tri))
    head = np.vstack([d.head.cranium, d.head.face])
    worst_gap, worst_head, hits = math.inf, math.inf, []
    per_angle = []
    for ang in np.arange(0.0, P.SWING_ANGLE + 1e-9, step):
        fv = rotate_open(fm.verts, d.pivot, ang)
        fb = _bvh(fv, ftris)
        ib = _bvh(rotate_open(iv, d.pivot, ang), it)
        ov = shell.overlap(fb) + shell.overlap(ib)
        if ov:
            hits.append((float(ang), len(ov)))
        # Neither board surface crosses the shell, so the plain distance from
        # the board's inner surface to the shell is the air gap.
        dd, _ = nearest(shell, rotate_open(inner, d.pivot, ang))
        gap = float(dd.min())
        dh, _ = nearest(fb, head[::3])
        hd = float(dh.min() - P.BOARD)
        worst_gap, worst_head = min(worst_gap, gap), min(worst_head, hd)
        per_angle.append((float(ang), gap, hd))
    return dict(intersections=hits, min_gap=worst_gap, min_head=worst_head, per_angle=per_angle,
                steps=len(per_angle))


def donning(d, dz_max=220.0, dz_step=4.0, dy_range=(-10.0, 70.0), dy_step=2.0, margin=1.0):
    """Can the helmet (faceplate open) be lowered onto the head? Searches a path
    of small moves: down in z, sliding back/forward in y, never touching."""
    sv, st = shell_parts(d)
    fv = rotate_open(d.faceplate.verts, d.pivot, P.SWING_ANGLE)
    off = len(sv)
    verts = list(sv) + list(fv)
    tris = list(st) + [tuple(i + off for i in t) for t in d.faceplate.triangles()]
    helmet = _bvh(verts, tris)
    head = np.vstack([d.head.cranium[::2], d.head.face[::2], d.head.nose[::2]])
    dzs = np.arange(dz_max, -1e-9, -dz_step)
    dys = np.arange(dy_range[0], dy_range[1] + 1e-9, dy_step)
    free = np.zeros((len(dzs), len(dys)), bool)
    for i, dz in enumerate(dzs):
        for j, dy in enumerate(dys):
            pts = head - np.array([0.0, dy, dz])   # move the head instead of the helmet
            ok = True
            for p in pts:
                loc, _, _, dist = helmet.find_nearest(Vector(p), P.BOARD + margin)
                if loc is not None:
                    ok = False
                    break
            free[i, j] = ok
    # reachable states: start anywhere free at the top, move dz down one step and
    # dy by at most 2 steps per move; must end at dz = 0, dy = 0
    reach = free[0].copy()
    for i in range(1, len(dzs)):
        prev = reach.copy()
        spread = prev.copy()
        for s in (1, 2):
            spread[s:] |= prev[:-s]
            spread[:-s] |= prev[s:]
        reach = spread & free[i]
        # allow sliding sideways (in y) at the same height
        for _ in range(len(dys)):
            nxt = reach.copy()
            nxt[1:] |= reach[:-1] & free[i][1:]
            nxt[:-1] |= reach[1:] & free[i][:-1]
            if (nxt == reach).all():
                break
            reach = nxt
    j0 = int(np.argmin(np.abs(dys)))
    blocked = [float(dzs[i]) for i in range(len(dzs)) if not free[i].any()]
    return dict(ok=bool(reach[j0] and free[-1][j0]), final_free=bool(free[-1][j0]),
                blocked_heights=blocked[:5], grid=(len(dzs), len(dys)))


def coverage(d, n_az=72, elev=(-15.0, 88.0), n_el=24):
    """Rays from the middle of the head: does the closed helmet cover them?"""
    sv, st = shell_parts(d)
    off = len(sv)
    verts = list(sv) + list(d.faceplate.verts)
    tris = list(st) + [tuple(i + off for i in t) for t in d.faceplate.triangles()]
    bvh = _bvh(verts, tris)
    o = Vector((0.0, 0.0, -20.0))
    miss = []
    total = 0
    for el in np.linspace(elev[0], elev[1], n_el):
        for az in np.linspace(0, 360, n_az, endpoint=False):
            e, a = math.radians(el), math.radians(az)
            dvec = Vector((math.cos(e) * math.sin(a), -math.cos(e) * math.cos(a), math.sin(e)))
            total += 1
            if bvh.ray_cast(o, dvec, 1000.0)[0] is None:
                miss.append((round(float(az), 1), round(float(el), 1)))
    return dict(rays=total, missed=len(miss), examples=miss[:6])


def _param_of(d, fi, p):
    """(x, phi) of point p on faceplate face fi."""
    f = d.faceplate.faces[fi]
    return d.face_grid.invert(tuple(f["cell"]) + (f["tri"],), p)


def vision(d):
    """Where does each eye's line of sight cross the closed faceplate?"""
    fm = d.faceplate
    tris = fm.triangles()
    tri_face = []
    for fi, f in enumerate(fm.faces):
        n = len(f["vi"]) - 2
        tri_face += [fi] * n
    bvh = _bvh(fm.verts, tris)
    holes = [poly for name, poly in d.holes["faceplate"] if name.startswith("eye")]
    out = {}
    for eye in ("eye_l", "eye_r"):
        o = d.head.landmarks[eye]
        seen = {}
        for h in range(-30, 31, 5):
            for v in range(-10, 11, 5):
                a, e = math.radians(h), math.radians(v)
                dvec = Vector((math.sin(a) * math.cos(e), -math.cos(a) * math.cos(e), math.sin(e)))
                loc, _, idx, _ = bvh.ray_cast(Vector(o), dvec, 500.0)
                if loc is None:
                    seen[(h, v)] = True
                    continue
                fi = tri_face[idx]
                if fm.faces[fi].get("cell") is None:
                    seen[(h, v)] = False
                    continue
                x, ph = _param_of(d, fi, np.array(loc))
                seen[(h, v)] = any(g2.point_in_poly((x, ph), poly) for poly in holes)
        out[eye] = dict(straight_ahead=seen[(0, 0)],
                        horizontal=[h for h in range(-30, 31, 5) if seen[(h, 0)]],
                        vertical=[v for v in range(-10, 11, 5) if seen[(0, v)]])
    return out


def mesh_checks(d):
    out = {}
    for m in (d.shell, d.faceplate):
        plan = max(m.planarity(f) for f in m.faces)
        edges = m.edges()
        nonmanifold = sum(1 for u in edges.values() if len(u) > 2)
        flips = 0
        for u in edges.values():   # neighbours must run a shared edge in opposite directions
            if len(u) == 2:
                (f1, i1), (f2, i2) = u
                a1 = m.faces[f1]["vi"][i1]
                a2 = m.faces[f2]["vi"][i2]
                flips += a1 == a2
        out[m.name] = dict(faces=len(m.faces), max_planarity_mm=plan, nonmanifold_edges=nonmanifold,
                           inconsistent_normals=flips)
    return out


def run_all(d, log=print):
    res = {}
    log("3D: mesh checks")
    res["mesh"] = mesh_checks(d)
    log("3D: clearances")
    res["clearance"] = clearance(d)
    log("3D: faceplate swing")
    res["swing"] = swing(d)
    log("3D: vision through eye slots")
    res["vision"] = vision(d)
    log("3D: coverage")
    res["coverage"] = coverage(d)
    log("3D: putting it on")
    res["donning"] = donning(d)
    return res
