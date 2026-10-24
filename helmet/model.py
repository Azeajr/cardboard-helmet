"""Turn the design into bodies (meshes + islands) and flat pieces."""

import math
import numpy as np

from . import params as P
from . import geom2d as g2
from .mesh import Mesh
from .pieces import Body, build_pieces, flat_piece, circle


def column_names(n):
    """Names of n grid columns, wearer's right to left: ..., R1, C, L1, ..."""
    mid = n // 2
    return ["C" if i == mid else (f"L{i - mid}" if i > mid else f"R{mid - i}") for i in range(n)]


def column_rank(name):
    """Build order: centre first, then each pair outwards, right before left."""
    return 0 if name == "C" else 2 * int(name[1:]) - (name[0] == "R")


def _panels(body, m, prefix, kind, base, title):
    """One island per grid facet: every facet is its own flat panel."""
    cols = column_names(max(f["cell"][0] for f in m.faces if f.get("cell")) + 1)
    for fi, f in enumerate(m.faces):
        if not f.get("cell"):
            continue
        i, j = f["cell"]
        col = cols[i]
        body.islands.append(dict(
            name=f"{prefix}-{col}-{j + 1}", faces=[fi], root=fi, kind=kind,
            allow=lambda a, b: False, order=base + column_rank(col) * 20 + j,
            title=f"{title} panel {col}, row {j + 1}"))
    return len(cols)


def shell_body(d):
    m = d.shell
    body = Body(m, d.shell_grid)
    ncol = _panels(body, m, "S", "shell", 100, "Shell")
    for fi, f in enumerate(m.faces):
        if f.get("plate"):
            side = f["plate"]
            body.islands.append(dict(name=f"S-SIDE-{side}", faces=[fi], root=fi, kind="shell",
                                     allow=lambda a, b: False, order=100 + 20 * ncol + (side == "L"),
                                     title=f"Shell side plate ({'right' if side == 'R' else 'left'})"))
            body.plate_holes[fi] = [_pivot_hole(d, f, m)]
    return body


def _pivot_hole(d, f, m):
    x = m.verts[f["vi"][0]][0]
    t = np.linspace(0, 2 * math.pi, 24, endpoint=False)
    r = P.PIVOT_HOLE / 2
    return [np.array([x, d.pivot[0] + r * math.cos(a), d.pivot[1] + r * math.sin(a)]) for a in t]


def face_body(d):
    """Faceplate panels like the shell's; the pivot arms are flat end panels."""
    m = d.faceplate
    body = Body(m, d.face_grid)
    ncol = _panels(body, m, "F", "face", 300, "Faceplate")
    for fi, f in enumerate(m.faces):
        if f.get("plate"):
            side = f["plate"]
            body.plate_holes[fi] = [_pivot_hole(d, f, m)]
            body.islands.append(dict(
                name=f"F-ARM-{side}", faces=[fi], root=fi, kind="face", allow=lambda a, b: False,
                order=300 + 20 * ncol + (side == "L"),
                title=f"Faceplate pivot arm ({'right' if side == 'R' else 'left'})"))
    body.param_holes = [poly for _, poly in d.holes["faceplate"]]
    return body


def overlay_bodies(d):
    bodies = []
    for k, (name, gname, poly) in enumerate(d.overlays):
        g = d.face_grid if gname == "face" else d.shell_grid
        m = Mesh(name)
        dense = g2.densify(poly, 1.0)
        for i in range(len(g.xs) - 1):
            for j in range(len(g.phis) - 1):
                x0, x1, p0, p1 = g.cell_rect(i, j)
                clip = g2.clip_rect([np.asarray(p, float) for p in dense], x0, x1, p0, p1)
                if len(clip) < 3 or abs(g2.area(clip)) < 1e-3:
                    continue
                pts = [g.point(p[0], p[1], (i, j)) for p in clip]
                m.add_face(pts, f"{name}:{i}.{j}", d._radial_out, cell=(i, j), plate=None)
        if len(m.faces) != 1:
            raise RuntimeError(f"{name} spans {len(m.faces)} panels; overlays must sit on one")
        body = Body(m, None)
        body.islands.append(dict(name=name, faces=[0], root=0, kind="overlay",
                                 allow=lambda a, b: False, order=500 + k,
                                 title=f"Overlay {name[2:].lower()}"))
        bodies.append(body)
    return bodies


def ear_pieces():
    o = np.zeros(2)
    return [flat_piece("E-RING", 600, circle(o, P.EAR_DISC_R), [circle(o, P.EAR_RING_IN)], count=2,
                       title="Ear ring, glued on each arm around the screw cap")]


def build(d):
    bodies = [shell_body(d), face_body(d)] + overlay_bodies(d)
    pieces, seams = build_pieces(bodies)
    pieces += ear_pieces()
    return bodies, pieces, seams
