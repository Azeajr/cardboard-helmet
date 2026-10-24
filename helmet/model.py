"""Turn the design into bodies (meshes + islands) and flat pieces."""

import math
import numpy as np

from . import params as P
from . import geom2d as g2
from .mesh import Mesh, point_at
from .pieces import Body, build_pieces, flat_piece, circle

SHELL_COLS = ["R3", "R2", "R1", "C", "L1", "L2", "L3"]   # x < 0 is the wearer's right
COL_RANK = {"C": 0, "R1": 1, "L1": 2, "R2": 3, "L2": 4, "R3": 5, "L3": 6}
MAX_STRIP = 210.0   # longest strip segment (mm) so it fits a page with its tabs
STAGGER = {"C": 0, "R1": 1, "L1": 1, "R2": 0, "L2": 0, "R3": 1, "L3": 1}


def _cell(f):
    return f.get("cell")


def shell_body(d):
    m = d.shell
    g = d.shell_grid
    body = Body(m, g)
    nb = len(g.phis) - 1
    faces_at = {}
    for fi, f in enumerate(m.faces):
        if _cell(f):
            faces_at[tuple(f["cell"])] = fi
    for i, col in enumerate(SHELL_COLS):
        mid = lambda j: (g.vertex(i, j) + g.vertex(i + 1, j)) / 2
        lens = [float(np.linalg.norm(mid(j + 1) - mid(j))) for j in range(nb)]
        cuts = split_rows(lens, MAX_STRIP, STAGGER[col])
        for s in range(len(cuts) - 1):
            faces = [faces_at[(i, j)] for j in range(cuts[s], cuts[s + 1])]
            body.islands.append(dict(
                name=f"S-{col}-{s + 1}", faces=faces, root=faces[0], kind="shell",
                allow=lambda a, b: a["cell"][0] == b["cell"][0],
                order=10 + COL_RANK[col] * 10 + s,
                title=f"Shell strip {col}, part {s + 1} of {len(cuts) - 1}"))
    for fi, f in enumerate(m.faces):
        if f.get("plate"):
            side = f["plate"]
            body.islands.append(dict(name=f"S-SIDE-{side}", faces=[fi], root=fi, kind="shell",
                                     allow=lambda a, b: False, order=90 + (side == "L"),
                                     title=f"Shell side plate ({'right' if side == 'R' else 'left'})"))
            body.plate_holes[fi] = [_pivot_hole(d, f, m)]
    return body


def split_rows(lens, max_len, stagger):
    """Fewest segments (each <= max_len) with the most even lengths; neighbouring
    columns shift their seams by a row so seams don't line up."""
    n = len(lens)
    cum = np.concatenate([[0.0], np.cumsum(lens)])
    for k in range(max(1, int(math.ceil(cum[-1] / max_len))), n + 1):
        best = np.full((k + 1, n + 1), math.inf)
        arg = np.zeros((k + 1, n + 1), int)
        best[0, 0] = 0.0
        for i in range(1, k + 1):
            for j in range(i, n + 1):
                for m in range(i - 1, j):
                    v = max(best[i - 1, m], cum[j] - cum[m])
                    if v < best[i, j]:
                        best[i, j], arg[i, j] = v, m
        if best[k, n] <= max_len:
            cuts = [n]
            for i in range(k, 0, -1):
                cuts.append(arg[i, cuts[-1]])
            cuts = cuts[::-1]
            if stagger:
                for c in range(1, len(cuts) - 1):
                    trial = cuts[:c] + [cuts[c] + 1] + cuts[c + 1:]
                    if trial[c] < trial[c + 1] and all(cum[b] - cum[a] <= max_len for a, b in zip(trial, trial[1:])):
                        cuts = trial
            return cuts
    raise RuntimeError("a single row is longer than a page")


def _pivot_hole(d, f, m):
    x = m.verts[f["vi"][0]][0]
    t = np.linspace(0, 2 * math.pi, 24, endpoint=False)
    r = P.PIVOT_HOLE / 2
    return [np.array([x, d.pivot[0] + r * math.cos(a), d.pivot[1] + r * math.sin(a)]) for a in t]


def face_body(d):
    """Faceplate strips run top to bottom like the shell's; the pivot arms are
    separate flat pieces tabbed onto the outer strips."""
    m = d.faceplate
    g = d.face_grid
    body = Body(m, g)
    e = P.EYE_ROW
    groups = {}
    for fi, f in enumerate(m.faces):
        if f.get("plate"):
            side = f["plate"]
            body.plate_holes[fi] = [_pivot_hole(d, f, m)]
            body.islands.append(dict(
                name=f"F-ARM-{side}", faces=[fi], root=fi, kind="face", allow=lambda a, b: False,
                order=190 + (side == "L"),
                title=f"Faceplate pivot arm ({'right' if side == 'R' else 'left'})"))
        else:
            groups.setdefault(SHELL_COLS[f["cell"][0]], []).append(fi)

    for col, faces in groups.items():
        root = next(fi for fi in faces if m.faces[fi]["cell"][1] == e)
        body.islands.append(dict(name=f"F-{col}", faces=faces, root=root, kind="face",
                                 allow=lambda a, b: True, order=100 + COL_RANK[col],
                                 title=f"Faceplate strip {col}"))
    body.param_holes = [poly for _, poly in d.holes["faceplate"]]
    return body


def overlay_bodies(d):
    bodies = []
    for k, (name, gname, poly) in enumerate(d.overlays):
        g = d.face_grid if gname == "face" else d.shell_grid
        m = Mesh(name)
        for key, _, pts in g.clip(g2.densify(poly, 1.0)):
            m.add_face(pts, f"{name}:{key[0]}.{key[1]}{key[2]}", d._radial_out, cell=key[:2], tri=key[2],
                       plate=None)
        body = Body(m, None)
        body.tabs = False
        faces = list(range(len(m.faces)))
        root = max(faces, key=lambda fi: len(m.faces[fi]["vi"]))
        body.islands.append(dict(name=name, faces=faces, root=root, kind="overlay",
                                 allow=lambda a, b: True, order=200 + k,
                                 title=f"Overlay {name[2:].lower()}"))
        bodies.append(body)
    return bodies


def ear_pieces():
    o = np.zeros(2)
    hole = circle(o, P.PIVOT_HOLE / 2, 24)
    return [
        flat_piece("E-SIDE", 300, circle(o, P.SIDE_DISC_R), [hole], count=2,
                   title="Pivot disc, glued outside each shell side plate"),
        flat_piece("E-DISC", 301, circle(o, P.EAR_DISC_R), [hole], count=2,
                   title="Ear disc, glued outside each faceplate arm"),
        flat_piece("E-RING", 302, circle(o, P.EAR_DISC_R), [circle(o, P.EAR_RING_IN)], count=4,
                   title="Ear ring, two stacked on each ear disc"),
    ]


def build(d):
    bodies = [shell_body(d), face_body(d)] + overlay_bodies(d)
    pieces, seams = build_pieces(bodies)
    pieces += ear_pieces()
    return bodies, pieces, seams
