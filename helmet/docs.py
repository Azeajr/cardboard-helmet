"""Cover, instructions and assembly-map pages (all vector, plus renders)."""

import math
import numpy as np

from . import params as P
from . import geom2d as g2
from . import sheets
from .pdf import text_width
from .sheets import INK, GREY, ACCENT
from .model import column_names

RED = (0.70, 0.16, 0.13)
GREEN = (0.10, 0.48, 0.22)


def pivot_post():
    """(grip, post): the thickness a Chicago screw clamps (washer, side plate,
    washer, arm, washer; the cap sits inside the hollow E-RING) and the post
    length to buy, a whole millimetre at least 0.5 mm shorter so tightening
    squeezes the board and the faceplate holds its angle."""
    grip = 2 * P.BOARD + 3 * P.PIVOT_WASHER
    return grip, math.floor(grip - 0.5)


def board_area(pieces):
    """Total panel area to cut, in square metres (slots not subtracted)."""
    return sum(abs(g2.area(poly)) * pc.count for pc in pieces for poly in pc.poly.values()) / 1e6


def wrap(text, size, width, bold=False):
    lines = []
    for para in text.split("\n"):
        words, cur = para.split(" "), ""
        for w in words:
            t = (cur + " " + w).strip()
            if text_width(t, size, bold) <= width or not cur:
                cur = t
            else:
                lines.append(cur)
                cur = w
        lines.append(cur)
    return lines


def para(pg, x, y, width, text, size=2.9, lead=1.42, bold=False, color=INK):
    for line in wrap(text, size, width, bold):
        pg.text(line, (x, y), size, bold=bold, anchor="start", color=color, vcenter=False)
        y -= size * lead
    return y


def heading(pg, x, y, text, size=3.6):
    pg.text(text, (x, y), size, bold=True, anchor="start", color=ACCENT, vcenter=False)
    return y - size * 1.7


# --- projected views ------------------------------------------------------------
def _basis(d, up):
    d = np.asarray(d, float) / np.linalg.norm(d)
    r = np.cross(d, up)
    r /= np.linalg.norm(r)
    return d, r, np.cross(r, d)


def draw_view(pg, layers, d, up, box, labels=True, light=(-0.4, 0.5, 0.75)):
    """layers: list of (verts Nx3, faces [vi lists], colour per face, label per face)."""
    d, r, u = _basis(d, up)
    L = np.asarray(light, float)
    L /= np.linalg.norm(L)
    items, allp = [], []
    for verts, faces, cols, labs in layers:
        V = np.asarray(verts, float)
        for vi, col, lab in zip(faces, cols, labs):
            p = V[vi]
            n = np.zeros(3)
            for a, b in zip(p, np.roll(p, -1, 0)):
                n += np.cross(a, b)
            n /= np.linalg.norm(n) + 1e-12
            front = np.dot(n, d) < 0
            shade = 0.62 + 0.38 * max(0.0, np.dot(n, L)) if front else 0.42
            c = tuple(min(1.0, ch * shade + (1 - shade) * 0.15) for ch in col)
            q = np.column_stack([p @ r, p @ u])
            items.append((float(np.mean(p @ d)), q, c, lab, front))
            allp.append(q)
    allp = np.vstack(allp)
    lo, hi = allp.min(0), allp.max(0)
    x, y, w, h = box
    s = min(w / (hi[0] - lo[0]), h / (hi[1] - lo[1]))
    off = np.array([x + (w - s * (hi[0] - lo[0])) / 2, y + (h - s * (hi[1] - lo[1])) / 2]) - lo * s
    acc = {}
    for depth, q, c, lab, front in sorted(items, key=lambda t: -t[0]):
        pts = q * s + off
        pg.polyline(pts, closed=True, fill=c, width=0.08, color=(0.15, 0.15, 0.15))
        if lab and front:
            a = abs(0.5 * np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - pts[:, 1] * np.roll(pts[:, 0], -1)))
            cc = pts.mean(0)
            A, C = acc.get(lab, (0.0, np.zeros(2)))
            acc[lab] = (A + a, C + cc * a)
    if labels:
        placed = []
        for lab, (A, C) in sorted(acc.items(), key=lambda kv: -kv[1][0]):
            if A < 60.0:
                continue
            c = C / A
            size = 2.2 if A < 250 else 2.6
            w = text_width(lab, size, True) + 1.2
            for _ in range(8):   # nudge down until it clears earlier labels
                if all(abs(c[0] - q[0]) > (w + q[2]) / 2 or abs(c[1] - q[1]) > 3.6 for q in placed):
                    break
                c = c + np.array([0.0, -3.8])
            placed.append((c[0], c[1], w))
            pg.polyline([(c[0] - w / 2, c[1] - 1.7), (c[0] + w / 2, c[1] - 1.7),
                         (c[0] + w / 2, c[1] + 1.7), (c[0] - w / 2, c[1] + 1.7)],
                        closed=True, fill=(1, 1, 1), width=0.08, color=(0.3, 0.3, 0.3))
            pg.text(lab, c, size, bold=True)
    return s


def piece_layers(d, pieces, open_angle=0.0, with_overlays=False):
    from .checks3d import rotate_open
    owner = {}
    for pc in pieces:
        for f in pc.faces:
            owner[(pc.body, f)] = pc.name
    layers = []
    for m, kind in ((d.shell, "shell"), (d.faceplate, "face")):
        verts = np.asarray(m.verts)
        if kind == "face" and open_angle:
            verts = rotate_open(verts, d.pivot, open_angle)
        faces, cols, labs = [], [], []
        for fi, f in enumerate(m.faces):
            name = owner.get((m.name, fi), "")
            faces.append(f["vi"])
            labs.append(name)
            if kind == "shell":
                base = (0.80, 0.30, 0.26) if "SIDE" in name else (
                    (0.93, 0.62, 0.55) if (f.get("cell") or (0,))[0] % 2 else (0.97, 0.80, 0.74))
            else:
                base = (0.86, 0.72, 0.40) if "ARM" in name else (
                    (0.98, 0.87, 0.55) if (f.get("cell") or (0,))[0] % 2 else (0.99, 0.94, 0.75))
            cols.append(base)
        layers.append((verts, faces, cols, labs))
    return layers


# --- pages ----------------------------------------------------------------------
def cover(pdf, d, pieces, renders, checks, total, n_sheets):
    pg = pdf.page()
    x0, W = P.MARGIN, P.PAGE_W - 2 * P.MARGIN
    top = P.PAGE_H - P.MARGIN
    pg.text("CARDBOARD HELMET", (x0, top - 8), 8.0, bold=True, anchor="start", vcenter=False)
    pg.text(f"1 / {total}", (P.PAGE_W - P.MARGIN, top - 4), 3.2, bold=True, anchor="end", color=ACCENT)
    pg.text("Flip-up faceplate on ear pivots  |  flat panels, numbered edges  |  full-size stencils from a 3D model",
            (x0, top - 14), 3.2, anchor="start", vcenter=False, color=GREY)
    s = d.summary()
    pg.text(f"Fitted to: forehead {P.HEAD_CIRCUMFERENCE / 10:.1f} cm around, chin-over-top-to-nape "
            f"{P.CHIN_TOP_NAPE / 10:.1f} cm, {P.BOARD:g} mm board. Outside: about "
            f"{2 * (d.arm_x + P.BOARD):.0f} mm wide.",
            (x0, top - 20), 3.0, anchor="start", vcenter=False)
    img_w = (W - 6) / 2
    y_img = top - 26 - img_w
    for k, key in enumerate(("three_quarter", "open_three_quarter")):
        if key in renders:
            path, pw, ph = renders[key]
            pg.image(open(path, "rb").read(), pw, ph, x0 + k * (img_w + 6), y_img, img_w, img_w)
    pg.text("closed", (x0 + img_w / 2, y_img + 3), 3.0, color=GREY)
    pg.text("faceplate open", (x0 + img_w * 1.5 + 6, y_img + 3), 3.0, color=GREY)

    y = y_img - 8
    colw = (W - 8) / 2
    yl = heading(pg, x0, y, "1. CHECK THE PRINT FIRST")
    yl = para(pg, x0, yl, colw,
              "Print every page on US Letter at 100% / Actual size (turn off 'fit to page'). "
              "Measure the bar at the bottom of any page: it must be exactly 100 mm. If it isn't, "
              "fix the printer setting before cutting anything.")
    yl -= 2.5
    yl = heading(pg, x0, yl, "2. WHAT YOU NEED")
    grip, post = pivot_post()
    yl = para(pg, x0, yl, colw,
              f"- {P.BOARD:g} mm double-wall corrugated cardboard: about {board_area(pieces):.2f} m\u00b2 of "
              "panels plus offcuts (large appliance or moving boxes)\n"
              f"- Glue stick or spray adhesive (to stick the {n_sheets} stencil pages on)\n"
              "- Hot glue gun and plenty of sticks; masking tape\n"
              "- Craft knife with fresh blades, steel ruler, cutting mat\n"
              f"- 2 Chicago screws (binding posts), flat heads: {P.PIVOT_HOLE:g} mm diameter, "
              f"{post} mm post (shorter than the {grip:g} mm they clamp, so they can be tightened); "
              "removable thread locker\n"
              f"- 6 fender washers for {P.PIVOT_HOLE:g} mm screws: about {P.PIVOT_WASHER_D:g} mm across, "
              f"{P.PIVOT_WASHER:g} mm thick\n"
              "- Optional: thin foam for padding (up to 8 mm keeps the fit)")
    xr = x0 + colw + 8
    yr = heading(pg, xr, y, "3. CHECKED ON THE 3D MODEL BEFORE PRINTING")
    for name, detail, ok in checks:
        pg.text("PASS" if ok else "FAIL", (xr, yr), 2.6, bold=True, anchor="start", vcenter=False,
                color=GREEN if ok else RED)
        yr = para(pg, xr + 10, yr, colw - 10, f"{name}: {detail}", size=2.5, lead=1.35)
        yr -= 0.8
    sheets.footer(pg, "GUIDE - DO NOT CUT")
    return pg


STEPS = [
    ("Stick, then cut",
     "Glue each printout onto the board with the printed side up. The printed side becomes the INSIDE of the "
     "helmet, so the outside stays clean for paint. Cut every solid line in several light passes with the "
     "blade held upright, so the edges come out square. Cut the eye and mouth slots and the pivot holes too."),
    ("How the joints work",
     "Every edge has a number, and the edge with the same number on another panel is its partner. Hold the "
     "two panels printed side down with those edges touching and tape along the seam on the outside. Fold "
     "them away from the tape: the outer corners stay together and a V opens on the inside. Fill it with hot glue."),
    ("Shell",
     "Join the centre panels S-C-1 to S-C-{n} front to back. Add the S-R1 and S-L1 panels alongside{shell_out}, "
     "then the side plates S-SIDE-R and S-SIDE-L. Each panel sets its neighbours' angles, so the shape locks "
     "in as the rows close up."),
    ("Try the shell on",
     "Before going further, put the shell on. There should be about a finger's width all round. The "
     "front edge sits just above your brows."),
    ("Faceplate",
     "Join F-C-1 to F-C-{m}, then the F-R1 and F-L1 panels (with the eye slots){face_out}, then the pivot "
     "arms F-ARM-R and F-ARM-L."),
    ("Pivots",
     "Put a washer on a Chicago screw post and push it from the inside through the side plate. Add a washer, "
     "the arm and a washer, then screw on the cap with a drop of thread locker. Tighten until it bottoms out: "
     "that is as tight as it goes, and it stops before it crushes the board."),
    ("Overlays and finish",
     "Glue an E-RING round each screw cap, then the overlays (positions on the next page). Tape over the cut "
     "edges so the flutes don't show, prime, then paint. Swing the faceplate fully open and closed to check "
     "nothing rubs."),
]


def instructions(pdf, d, pieces, renders, page_no, total):
    pg = pdf.page()
    x0, W = P.MARGIN, P.PAGE_W - 2 * P.MARGIN
    sheets.header(pg, "CARDBOARD HELMET - HOW TO BUILD", f"{page_no} / {total}",
                  "Read once before cutting. Piece names are printed on every stencil.")
    y = P.PAGE_H - P.MARGIN - P.HEADER_H - 6
    colw = W * 0.56
    outwards = lambda g, p: "".join(f", then {p}-R{k} and {p}-L{k}" for k in range(2, (len(g.xs) - 2) // 2 + 1))
    rows = {"n": len(d.shell_grid.phis) - 1, "m": len(d.face_grid.phis) - 1,
            "shell_out": outwards(d.shell_grid, "S"), "face_out": outwards(d.face_grid, "F")}
    for k, (title, text) in enumerate(STEPS, 1):
        text = text.format(**rows)
        pg.text(f"{k}", (x0 + 2.5, y - 1.5), 5.0, bold=True, color=ACCENT)
        pg.text(title.upper(), (x0 + 9, y), 3.1, bold=True, anchor="start", vcenter=False)
        y = para(pg, x0 + 9, y - 4.6, colw - 9, text, size=2.75, lead=1.4) - 3.0

    xr = x0 + colw + 6
    wr = W - colw - 6
    imgs = [("side_closed", "closed (from the left)"), ("side_open", f"open {P.SWING_ANGLE:.0f} deg")]
    yi = P.PAGE_H - P.MARGIN - P.HEADER_H - 4
    for key, cap in imgs:
        if key in renders:
            path, pw, ph = renders[key]
            pg.image(open(path, "rb").read(), pw, ph, xr, yi - wr, wr, wr)
            pg.text(cap, (xr + wr / 2, yi - wr + 3), 2.6, color=GREY)
            yi -= wr + 3
    # pivot stack
    yi -= 4
    grip, post = pivot_post()
    pg.text("PIVOT STACK (section through the pivot)", (xr, yi), 2.8, bold=True, anchor="start",
            vcenter=False, color=ACCENT)
    pg.text(f"Chicago screw: {post} mm post, clamps {grip:g} mm", (xr, yi - 4.2), 2.3, anchor="start",
            vcenter=False, color=GREY)
    yi -= 12
    steel = (0.6, 0.6, 0.62)
    layers = [("washer", P.PIVOT_WASHER, steel), ("side plate", P.BOARD, (0.8, 0.3, 0.26)),
              ("washer", P.PIVOT_WASHER, steel), ("arm", P.BOARD, (0.86, 0.72, 0.4)),
              ("washer", P.PIVOT_WASHER, steel), ("E-RING", P.BOARD, (0.86, 0.72, 0.4))]
    sc = 2.6
    x_in = xr + 4 + P.PIVOT_HEAD * sc
    xx = x_in
    hgt = 26.0
    vs = hgt / (2 * P.EAR_DISC_R)                   # vertical scale across the arm's round end
    ring = (P.EAR_DISC_R - P.EAR_RING_IN) * vs
    for name, t, col in layers:
        h = P.PIVOT_WASHER_D * vs if name == "washer" else hgt
        if name == "E-RING":                        # hollow: the cap sits in the middle
            pg.rect(xx, yi - ring, t * sc, ring, fill=col, width=0.1)
            pg.rect(xx, yi - hgt, t * sc, ring, fill=col, width=0.1)
        else:
            pg.rect(xx, yi - hgt / 2 - h / 2, t * sc, h, fill=col, width=0.1)
        pg.text(name, (xx + t * sc / 2, yi - hgt - 3.5), 2.1, angle=-60, anchor="start")
        xx += t * sc
    x_out = x_in + grip * sc
    head, hh, dark = P.PIVOT_HEAD * sc, 12.0 * vs, (0.45, 0.45, 0.47)
    pg.rect(x_in, yi - hgt / 2 - 1.2, x_out - x_in, 2.4, fill=dark, width=0.1)
    pg.rect(x_in - head, yi - hgt / 2 - hh / 2, head, hh, fill=dark, width=0.1)
    pg.rect(x_out, yi - hgt / 2 - hh / 2, head, hh, fill=dark, width=0.1)
    pg.text("inside (ear)", (xr - 1, yi + 2.5), 2.1, anchor="start", color=GREY)
    pg.text("outside", (xx, yi + 2.5), 2.1, anchor="end", color=GREY)
    sheets.footer(pg, "GUIDE - DO NOT CUT")
    return pg


def piece_map(pdf, d, pieces, page_no, total):
    pg = pdf.page()
    sheets.header(pg, "CARDBOARD HELMET - WHERE EACH PIECE GOES", f"{page_no} / {total}",
                  "Outside views of the finished helmet. Seam numbers on the stencils say exactly which "
                  "edges meet.")
    x0, W = P.MARGIN, P.PAGE_W - 2 * P.MARGIN
    top = P.PAGE_H - P.MARGIN - P.HEADER_H - 2
    bottom = P.MARGIN + P.FOOTER_H + 2
    cw = (W - 6) / 2
    ch = (top - bottom - 26) / 2
    closed = piece_layers(d, pieces)
    views = [
        ("FRONT", (0, 1, 0), closed),
        ("LEFT SIDE", (-1, 0, 0), closed),
        ("BACK", (0, -1, 0), closed),
        ("TOP (front is down)", (0, 0, -1), closed),
    ]
    for k, (title, dvec, layers) in enumerate(views):
        cx = x0 + (k % 2) * (cw + 6)
        cy = top - (k // 2 + 1) * (ch + 8) + 2
        up = (0, -1, 0) if title.startswith("TOP") else (0, 0, 1)
        draw_view(pg, layers, dvec, up, (cx, cy, cw, ch - 2))
        pg.text(title, (cx + cw / 2, cy + ch + 1.5), 3.0, bold=True, color=ACCENT)
    # overlay positions
    spots = []
    for name, gname, poly in d.overlays:
        g = d.face_grid if gname == "face" else d.shell_grid
        i, j = g.cell_of(*np.mean(np.asarray(poly, float), axis=0))
        spots.append(f"{name} on {'F' if gname == 'face' else 'S'}-{column_names(len(g.xs) - 1)[i]}-{j + 1}")
    para(pg, x0, bottom + 7, W, "Overlays (centred on their panel): " + ", ".join(spots) + ".",
         size=2.4, color=GREY)
    sheets.footer(pg, "GUIDE - DO NOT CUT")
    return pg
