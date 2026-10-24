"""Cover, instructions and assembly-map pages (all vector, plus renders)."""

import math
import numpy as np

from . import params as P
from . import sheets
from .pdf import text_width
from .sheets import INK, GREY, ACCENT

RED = (0.70, 0.16, 0.13)
GREEN = (0.10, 0.48, 0.22)


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
            elif f.get("paint") == "red":
                base = (0.88, 0.45, 0.38) if (f.get("cell") or (0,))[0] % 2 else (0.93, 0.55, 0.48)
            else:
                base = (0.98, 0.87, 0.55) if (f.get("cell") or (0,))[0] % 2 else (0.99, 0.94, 0.75)
            cols.append(base)
        layers.append((verts, faces, cols, labs))
    return layers


# --- pages ----------------------------------------------------------------------
def cover(pdf, d, renders, checks, total, n_sheets):
    pg = pdf.page()
    x0, W = P.MARGIN, P.PAGE_W - 2 * P.MARGIN
    top = P.PAGE_H - P.MARGIN
    pg.text("CARDBOARD HELMET", (x0, top - 8), 8.0, bold=True, anchor="start", vcenter=False)
    pg.text(f"1 / {total}", (P.PAGE_W - P.MARGIN, top - 4), 3.2, bold=True, anchor="end", color=ACCENT)
    pg.text("Flip-up faceplate on ear pivots  |  numbered glue tabs  |  full-size stencils from a 3D model",
            (x0, top - 14), 3.2, anchor="start", vcenter=False, color=GREY)
    s = d.summary()
    pg.text(f"Fitted to: forehead {P.HEAD_CIRCUMFERENCE / 10:.1f} cm around, chin-over-top-to-nape "
            f"{P.CHIN_TOP_NAPE / 10:.1f} cm, {P.BOARD} mm board. Outside: about "
            f"{2 * (d.arm_x + 3 * P.BOARD):.0f} mm wide.",
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
    yl = para(pg, x0, yl, colw,
              f"- {P.BOARD} mm chipboard (mat board / sketch-pad backing): {n_sheets} Letter-size pieces\n"
              "- Glue stick or spray adhesive (to stick the printouts on)\n"
              "- PVA or hot glue for the tabs; craft knife, steel ruler, cutting mat\n"
              "- Empty ballpoint pen for scoring\n"
              f"- 2 Chicago screws (binding posts) {P.PIVOT_HOLE:.0f} mm x 10 mm, 2 thin washers\n"
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
     "Glue each printout onto board with the printed side up. The printed side becomes the INSIDE of the "
     "helmet, so the outside stays clean for paint. Cut every solid line, including the three outer sides "
     "of each grey tab, the eye and mouth slots and the pivot holes."),
    ("Score and pre-fold",
     "Dashed lines are valley folds: run an empty ballpoint along them on the printed side and fold "
     "towards you (printed side inside the fold). Dash-dot lines are mountain folds: score lightly with "
     "the knife (about a third of the way through) and fold away from you. Fold the base of every tab."),
    ("Glue tabs",
     "Every seam has a number printed on both pieces. The piece with the grey tab goes UNDER the edge "
     "with the same number: glue the tab, slide it under, press from outside until set. Edges meet flush."),
    ("Shell strips",
     "Join S-C-1, S-C-2 and S-C-3 into the centre strip (front at S-C-1). Add S-R1 and S-L1 strips on "
     "each side, then R2/L2, then R3/L3, working front to back. Each strip's tabs lie along its inner "
     "edge. Finish with the side plates S-SIDE-R and S-SIDE-L."),
    ("Try the shell on",
     "Before going further, put the shell on. There should be about a finger's width all round. The "
     "front edge sits just above your brows."),
    ("Faceplate",
     "Build F-C, then F-R1 and F-L1, F-R2 and F-L2 (the eye slots line up across the seam), F-R3 and "
     "F-L3, then glue the pivot arms F-ARM-R and F-ARM-L onto the outer strips."),
    ("Pivots",
     "Glue an E-SIDE disc outside each side plate and an E-DISC outside each arm, holes aligned; add two "
     "E-RINGs on each E-DISC. Push a Chicago screw from inside through side plate, E-SIDE, washer, arm, "
     "E-DISC and rings, and screw on the cap. Tighten until the faceplate stays where you put it."),
    ("Overlays and finish",
     "Glue O-BROW, O-CHEEK and O-CHIN onto the faceplate and O-CREST along the back centre strip "
     "(positions are shown on the next page). Seal with PVA or primer. Paint the shell red, the face "
     "gold, and the faceplate facets marked RED on the stencils red: they form the jaw and cheek "
     "plates (see the piece map). Swing the faceplate fully open and closed to check nothing rubs."),
]


def instructions(pdf, d, pieces, renders, page_no, total):
    pg = pdf.page()
    x0, W = P.MARGIN, P.PAGE_W - 2 * P.MARGIN
    sheets.header(pg, "CARDBOARD HELMET - HOW TO BUILD", f"{page_no} / {total}",
                  "Read once before cutting. Piece names are printed on every stencil.")
    y = P.PAGE_H - P.MARGIN - P.HEADER_H - 6
    colw = W * 0.56
    for k, (title, text) in enumerate(STEPS, 1):
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
    pg.text("PIVOT STACK (section through the ear disc)", (xr, yi), 2.8, bold=True, anchor="start",
            vcenter=False, color=ACCENT)
    yi -= 6
    layers = [("side plate", P.BOARD, (0.8, 0.3, 0.26)), ("E-SIDE", P.BOARD, (0.8, 0.3, 0.26)),
              ("washer", P.PIVOT_WASHER, (0.6, 0.6, 0.62)), ("arm", P.BOARD, (0.86, 0.72, 0.4)),
              ("E-DISC", P.BOARD, (0.86, 0.72, 0.4)), ("E-RING x2", 2 * P.BOARD, (0.86, 0.72, 0.4))]
    sc = 3.2
    xx = xr + 4
    hgt = 26.0
    for name, t, col in layers:
        pg.rect(xx, yi - hgt, t * sc, hgt, fill=col, width=0.1)
        pg.text(name, (xx + t * sc / 2, yi - hgt - 3.5), 2.1, angle=-60, anchor="start")
        xx += t * sc
    pg.rect(xr + 1, yi - hgt / 2 - 1.2, xx - xr + 2, 2.4, fill=(0.5, 0.5, 0.52), width=0.1)
    pg.text("Chicago screw", (xx + 4, yi - hgt / 2), 2.3, anchor="start")
    pg.text("inside (ear)", (xr - 1, yi + 2.5), 2.1, anchor="start", color=GREY)
    pg.text("outside", (xx - 2, yi + 2.5), 2.1, anchor="end", color=GREY)
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
    # overlay positions note
    pg.text("Overlays: O-BROW-L/R along the faceplate top band, O-CHEEK-L/R below the eye slots on F-L2/F-R2, "
            "O-CHIN at the bottom of F-C, O-CREST-1/2 on the back centre strip.",
            (x0, bottom + 1), 2.4, anchor="start", vcenter=False, color=GREY)
    sheets.footer(pg, "GUIDE - DO NOT CUT")
    return pg
